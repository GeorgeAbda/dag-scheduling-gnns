"""
Ablation experiments for the GNN (GIN) agent architecture.

This script trains several agent variants where specific architectural
components are removed/modified, and compares performance across:
- Makespan
- Total energy
- Active energy
- Idle energy

Usage example (matches your requested dataset settings):

  python -m scheduler.rl_model.ablation_gnn \
    --exp_name gnn_ablation_linear \
    --dataset.dag_method linear \
    --dataset.gnp_min_n 12 \
    --dataset.gnp_max_n 24 \
    --dataset.workflow_count 10 \
    --dataset.host_count 4 \
    --dataset.vm_count 10 \
    --total_timesteps 200000 \
    --test_every_iters 10 \
    --test_iterations 4 \
    --csv_reward_tag only_energy

Outputs:
- Per-variant CSVs under logs/<run>/ablation/per_variant/ (separate from the global csv/ directory)
- A combined summary CSV at logs/<run>/ablation/summary.csv
- Comparison plots at logs/<run>/ablation/:
  - bars_makespan.png, bars_total_energy.png, bars_active_energy.png, bars_idle_energy.png, bars_active_plus_idle.png
  - lines_makespan.png, lines_total_energy.png, lines_active_energy.png, lines_idle_energy.png, lines_active_plus_idle.png

Includes both GIN and attention-based GATv2 graph encoders in the ablation variants.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace as _dc_replace
import csv as _csv
from pathlib import Path
from typing import Optional, Tuple, List
import os
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import softmax
from torch.distributions.categorical import Categorical

from torch_geometric.nn.models import GIN
from torch_geometric.nn.conv import (
    GATv2Conv,
    SAGEConv,
    PNAConv,
    TransformerConv,
    NNConv,
    EdgeConv,
)
from torch_geometric.nn import HeteroConv
from torch_geometric.nn.glob import global_mean_pool, GlobalAttention

import gymnasium as gym
import tyro
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Local imports via path-relative strategy (consistent with train.py)
import sys as _sys
_grandparent_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
if _grandparent_dir not in _sys.path:
    _sys.path.insert(0, _grandparent_dir)

from cogito.config.settings import MIN_TESTING_DS_SEED, MAX_OBS_SIZE
from cogito.dataset_generator.gen_dataset import DatasetArgs
from cogito.gnn_deeprl_model.core.env.gym_env import CloudSchedulingGymEnvironment
from cogito.gnn_deeprl_model.agents.agent import Agent
from cogito.gnn_deeprl_model.agents.gin_agent.mapper import GinAgentMapper, GinAgentObsTensor


# --------------------------------------------------------------------------------------
# Ablation flags and variants
# --------------------------------------------------------------------------------------

@dataclass
class AblationVariant:
    name: str
    use_batchnorm: bool = True
    use_task_dependencies: bool = True
    use_actor_global_embedding: bool = True
    gin_num_layers: int = 2  # used when graph_type == "gin"
    mlp_only: bool = False   # if True, skip graph network entirely (identity)
    # Graph architecture selection
    graph_type: str = "gin"  # one of {"gin", "gatv2", "sage", "pna", "transformer", "hetero", "nnconv", "edgeconv"}
    gat_heads: int = 4       # used when graph_type == "gatv2"
    gat_dropout: float = 0.0 # attention dropout for GATv2
    # PNA options
    pna_aggregators: Tuple[str, ...] = ("mean", "min", "max", "std")
    pna_scalers: Tuple[str, ...] = ("identity", "amplification", "attenuation")
    # Residuals / norms / activations
    use_residual: bool = True
    use_layernorm: bool = False
    use_preactivation: bool = False
    # Global pooling
    use_attention_pool: bool = False
    # Hetero specific
    hetero_base: str = "sage"  # base conv inside HeteroConv
    # Edge features for edge-aware MPNNs
    use_edge_features: bool = True
    # Low-pass regularization knob (Dirichlet smoothness on node embeddings)
    lowpass_reg_lambda: float = 0.0
    # When True, cache node embeddings from actor.forward() to reuse for low-pass penalty (experiment only)
    cache_lowpass_from_forward: bool = False


def _maybe_bn(use_bn: bool, dim: int) -> nn.Module:
    return nn.BatchNorm1d(dim) if use_bn else nn.Identity()


class AblationBaseNetwork(nn.Module):
    def __init__(self, hidden_dim: int, embedding_dim: int, device: torch.device,
                 num_layers: int = 2, use_batchnorm: bool = True,
                 use_task_dependencies: bool = True,
                 graph_type: str = "gin",
                 gat_heads: int = 4,
                 gat_dropout: float = 0.0,
                 pna_aggregators: Tuple[str, ...] = ("mean", "min", "max", "std"),
                 pna_scalers: Tuple[str, ...] = ("identity", "amplification", "attenuation"),
                 use_residual: bool = True,
                 use_layernorm: bool = False,
                 use_preactivation: bool = False,
                 hetero_base: str = "sage",
                 use_edge_features: bool = True,
                 use_attention_pool: bool = False) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.device = device
        self.use_task_dependencies = use_task_dependencies
        self.graph_type = graph_type
        self.use_residual = use_residual
        self.use_layernorm = use_layernorm
        self.use_preactivation = use_preactivation
        self.use_edge_features = use_edge_features
        self.use_attention_pool = use_attention_pool

        self.task_encoder = nn.Sequential(
            nn.Linear(6, hidden_dim),
            _maybe_bn(use_batchnorm, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            _maybe_bn(use_batchnorm, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        ).to(self.device)
        self.vm_encoder = nn.Sequential(
            nn.Linear(10, hidden_dim),
            _maybe_bn(use_batchnorm, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            _maybe_bn(use_batchnorm, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        ).to(self.device)

        self.use_gnn = num_layers > 0
        self._gat_layers: nn.ModuleList | None = None
        self._sage_layers: nn.ModuleList | None = None
        self._pna_layers: nn.ModuleList | None = None
        self._trans_layers: nn.ModuleList | None = None
        self._hetero: HeteroConv | None = None
        self._nnconv_layers: nn.ModuleList | None = None
        self._edgeconv_layers: nn.ModuleList | None = None
        self._layernorm: nn.Module | None = (nn.LayerNorm(embedding_dim) if use_layernorm else None)
        self._attn_pool: GlobalAttention | None = (
            GlobalAttention(nn.Sequential(nn.Linear(embedding_dim, 1))) if use_attention_pool else None
        )
        if self._attn_pool is not None:
            self._attn_pool = self._attn_pool.to(self.device)
        if not self.use_gnn:
            self.graph_network = nn.Identity()
        else:
            if graph_type == "gin":
                self.graph_network = GIN(
                    in_channels=embedding_dim,
                    hidden_channels=hidden_dim,
                    num_layers=num_layers,
                    out_channels=embedding_dim,
                ).to(self.device)
            elif graph_type == "gatv2":
                # Build a simple GATv2 stack that preserves embedding_dim
                heads = max(1, int(gat_heads))
                per_head_out = max(1, embedding_dim // heads)
                # If not divisible, fallback to 1 head
                if heads * per_head_out != embedding_dim:
                    heads, per_head_out = 1, embedding_dim
                in_dim = embedding_dim
                layers = []
                for _ in range(num_layers):
                    layers.append(
                        GATv2Conv(
                            in_channels=in_dim,
                            out_channels=per_head_out,
                            heads=heads,
                            dropout=float(gat_dropout),
                            add_self_loops=True,
                            concat=True,
                            edge_dim=None,
                        )
                    )
                    layers.append(nn.ELU())
                    in_dim = heads * per_head_out
                self._gat_layers = nn.ModuleList(layers).to(self.device)
                self.graph_network = nn.Identity()  # not used; handled in forward
            elif graph_type == "sage":
                layers = []
                in_dim = embedding_dim
                for _ in range(num_layers):
                    conv = SAGEConv(in_dim, embedding_dim)
                    layers += [conv, nn.ReLU()]
                self._sage_layers = nn.ModuleList(layers).to(self.device)
                self.graph_network = nn.Identity()
            elif graph_type == "pna":
                # Degree scalers need deg stats; approximate with fixed vector for simplicity
                deg = torch.tensor([1, 2, 3, 4, 8, 16], dtype=torch.long)
                layers = []
                for _ in range(num_layers):
                    conv = PNAConv(
                        in_channels=embedding_dim,
                        out_channels=embedding_dim,
                        aggregators=list(pna_aggregators),
                        scalers=list(pna_scalers),
                        deg=deg,
                    )
                    layers += [conv, nn.ReLU()]
                self._pna_layers = nn.ModuleList(layers).to(self.device)
                self.graph_network = nn.Identity()
            elif graph_type == "transformer":
                layers = []
                for _ in range(num_layers):
                    conv = TransformerConv(embedding_dim, embedding_dim // 2, heads=2, dropout=float(gat_dropout))
                    layers += [conv, nn.ReLU()]
                self._trans_layers = nn.ModuleList(layers).to(self.device)
                self.graph_network = nn.Identity()
            elif graph_type == "hetero":
                # Build hetero conv with a base conv per relation
                def _base():
                    if hetero_base == "sage":
                        return SAGEConv((-1, -1), embedding_dim)
                    elif hetero_base == "gatv2":
                        return GATv2Conv((-1, -1), embedding_dim // 2, heads=2)
                    else:
                        return SAGEConv((-1, -1), embedding_dim)
                conv_dict = {
                    ("task", "dep", "task"): _base(),
                    ("task", "assign", "vm"): _base(),
                    ("vm", "rev_assign", "task"): _base(),
                }
                self._hetero = HeteroConv(conv_dict, aggr="sum").to(self.device)
                self.graph_network = nn.Identity()
            elif graph_type == "nnconv":
                # edge-conditioned NNConv (requires edge_attr)
                layers = []
                net = nn.Sequential(nn.Linear(1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, embedding_dim * embedding_dim))
                for _ in range(num_layers):
                    layers += [NNConv(embedding_dim, embedding_dim, net), nn.ReLU()]
                self._nnconv_layers = nn.ModuleList(layers).to(self.device)
                self.graph_network = nn.Identity()
            elif graph_type == "edgeconv":
                layers = []
                # EdgeConv uses an MLP on (x_i || x_j - x_i)
                mlp = nn.Sequential(nn.Linear(2 * embedding_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, embedding_dim))
                for _ in range(num_layers):
                    layers += [EdgeConv(mlp), nn.ReLU()]
                self._edgeconv_layers = nn.ModuleList(layers).to(self.device)
                self.graph_network = nn.Identity()
            else:
                # Unknown type -> identity
                self.graph_network = nn.Identity()

    def forward(self, obs: GinAgentObsTensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_tasks = obs.task_state_scheduled.shape[0]
        num_vms = obs.vm_completion_time.shape[0]

        max_vm_cores = max(obs.vm_cpu_cores.max().item(), 1.0)

        task_features = [
            obs.task_state_scheduled,
            obs.task_state_ready,
            obs.task_length,
            obs.task_completion_time,
            obs.task_memory_req_mb / 1000.0,
            obs.task_cpu_req_cores / max_vm_cores,
        ]
        vm_features = [
            obs.vm_completion_time,
            1 / (obs.vm_speed + 1e-8),
            obs.vm_energy_rate,
            obs.vm_memory_mb / 1000.0,
            obs.vm_available_memory_mb / 1000.0,
            obs.vm_used_memory_fraction,
            obs.vm_active_tasks_count,
            obs.vm_cpu_cores / max_vm_cores,
            obs.vm_available_cpu_cores / max_vm_cores,
            obs.vm_used_cpu_fraction_cores,
        ]

        task_x = torch.stack(task_features, dim=-1)
        vm_x = torch.stack(vm_features, dim=-1)
        task_h: torch.Tensor = self.task_encoder(task_x)
        vm_h: torch.Tensor = self.vm_encoder(vm_x)

        # Build edges (homogeneous: shift VM indices; hetero: keep unshifted)
        task_vm_edges = obs.compatibilities.clone()
        task_vm_edges[1] = task_vm_edges[1] + num_tasks  # shifted for homogeneous graph
        hetero_task_vm_edges = obs.compatibilities.clone()  # unshifted for HeteroConv

        node_x = torch.cat([task_h, vm_h])
        if self.use_task_dependencies and obs.task_dependencies.numel() > 0:
            edge_index = torch.cat([task_vm_edges, obs.task_dependencies], dim=-1)
        else:
            edge_index = task_vm_edges

        batch = torch.zeros(num_tasks + num_vms, dtype=torch.long, device=self.device)
        if isinstance(self.graph_network, nn.Identity) and all(m is None for m in [self._gat_layers, self._sage_layers, self._pna_layers, self._trans_layers, self._hetero, self._nnconv_layers, self._edgeconv_layers]):
            node_embeddings = node_x
        elif self._gat_layers is not None and self.use_gnn and self.graph_type == "gatv2":
            h = node_x
            # Apply conv and activation alternately
            for layer in self._gat_layers:
                if isinstance(layer, GATv2Conv):
                    h = layer(h, edge_index)
                else:
                    h = layer(h)
            node_embeddings = h
        elif self._sage_layers is not None and self.use_gnn and self.graph_type == "sage":
            h = node_x
            for layer in self._sage_layers:
                if isinstance(layer, SAGEConv):
                    h_res = h
                    h = layer(h, edge_index)
                    if self.use_residual:
                        h = h + h_res
                else:
                    if self.use_preactivation:
                        h = layer(h)
                    else:
                        h = layer(h)
                if self._layernorm is not None:
                    h = self._layernorm(h)
            node_embeddings = h
        elif self._pna_layers is not None and self.use_gnn and self.graph_type == "pna":
            h = node_x
            for layer in self._pna_layers:
                if isinstance(layer, PNAConv):
                    h_res = h
                    h = layer(h, edge_index)
                    if self.use_residual:
                        h = h + h_res
                else:
                    h = layer(h)
                if self._layernorm is not None:
                    h = self._layernorm(h)
            node_embeddings = h
        elif self._trans_layers is not None and self.use_gnn and self.graph_type == "transformer":
            h = node_x
            for layer in self._trans_layers:
                if isinstance(layer, TransformerConv):
                    h_res = h
                    h = layer(h, edge_index)
                    if self.use_residual:
                        h = h + h_res
                else:
                    h = layer(h)
                if self._layernorm is not None:
                    h = self._layernorm(h)
            node_embeddings = h
        elif self._hetero is not None and self.use_gnn and self.graph_type == "hetero":
            # Build hetero inputs
            x_dict = {"task": task_h, "vm": vm_h}
            edge_index_dict = {}
            if obs.task_dependencies.numel() > 0 and self.use_task_dependencies:
                edge_index_dict[("task", "dep", "task")] = obs.task_dependencies
            edge_index_dict[("task", "assign", "vm")] = hetero_task_vm_edges
            # Add reverse relation vm <- task
            rev_edges = torch.stack([hetero_task_vm_edges[1], hetero_task_vm_edges[0]], dim=0)
            edge_index_dict[("vm", "rev_assign", "task")] = rev_edges
            # Provide explicit sizes for bipartite relations via size_dict
            hetero_sizes: dict = {}
            if ("task", "dep", "task") in edge_index_dict:
                hetero_sizes[("task", "dep", "task")] = (num_tasks, num_tasks)
            if ("task", "assign", "vm") in edge_index_dict:
                hetero_sizes[("task", "assign", "vm")] = (num_tasks, num_vms)
            if ("vm", "rev_assign", "task") in edge_index_dict:
                hetero_sizes[("vm", "rev_assign", "task")] = (num_vms, num_tasks)
            out_dict = self._hetero(x_dict, edge_index_dict=edge_index_dict, size_dict=hetero_sizes)
            # Concat to homogeneous ordering [tasks, vms]
            node_embeddings = torch.cat([out_dict["task"], out_dict["vm"]], dim=0)
        elif self._nnconv_layers is not None and self.use_gnn and self.graph_type == "nnconv":
            # Build simple edge attributes: compat edges get est runtime (task_len / vm_speed), deps get 1.0
            h = node_x
            # compat edge attrs
            t_idx = task_vm_edges[0]
            v_idx = task_vm_edges[1] - num_tasks
            est_runtime = (task_x[t_idx, 2] / (vm_x[v_idx, 1] + 1e-8)).unsqueeze(-1)  # task_length / inv_speed
            edge_attr = est_runtime.clamp_min(0.0)
            for layer in self._nnconv_layers:
                if isinstance(layer, NNConv):
                    h_res = h
                    h = layer(h, task_vm_edges, edge_attr)
                    if self.use_residual:
                        h = h + h_res
                else:
                    h = layer(h)
            node_embeddings = h
            edge_index = task_vm_edges  # ensure edge_embeddings computed below align
        elif self._edgeconv_layers is not None and self.use_gnn and self.graph_type == "edgeconv":
            # EdgeConv requires k-NN style graph; we reuse existing edges
            h = node_x
            for layer in self._edgeconv_layers:
                if isinstance(layer, EdgeConv):
                    h_res = h
                    h = layer(h, edge_index)
                    if self.use_residual:
                        h = h + h_res
                else:
                    h = layer(h)
            node_embeddings = h
        else:
            node_embeddings = self.graph_network(node_x, edge_index=edge_index)
        edge_embeddings = torch.cat([node_embeddings[edge_index[0]], node_embeddings[edge_index[1]]], dim=1)
        if self._attn_pool is not None:
            # attention gate over nodes
            graph_embedding = self._attn_pool(node_embeddings, batch)
        else:
            graph_embedding = global_mean_pool(node_embeddings, batch=batch)

        return node_embeddings, edge_embeddings, graph_embedding


class AblationActor(nn.Module):
    def __init__(self, hidden_dim: int, embedding_dim: int, device: torch.device,
                 num_layers: int, use_batchnorm: bool,
                 use_task_dependencies: bool,
                 use_actor_global_embedding: bool,
                 mlp_only: bool = False,
                 graph_type: str = "gin",
                 gat_heads: int = 4,
                 gat_dropout: float = 0.0,
                 pna_aggregators: Tuple[str, ...] = ("mean", "min", "max", "std"),
                 pna_scalers: Tuple[str, ...] = ("identity", "amplification", "attenuation"),
                 use_residual: bool = True,
                 use_layernorm: bool = False,
                 use_preactivation: bool = False,
                 hetero_base: str = "sage",
                 use_edge_features: bool = True,
                 use_attention_pool: bool = False):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.device = device
        self.use_actor_global_embedding = use_actor_global_embedding
        self.mlp_only = mlp_only

        self.network = AblationBaseNetwork(
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            device=device,
            num_layers=0 if mlp_only else num_layers,
            use_batchnorm=use_batchnorm,
            use_task_dependencies=use_task_dependencies,
            graph_type=graph_type,
            gat_heads=gat_heads,
            gat_dropout=gat_dropout,
            pna_aggregators=pna_aggregators,
            pna_scalers=pna_scalers,
            use_residual=use_residual,
            use_layernorm=use_layernorm,
            use_preactivation=use_preactivation,
            hetero_base=hetero_base,
            use_edge_features=use_edge_features,
            use_attention_pool=use_attention_pool,
        )
        in_dim = (2 + (1 if use_actor_global_embedding else 0)) * embedding_dim
        self.edge_scorer = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            _maybe_bn(use_batchnorm, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            _maybe_bn(use_batchnorm, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        ).to(self.device)

    def forward(self, obs: GinAgentObsTensor) -> torch.Tensor:
        num_tasks = obs.task_completion_time.shape[0]
        num_vms = obs.vm_completion_time.shape[0]

        node_embeddings, edge_embeddings, graph_embedding = self.network(obs)

        # Optional: cache node embeddings for low-pass penalty reuse in PPO update
        if getattr(self, "cache_lowpass_from_forward", False):
            if not hasattr(self, "_cached_batch_node_embeddings"):
                self._cached_batch_node_embeddings: list[torch.Tensor] = []
            # Store with gradient to allow regularizer to backprop
            self._cached_batch_node_embeddings.append(node_embeddings)

        if self.use_actor_global_embedding:
            rep_graph_embedding = graph_embedding.expand(edge_embeddings.shape[0], self.embedding_dim)
            edge_embeddings = torch.cat([edge_embeddings, rep_graph_embedding], dim=1)

        scores: torch.Tensor = self.edge_scorer(edge_embeddings).flatten()
        # Keep only task-vm edge scores (first E entries)
        scores = scores[: obs.compatibilities.shape[1]]

        action_scores = torch.ones((num_tasks, num_vms), dtype=torch.float32).to(self.device) * -1e8
        action_scores[obs.compatibilities[0], obs.compatibilities[1]] = scores
        action_scores[obs.task_state_ready == 0, :] = -1e8
        action_scores[obs.task_state_scheduled == 1, :] = -1e8
        return action_scores


class AblationCritic(nn.Module):
    def __init__(self, hidden_dim: int, embedding_dim: int, device: torch.device,
                 num_layers: int, use_batchnorm: bool,
                 use_task_dependencies: bool,
                 graph_type: str = "gin",
                 gat_heads: int = 4,
                 gat_dropout: float = 0.0,
                 pna_aggregators: Tuple[str, ...] = ("mean", "min", "max", "std"),
                 pna_scalers: Tuple[str, ...] = ("identity", "amplification", "attenuation"),
                 use_residual: bool = True,
                 use_layernorm: bool = False,
                 use_preactivation: bool = False,
                 hetero_base: str = "sage",
                 use_edge_features: bool = True,
                 use_attention_pool: bool = True):
        super().__init__()
        self.device = device
        self.network = AblationBaseNetwork(
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            device=device,
            num_layers=num_layers,
            use_batchnorm=use_batchnorm,
            use_task_dependencies=use_task_dependencies,
            graph_type=graph_type,
            gat_heads=gat_heads,
            gat_dropout=gat_dropout,
            pna_aggregators=pna_aggregators,
            pna_scalers=pna_scalers,
            use_residual=use_residual,
            use_layernorm=use_layernorm,
            use_preactivation=use_preactivation,
            hetero_base=hetero_base,
            use_edge_features=use_edge_features,
            use_attention_pool=use_attention_pool,
        )
        self.graph_scorer = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        ).to(self.device)

    def forward(self, obs: GinAgentObsTensor) -> torch.Tensor:
        _, _, graph_embedding = self.network(obs)
        return self.graph_scorer(graph_embedding.unsqueeze(0))


# --------------------------------------------------------------------------------------
# Low-pass regularization helpers
# --------------------------------------------------------------------------------------

def _build_edge_index_from_obs(obs: GinAgentObsTensor, use_task_deps: bool) -> torch.Tensor:
    """Construct homogeneous edge_index [2, E] from compatibilities (+ optional task deps)."""
    num_tasks = obs.task_state_scheduled.shape[0]
    task_vm_edges = obs.compatibilities.clone()
    task_vm_edges[1] = task_vm_edges[1] + num_tasks  # shift VM indices
    if use_task_deps and obs.task_dependencies.numel() > 0:
        edge_index = torch.cat([task_vm_edges, obs.task_dependencies], dim=-1)
    else:
        edge_index = task_vm_edges
    return edge_index


def _normalized_laplacian(edge_index: torch.Tensor, num_nodes: int, device: torch.device) -> torch.Tensor:
    """Compute L = I - D^{-1/2} A D^{-1/2} from edge_index (undirected assumed by symmetrizing)."""
    if edge_index.numel() == 0:
        return torch.eye(num_nodes, device=device)
    src = edge_index[0].long()
    dst = edge_index[1].long()
    # Symmetrize
    src_all = torch.cat([src, dst], dim=0)
    dst_all = torch.cat([dst, src], dim=0)
    # Build degree
    deg = torch.bincount(src_all, minlength=num_nodes).float()
    deg += torch.bincount(dst_all, minlength=num_nodes).float()
    deg = deg.clamp_min(1.0)
    inv_sqrt_deg = deg.pow(-0.5)
    # Sparse-like multiply without building sparse tensor: accumulate normalized adjacency
    A_norm = torch.zeros((num_nodes, num_nodes), device=device)
    vals = inv_sqrt_deg[src_all] * inv_sqrt_deg[dst_all]
    A_norm.index_put_((src_all, dst_all), vals, accumulate=True)
    I = torch.eye(num_nodes, device=device)
    L = I - A_norm
    return L


def _dirichlet_energy(H: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """Return tr(H^T L H). H: [N, D], L: [N, N]."""
    return torch.trace(H.t().matmul(L).matmul(H))


class AblationGinAgent(Agent, nn.Module):
    def __init__(self, device: torch.device, variant: AblationVariant,
                 hidden_dim: int = 32, embedding_dim: int = 16) -> None:
        super().__init__()
        self.device = device
        self.mapper = GinAgentMapper(MAX_OBS_SIZE)
        self.variant = variant

        # Patch actor to carry graph_type configuration
        actor = AblationActor(
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            device=device,
            num_layers=variant.gin_num_layers if variant.graph_type == "gin" else variant.gin_num_layers,
            use_batchnorm=variant.use_batchnorm,
            use_task_dependencies=variant.use_task_dependencies,
            use_actor_global_embedding=variant.use_actor_global_embedding,
            mlp_only=variant.mlp_only,
            graph_type=variant.graph_type,
            gat_heads=variant.gat_heads,
            gat_dropout=variant.gat_dropout,
        )
        # Enable optional caching for experiment
        try:
            actor.cache_lowpass_from_forward = bool(variant.cache_lowpass_from_forward)
        except Exception:
            actor.cache_lowpass_from_forward = False
        self.actor = actor
        self.critic = AblationCritic(
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            device=device,
            num_layers=max(1, variant.gin_num_layers if not variant.mlp_only else 0),
            use_batchnorm=variant.use_batchnorm,
            use_task_dependencies=variant.use_task_dependencies,
            graph_type=variant.graph_type,
            gat_heads=variant.gat_heads,
            gat_dropout=variant.gat_dropout,
        )

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        values = []
        for b in range(x.shape[0]):
            obs = self.mapper.unmap(x[b])
            values.append(self.critic(obs))
        return torch.stack(values).to(self.device)

    def get_action_and_value(self, x: torch.Tensor, action: Optional[torch.Tensor] = None):
        x = x.to(self.device)
        all_actions, all_log_probs, all_entropies, all_values = [], [], [], []
        for b in range(x.shape[0]):
            obs = self.mapper.unmap(x[b])
            action_scores = self.actor(obs).flatten()
            probs_vec = softmax(action_scores, dim=0)

            # Valid mask (READY ∧ NOT-SCHEDULED) ∧ COMPATIBLE
            num_tasks = obs.task_state_ready.shape[0]
            num_vms = obs.vm_completion_time.shape[0]
            valid_task_mask = obs.task_state_ready.bool() & (obs.task_state_scheduled == 0)
            ready_mask_flat = valid_task_mask.repeat_interleave(num_vms)

            compat_mask_flat = torch.zeros(num_tasks * num_vms, dtype=torch.bool, device=self.device)
            if obs.compatibilities.numel() > 0:
                comp_t = obs.compatibilities[0].to(torch.long)
                comp_v = obs.compatibilities[1].to(torch.long)
                comp_flat_idx = comp_t * num_vms + comp_v
                good = (comp_flat_idx >= 0) & (comp_flat_idx < num_tasks * num_vms)
                compat_mask_flat[comp_flat_idx[good]] = True

            mask_flat = ready_mask_flat & compat_mask_flat

            mask_f = mask_flat.to(probs_vec.dtype)
            probs_vec = probs_vec * mask_f
            total = probs_vec.sum()
            if total.item() <= 0:
                if mask_flat.any():
                    probs_vec = mask_f / mask_f.sum()
                else:
                    probs_vec = torch.ones_like(probs_vec) / probs_vec.numel()
            dist = Categorical(probs_vec)
            chosen = action[b] if action is not None else dist.sample()
            value = self.critic(obs)

            all_actions.append(chosen)
            all_log_probs.append(dist.log_prob(chosen))
            all_entropies.append(dist.entropy())
            all_values.append(value)

        return (
            torch.stack(all_actions).to(self.device),
            torch.stack(all_log_probs).to(self.device),
            torch.stack(all_entropies).to(self.device),
            torch.stack(all_values).to(self.device),
        )


# --------------------------------------------------------------------------------------
# Diagnostics: structural overfitting checks
# --------------------------------------------------------------------------------------

def _clone_agent_with_overrides(src: AblationGinAgent, **overrides) -> AblationGinAgent:
    """Clone an agent architecture with variant overrides and load intersecting weights.
    Useful to test reliance on specific architectural features (e.g., task deps).
    """
    v = _dc_replace(src.variant, **overrides)
    dst = AblationGinAgent(src.device, v)
    # Load intersecting keys to avoid shape mismatches
    src_sd = src.state_dict()
    dst_sd = dst.state_dict()
    intersect = {k: v for k, v in src_sd.items() if k in dst_sd and dst_sd[k].shape == v.shape}
    dst_sd.update(intersect)
    dst.load_state_dict(dst_sd, strict=False)
    return dst


def _create_eval_config(base: Args, dag: str | None = None, size_scale: float | None = None) -> Args:
    cfg = _dc_replace(base)
    if dag is not None:
        cfg.dataset.dag_method = dag
    if size_scale is not None and size_scale != 1.0:
        # Scale DAG size only to avoid exploding action space and episode cost
        cfg.dataset.gnp_min_n = max(4, int(cfg.dataset.gnp_min_n * size_scale))
        cfg.dataset.gnp_max_n = max(cfg.dataset.gnp_min_n + 1, int(cfg.dataset.gnp_max_n * size_scale))
        # Keep workflow_count, host_count, vm_count unchanged
    return cfg


def run_structure_diagnostics(trained_agent: AblationGinAgent, base_args: Args, out_csv: Path) -> None:
    """Evaluate sensitivity to structural changes as an overfitting diagnostic.
    We compare:
      - In-domain (original eval config)
      - Size-shifted (+50% scale)
      - DAG-shifted (switch linear<->gnp)
      - No-dependencies agent (variant.use_task_dependencies=False)
    """
    rows = []
    headers = ['setting','makespan','total_energy','avg_active','avg_idle','active_plus_idle']

    def _safe_write():
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open('w', newline='') as f:
            w = _csv.writer(f)
            w.writerow(headers)
            w.writerows(rows)

    def _eval(agent: AblationGinAgent, args: Args, setting: str):
        try:
            mk, eobs, etot, mm = _test_agent(agent, args)
            ea = (mm or {}).get('avg_active_energy')
            ei = (mm or {}).get('avg_idle_energy')
            api = (ea or 0.0) + (ei or 0.0)
            rows.append([setting, mk, (etot if etot > 0 else eobs), ea or 0.0, ei or 0.0, api])
            # Incremental write to avoid loss if subsequent steps are slow/crash
            _safe_write()
        except Exception as e:
            print(f"[diagnostics] Warning: failed setting '{setting}': {e}")
    print("[diagnostics] Running for", base_args.dataset.dag_method)
    # In-domain
    _eval(trained_agent, base_args, 'in_domain')
    print("[diagnostics] In-domain done")
    # Size-shift (+50%)
    _eval(trained_agent, _create_eval_config(base_args, size_scale=1.5), 'size_shift_1p5x')
    print("[diagnostics] Size-shift done")
    # DAG-shift
    other = 'gnp' if base_args.dataset.dag_method == 'linear' else 'linear'
    _eval(trained_agent, _create_eval_config(base_args, dag=other), f'dag_shift_{other}')
    print("[diagnostics] DAG-shift done")
    # No-dependencies agent in-domain
    no_dep_agent = _clone_agent_with_overrides(trained_agent, use_task_dependencies=False)
    _eval(no_dep_agent, base_args, 'no_task_deps_in_domain')
    print("[diagnostics] No-dependencies done")

    # Final write (ensures file exists even if nothing ran)
    _safe_write()
    print(f"[diagnostics] wrote {out_csv}")


# --------------------------------------------------------------------------------------
# Training/Eval args and loop (adapted from train.py with agent injection)
# --------------------------------------------------------------------------------------

@dataclass
class Args:
    exp_name: str = "gnn_ablation"
    seed: int = 1
    output_dir: str = "logs"
    device: str = "cpu"
    torch_deterministic: bool = True
    torch_num_threads: int | None = None

    capture_video: bool = False
    env_mode: str = "async"

    # TensorBoard logging control (similar to train.py)
    no_tensorboard: bool = True
    # Generate offline plots from CSV metrics after training completes
    offline_plots_after_training: bool = True
    # Logging cadence controls (iterations)
    log_loss_every: int = 10
    grad_log_every: int = 10
    log_grad_norms: bool = True
    # Match CPU threads to number of envs (overridden if torch_num_threads is set)
    cpus_match_envs: bool = True

    # Sparse reward option: give only end-of-episode energy-based reward
    # If True, ignore step rewards; at termination, set reward = -(active+idle energy)
    sparse_reward: bool = False

    total_timesteps: int = 200_000
    learning_rate: float = 2.5e-4
    num_envs: int = 8
    num_steps: int = 256
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None

    test_every_iters: int = 10
    test_iterations: int = 4

    # Where to store ablation outputs (separate from global csv/)
    ablation_subdir: str = "ablation"

    dataset: DatasetArgs = field(
        default_factory=lambda: DatasetArgs(
            host_count=4,
            vm_count=10,
            workflow_count=10,
            gnp_min_n=12,
            gnp_max_n=24,
            max_memory_gb=10,
            min_cpu_speed=500,
            max_cpu_speed=5000,
            min_task_length=500,
            max_task_length=100_000,
            task_arrival="static",
            dag_method="linear",
        )
    )

    # runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0
    run_name: str = ""
    # Train-only-baseline switch (skip full ablation suite and heavy evals)
    train_only_baseline: bool = False
    # Train only a single variant by name (e.g., "no_global_actor"). If set, overrides train_only_baseline.
    train_only_variant: str | None = None
    # Skip per-variant final evaluation and summary write at the end of training
    skip_variant_summary_eval: bool = True
    # Skip the additional ablation evaluation sweeps (balanced/complex/constrained)
    skip_additional_eval: bool = True
    # Low-pass experiment controls (applied to constructed AblationVariant instances)
    variant_lowpass_reg_lambda: float = 0.0
    variant_cache_lowpass_from_forward: bool = False
    # Checkpointing frequency (iterations). 0 disables periodic checkpoints; initial/final/best still saved.
    checkpoint_every: int = 0


# Environment helpers

def _make_env(idx: int, args: Args):
    env: gym.Env = CloudSchedulingGymEnvironment(
        dataset_args=args.dataset,
        collect_timelines=False,
        compute_metrics=False,
        profile=False,
        fixed_env_seed=True,
    )
    # Reuse same wrapper as train.py to normalize obs/action mapping
    from cogito.gnn_deeprl_model.agents.gin_agent.wrapper import GinAgentWrapper
    env = GinAgentWrapper(env)
    from gymnasium.wrappers import RecordEpisodeStatistics
    return RecordEpisodeStatistics(env)


def _make_env_thunk(i: int, args: Args):
    def _thunk():
        return _make_env(i, args)
    return _thunk


def _make_test_env(args: Args):
    env: gym.Env = CloudSchedulingGymEnvironment(
        dataset_args=args.dataset,
        collect_timelines=False,
        compute_metrics=True,
        profile=False,
        fixed_env_seed=True,
    )
    from cogito.gnn_deeprl_model.agents.gin_agent.wrapper import GinAgentWrapper
    return GinAgentWrapper(env)


def _pick_device(choice: str) -> torch.device:
    choice = (choice or "auto").lower()
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if choice == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _test_agent(agent: Agent, args: Args):
    total_makespan = 0.0
    total_energy_obs = 0.0
    total_energy_full = 0.0
    total_active = 0.0
    total_idle = 0.0
    total_return = 0.0
    total_active_energy_return = 0.0
    total_makespan_return = 0.0

    # Evaluation seeds identical to training dataset seed
    seed_base = int(getattr(args.dataset, 'seed', MIN_TESTING_DS_SEED))
    for s in range(args.test_iterations):

        print("[diagnostics] Testing iteration", s)
        env = _make_test_env(args)
        # Use identical seed across episodes to keep configuration fixed
        next_obs, _ = env.reset(seed=seed_base)
        final_info: dict | None = None
        while True:
            obs_tensor = torch.from_numpy(np.asarray(next_obs, dtype=np.float32).reshape(1, -1))
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(obs_tensor)
            next_obs, rew, terminated, truncated, info = env.step(int(action.item()))
            try:
                total_return += float(rew)
            except Exception:
                pass
            if terminated or truncated:
                final_info = info
                break
        assert env.prev_obs is not None
        total_makespan += float(env.prev_obs.makespan())
        total_energy_obs += float(env.prev_obs.energy_consumption())
        if isinstance(final_info, dict):
            total_energy_full += float(final_info.get("total_energy", 0.0))
            total_active += float(final_info.get("total_energy_active", 0.0))
            total_idle += float(final_info.get("total_energy_idle", 0.0))
            # Aggregate episodic returns if provided by wrapper
            if "active_energy_return" in final_info:
                try:
                    total_active_energy_return += float(final_info["active_energy_return"])
                except Exception:
                    pass
            if "makespan_return" in final_info:
                try:
                    total_makespan_return += float(final_info["makespan_return"])
                except Exception:
                    pass
        env.close()

    n = max(1, args.test_iterations)
    return (
        total_makespan / n,
        total_energy_obs / n,
        total_energy_full / n,
        {
            "avg_active_energy": (total_active / n) if total_active > 0 else None,
            "avg_idle_energy": (total_idle / n) if total_idle > 0 else None,
            "avg_return": (total_return / n),
            "avg_active_energy_return": (total_active_energy_return / n) if total_active_energy_return != 0.0 else None,
            "avg_makespan_return": (total_makespan_return / n) if total_makespan_return != 0.0 else None,
        },
    )


def _train_one_variant(args: Args, variant: AblationVariant, device: torch.device, per_variant_dir: Path):
    run_tag = f"{variant.name}"
    print(f"[ablation] Training variant: {run_tag}")

    # Configure CPU threading: prefer explicit torch_num_threads; otherwise, match num_envs if enabled
    try:
        thread_count: int | None = None
        if args.torch_num_threads and int(args.torch_num_threads) > 0:
            thread_count = int(args.torch_num_threads)
        elif bool(getattr(args, 'cpus_match_envs', True)):
            thread_count = int(max(1, args.num_envs))
        if thread_count is not None:
            torch.set_num_threads(thread_count)
            # Export common BLAS/OMP envs to the same value for consistency
            os.environ.setdefault("OMP_NUM_THREADS", str(thread_count))
            os.environ.setdefault("MKL_NUM_THREADS", str(thread_count))
            os.environ.setdefault("OPENBLAS_NUM_THREADS", str(thread_count))
            os.environ.setdefault("NUMEXPR_NUM_THREADS", str(thread_count))
            os.environ.setdefault("TORCH_NUM_THREADS", str(thread_count))
            # Report CPU/threads configuration
            try:
                avail_cpus = os.cpu_count()
            except Exception:
                avail_cpus = None
            try:
                torch_threads = torch.get_num_threads()
            except Exception:
                torch_threads = thread_count
            try:
                interop_threads = torch.get_num_interop_threads()
            except Exception:
                interop_threads = None
            print(
                f"[ablation] CPU config: available_cpus={avail_cpus} | num_envs={args.num_envs} | "
                f"torch_num_threads={torch_threads} | torch_num_interop_threads={interop_threads} | "
                f"cpus_match_envs={getattr(args, 'cpus_match_envs', True)}"
            )
    except Exception as _e_threads:
        print(f"[ablation] Warning: failed to set CPU threads: {_e_threads}")

    batch_size = int(args.num_envs * args.num_steps)
    minibatch_size = max(1, int(batch_size // max(1, args.num_minibatches)))
    num_iterations = max(1, int(args.total_timesteps // max(1, batch_size)))

    # Ensure output directory exists early (for periodic checkpointing)
    per_variant_dir.mkdir(parents=True, exist_ok=True)

    envs = gym.vector.AsyncVectorEnv([_make_env_thunk(i, args) for i in range(args.num_envs)])
    obs_space = envs.single_observation_space
    act_space = envs.single_action_space
    assert isinstance(act_space, gym.spaces.Discrete)

    agent = AblationGinAgent(device, variant)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    obs = torch.zeros((args.num_steps, args.num_envs) + obs_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + act_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs_tensor = torch.Tensor(next_obs).to(device)
    next_done_tensor = torch.zeros(args.num_envs).to(device)

    pbar = tqdm(total=args.total_timesteps)
    # Periodic checkpointing frequency
    if getattr(args, "checkpoint_every", 0) and int(getattr(args, "checkpoint_every")) > 0:
        save_every = int(getattr(args, "checkpoint_every"))
    else:
        # Default: about 20 checkpoints across the run
        save_every = max(1, int(num_iterations // 20) or 1)
    
    # Best model tracking based on active+idle energy
    best_active_idle_energy = float('inf')
    best_model_state = None
    best_iteration = -1

    # Save initial model at iteration 0
    try:
        initial_ckpt_path = per_variant_dir / f"{variant.name}_iter00000.pt"
        torch.save(agent.state_dict(), initial_ckpt_path)
        print(f"[ablation] Saved initial model at iteration 0 to: {initial_ckpt_path}")
    except Exception as e:
        print(f"[ablation] Warning: failed to save initial model: {e}")

    # TensorBoard writer (per variant)
    writer: SummaryWriter | None = None
    if not args.no_tensorboard:
        try:
            tb_dir = per_variant_dir.parent.parent / "tb" / variant.name
            tb_dir.mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(log_dir=str(tb_dir))
            writer.add_text("variant", variant.name)
        except Exception as _e:
            print(f"[ablation] TensorBoard disabled due to init failure: {_e}")
            writer = None

    try:
        for iteration in range(1, num_iterations + 1):
            if args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / num_iterations
                optimizer.param_groups[0]["lr"] = frac * args.learning_rate



            for step in range(0, args.num_steps):
                global_step += args.num_envs
                obs[step] = next_obs_tensor
                dones[step] = next_done_tensor
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs_tensor)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
                next_done = np.logical_or(terminations, truncations)
                # Reward handling: dense (default) or sparse (end-of-episode energy)
                if not args.sparse_reward:
                    rewards[step] = torch.Tensor(reward).to(device).view(-1)
                else:
                    # Build sparse reward vector aligned with envs
                    sparse_r = np.zeros_like(reward, dtype=float)
                    if "final_info" in infos:
                        for i, finfo in enumerate(infos["final_info"]):
                            if isinstance(finfo, dict):
                                # Prefer active+idle energy if available; fallback to total_energy
                                ae = finfo.get("total_energy_active", None)
                                ie = finfo.get("total_energy_idle", None)
                                if ae is not None and ie is not None:
                                    sparse_r[i] = -float(ae) - float(ie)
                                else:
                                    te = finfo.get("total_energy", None)
                                    if te is not None:
                                        sparse_r[i] = -float(te)
                    rewards[step] = torch.Tensor(sparse_r).to(device).view(-1)
                next_obs_tensor, next_done_tensor = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

                if "final_info" in infos:
                    for info in infos["final_info"]:
                        if info and "episode" in info:
                            pbar.update(global_step - pbar.n)

            with torch.no_grad():
                next_value = agent.get_value(next_obs_tensor).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done_tensor
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values

            b_obs = obs.reshape((-1,) + obs_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + act_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            b_inds = np.arange(batch_size)
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, batch_size, minibatch_size):
                    end = start + minibatch_size
                    mb_inds = b_inds[start:end]
                    # Clear any previous cache before forward on this minibatch
                    if hasattr(agent.actor, "_cached_batch_node_embeddings"):
                        agent.actor._cached_batch_node_embeddings.clear()
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()
                    with torch.no_grad():
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                    pg_loss = torch.max(
                        -mb_advantages * ratio,
                        -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef),
                    ).mean()
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef)
                        v_loss = 0.5 * torch.max(v_loss_unclipped, (v_clipped - b_returns[mb_inds]) ** 2).mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss
                    # Low-pass regularization on node embeddings (Dirichlet energy)
                    lp_lambda = float(agent.variant.lowpass_reg_lambda) if hasattr(agent, 'variant') else 0.0
                    if lp_lambda > 0.0:
                        # Use cached embeddings if available (no second forward), else compute on the fly
                        emb_list = getattr(agent.actor, "_cached_batch_node_embeddings", None) if getattr(agent.actor, "cache_lowpass_from_forward", False) else None
                        H_energy = 0.0
                        count_graphs = 0
                        for b in range(b_obs[mb_inds].shape[0]):
                            obs_tensor = b_obs[mb_inds][b]
                            obs_obj = agent.mapper.unmap(obs_tensor)
                            if emb_list is not None and b < len(emb_list):
                                node_h = emb_list[b]
                            else:
                                node_h, _edge_h, _g = agent.actor.network(obs_obj)
                            ei = _build_edge_index_from_obs(obs_obj, use_task_deps=bool(agent.variant.use_task_dependencies))
                            num_nodes = node_h.shape[0]
                            L = _normalized_laplacian(ei, num_nodes=num_nodes, device=node_h.device)
                            H_energy = H_energy + _dirichlet_energy(node_h, L)
                            count_graphs += 1
                        if count_graphs > 0:
                            smooth_penalty = (H_energy / count_graphs)
                            loss = loss + lp_lambda * smooth_penalty

                    # Periodic CSV logger helper (opened lazily)
                    def _log_scalar_csv(metric_name: str, value: float):
                        path = per_variant_dir / f"{variant.name}_{metric_name}.csv"
                        new_file = not path.exists()
                        with path.open("a", newline="") as f:
                            w = _csv.writer(f)
                            if new_file:
                                w.writerow(["Wall time", "Step", "Value"])
                            w.writerow([time.time(), int(global_step), float(value)])

                    # Compute and log losses/entropy/KL periodically (once per iteration, first minibatch)
                    if (iteration % max(1, int(getattr(args, 'log_loss_every', 10))) == 0) and start == 0:
                        try:
                            ent_val = float(entropy.detach().mean().item())
                        except Exception:
                            ent_val = float('nan')
                        try:
                            kl_val = float(approx_kl.detach().item())
                        except Exception:
                            kl_val = float('nan')
                        try:
                            pg_val = float(pg_loss.detach().item())
                        except Exception:
                            pg_val = float('nan')
                        try:
                            v_val = float(v_loss.detach().item())
                        except Exception:
                            v_val = float('nan')
                        _log_scalar_csv("policy_entropy", ent_val)
                        _log_scalar_csv("kl", kl_val)
                        _log_scalar_csv("pg_loss", pg_val)
                        _log_scalar_csv("value_loss", v_val)

                    optimizer.zero_grad()
                    loss.backward()

                    # Gradient norm logging (pre- and post-clip), first minibatch of logging iterations
                    do_grad_log = bool(getattr(args, 'log_grad_norms', True)) and (iteration % max(1, int(getattr(args, 'grad_log_every', 10))) == 0) and start == 0
                    if do_grad_log:
                        def _norm(ps):
                            s = 0.0
                            for p in ps:
                                if p.grad is not None:
                                    g = p.grad.detach()
                                    s += float(torch.sum(g * g).item())
                            return float(s ** 0.5)
                        pre_actor = _norm(agent.actor.parameters())
                        pre_critic = _norm(agent.critic.parameters())
                        pre_total = (pre_actor ** 2 + pre_critic ** 2) ** 0.5
                        _log_scalar_csv("grads_preclip_actor", pre_actor)
                        _log_scalar_csv("grads_preclip_critic", pre_critic)
                        _log_scalar_csv("grads_preclip_total", pre_total)

                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)

                    if do_grad_log:
                        def _norm(ps):
                            s = 0.0
                            for p in ps:
                                if p.grad is not None:
                                    g = p.grad.detach()
                                    s += float(torch.sum(g * g).item())
                            return float(s ** 0.5)
                        post_actor = _norm(agent.actor.parameters())
                        post_critic = _norm(agent.critic.parameters())
                        post_total = (post_actor ** 2 + post_critic ** 2) ** 0.5
                        _log_scalar_csv("grads_postclip_actor", post_actor)
                        _log_scalar_csv("grads_postclip_critic", post_critic)
                        _log_scalar_csv("grads_postclip_total", post_total)
                    optimizer.step()
                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

            if args.test_every_iters > 0 and (iteration % args.test_every_iters == 0 or iteration == num_iterations):
                with torch.no_grad():
                    avg_mk, avg_energy_obs, avg_total_energy, m = _test_agent(agent, args)
                    print(f"[ablation][{run_tag}] iter {iteration}: mk={avg_mk:.4g}, Eobs={avg_energy_obs:.4g}, Etot={avg_total_energy:.4g}, Ea={m.get('avg_active_energy')}, Ei={m.get('avg_idle_energy')}")
                    # Write per-variant CSV metrics into logs/<run>/ablation/per_variant/
                    ts = time.time()
                    def _append_row(metric_name: str, value: float):
                        path = per_variant_dir / f"{variant.name}_{metric_name}.csv"
                        new_file = not path.exists()
                        with path.open("a", newline="") as f:
                            w = _csv.writer(f)
                            if new_file:
                                w.writerow(["Wall time", "Step", "Value"])
                            w.writerow([ts, int(global_step), float(value)])
                    _append_row("makespan", avg_mk)
                    _append_row("total_energy", avg_total_energy if avg_total_energy > 0 else avg_energy_obs)
                    has_active = m.get("avg_active_energy") is not None
                    has_idle = m.get("avg_idle_energy") is not None
                    has_return = m.get("avg_return") is not None
                    if has_active:
                        _append_row("active_energy", float(m["avg_active_energy"]))
                    if has_idle:
                        _append_row("idle_energy", float(m["avg_idle_energy"]))
                    if has_return:
                        _append_row("episodic_return", float(m["avg_return"]))
                    if has_active and has_idle:
                        active_plus_idle = float(m["avg_active_energy"]) + float(m["avg_idle_energy"])
                        _append_row("active_plus_idle", active_plus_idle)
                        
                        # Track best model based on active+idle energy (lower is better)
                        if active_plus_idle < best_active_idle_energy:
                            best_active_idle_energy = active_plus_idle
                            best_model_state = agent.state_dict().copy()
                            best_iteration = iteration
                            print(f"[ablation] New best model at iteration {iteration}: active+idle={active_plus_idle:.4f}")

                    # TensorBoard logging similar to train.py
                    if writer is not None:
                        try:
                            # Core metrics
                            writer.add_scalar(f"tests/{variant.name}/makespan", float(avg_mk), int(global_step))
                            writer.add_scalar(f"tests/{variant.name}/total_energy", float(avg_total_energy if avg_total_energy > 0 else avg_energy_obs), int(global_step))
                            if has_active:
                                writer.add_scalar(f"tests/{variant.name}/active_energy", float(m.get("avg_active_energy", 0.0)), int(global_step))
                            if has_idle:
                                writer.add_scalar(f"tests/{variant.name}/idle_energy", float(m.get("avg_idle_energy", 0.0)), int(global_step))
                            # Bottleneck and refined metrics (if provided by env)
                            for key, tbtag in [
                                ("bneck_steps_ratio", f"tests/{variant.name}/bottleneck_steps_ratio"),
                                ("ready_ratio", f"tests/{variant.name}/ready_bottleneck_ratio"),
                                ("avg_wait_time", f"tests/{variant.name}/avg_wait_time"),
                                ("avg_bneck_steps", f"tests/{variant.name}/avg_bottleneck_steps"),
                                ("avg_decision_steps", f"tests/{variant.name}/avg_decision_steps"),
                                ("avg_ready", f"tests/{variant.name}/avg_ready_tasks"),
                                ("avg_ready_blocked", f"tests/{variant.name}/avg_blocked_ready_tasks"),
                                ("refined_steps_ratio", f"tests/{variant.name}/refined_bottleneck_steps_ratio"),
                                ("refined_ready_ratio", f"tests/{variant.name}/refined_ready_bottleneck_ratio"),
                                ("avg_refined_steps", f"tests/{variant.name}/avg_refined_bottleneck_steps"),
                                ("avg_ready_refined", f"tests/{variant.name}/avg_ready_tasks_refined"),
                                ("avg_ready_blocked_refined", f"tests/{variant.name}/avg_blocked_ready_tasks_refined"),
                                ("wait_time_cp", f"tests/{variant.name}/wait_time_cp"),
                                ("wait_time_offcp", f"tests/{variant.name}/wait_time_offcp"),
                            ]:
                                if m.get(key) is not None:
                                    writer.add_scalar(tbtag, float(m[key]), int(global_step))
                            writer.flush()
                        except Exception as _e:
                            print(f"[ablation] TB log failed: {_e}")

            # Periodic checkpointing (disabled if checkpoint_every == 0)
            if save_every > 0 and ((iteration % save_every == 0) or (iteration == num_iterations)):
                try:
                    ckpt_iter_path = per_variant_dir / f"{variant.name}_iter{iteration:05d}.pt"
                    torch.save(agent.state_dict(), ckpt_iter_path)
                    print(f"[ablation] Saved checkpoint at iteration {iteration} to: {ckpt_iter_path}")
                except Exception as e:
                    print(f"[ablation] Warning: failed to save periodic checkpoint at iter {iteration}: {e}")
    finally:
        # Helper: offline plotting from per-variant CSVs
        def _offline_plot_per_variant_metrics(out_dir: Path, var_name: str) -> None:
            try:
                import pandas as pd
                import matplotlib.pyplot as plt
            except Exception as e:
                print(f"[ablation][offline-plots] Skipping plots due to missing deps: {e}")
                return

            # Core performance metrics (already collected during eval cadence)
            metrics = [
                ("makespan", "Makespan"),
                ("total_energy", "Total Energy"),
                ("active_energy", "Active Energy"),
                ("idle_energy", "Idle Energy"),
                ("active_plus_idle", "Active + Idle Energy"),
                ("episodic_return", "Episodic Return"),
            ]
            for key, title in metrics:
                csv_path = out_dir / f"{var_name}_{key}.csv"
                if not csv_path.exists() or (csv_path.exists() and csv_path.stat().st_size == 0):
                    continue
                try:
                    df = pd.read_csv(csv_path)
                    if {"Step", "Value"}.issubset(df.columns):
                        plt.figure(figsize=(8, 4))
                        plt.plot(df["Step"], df["Value"], marker="o", linewidth=2, alpha=0.8)
                        plt.xlabel("Global Step")
                        plt.ylabel(title)
                        plt.title(f"{var_name} - {title}")
                        plt.grid(True, alpha=0.3)
                        plt.tight_layout()
                        out_png = out_dir / f"{var_name}_timeseries_{key}.png"
                        plt.savefig(out_png, dpi=180)
                        plt.close()
                        print(f"[ablation][offline-plots] Saved {out_png}")
                except Exception as e:
                    print(f"[ablation][offline-plots] Failed to plot {csv_path.name}: {e}")

            # Optimization diagnostics: entropy, KL, losses
            opt_metrics = [
                ("policy_entropy", "Policy Entropy"),
                ("kl", "Approx KL"),
                ("pg_loss", "Policy Loss"),
                ("value_loss", "Value Loss"),
            ]
            for key, title in opt_metrics:
                csv_path = out_dir / f"{var_name}_{key}.csv"
                if not csv_path.exists() or (csv_path.exists() and csv_path.stat().st_size == 0):
                    continue
                try:
                    df = pd.read_csv(csv_path)
                    if {"Step", "Value"}.issubset(df.columns):
                        plt.figure(figsize=(8, 4))
                        plt.plot(df["Step"], df["Value"], linewidth=2, alpha=0.9)
                        plt.xlabel("Global Step")
                        plt.ylabel(title)
                        plt.title(f"{var_name} - {title}")
                        plt.grid(True, alpha=0.3)
                        plt.tight_layout()
                        out_png = out_dir / f"{var_name}_timeseries_{key}.png"
                        plt.savefig(out_png, dpi=180)
                        plt.close()
                        print(f"[ablation][offline-plots] Saved {out_png}")
                except Exception as e:
                    print(f"[ablation][offline-plots] Failed to plot {csv_path.name}: {e}")

            # Gradient norms: pre-clip and post-clip, actor/critic/total
            grad_groups = [
                ("grads_preclip_actor", "Grad Norm (Actor, pre-clip)"),
                ("grads_preclip_critic", "Grad Norm (Critic, pre-clip)"),
                ("grads_preclip_total", "Grad Norm (Total, pre-clip)"),
                ("grads_postclip_actor", "Grad Norm (Actor, post-clip)"),
                ("grads_postclip_critic", "Grad Norm (Critic, post-clip)"),
                ("grads_postclip_total", "Grad Norm (Total, post-clip)"),
            ]
            for key, title in grad_groups:
                csv_path = out_dir / f"{var_name}_{key}.csv"
                if not csv_path.exists() or (csv_path.exists() and csv_path.stat().st_size == 0):
                    continue
                try:
                    df = pd.read_csv(csv_path)
                    if {"Step", "Value"}.issubset(df.columns):
                        plt.figure(figsize=(8, 4))
                        plt.plot(df["Step"], df["Value"], linewidth=2, alpha=0.9)
                        plt.xlabel("Global Step")
                        plt.ylabel(title)
                        plt.title(f"{var_name} - {title}")
                        plt.grid(True, alpha=0.3)
                        plt.tight_layout()
                        out_png = out_dir / f"{var_name}_timeseries_{key}.png"
                        plt.savefig(out_png, dpi=180)
                        plt.close()
                        print(f"[ablation][offline-plots] Saved {out_png}")
                except Exception as e:
                    print(f"[ablation][offline-plots] Failed to plot {csv_path.name}: {e}")

        try:
            envs.close()
        except Exception:
            pass
        try:
            pbar.close()
        except Exception:
            pass
        # Save trained agent checkpoint for this variant
        try:
            ckpt_path = per_variant_dir / f"{variant.name}_model.pt"
            torch.save(agent.state_dict(), ckpt_path)
            print(f"[ablation] Saved final model to: {ckpt_path}")
        except Exception as e:
            print(f"[ablation] Warning: failed to save final model: {e}")
        
        # Save best model based on active+idle energy
        if best_model_state is not None:
            try:
                best_ckpt_path = per_variant_dir / f"{variant.name}_best_model.pt"
                torch.save(best_model_state, best_ckpt_path)
                print(f"[ablation] Saved best model (iter {best_iteration}, active+idle={best_active_idle_energy:.4f}) to: {best_ckpt_path}")
            except Exception as e:
                print(f"[ablation] Warning: failed to save best model: {e}")
        else:
            print(f"[ablation] Warning: No best model found (no active+idle energy measurements)")
        try:
            if writer is not None:
                writer.close()
        except Exception:
            pass

        # Generate offline plots from CSVs right after training this variant
        if getattr(args, "offline_plots_after_training", True):
            _offline_plot_per_variant_metrics(per_variant_dir, variant.name)


def main(args: Args):
    # Seed and device
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = _pick_device(args.device)

    args.run_name = f"{int(time.time())}_{args.exp_name}"
    run_dir = Path(args.output_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    # Dedicated ablation directory for this run
    ablation_dir = run_dir / args.ablation_subdir
    per_variant_root = ablation_dir / "per_variant"
    ablation_dir.mkdir(parents=True, exist_ok=True)
    per_variant_root.mkdir(parents=True, exist_ok=True)

    # Define ablation suite (single-variant or baseline-only when requested)
    if args.train_only_variant is not None:
        vname = args.train_only_variant.strip().lower()
        if vname == "baseline":
            variants: List[AblationVariant] = [AblationVariant(name="baseline", gin_num_layers=3)]
        elif vname == "no_global_actor":
            variants = [AblationVariant(name="no_global_actor", use_actor_global_embedding=False)]
        elif vname == "no_bn":
            variants = [AblationVariant(name="no_bn", use_batchnorm=False)]
        elif vname == "no_task_deps":
            variants = [AblationVariant(name="no_task_deps", use_task_dependencies=False)]
        elif vname == "shallow_gnn":
            variants = [AblationVariant(name="shallow_gnn", gin_num_layers=1)]
        elif vname == "mlp_only":
            variants = [AblationVariant(name="mlp_only", mlp_only=True, gin_num_layers=0, use_actor_global_embedding=False)]
        elif vname == "hetero":
            variants = [AblationVariant(name="hetero", graph_type="hetero", use_task_dependencies=True, gin_num_layers=3)]
        elif vname == "hetero_noglobal":
            variants = [AblationVariant(
                name="hetero_noglobal", 
                graph_type="hetero", 
                use_task_dependencies=True,
                gin_num_layers=3,
                use_actor_global_embedding=False,
            )]
        else:
            raise SystemExit(f"Unknown train_only_variant '{args.train_only_variant}'. Supported: baseline, no_global_actor, no_bn, no_task_deps, shallow_gnn, mlp_only, hetero, hetero_noglobal")
    elif args.train_only_baseline:
        variants = [AblationVariant(name="baseline", gin_num_layers=3)]
    else:
        variants = [
            AblationVariant(name="baseline", gin_num_layers=3),
            AblationVariant(name="no_bn", use_batchnorm=False),
            AblationVariant(name="no_task_deps", use_task_dependencies=False),
            AblationVariant(name="no_global_actor", use_actor_global_embedding=False),
            AblationVariant(name="shallow_gnn", gin_num_layers=1),
            AblationVariant(name="mlp_only", mlp_only=True, gin_num_layers=0, use_actor_global_embedding=False),
            AblationVariant(name="hetero", graph_type="hetero", use_task_dependencies=True, gin_num_layers=3),
            # Attention-based (GATv2) variants; heads must divide embedding_dim (default 8)
            AblationVariant(name="gatv2_h4_l2", graph_type="gatv2", gin_num_layers=2, gat_heads=4, gat_dropout=0.1),
            AblationVariant(name="gatv2_h8_l2", graph_type="gatv2", gin_num_layers=2, gat_heads=8, gat_dropout=0.1),
            AblationVariant(name="gatv2_shallow", graph_type="gatv2", gin_num_layers=1, gat_heads=4, gat_dropout=0.1),
            # GraphSAGE
            AblationVariant(name="sage_l2", graph_type="sage", gin_num_layers=2),
            AblationVariant(name="sage_l1", graph_type="sage", gin_num_layers=1),
            # PNA
            AblationVariant(name="pna_l2", graph_type="pna", gin_num_layers=2),
            # Graph Transformer
            AblationVariant(name="transformer_l2", graph_type="transformer", gin_num_layers=2, gat_dropout=0.1),
            AblationVariant(name="transformer_l1", graph_type="transformer", gin_num_layers=1, gat_dropout=0.1),
            # Hetero (task/vm types)
            AblationVariant(name="hetero_sage_l2", graph_type="hetero", gin_num_layers=2, hetero_base="sage"),
            AblationVariant(name="hetero_gatv2_l2", graph_type="hetero", gin_num_layers=2, hetero_base="gatv2"),
            # Edge-aware MPNNs
            AblationVariant(name="nnconv_l2", graph_type="nnconv", gin_num_layers=2),
            AblationVariant(name="edgeconv_l2", graph_type="edgeconv", gin_num_layers=2),
            # Pooling comparison (attention vs mean)
            AblationVariant(name="baseline_no_attnpool", use_actor_global_embedding=True),
        ]

    # Apply low-pass experiment flags to all variants
    try:
        for _v in variants:
            _v.lowpass_reg_lambda = float(args.variant_lowpass_reg_lambda)
            _v.cache_lowpass_from_forward = bool(args.variant_cache_lowpass_from_forward)
    except Exception:
        pass

    # Train variants and optionally run per-variant summary evaluation
    import csv as _csv
    summary_path = ablation_dir / "summary.csv"
    if not args.skip_variant_summary_eval:
        with summary_path.open("w", newline="") as fsum:
            w = _csv.writer(fsum)
            w.writerow(["variant", "makespan", "total_energy", "active_energy", "idle_energy", "active_plus_idle"])  # avg over final eval

    for v in variants:
        v_dir = per_variant_root
        _train_one_variant(args, v, device, v_dir)
        if not args.skip_variant_summary_eval:
            # After training, run a final evaluation to log summary
            agent = AblationGinAgent(device, v)
            mk, eobs, etot, mm = _test_agent(agent, args)
            ea = mm.get("avg_active_energy") if mm else None
            ei = mm.get("avg_idle_energy") if mm else None
            api = (ea or 0.0) + (ei or 0.0)
            with summary_path.open("a", newline="") as fsum:
                w = _csv.writer(fsum)
                w.writerow([v.name, mk, etot if etot > 0 else eobs, ea or 0.0, ei or 0.0, api])
            print(f"[ablation] {v.name}: makespan={mk:.4g}, total_energy={(etot if etot>0 else eobs):.4g}, active={ea}, idle={ei}")
        else:
            print(f"[ablation] Skipping summary eval for variant '{v.name}' (skip_variant_summary_eval=True)")

    if not args.skip_variant_summary_eval:
        # Convenience: copy baseline checkpoint to run_dir/model.pt if available
        try:
            baseline_ckpt = per_variant_root / "baseline_model.pt"
            if baseline_ckpt.exists():
                out_ckpt = run_dir / "model.pt"
                # Load and resave to ensure portability
                state = torch.load(str(baseline_ckpt), map_location=_pick_device(args.device))
                torch.save(state, str(out_ckpt))
                print(f"[ablation] Exported baseline checkpoint to: {out_ckpt}")
            else:
                print("[ablation] baseline_model.pt not found; skipping export to model.pt")
        except Exception as e:
            print(f"[ablation] Warning: failed to export baseline checkpoint: {e}")
    else:
        print("[ablation] Skipping baseline checkpoint export (skip_variant_summary_eval=True)")

    if (not args.train_only_baseline) and (not args.skip_additional_eval):
        try:
            # Generate additional evaluation artifacts (optional)
            import matplotlib.pyplot as plt
            import pandas as pd

            def _create_eval_config(base: Args, config_name: str, dag_method: str) -> Args:
                """Create one of three evaluation configurations.
                
                Args:
                    config_name: One of 'balanced', 'complex', or 'constrained'
                    dag_method: 'linear', 'gnp', or 'mixed'
                """
                new = _dc_replace(base)
                
                # Common settings for all eval configs
                eval_base = {
                    'test_iterations': 5,
                    'workflow_count': 8 if dag_method != 'mixed' else 4
                }
                
                # Configuration-specific settings
                configs = {
                    # 1) Balanced: Similar to training but with different seeds
                    'balanced': {
                        'gnp_min_n': base.dataset.gnp_min_n,
                        'gnp_max_n': base.dataset.gnp_max_n,
                        'host_count': base.dataset.host_count,
                        'vm_count': base.dataset.vm_count,
                        'suffix': 'balanced'
                    },
                    # 2) Complex: Larger workflows and more resources
                    'complex': {
                        'gnp_min_n': 20,  # Larger workflows
                        'gnp_max_n': 35,
                        'host_count': 8,   # More hosts
                        'vm_count': 20,    # More VMs
                        'suffix': 'complex'
                    },
                    # 3) Resource-constrained: Fewer resources than training
                    'constrained': {
                        'gnp_min_n': base.dataset.gnp_min_n,
                        'gnp_max_n': base.dataset.gnp_max_n,
                        'host_count': max(2, base.dataset.host_count - 2),  # Fewer hosts
                        'vm_count': max(5, base.dataset.vm_count - 5),      # Fewer VMs
                        'suffix': 'constrained'
                    }
                }
                
                cfg = configs[config_name]
                new.dataset = _dc_replace(
                    base.dataset,
                    workflow_count=eval_base['workflow_count'],
                    dag_method=dag_method,
                    gnp_min_n=cfg['gnp_min_n'],
                    gnp_max_n=cfg['gnp_max_n'],
                    host_count=cfg['host_count'],
                    vm_count=cfg['vm_count']
                )
                new.test_iterations = eval_base['test_iterations']
                new.eval_suffix = cfg['suffix']
                return new

            def _eval_variants(eval_args: Args, variants_list: List[AblationVariant]) -> List[tuple]:
                rows = []
                for v in variants_list:
                    agent = AblationGinAgent(device, v)
                    mk, eobs, etot, mm = _test_agent(agent, eval_args)
                    ea = mm.get("avg_active_energy") if mm else None
                    ei = mm.get("avg_idle_energy") if mm else None
                    api = (ea or 0.0) + (ei or 0.0)
                    rows.append((v.name, mk, (etot if etot > 0 else eobs), ea or 0.0, ei or 0.0, api))
                return rows

            def _write_and_plot(rows: List[tuple], out_dir: Path, label: str):
                out_dir.mkdir(parents=True, exist_ok=True)
                cat_summary = out_dir / "summary.csv"
                import csv as _csv
                with cat_summary.open("w", newline="") as f:
                    w = _csv.writer(f)
                    w.writerow(["variant", "makespan", "total_energy", "active_energy", "idle_energy", "active_plus_idle"])
                    for r in rows:
                        w.writerow(list(r))
                # Bars
                dfc = pd.read_csv(cat_summary)
                def _bar(metric: str, ylabel: str, fname: str):
                    plt.figure(figsize=(8, 4))
                    xs = np.arange(len(dfc["variant"]))
                    plt.bar(xs, dfc[metric])
                    plt.xticks(xs, dfc["variant"], rotation=45, ha="right")
                    plt.title(label)
                    plt.ylabel(ylabel)
                    plt.tight_layout()
                    plt.savefig(out_dir / fname, dpi=180)
                    plt.close()
                _bar("makespan", "Avg makespan (test)", "bars_makespan.png")
                _bar("total_energy", "Avg total energy (test)", "bars_total_energy.png")
                _bar("active_energy", "Avg active energy (test)", "bars_active_energy.png")
                _bar("idle_energy", "Avg idle energy (test)", "bars_idle_energy.png")
                _bar("active_plus_idle", "Avg active+idle (test)", "bars_active_plus_idle.png")

                # Lines (across variants) for quick visual comparison within the category
                def _lines(metric: str, ylabel: str, fname: str):
                    plt.figure(figsize=(8, 4))
                    xs = np.arange(len(dfc["variant"]))
                    plt.plot(xs, dfc[metric], marker="o")
                    plt.xticks(xs, dfc["variant"], rotation=45, ha="right")
                    plt.title(label)
                    plt.ylabel(ylabel)
                    plt.tight_layout()
                    plt.savefig(out_dir / fname, dpi=180)
                    plt.close()
                _lines("makespan", "Avg makespan (test)", "lines_makespan.png")
                _lines("total_energy", "Avg total energy (test)", "lines_total_energy.png")
                _lines("active_energy", "Avg active energy (test)", "lines_active_energy.png")
                _lines("idle_energy", "Avg idle energy (test)", "lines_idle_energy.png")
                _lines("active_plus_idle", "Avg active+idle (test)", "lines_active_plus_idle.png")

            # Run all three evaluation configurations
            for eval_config in ['balanced', 'complex', 'constrained']:
                # Create output directory for this config
                config_dir = ablation_dir / f"eval_{eval_config}"
                
                # 1) Linear workflows
                linear_args = _create_eval_config(args, eval_config, "linear")
                rows_linear = _eval_variants(linear_args, variants)
                _write_and_plot(
                    rows_linear, 
                    config_dir / "linear",
                    f"Eval ({eval_config}): Linear workflows | "
                    f"Tasks: {linear_args.dataset.gnp_min_n}-{linear_args.dataset.gnp_max_n} | "
                    f"Hosts: {linear_args.dataset.host_count} | "
                    f"VMs: {linear_args.dataset.vm_count}"
                )
                
                # 2) GNP workflows
                gnp_args = _create_eval_config(args, eval_config, "gnp")
                rows_gnp = _eval_variants(gnp_args, variants)
                _write_and_plot(
                    rows_gnp,
                    config_dir / "gnp",
                    f"Eval ({eval_config}): GNP workflows | "
                    f"Tasks: {gnp_args.dataset.gnp_min_n}-{gnp_args.dataset.gnp_max_n} | "
                    f"Hosts: {gnp_args.dataset.host_count} | "
                    f"VMs: {gnp_args.dataset.vm_count}"
                )
                
                # 3) Mixed workflows (half linear, half GNP)
                def _clone_for_mixed(base: Args) -> Args:
                    mixed = _create_eval_config(base, eval_config, "gnp")  # Just to get config
                    mixed.dataset.workflow_count = 4  # 4 of each type for mixed8
                    return mixed
                    
                lin_args = _clone_for_mixed(args)
                lin_args.dataset.dag_method = "linear"
                gnp_args = _clone_for_mixed(args)
                
                rows_mixed = []
                for v in variants:
                    agent = AblationGinAgent(device, v)
                    # Test on linear portion
                    mk_lin, eobs_lin, etot_lin, mm_lin = _test_agent(agent, lin_args)
                    # Test on GNP portion
                    mk_gnp, eobs_gnp, etot_gnp, mm_gnp = _test_agent(agent, gnp_args)
                    
                    # Combine metrics (simple average of both types)
                    mk = 0.5 * (mk_lin + mk_gnp)
                    te = 0.5 * (
                        (etot_lin if etot_lin > 0 else eobs_lin) + 
                        (etot_gnp if etot_gnp > 0 else eobs_gnp)
                    )
                    ea_lin = mm_lin.get("avg_active_energy", 0) if mm_lin else 0.0
                    ea_gnp = mm_gnp.get("avg_active_energy", 0) if mm_gnp else 0.0
                    ei_lin = mm_lin.get("avg_idle_energy", 0) if mm_lin else 0.0
                    ei_gnp = mm_gnp.get("avg_idle_energy", 0) if mm_gnp else 0.0
                    
                    rows_mixed.append((
                        v.name, 
                        mk, 
                        te,
                        0.5 * (ea_lin + ea_gnp),  # avg active energy
                        0.5 * (ei_lin + ei_gnp),  # avg idle energy
                        0.5 * (ea_lin + ea_gnp + ei_lin + ei_gnp)  # total energy
                    ))
                
                _write_and_plot(
                    rows_mixed,
                    config_dir / "mixed",
                    f"Eval ({eval_config}): Mixed workflows | "
                    f"Tasks: {lin_args.dataset.gnp_min_n}-{lin_args.dataset.gnp_max_n} | "
                    f"Hosts: {lin_args.dataset.host_count} | "
                    f"VMs: {lin_args.dataset.vm_count} | "
                    f"(4 linear + 4 GNP)"
                )
                
                print(f"[ablation] {eval_config} evaluation complete: {config_dir}")
                
            print(f"\n[ablation] All evaluations completed. Results saved under: {ablation_dir}/eval_*")
        except Exception as e:
            print("[ablation] Additional evaluation generation failed:", e)
    else:
        print("[ablation] Skipping additional evaluation sweeps (skip_additional_eval=True or train_only_baseline=True)")

    print(f"\nAblation experiments complete. Summary: {summary_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
