"""
Adversarial MDP2 selection to maximize gradient conflict with MDP1.

Given a trained policy θ and a fixed preference α (makespan-energy trade-off),
this module searches for an MDP2 that produces the most conflicting gradients
with a reference MDP1.
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, List
from dataclasses import dataclass

from cogito.dataset_generator.gen_dataset import DatasetArgs
from cogito.gnn_deeprl_model.core.env.gym_env import CloudSchedulingGymEnvironment
from cogito.gnn_deeprl_model.agents.gin_agent.wrapper import GinAgentWrapper


@dataclass
class AdversarialMDPConfig:
    """Configuration for adversarial MDP2 search."""
    num_candidates: int = 20  # Number of candidate MDPs to try
    rollout_steps: int = 128  # Steps to collect per candidate
    seed_pool_start: int = 200000  # Starting seed for candidate pool
    conflict_metric: str = "cosine"  # "cosine" or "dot"
    
    # Diversity parameters for candidate generation
    vary_structure: bool = True  # If True, vary DAG structure parameters
    gnp_p_range: tuple = (0.05, 0.85)  # Range of edge probabilities to try
    task_count_range: tuple = (8, 15)  # Range of task counts to try
    style_pool: list = None  # Pool of DAG styles to try (None = use all)
    
    def __post_init__(self):
        if self.style_pool is None:
            self.style_pool = ["long_cp", "wide", "generic"]
    

def compute_actor_gradient(
    agent: nn.Module,
    env,
    num_steps: int,
    device: torch.device,
    alpha_makespan: float = 0.5,
    alpha_energy: float = 0.5,
) -> torch.Tensor:
    """
    Collect a mini-rollout and compute the actor gradient for the scalarized objective.
    
    Args:
        agent: The policy network
        env: Gym environment
        num_steps: Number of steps to roll out
        device: torch device
        alpha_makespan: Weight for makespan objective
        alpha_energy: Weight for energy objective
        
    Returns:
        Flattened actor gradient tensor
    """
    agent.eval()
    
    # Collect rollout
    obs_list = []
    action_list = []
    reward_makespan_list = []
    reward_energy_list = []
    
    obs, _ = env.reset()
    for _ in range(num_steps):
        obs_tensor = torch.from_numpy(np.asarray(obs, dtype=np.float32).reshape(1, -1)).to(device)
        
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(obs_tensor)
        
        obs_list.append(obs_tensor)
        action_list.append(action)
        
        obs, reward, terminated, truncated, info = env.step(int(action.item()))
        
        # Extract makespan and energy from reward/info
        # Assuming reward is -makespan and energy is in info
        reward_makespan_list.append(reward)
        reward_energy_list.append(-info.get("total_energy_active", 0.0))
        
        if terminated or truncated:
            obs, _ = env.reset()
    
    # Compute scalarized returns (simple sum for now, can use GAE)
    returns_makespan = torch.tensor(reward_makespan_list, device=device, dtype=torch.float32)
    returns_energy = torch.tensor(reward_energy_list, device=device, dtype=torch.float32)
    
    # Scalarize with alpha
    returns_scalarized = alpha_makespan * returns_makespan + alpha_energy * returns_energy
    
    # Compute policy gradient (REINFORCE-style)
    agent.zero_grad()
    loss = 0.0
    
    for i, (obs_t, action_t, ret) in enumerate(zip(obs_list, action_list, returns_scalarized)):
        _, log_prob, _, _ = agent.get_action_and_value(obs_t, action_t)
        loss = loss - log_prob * ret
    
    loss = loss / len(obs_list)
    loss.backward()
    
    # Extract actor gradients
    grad_parts = []
    for p in agent.actor.parameters():
        if p.grad is not None:
            grad_parts.append(p.grad.view(-1).detach().clone())
    
    agent.zero_grad()
    agent.train()
    
    if not grad_parts:
        return torch.zeros(1, device=device)
    
    return torch.cat(grad_parts)


def generate_diverse_candidate(
    base_args: DatasetArgs,
    candidate_idx: int,
    config: AdversarialMDPConfig,
    rng: np.random.RandomState,
) -> Tuple[DatasetArgs, int]:
    """
    Generate a diverse candidate MDP by varying structure parameters.
    
    Args:
        base_args: Base dataset configuration
        candidate_idx: Index of this candidate
        config: Adversarial search config
        rng: Random state for reproducibility
        
    Returns:
        (modified_args, seed): Modified dataset args and seed for this candidate
    """
    seed = config.seed_pool_start + candidate_idx
    
    if not config.vary_structure:
        # Just vary seed, keep structure same
        return base_args, seed
    
    # Create a copy to modify
    import copy
    args = copy.deepcopy(base_args)
    
    # Vary DAG style
    if config.style_pool and len(config.style_pool) > 0:
        args.style = rng.choice(config.style_pool)
    
    # Vary edge probability (controls DAG density)
    if config.gnp_p_range:
        p_min, p_max = config.gnp_p_range
        args.gnp_p = rng.uniform(p_min, p_max)
    
    # Vary task count
    if config.task_count_range:
        n_min, n_max = config.task_count_range
        n_tasks = rng.randint(n_min, n_max + 1)
        args.gnp_min_n = n_tasks
        args.gnp_max_n = n_tasks
    
    return args, seed


def select_adversarial_mdp2(
    agent: nn.Module,
    mdp1_dataset_args: DatasetArgs,
    mdp1_seed: int,
    config: AdversarialMDPConfig,
    device: torch.device,
    alpha_makespan: float = 0.5,
    alpha_energy: float = 0.5,
) -> Tuple[DatasetArgs, int, float]:
    """
    Search for an MDP2 that maximizes gradient conflict with MDP1.
    
    Args:
        agent: Current policy network
        mdp1_dataset_args: Dataset config for MDP1
        mdp1_seed: Seed for MDP1
        config: Adversarial search config
        device: torch device
        alpha_makespan: Preference weight for makespan
        alpha_energy: Preference weight for energy
        
    Returns:
        (best_args, best_seed, conflict_score): Dataset args, seed, and conflict metric for MDP2
    """
    # Compute reference gradient from MDP1
    env1 = CloudSchedulingGymEnvironment(
        dataset_args=mdp1_dataset_args,
        collect_timelines=False,
        compute_metrics=False,
    )
    env1 = GinAgentWrapper(env1)
    env1.reset(seed=mdp1_seed)
    
    grad_mdp1 = compute_actor_gradient(
        agent, env1, config.rollout_steps, device, alpha_makespan, alpha_energy
    )
    env1.close()
    
    norm_mdp1 = grad_mdp1.norm().item()
    if norm_mdp1 < 1e-9:
        print("[AdversarialMDP] Warning: MDP1 gradient is near-zero, returning random config")
        return mdp1_dataset_args, config.seed_pool_start, 0.0
    
    # Search over candidate MDP2s with diverse structures
    best_args = mdp1_dataset_args
    best_seed = config.seed_pool_start
    best_conflict = float("inf")  # We want most negative cosine
    
    rng = np.random.RandomState(42)  # Fixed seed for reproducible candidate generation
    
    print(f"[AdversarialMDP] Searching {config.num_candidates} candidates for max conflict with MDP1")
    print(f"  MDP1: style={mdp1_dataset_args.style}, p={getattr(mdp1_dataset_args, 'gnp_p', 'N/A')}, "
          f"n={mdp1_dataset_args.gnp_min_n}-{mdp1_dataset_args.gnp_max_n}, seed={mdp1_seed}")
    
    if config.vary_structure:
        print(f"  Varying: style={config.style_pool}, p={config.gnp_p_range}, n={config.task_count_range}")
    
    for cand_idx in range(config.num_candidates):
        # Generate diverse candidate
        mdp2_dataset_args, cand_seed = generate_diverse_candidate(
            mdp1_dataset_args, cand_idx, config, rng
        )
        
        env2 = CloudSchedulingGymEnvironment(
            dataset_args=mdp2_dataset_args,
            collect_timelines=False,
            compute_metrics=False,
        )
        env2 = GinAgentWrapper(env2)
        env2.reset(seed=cand_seed)
        
        try:
            grad_mdp2 = compute_actor_gradient(
                agent, env2, config.rollout_steps, device, alpha_makespan, alpha_energy
            )
            env2.close()
            
            norm_mdp2 = grad_mdp2.norm().item()
            if norm_mdp2 < 1e-9:
                continue
            
            # Compute conflict metric
            if config.conflict_metric == "cosine":
                cosine_sim = torch.dot(grad_mdp1, grad_mdp2).item() / (norm_mdp1 * norm_mdp2 + 1e-9)
                conflict_score = cosine_sim  # More negative = more conflict
            else:  # dot product
                conflict_score = torch.dot(grad_mdp1, grad_mdp2).item()
            
            if conflict_score < best_conflict:
                best_conflict = conflict_score
                best_seed = cand_seed
                best_args = mdp2_dataset_args
                
                print(f"  → New best: style={mdp2_dataset_args.style}, "
                      f"p={getattr(mdp2_dataset_args, 'gnp_p', 'N/A'):.2f}, "
                      f"n={mdp2_dataset_args.gnp_min_n}, seed={cand_seed}, conflict={conflict_score:.4f}")
        
        except Exception as e:
            print(f"[AdversarialMDP] Candidate {cand_idx} failed: {e}")
            continue
    
    print(f"[AdversarialMDP] Selected MDP2: style={best_args.style}, "
          f"p={getattr(best_args, 'gnp_p', 'N/A'):.2f}, n={best_args.gnp_min_n}, "
          f"seed={best_seed}, conflict={best_conflict:.4f}")
    
    return best_args, best_seed, best_conflict


def generate_adversarial_mdp_pair(
    agent: nn.Module,
    base_dataset_args: DatasetArgs,
    device: torch.device,
    mdp1_seed: int = 101001,
    alpha_makespan: float = 0.5,
    alpha_energy: float = 0.5,
    config: AdversarialMDPConfig = None,
) -> Tuple[int, DatasetArgs, int, float]:
    """
    Generate a pair of MDPs (MDP1, MDP2) where MDP2 is adversarially selected
    to maximize gradient conflict with MDP1.
    
    Args:
        agent: Current policy
        base_dataset_args: Base dataset configuration (for MDP1)
        device: torch device
        mdp1_seed: Seed for MDP1 (fixed)
        alpha_makespan: Preference weight for makespan
        alpha_energy: Preference weight for energy
        config: Adversarial search config
        
    Returns:
        (mdp1_seed, mdp2_args, mdp2_seed, conflict_score)
    """
    if config is None:
        config = AdversarialMDPConfig()
    
    mdp2_args, mdp2_seed, conflict = select_adversarial_mdp2(
        agent=agent,
        mdp1_dataset_args=base_dataset_args,
        mdp1_seed=mdp1_seed,
        config=config,
        device=device,
        alpha_makespan=alpha_makespan,
        alpha_energy=alpha_energy,
    )
    
    return mdp1_seed, mdp2_args, mdp2_seed, conflict
