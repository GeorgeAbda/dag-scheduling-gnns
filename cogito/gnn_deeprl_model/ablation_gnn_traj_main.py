"""
Trajectory-enabled ablation training script.

This script mirrors scheduler/rl_model/ablation_gnn.py behavior (no live TensorBoard by default,
CSV logging + offline plots), and adds actor trajectory collection/visualization.

Usage (example):

  python -m scheduler.rl_model.ablation_gnn_traj_main \
    --exp_name hetero_cpu_traj \
    --seed 12345 \
    --train_only_variant hetero \
    --dataset.dag_method gnp \
    --dataset.gnp_min_n 12 \
    --dataset.gnp_max_n 16 \
    --dataset.host_count 4 \
    --dataset.vm_count 10 \
    --dataset.workflow_count 10 \
    --dataset.gnp_p 0.3 \
    --total_timesteps 2000000 \
    --device cpu \
    --test_every_iters 10 \
    --trajectory.enabled True \
    --trajectory.collect_every 50

Notes:
- Deterministic evaluation, offline CSV/PNG plots.
- CPU threads auto-match num_envs unless --torch_num_threads is set or --cpus_match_envs False.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from dataclasses import replace as _dc_replace
from pathlib import Path
import json
from typing import Optional
import os
import time
import math
import numpy as np
import torch
import gymnasium as gym
import tyro
import csv as _csv
from torch.utils.tensorboard import SummaryWriter

# Import base ablation components
from cogito.gnn_deeprl_model import ablation_gnn as AG
from cogito.dataset_generator.gen_dataset import DatasetArgs
from cogito.dataset_generator.core.gen_dataset import generate_dataset
from cogito.dataset_generator.core.models import Dataset
from cogito.gnn_deeprl_model.core.env.gym_env import CloudSchedulingGymEnvironment
from cogito.gnn_deeprl_model.agents.gin_agent.wrapper import GinAgentWrapper
from gymnasium.wrappers import RecordEpisodeStatistics
from cogito.config.settings import HOST_SPECS_PATH
from cogito.gnn_deeprl_model.ablation_gnn_with_trajectory import (
    TrajectoryConfig,
    integrate_trajectory_collection,
    visualize_trajectory,
    read_seed_file,
)
@dataclass
class Args:
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)

    exp_name: str = "gnn_ablation"
    seed: int = 1
    output_dir: str = "logs"
    device: str = "cpu"
    nn_device: str = "same"  # use same as device unless overridden
    torch_deterministic: bool = True
    torch_num_threads: int | None = None

    capture_video: bool = False
    env_mode: str = "async"

    # TensorBoard logging control (similar to train.py)
    no_tensorboard: bool = False
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
    num_envs: int = 10
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
    test_iterations: int = 1

    # Where to store ablation outputs (separate from global csv/)
    ablation_subdir: str = "ablation"

    dataset: DatasetArgs = field(
        default_factory=lambda: DatasetArgs(
            host_count=10,
            vm_count=10,
            workflow_count=1,
            gnp_min_n=20,
            gnp_max_n=20,
            max_memory_gb=10,
            min_cpu_speed=500,
            max_cpu_speed=5000,
            min_task_length=500,
            max_task_length=100_000,
            task_arrival="static",
            dag_method="gnp",
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
    # Robust multi-seed evaluation flags
    robust_eval_enable: bool = True
    robust_eval_alpha: float = 0.25
    robust_eval_seeds: list[int] | None = None
    hv_ref_margin: float = 1.05
    robust_eval_seeds_file: str | None = None

    training_seed_mode: str = "uncontrolled"
    train_seeds_file: str | None = None
    reseed_every: int = 5
    train_seed_pool_size: int = 128

    # Optional RL config JSONs to derive DatasetArgs per domain (wide/longcp) and seed lists
    wide_config: str | None = None
    longcp_config: str | None = None
    # Optional: use a fixed dataset (JSON with {"workflows": [...]}) for training envs
    dataset_json: str | None = None

    # Queue regime: optional divisor (1/alpha). If set, envs will generate datasets with this req_divisor
    # regardless of style, using the generic generator (gnp params still honored)
    dataset_req_divisor: int | None = None




def _log_scalar_csv(per_variant_dir: Path, variant: str, global_step: int, metric_name: str, value: float) -> None:
    path = per_variant_dir / f"{variant}_{metric_name}.csv"
    new_file = not path.exists()
    with path.open("a", newline="") as f:
        w = _csv.writer(f)
        if new_file:
            w.writerow(["Wall time", "Step", "Value"])
        w.writerow([time.time(), int(global_step), float(value)])


def _offline_plot_per_variant_metrics(out_dir: Path, var_name: str) -> None:
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[ablation][offline-plots] Skipping plots due to missing deps: {e}")
        return

    metrics = [
        ("makespan", "Makespan"),
        ("total_energy", "Total Energy"),
        ("active_energy", "Active Energy"),
        ("idle_energy", "Idle Energy"),
        ("active_plus_idle", "Active + Idle Energy"),
        ("active_energy_return", "Active Energy Return"),
        ("makespan_return", "Makespan Return"),
        ("policy_entropy", "Policy Entropy"),
        ("kl", "Approx KL"),
        ("pg_loss", "Policy Loss"),
        ("value_loss", "Value Loss"),
        ("grads_preclip_actor", "Grad Norm (Actor, pre-clip)"),
        ("grads_preclip_critic", "Grad Norm (Critic, pre-clip)"),
        ("grads_preclip_total", "Grad Norm (Total, pre-clip)"),
        ("grads_postclip_actor", "Grad Norm (Actor, post-clip)"),
        ("grads_postclip_critic", "Grad Norm (Critic, post-clip)"),
        ("grads_postclip_total", "Grad Norm (Total, post-clip)"),
    ]
    for key, title in metrics:
        csv_path = out_dir / f"{var_name}_{key}.csv"
        if not csv_path.exists():
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


def _train_one_variant_with_traj(args: Args, variant: AG.AblationVariant, device: torch.device, per_variant_dir: Path):
    run_tag = f"{variant.name}"
    print(f"[ablation+traj] Training variant: {run_tag}")

    # CPU threading config: prefer explicit torch_num_threads; else match num_envs if enabled
    try:
        class NullWriter:
            def add_text(self, *args, **kwargs):
                pass

            def add_scalar(self, *args, **kwargs):
                pass

            def close(self):
                pass

        if args.no_tensorboard:
            writer = NullWriter()
        else:
            tb_dir = Path(args.output_dir) / args.exp_name / "tb" / variant.name
            writer = SummaryWriter(str(tb_dir))
            try:
                writer.add_text("variant", variant.name)
            except Exception:
                pass
        # Pareto archive (makespan, active_energy)
        pareto_points: list[tuple[float, float, int, int]] = []
        pareto_models: list[dict] = []

        # Robust fronts (across evaluation seeds)
        mean_pf_points: list[tuple[float, float, int, int, str]] = []
        worst_pf_points: list[tuple[float, float, int, int, str]] = []
        cvar_pf_points: list[tuple[float, float, int, int, str]] = []

        def _dominates(p: tuple[float, float], q: tuple[float, float]) -> bool:
            return (p[0] <= q[0] and p[1] <= q[1]) and (p[0] < q[0] or p[1] < q[1])

        def _update_pareto(mk: float, ae: float, iter_no: int, step_no: int, model_sd: dict) -> bool:
            nonlocal pareto_points, pareto_models
            new_p = (float(mk), float(ae))
            # Remove dominated existing points
            keep_idx: list[int] = []
            for i, (pmk, pae, _it, _st) in enumerate(pareto_points):
                if _dominates(new_p, (pmk, pae)):
                    continue
                keep_idx.append(i)
            updated = (len(keep_idx) != len(pareto_points))
            pareto_points = [pareto_points[i] for i in keep_idx]
            pareto_models = [pareto_models[i] for i in keep_idx]
            # If new point is dominated by any kept point, do not add
            for (pmk, pae, _it, _st) in pareto_points:
                if _dominates((pmk, pae), new_p):
                    return updated
            # Add new non-dominated point
            pareto_points.append((new_p[0], new_p[1], int(iter_no), int(step_no)))
            pareto_models.append({k: v.cpu() for k, v in model_sd.items()})
            return True

        def _persist_pareto():
            pcsv = per_variant_dir / f"{variant.name}_pareto.csv"
            pcsv.parent.mkdir(parents=True, exist_ok=True)
            with pcsv.open("w", newline="") as f:
                w = _csv.writer(f)
                w.writerow(["makespan", "active_energy", "iteration", "global_step", "checkpoint"])
                for i, (mk, ae, it, st) in enumerate(pareto_points):
                    # Stable ID using training coordinates instead of a running index
                    ckpt_name = f"{variant.name}_pareto_it{int(it):06d}_st{int(st):09d}.pt"
                    ckpt = per_variant_dir / ckpt_name
                    torch.save({
                        "state_dict": pareto_models[i],
                        "metrics": {
                            "makespan": float(mk),
                            "active_energy": float(ae),
                            "iteration": int(it),
                            "global_step": int(st),
                        },
                    }, ckpt)
                    w.writerow([mk, ae, it, st, ckpt_name])

        def _dom(p: tuple[float, float], q: tuple[float, float]) -> bool:
            return (p[0] <= q[0] and p[1] <= q[1]) and (p[0] < q[0] or p[1] < q[1])

        def _update_front(front: list[tuple[float, float, int, int, str]], mk: float, en: float, it: int, st: int, ckpt: str) -> bool:
            keep: list[int] = []
            for i, (pmk, pen, _it, _st, _ck) in enumerate(front):
                if _dom((mk, en), (pmk, pen)):
                    continue
                keep.append(i)
            updated = (len(keep) != len(front))
            front[:] = [front[i] for i in keep]
            for (pmk, pen, _it, _st, _ck) in front:
                if _dom((pmk, pen), (mk, en)):
                    return updated
            front.append((float(mk), float(en), int(it), int(st), str(ckpt)))
            return True

        def _write_front_csv(front: list[tuple[float, float, int, int, str]], path: Path) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="") as f:
                w = _csv.writer(f)
                w.writerow(["makespan", "energy", "iteration", "global_step", "checkpoint"])
                for (pmk, pen, pit, pst, pck) in front:
                    w.writerow([pmk, pen, pit, pst, pck])

        def _hv_and_contrib(points: list[tuple[float, float, int, int, str]]):
            if not points:
                return 0.0, []
            ref_mk = max(p[0] for p in points) * float(getattr(args, "hv_ref_margin", 1.05))
            ref_en = max(p[1] for p in points) * float(getattr(args, "hv_ref_margin", 1.05))
            pts = sorted([(p[0], p[1]) for p in points], key=lambda x: (x[0], x[1]))

            def _hv_of(pts2: list[tuple[float, float]]):
                if not pts2:
                    return 0.0
                pts2s = sorted(pts2, key=lambda x: (x[0], x[1]))
                hv = 0.0
                prev_en = ref_en
                for mk_i, en_i in pts2s:
                    width = max(0.0, ref_mk - mk_i)
                    height = max(0.0, prev_en - en_i)
                    hv += width * height
                    prev_en = min(prev_en, en_i)
                return hv

            hv_total = _hv_of(pts)
            contrib = []
            for i in range(len(pts)):
                hv_wo = _hv_of([pts[j] for j in range(len(pts)) if j != i])
                contrib.append(max(0.0, hv_total - hv_wo))
            return hv_total, contrib

        def _knee_index(points: list[tuple[float, float, int, int, str]]):
            if len(points) == 0:
                return -1, 0.0
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            ix = int(np.argmin(xs))
            iy = int(np.argmin(ys))
            if ix == iy:
                return ix, 0.0
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            eps = 1e-9
            x_norm = [(x - x_min) / (x_max - x_min + eps) for x in xs]
            y_norm = [(y - y_min) / (y_max - y_min + eps) for y in ys]
            x1, y1 = x_norm[ix], y_norm[ix]
            x2, y2 = x_norm[iy], y_norm[iy]
            dx = x2 - x1
            dy = y2 - y1
            denom = math.sqrt(dx * dx + dy * dy) + 1e-12
            best_i = 0
            best_d = -1.0
            for i in range(len(points)):
                x0 = x_norm[i]
                y0 = y_norm[i]
                num = abs(dy * x0 - dx * y0 + (x2 * y1 - y2 * x1))
                d = num / denom
                if d > best_d:
                    best_d = d
                    best_i = i
            return best_i, best_d

        def _write_vm_specs_csv(_dir: Path, _name: str, _vms):
            try:
                p = _dir / _name
                if p.exists():
                    return
                with p.open("w", newline="") as f:
                    w = _csv.writer(f)
                    w.writerow(["vm_id", "cpu_cores", "memory_mb", "cpu_speed_mips"])
                    for v in _vms:
                        w.writerow([
                            int(getattr(v, "id", -1)),
                            int(getattr(v, "cpu_cores", -1)),
                            int(getattr(v, "memory_mb", -1)),
                            int(getattr(v, "cpu_speed_mips", -1)),
                        ])
                print(f"[vmspec] wrote {p}")
            except Exception as _e_vm:
                print(f"[vmspec] warn: failed to write VM specs: {_e_vm}")

        def _write_host_specs_csv(_dir: Path, _name: str, _hosts):
            try:
                p = _dir / _name
                if p.exists():
                    return
                with p.open("w", newline="") as f:
                    w = _csv.writer(f)
                    w.writerow(["host_id", "cores", "memory_mb", "cpu_speed_mips", "power_idle_watt", "power_peak_watt"])
                    for h in _hosts:
                        w.writerow([
                            int(getattr(h, "id", -1)),
                            int(getattr(h, "cores", -1)),
                            int(getattr(h, "memory_mb", -1)),
                            int(getattr(h, "cpu_speed_mips", -1)),
                            int(getattr(h, "power_idle_watt", -1)),
                            int(getattr(h, "power_peak_watt", -1)),
                        ])
                print(f"[vmspec] wrote {p}")
            except Exception as _e_host:
                print(f"[vmspec] warn: failed to write Host specs: {_e_host}")

        thread_count: Optional[int] = None
        if args.torch_num_threads and int(args.torch_num_threads) > 0:
            thread_count = int(args.torch_num_threads)
        elif bool(getattr(args, 'cpus_match_envs', True)):
            thread_count = int(max(1, args.num_envs))
        if thread_count is not None:
            torch.set_num_threads(thread_count)
            os.environ.setdefault("OMP_NUM_THREADS", str(thread_count))
            os.environ.setdefault("MKL_NUM_THREADS", str(thread_count))
            os.environ.setdefault("OPENBLAS_NUM_THREADS", str(thread_count))
            os.environ.setdefault("NUMEXPR_NUM_THREADS", str(thread_count))
            os.environ.setdefault("TORCH_NUM_THREADS", str(thread_count))
            try:
                avail_cpus = os.cpu_count()
            except Exception:
                avail_cpus = None
            print(
                f"[ablation+traj] CPU config: available_cpus={avail_cpus} | num_envs={args.num_envs} | threads={thread_count}"
            )
    except Exception as _e_threads:
        print(f"[ablation+traj] Warning: failed to set CPU threads: {_e_threads}")

    batch_size = int(args.num_envs * args.num_steps)
    minibatch_size = max(1, int(batch_size // max(1, args.num_minibatches)))
    num_iterations = max(1, int(args.total_timesteps // max(1, batch_size)))

    per_variant_dir.mkdir(parents=True, exist_ok=True)

    # Domain-aware env construction: if RL configs provided, build per-env Args with matching DatasetArgs
    # If dataset_json is provided, override and build envs from that dataset
    domain_mode = (bool(getattr(args, 'wide_config', None)) or bool(getattr(args, 'longcp_config', None))) and not bool(getattr(args, 'dataset_json', None))

    def _load_rl_cfg(path_str: str | None):
        if not path_str:
            return set(), None
        p = Path(path_str)
        if not p.exists():
            return set(), None
        with p.open('r') as f:
            cfg = json.load(f)
        train = cfg.get('train', {}) if isinstance(cfg, dict) else {}
        seeds = set(int(x) for x in train.get('seeds', []) if isinstance(x, (int, float, str)))
        ds = dict(train.get('dataset', {}))
        ds_args = DatasetArgs(
            style=str(ds.get('style', getattr(args.dataset, 'style', 'generic'))),
            host_count=int(ds.get('host_count', args.dataset.host_count)),
            vm_count=int(ds.get('vm_count', args.dataset.vm_count)),
            max_memory_gb=int(ds.get('max_memory_gb', args.dataset.max_memory_gb)),
            min_cpu_speed=int(ds.get('min_cpu_speed', args.dataset.min_cpu_speed)),
            max_cpu_speed=int(ds.get('max_cpu_speed', args.dataset.max_cpu_speed)),
            workflow_count=int(ds.get('workflow_count', args.dataset.workflow_count)),
            dag_method=str(ds.get('dag_method', args.dataset.dag_method)),
            gnp_min_n=int(ds.get('gnp_min_n', args.dataset.gnp_min_n)),
            gnp_max_n=int(ds.get('gnp_max_n', args.dataset.gnp_max_n)),
            gnp_p=float(ds['gnp_p']) if ('gnp_p' in ds and ds['gnp_p'] is not None) else args.dataset.gnp_p,
            task_length_dist=str(ds.get('task_length_dist', args.dataset.task_length_dist)),
            min_task_length=int(ds.get('min_task_length', args.dataset.min_task_length)),
            max_task_length=int(ds.get('max_task_length', args.dataset.max_task_length)),
            task_arrival=str(ds.get('task_arrival', args.dataset.task_arrival)),
            arrival_rate=float(ds.get('arrival_rate', args.dataset.arrival_rate)),
        )
        # Apply global req_divisor override if requested at top level
        try:
            req_div = getattr(args, 'dataset_req_divisor', None)
            
        except Exception as e:
            req_div = None
            print(f"Warning: Could not get dataset_req_divisor from args: {e}")
        if req_div is not None and ds_args is not None:
            ds_args = _dc_replace(ds_args, req_divisor=int(req_div))
        return seeds, ds_args

    seeds_wide_set, ds_wide = _load_rl_cfg(getattr(args, 'wide_config', None))
    seeds_long_set, ds_long = _load_rl_cfg(getattr(args, 'longcp_config', None))

    # Build per-env Args with fixed domain assignment (alternating if both domains provided)
    envs_args: list[Args] = []
    env_domains: list[str] = []  # 'wide' | 'longcp' | 'default'
    if domain_mode:
        for i in range(int(args.num_envs)):
            if (ds_wide is not None) and (ds_long is not None):
                dom = 'wide' if (i % 2 == 0) else 'longcp'
            elif ds_wide is not None:
                dom = 'wide'
            elif ds_long is not None:
                dom = 'longcp'
            else:
                dom = 'default'
            env_domains.append(dom)
            args_i = _dc_replace(args)
            if dom == 'wide' and ds_wide is not None:
                # Use req_divisor from ds_wide (set by _load_rl_cfg)
                req_div_val = getattr(ds_wide, 'req_divisor', None)
                args_i.dataset = _dc_replace(args.dataset,
                                             style=ds_wide.style,
                                             host_count=ds_wide.host_count,
                                             vm_count=ds_wide.vm_count,
                                             max_memory_gb=ds_wide.max_memory_gb,
                                             min_cpu_speed=ds_wide.min_cpu_speed,
                                             max_cpu_speed=ds_wide.max_cpu_speed,
                                             workflow_count=ds_wide.workflow_count,
                                             dag_method=ds_wide.dag_method,
                                             gnp_min_n=ds_wide.gnp_min_n,
                                             gnp_max_n=ds_wide.gnp_max_n,
                                             gnp_p=ds_wide.gnp_p,
                                             task_length_dist=ds_wide.task_length_dist,
                                             min_task_length=ds_wide.min_task_length,
                                             max_task_length=ds_wide.max_task_length,
                                             task_arrival=ds_wide.task_arrival,
                                             arrival_rate=ds_wide.arrival_rate,
                                             req_divisor=req_div_val,
                                             )
            elif dom == 'longcp' and ds_long is not None:
                # Use req_divisor from ds_long (set by _load_rl_cfg)
                req_div_val = getattr(ds_long, 'req_divisor', None)
                args_i.dataset = _dc_replace(args.dataset,
                                             style=ds_long.style,
                                             host_count=ds_long.host_count,
                                             vm_count=ds_long.vm_count,
                                             max_memory_gb=ds_long.max_memory_gb,
                                             min_cpu_speed=ds_long.min_cpu_speed,
                                             max_cpu_speed=ds_long.max_cpu_speed,
                                             workflow_count=ds_long.workflow_count,
                                             dag_method=ds_long.dag_method,
                                             gnp_min_n=ds_long.gnp_min_n,
                                             gnp_max_n=ds_long.gnp_max_n,
                                             gnp_p=ds_long.gnp_p,
                                             task_length_dist=ds_long.task_length_dist,
                                             min_task_length=ds_long.min_task_length,
                                             max_task_length=ds_long.max_task_length,
                                             task_arrival=ds_long.task_arrival,
                                             arrival_rate=ds_long.arrival_rate,
                                             req_divisor=req_div_val,
                                             )
            else:
                # default fallback
                pass
            envs_args.append(args_i)
    else:
        env_domains = ['default'] * int(args.num_envs)
        # If requested, override dataset.req_divisor on the base args before replication
        try:
            req_div = getattr(args, 'dataset_req_divisor', None)
        except Exception:
            req_div = None
        if req_div is not None:
            args.dataset = _dc_replace(args.dataset, req_divisor=int(req_div))
        envs_args = [args] * int(args.num_envs)

    if getattr(args, 'dataset_json', None):
        # Load dataset and construct fixed-dataset envs
        p = Path(str(args.dataset_json))
        with p.open('r') as f:
            data = json.load(f)
        # If vms/hosts missing, synthesize them using host_specs.json (and ensure memory sufficiency)
        
        if not isinstance(data, dict):
            raise RuntimeError("dataset_json must be a JSON object with at least 'workflows'")
        if ("vms" not in data) or ("hosts" not in data):
            rng = np.random.default_rng(int(getattr(args, 'seed', 0)))
            workflows = list(data.get("workflows", []))
            # Compute max task memory and cores requirement across workflows
            max_req_mb = 0
            max_req_cores = 0
            for wf in workflows:
                for t in wf.get("tasks", []):
                    try:
                        max_req_mb = max(max_req_mb, int(t.get("req_memory_mb", 0)))
                        max_req_cores = max(max_req_cores, int(t.get("req_cpu_cores", 0)))
                    except Exception:
                        pass
            # Load available host specs
            with open(HOST_SPECS_PATH, 'r') as hf:
                specs = json.load(hf)
            # Ensure at least one host can satisfy max_req_mb
            specs_sorted = sorted(specs, key=lambda s: int(s.get("memory_gb", 0)), reverse=True)
            host_count = int(getattr(args.dataset, 'host_count', 4))
            vm_count = int(getattr(args.dataset, 'vm_count', 10))
            hosts: list[dict] = []
            # Place the largest-memory host first
            if len(specs_sorted) == 0:
                raise RuntimeError(f"Empty HOST_SPECS_PATH: {HOST_SPECS_PATH}")
            largest = specs_sorted[0]
            # Convert GB to MB etc.
            def _host_from_spec(i: int, spec: dict) -> dict:
                return {
                    "id": int(i),
                    "cores": int(spec.get("cores", 1)),
                    "cpu_speed_mips": int(float(spec.get("cpu_speed_gips", 1.0)) * 1e3),
                    "memory_mb": int(float(spec.get("memory_gb", 1.0)) * 1024),
                    "disk_mb": int(float(spec.get("disk_tb", 0.0)) * 1e6),
                    "bandwidth_mbps": int(float(spec.get("bandwidth_gbps", 1.0)) * 1024),
                    "power_idle_watt": int(spec.get("power_idle_watt", 0)),
                    "power_peak_watt": int(spec.get("power_peak_watt", 0)),
                }
            hosts.append(_host_from_spec(0, largest))
            # Fill remaining hosts by random sampling from specs
            for i in range(1, max(1, host_count)):
                spec = specs_sorted[i % len(specs_sorted)]
                hosts.append(_host_from_spec(i, spec))

            # Build VMs: one per requested vm_count, ensure at least one on the largest host (hosts[0])
            vms: list[dict] = []
            for i in range(vm_count):
                h = hosts[i % len(hosts)]
                vms.append({
                    "id": int(i),
                    "host_id": int(h["id"]),
                    "cpu_speed_mips": int(h["cpu_speed_mips"]),
                    "memory_mb": int(h["memory_mb"]),
                    "cpu_cores": int(h["cores"]),
                    "disk_mb": int(h["disk_mb"]),
                    "bandwidth_mbps": int(h["bandwidth_mbps"]),
                    "vmm": "Xen",
                })
            # If none of the hosts can satisfy memory/cores, upscale the first host (rare fallback)

        else:
            fixed_ds = Dataset.from_json(data)
        try:
            _write_vm_specs_csv(per_variant_dir, f"{variant.name}_vm_specs_train.csv", fixed_ds.vms)
            _write_host_specs_csv(per_variant_dir, f"{variant.name}_host_specs_train.csv", fixed_ds.hosts)
        except Exception as _e_log_train_ds:
            print(f"[vmspec][train] warn: failed to log dataset_json specs: {_e_log_train_ds}")
        def _make_env_dataset():
            env: gym.Env = CloudSchedulingGymEnvironment(
                dataset=fixed_ds,
                collect_timelines=False,
                compute_metrics=False,
                profile=False,
                fixed_env_seed=False,
                dataset_episode_mode="single",
            )
            env = GinAgentWrapper(env)
            return RecordEpisodeStatistics(env)
        envs = gym.vector.SyncVectorEnv([_make_env_dataset for _ in range(int(args.num_envs))])
    else:
        envs = gym.vector.SyncVectorEnv([AG._make_env_thunk(i, envs_args[i]) for i in range(int(args.num_envs))])
    obs_space = envs.single_observation_space
    act_space = envs.single_action_space
    assert isinstance(act_space, gym.spaces.Discrete)
    if domain_mode:
        try:
            if ds_wide is not None:
                print(f"Generating wide dataset with seed {args.seed}")
                _dsw = CloudSchedulingGymEnvironment.gen_dataset(int(args.seed), ds_wide)
                _write_vm_specs_csv(per_variant_dir, f"{variant.name}_vm_specs_train_wide.csv", _dsw.vms)
                _write_host_specs_csv(per_variant_dir, f"{variant.name}_host_specs_train_wide.csv", _dsw.hosts)
            if ds_long is not None:
                print(f"Generating long dataset with seed {args.seed}")
                _dsl = CloudSchedulingGymEnvironment.gen_dataset(int(args.seed), ds_long)
                _write_vm_specs_csv(per_variant_dir, f"{variant.name}_vm_specs_train_longcp.csv", _dsl.vms)
                _write_host_specs_csv(per_variant_dir, f"{variant.name}_host_specs_train_longcp.csv", _dsl.hosts)
        except Exception as _e_log_train_ds:
            print(f"[vmspec][train] warn: failed to log domain-mode specs: {_e_log_train_ds}")
    # Ensure observation/action spaces are available when using a fixed dataset as well
    if 'obs_space' not in locals() or 'act_space' not in locals():
        obs_space = envs.single_observation_space
        act_space = envs.single_action_space
        assert isinstance(act_space, gym.spaces.Discrete)

    # Instantiate agent once environments and spaces are ready
    agent = AG.AblationGinAgent(device, variant)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Trajectory collection
    traj_collector = integrate_trajectory_collection(
        actor_model=agent.actor,
        variant_name=variant.name,
        log_dir=per_variant_dir,
        config=args.trajectory,
    )

    obs = torch.zeros((args.num_steps, args.num_envs) + obs_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + act_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    next_obs_tensor = None
    next_done_tensor = None

    seed_pool = []
    # Domain-specific seed pools (if domain mode)
    pool_wide: list[int] = []
    pool_long: list[int] = []
    current_block_idx = -1
    block_len = int(getattr(args, 'reseed_every', 5))
    if str(getattr(args, 'training_seed_mode', 'uncontrolled')) == 'controlled':
        fp = getattr(args, 'train_seeds_file', None)
        if fp and isinstance(fp, str):
            tmp = read_seed_file(fp)
            seed_pool = list(tmp) if isinstance(tmp, list) else []
        # Domain pools from provided lists intersected with RL config seeds when available
        if domain_mode:
            if isinstance(seed_pool, list) and len(seed_pool) > 0:
                pool_wide = [int(s) for s in seed_pool if (s in seeds_wide_set)]
                pool_long = [int(s) for s in seed_pool if (s in seeds_long_set)]
            else:
                pool_wide = sorted(list(seeds_wide_set)) if len(seeds_wide_set) > 0 else []
                pool_long = sorted(list(seeds_long_set)) if len(seeds_long_set) > 0 else []
        if len(seed_pool) == 0 and not domain_mode:
            try:
                rng = np.random.default_rng(int(args.seed))
                n = int(getattr(args, 'train_seed_pool_size', 128))
                seed_pool = rng.integers(1, 2**31 - 1, size=n, dtype=np.int64).tolist()
            except Exception:
                seed_pool = [int(args.seed + i) for i in range(int(args.num_envs))]
        try:
            _src = (f"file:{fp}" if (fp and isinstance(fp, str) and len(seed_pool) > 0) else ("generated" if len(seed_pool) > 0 else "none"))
        except Exception:
            _src = "unknown"
        try:
            _head = seed_pool[:min(10, len(seed_pool))]
        except Exception:
            _head = []
        print(f"[seeds][train] mode=controlled source={_src} count={len(seed_pool)} head={_head}")
        if domain_mode:
            try:
                print(f"[seeds][train][domain] wide_pool={len(pool_wide)} long_pool={len(pool_long)} domains={env_domains}")
            except Exception:
                pass

    from tqdm import tqdm
    pbar = tqdm(total=args.total_timesteps)

    # Helpers to pick per-seed DatasetArgs during eval
    def _ds_for_seed(sd: int):
        if not domain_mode:
            return args.dataset
        if (sd in seeds_wide_set) and (ds_wide is not None):
            return ds_wide
        if (sd in seeds_long_set) and (ds_long is not None):
            return ds_long
        # Heuristic fallback by seed range
        try:
            if sd >= 200000 and ds_wide is not None:
                return ds_wide
            if sd < 200000 and ds_long is not None:
                return ds_long
        except Exception:
            pass
        return args.dataset

    # Single-episode eval with active energy only
    def _eval_one_episode(agent: AG.Agent, args: Args, seed: int, ds_override: DatasetArgs | None = None) -> tuple[float, float]:
        # Build a temp Args with dataset override if provided
        if ds_override is not None:
            tmp_args = _dc_replace(args)
            tmp_args.dataset = _dc_replace(args.dataset,
                                           host_count=ds_override.host_count,
                                           vm_count=ds_override.vm_count,
                                           max_memory_gb=ds_override.max_memory_gb,
                                           min_cpu_speed=ds_override.min_cpu_speed,
                                           max_cpu_speed=ds_override.max_cpu_speed,
                                           workflow_count=ds_override.workflow_count,
                                           dag_method=ds_override.dag_method,
                                           gnp_min_n=ds_override.gnp_min_n,
                                           gnp_max_n=ds_override.gnp_max_n,
                                           gnp_p=ds_override.gnp_p,
                                           task_length_dist=ds_override.task_length_dist,
                                           min_task_length=ds_override.min_task_length,
                                           max_task_length=ds_override.max_task_length,
                                           task_arrival=ds_override.task_arrival,
                                           arrival_rate=ds_override.arrival_rate)
            env = AG._make_test_env(tmp_args)
        else:
            # If a fixed dataset JSON was used for training, evaluate on the same fixed dataset
            if getattr(args, 'dataset_json', None):
                p = Path(str(args.dataset_json))
                with p.open('r') as f:
                    data = json.load(f)
                if not isinstance(data, dict):
                    raise RuntimeError("dataset_json must be a JSON object with at least 'workflows'")
                if ("vms" not in data) or ("hosts" not in data):
                    rng = np.random.default_rng(int(getattr(args, 'seed', 0)))
                    workflows = list(data.get("workflows", []))
                    # Compute max task memory and cores requirement across workflows
                    max_req_mb = 0
                    max_req_cores = 0
                    for wf in workflows:
                        for t in wf.get("tasks", []):
                            try:
                                max_req_mb = max(max_req_mb, int(t.get("req_memory_mb", 0)))
                                max_req_cores = max(max_req_cores, int(t.get("req_cpu_cores", 0)))
                            except Exception:
                                pass
                    # Load available host specs
                    with open(HOST_SPECS_PATH, 'r') as hf:
                        specs = json.load(hf)
                    # Ensure at least one host can satisfy max_req_mb
                    specs_sorted = sorted(specs, key=lambda s: int(s.get("memory_gb", 0)), reverse=True)
                    host_count = int(getattr(args.dataset, 'host_count', 4))
                    vm_count = int(getattr(args.dataset, 'vm_count', 10))
                    hosts: list[dict] = []
                    # Place the largest-memory host first
                    if len(specs_sorted) == 0:
                        raise RuntimeError(f"Empty HOST_SPECS_PATH: {HOST_SPECS_PATH}")
                    largest = specs_sorted[0]
                    # Convert GB to MB etc.
                    def _host_from_spec(i: int, spec: dict) -> dict:
                        return {
                            "id": int(i),
                            "cores": int(spec.get("cores", 1)),
                            "cpu_speed_mips": int(float(spec.get("cpu_speed_gips", 1.0)) * 1e3),
                            "memory_mb": int(float(spec.get("memory_gb", 1.0)) * 1024),
                            "disk_mb": int(float(spec.get("disk_tb", 0.0)) * 1e6),
                            "bandwidth_mbps": int(float(spec.get("bandwidth_gbps", 1.0)) * 1024),
                            "power_idle_watt": int(spec.get("power_idle_watt", 0)),
                            "power_peak_watt": int(spec.get("power_peak_watt", 0)),
                        }
                    hosts.append(_host_from_spec(0, largest))
                    # Fill remaining hosts by random sampling from specs
                    for i in range(1, max(1, host_count)):
                        spec = specs_sorted[i % len(specs_sorted)]
                        hosts.append(_host_from_spec(i, spec))
                    # Build VMs: one per requested vm_count, ensure at least one on the largest host (hosts[0])
                    vms: list[dict] = []
                    for i in range(vm_count):
                        h = hosts[i % len(hosts)]
                        vms.append({
                            "id": int(i),
                            "host_id": int(h["id"]),
                            "cpu_speed_mips": int(h["cpu_speed_mips"]),
                            "memory_mb": int(h["memory_mb"]),
                            "cpu_cores": int(h["cores"]),
                            "disk_mb": int(h["disk_mb"]),
                            "bandwidth_mbps": int(h["bandwidth_mbps"]),
                            "vmm": "Xen",
                        })
                    # If none of the hosts can satisfy memory/cores, upscale the first host
                    if max_req_mb > max(h["memory_mb"] for h in hosts):
                        scale_mb = int(max_req_mb)
                        hosts[0]["memory_mb"] = scale_mb
                        for v in vms:
                            if v["host_id"] == hosts[0]["id"]:
                                v["memory_mb"] = scale_mb
                    if max_req_cores > max(h["cores"] for h in hosts):
                        scale_cores = int(max_req_cores)
                        hosts[0]["cores"] = scale_cores
                        for v in vms:
                            if v["host_id"] == hosts[0]["id"]:
                                v["cpu_cores"] = scale_cores
                    full = {"workflows": workflows, "vms": vms, "hosts": hosts}
                    fixed_ds = Dataset.from_json(full)
                else:
                    fixed_ds = Dataset.from_json(data)
                env: gym.Env = CloudSchedulingGymEnvironment(
                    dataset=fixed_ds,
                    collect_timelines=False,
                    compute_metrics=True,
                    profile=False,
                    fixed_env_seed=False,
                    dataset_episode_mode="single",
                )
                env = GinAgentWrapper(env)
            else:
                env = AG._make_test_env(args)
        next_obs, _ = env.reset(seed=int(seed))
        final_info: dict | None = None
        while True:
            obs_tensor = torch.from_numpy(np.asarray(next_obs, dtype=np.float32).reshape(1, -1))
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(obs_tensor)
            next_obs, _rew, terminated, truncated, info = env.step(int(action.item()))
            if terminated or truncated:
                final_info = info
                break
        assert env.prev_obs is not None
        mk = float(env.prev_obs.makespan())
        ae = float(final_info.get("total_energy_active", 0.0)) if isinstance(final_info, dict) and ("total_energy_active" in final_info) else 0.0
        env.close()
        return mk, ae

    try:
        for iteration in range(1, num_iterations + 1):
            if args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / num_iterations
                optimizer.param_groups[0]["lr"] = frac * args.learning_rate

            if str(getattr(args, 'training_seed_mode', 'uncontrolled')) == 'controlled':
                blk_idx = (iteration - 1) // max(1, block_len)
                if blk_idx != current_block_idx and ((len(seed_pool) > 0) or domain_mode):
                    current_block_idx = blk_idx
                    if domain_mode:
                        # Build seeds per env based on its domain assignment
                        seeds_block: list[int] = []
                        # simple rolling indices
                        if not hasattr(_train_one_variant_with_traj, "_wi"):
                            setattr(_train_one_variant_with_traj, "_wi", 0)
                            setattr(_train_one_variant_with_traj, "_li", 0)
                        wi = int(getattr(_train_one_variant_with_traj, "_wi"))
                        li = int(getattr(_train_one_variant_with_traj, "_li"))
                        nw = max(1, len(pool_wide)) if len(pool_wide) > 0 else 1
                        nl = max(1, len(pool_long)) if len(pool_long) > 0 else 1
                        for i_env, dom in enumerate(env_domains):
                            if dom == 'wide' and len(pool_wide) > 0:
                                s = int(pool_wide[wi % nw]); wi += 1
                            elif dom == 'longcp' and len(pool_long) > 0:
                                s = int(pool_long[li % nl]); li += 1
                            else:
                                # fallback from flat pool or deterministic default
                                if len(seed_pool) > 0:
                                    base = (blk_idx * int(args.num_envs) + i_env) % len(seed_pool)
                                    s = int(seed_pool[base])
                                else:
                                    s = int(args.seed + i_env)
                            seeds_block.append(s)
                        setattr(_train_one_variant_with_traj, "_wi", wi)
                        setattr(_train_one_variant_with_traj, "_li", li)
                    else:
                        start = (blk_idx * int(args.num_envs)) % max(1, len(seed_pool))
                        seeds_block = [int(seed_pool[(start + j) % len(seed_pool)]) for j in range(int(args.num_envs))]
                    next_obs, _ = envs.reset(seed=seeds_block)
                    next_obs_tensor = torch.Tensor(next_obs).to(device)
                    next_done_tensor = torch.zeros(args.num_envs).to(device)
                    try:
                        print(f"[seeds][train] iteration={iteration} block_idx={blk_idx} seeds={seeds_block}")
                    except Exception:
                        pass
            else:
                if iteration == 1:
                    if getattr(args, 'dataset_json', None):
                        # Diversify sub-env starting workflow by unique seeds per env
                        seeds_block = [int(args.seed + i) for i in range(int(args.num_envs))]
                        print(f"[dataset][train] fixed_dataset='{args.dataset_json}' episodes=per-workflow seeds={seeds_block}")
                        next_obs, _ = envs.reset(seed=seeds_block)
                    else:
                        next_obs, _ = envs.reset(seed=args.seed)
                    next_obs_tensor = torch.Tensor(next_obs).to(device)
                    next_done_tensor = torch.zeros(args.num_envs).to(device)
                    try:
                        if not getattr(args, 'dataset_json', None):
                            print(f"[seeds][train] mode=uncontrolled initial_seed={int(args.seed)}")
                    except Exception:
                        pass

            for step in range(args.num_steps):
                global_step += args.num_envs
                obs[step] = next_obs_tensor
                dones[step] = next_done_tensor
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs_tensor)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                next_obs, reward, terminated, truncated, infos = envs.step(actions[step].cpu().numpy())
                next_done = np.logical_or(terminated, truncated)
                rewards[step] = torch.Tensor(reward).to(device).view(-1)
                next_obs_tensor, next_done_tensor = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

                if isinstance(infos, dict) and "final_info" in infos:
                    for info in infos["final_info"]:
                        if info and "episode" in info:
                            try:
                                pbar.update(global_step - pbar.n)
                            except Exception:
                                pass
                            try:
                                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                            except Exception:
                                pass

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
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()
                    with torch.no_grad():
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

                    # Periodic CSV logs (entropy/KL/losses)
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
                        _log_scalar_csv(per_variant_dir, variant.name, int(global_step), "policy_entropy", ent_val)
                        _log_scalar_csv(per_variant_dir, variant.name, int(global_step), "kl", kl_val)
                        _log_scalar_csv(per_variant_dir, variant.name, int(global_step), "pg_loss", pg_val)
                        _log_scalar_csv(per_variant_dir, variant.name, int(global_step), "value_loss", v_val)

                    optimizer.zero_grad()
                    loss.backward()

                    # Gradient norm logging pre/post clip
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
                        _log_scalar_csv(per_variant_dir, variant.name, int(global_step), "grads_preclip_actor", pre_actor)
                        _log_scalar_csv(per_variant_dir, variant.name, int(global_step), "grads_preclip_critic", pre_critic)
                        _log_scalar_csv(per_variant_dir, variant.name, int(global_step), "grads_preclip_total", pre_total)

                    torch.nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)

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
                        _log_scalar_csv(per_variant_dir, variant.name, int(global_step), "grads_postclip_actor", post_actor)
                        _log_scalar_csv(per_variant_dir, variant.name, int(global_step), "grads_postclip_critic", post_critic)
                        _log_scalar_csv(per_variant_dir, variant.name, int(global_step), "grads_postclip_total", post_total)

                    optimizer.step()

                # Collect trajectory snapshot after finishing the full PPO update (once per iteration)
                if traj_collector is not None and (iteration % max(1, int(args.trajectory.collect_every)) == 0):
                    try:
                        named_params = dict(agent.actor.named_parameters())
                        traj_collector.collect(named_params, step=iteration)
                    except Exception as e:
                        print(f"[traj][{run_tag}] warn: failed to collect snapshot: {e}")

            # Guarantee snapshots for short runs / delayed init:
            if traj_collector is not None:
                try:
                    # If initial snapshot was skipped (e.g., uninitialized params), take one at iter 1
                    if iteration == 1 and len(traj_collector.get_parameters()) == 0:
                        named_params = dict(agent.actor.named_parameters())
                        traj_collector.collect(named_params, step=iteration)
                    # On the final iteration, ensure we have at least two snapshots for visualization
                    if iteration == num_iterations and len(traj_collector.get_parameters()) < 2:
                        named_params = dict(agent.actor.named_parameters())
                        traj_collector.collect(named_params, step=iteration)
                except Exception as e:
                    print(f"[traj][{run_tag}] warn: post-iter snapshot failed: {e}")

            if args.target_kl is not None and float(approx_kl.detach().item()) > args.target_kl:
                pass

            # Periodic eval with robust multi-seed metrics only
            if args.test_every_iters > 0 and (iteration % args.test_every_iters == 0 or iteration == num_iterations):
                with torch.no_grad():
                    if bool(getattr(args, "robust_eval_enable", True)):
                        eval_seeds: list[int]
                        _eval_src = "default"
                        if args.robust_eval_seeds and len(args.robust_eval_seeds) > 0:
                            eval_seeds = [int(s) for s in list(args.robust_eval_seeds)]
                            _eval_src = "cli:list"
                        else:
                            fp = getattr(args, "robust_eval_seeds_file", None)
                            if fp and isinstance(fp, str):
                                tmp = read_seed_file(fp)
                                if isinstance(tmp, list) and len(tmp) > 0:
                                    eval_seeds = [int(s) for s in tmp]
                                    _eval_src = f"file:{fp}"
                                else:
                                    eval_seeds = [int(args.seed + i) for i in range(int(args.num_envs))]
                                    _eval_src = "default"
                            else:
                                eval_seeds = [int(args.seed + i) for i in range(int(args.num_envs))]
                                _eval_src = "default"
                        try:
                            print(f"[seeds][eval] source={_eval_src} count={len(eval_seeds)} seeds={eval_seeds}")
                        except Exception:
                            pass
                        try:
                            if getattr(args, 'dataset_json', None):
                                _p = Path(str(args.dataset_json))
                                with _p.open('r') as _f:
                                    _data = json.load(_f)
                                if isinstance(_data, dict) and ("vms" in _data) and ("hosts" in _data):
                                    from cogito.dataset_generator.core.models import Dataset as _Ds
                                    _fixed = _Ds.from_json(_data)
                                else:
                                    _rng = np.random.default_rng(int(getattr(args, 'seed', 0)))
                                    _wfs = list(_data.get("workflows", []))
                                    _max_mb = 0
                                    _max_cores = 0
                                    for _wf in _wfs:
                                        for _t in _wf.get("tasks", []):
                                            try:
                                                _max_mb = max(_max_mb, int(_t.get("req_memory_mb", 0)))
                                                _max_cores = max(_max_cores, int(_t.get("req_cpu_cores", 0)))
                                            except Exception:
                                                pass
                                    with open(HOST_SPECS_PATH, 'r') as _hf:
                                        _specs = json.load(_hf)
                                    _specs_sorted = sorted(_specs, key=lambda s: int(s.get("memory_gb", 0)), reverse=True)
                                    _H = int(getattr(args.dataset, 'host_count', 4))
                                    _V = int(getattr(args.dataset, 'vm_count', 10))
                                    _hosts: list[dict] = []
                                    _largest = _specs_sorted[0]
                                    def _h_from(i: int, s: dict) -> dict:
                                        return {
                                            "id": int(i),
                                            "cores": int(s.get("cores", 1)),
                                            "cpu_speed_mips": int(float(s.get("cpu_speed_gips", 1.0)) * 1e3),
                                            "memory_mb": int(float(s.get("memory_gb", 1.0)) * 1024),
                                            "disk_mb": int(float(s.get("disk_tb", 0.0)) * 1e6),
                                            "bandwidth_mbps": int(float(s.get("bandwidth_gbps", 1.0)) * 1024),
                                            "power_idle_watt": int(s.get("power_idle_watt", 0)),
                                            "power_peak_watt": int(s.get("power_peak_watt", 0)),
                                        }
                                    _hosts.append(_h_from(0, _largest))
                                    for _i in range(1, max(1, _H)):
                                        _spec = _specs_sorted[_i % len(_specs_sorted)]
                                        _hosts.append(_h_from(_i, _spec))
                                    _vms: list[dict] = []
                                    for _i in range(_V):
                                        _h = _hosts[_i % len(_hosts)]
                                        _vms.append({
                                            "id": int(_i),
                                            "host_id": int(_h["id"]),
                                            "cpu_speed_mips": int(_h["cpu_speed_mips"]),
                                            "memory_mb": int(_h["memory_mb"]),
                                            "cpu_cores": int(_h["cores"]),
                                            "disk_mb": int(_h["disk_mb"]),
                                            "bandwidth_mbps": int(_h["bandwidth_mbps"]),
                                            "vmm": "Xen",
                                        })
                                    if _max_mb > max(h["memory_mb"] for h in _hosts):
                                        _scale_mb = int(_max_mb)
                                        _hosts[0]["memory_mb"] = _scale_mb
                                        for _v in _vms:
                                            if _v["host_id"] == _hosts[0]["id"]:
                                                _v["memory_mb"] = _scale_mb
                                    if _max_cores > max(h["cores"] for h in _hosts):
                                        _scale_c = int(_max_cores)
                                        _hosts[0]["cores"] = _scale_c
                                        for _v in _vms:
                                            if _v["host_id"] == _hosts[0]["id"]:
                                                _v["cpu_cores"] = _scale_c
                                    from cogito.dataset_generator.core.models import Dataset as _Ds
                                    _fixed = _Ds.from_json({"workflows": _wfs, "vms": _vms, "hosts": _hosts})
                                _write_vm_specs_csv(per_variant_dir, f"{variant.name}_vm_specs_eval.csv", _fixed.vms)
                                _write_host_specs_csv(per_variant_dir, f"{variant.name}_host_specs_eval.csv", _fixed.hosts)
                            else:
                                if ds_wide is not None:
                                    _dsw = CloudSchedulingGymEnvironment.gen_dataset(int(args.seed), ds_wide)
                                    _write_vm_specs_csv(per_variant_dir, f"{variant.name}_vm_specs_eval_wide.csv", _dsw.vms)
                                    _write_host_specs_csv(per_variant_dir, f"{variant.name}_host_specs_eval_wide.csv", _dsw.hosts)
                                if ds_long is not None:
                                    _dsl = CloudSchedulingGymEnvironment.gen_dataset(int(args.seed), ds_long)
                                    _write_vm_specs_csv(per_variant_dir, f"{variant.name}_vm_specs_eval_longcp.csv", _dsl.vms)
                                    _write_host_specs_csv(per_variant_dir, f"{variant.name}_host_specs_eval_longcp.csv", _dsl.hosts)
                        except Exception as _e_eval_specs:
                            print(f"[vmspec][eval] warn: failed to log eval specs: {_e_eval_specs}")
                        mk_list: list[float] = []
                        en_list: list[float] = []
                        for sd in eval_seeds:
                            ds_over = _ds_for_seed(int(sd)) if domain_mode else None
                            mk_i, en_i = _eval_one_episode(agent, args, int(sd), ds_over)
                            mk_list.append(float(mk_i))
                            en_list.append(float(en_i))
                        if len(mk_list) > 0 and len(en_list) > 0:
                            mean_mk = float(np.mean(mk_list))
                            mean_en = float(np.mean(en_list))
                            worst_mk = float(np.max(mk_list))
                            worst_en = float(np.max(en_list))
                            alpha = float(getattr(args, "robust_eval_alpha", 0.25))
                            k = max(1, int(math.ceil(alpha * max(1, len(mk_list)))))
                            mk_sorted = sorted(mk_list, reverse=True)
                            en_sorted = sorted(en_list, reverse=True)
                            cvar_mk = float(np.mean(mk_sorted[:k]))
                            cvar_en = float(np.mean(en_sorted[:k]))

                            try:
                                writer.add_scalar(f"eval/{variant.name}/mean_makespan", mean_mk, int(global_step))
                                writer.add_scalar(f"eval/{variant.name}/mean_active_energy", mean_en, int(global_step))
                                writer.add_scalar(f"eval/{variant.name}/worst_makespan", worst_mk, int(global_step))
                                writer.add_scalar(f"eval/{variant.name}/worst_active_energy", worst_en, int(global_step))
                                writer.add_scalar(f"eval/{variant.name}/cvar_makespan", cvar_mk, int(global_step))
                                writer.add_scalar(f"eval/{variant.name}/cvar_active_energy", cvar_en, int(global_step))
                            except Exception:
                                pass

                            ck_mean = f"{variant.name}_mean_pf_it{int(iteration):05d}_st{int(global_step):09d}.pt"
                            ck_worst = f"{variant.name}_worst_pf_it{int(iteration):05d}_st{int(global_step):09d}.pt"
                            ck_cvar = f"{variant.name}_cvar_pf_it{int(iteration):05d}_st{int(global_step):09d}.pt"

                            # Log key scalar metrics for offline plotting (time series)
                            try:
                                _log_scalar_csv(per_variant_dir, variant.name, int(global_step), "makespan", mean_mk)
                                _log_scalar_csv(per_variant_dir, variant.name, int(global_step), "active_energy", mean_en)
                            except Exception as _e_log_eval:
                                print(f"[ablation][eval] warn: failed to log eval scalars: {_e_log_eval}")

                            um = _update_front(mean_pf_points, mean_mk, mean_en, iteration, int(global_step), ck_mean)
                            if um:
                                try:
                                    torch.save(agent.state_dict(), per_variant_dir / ck_mean)
                                except Exception as _e_svm:
                                    print(f"[ablation+traj] Warning: failed to save mean PF checkpoint: {_e_svm}")
                                _write_front_csv(mean_pf_points, per_variant_dir / f"{variant.name}_mean_pareto.csv")

                            uw = _update_front(worst_pf_points, worst_mk, worst_en, iteration, int(global_step), ck_worst)
                            if uw:
                                try:
                                    torch.save(agent.state_dict(), per_variant_dir / ck_worst)
                                except Exception as _e_svw:
                                    print(f"[ablation+traj] Warning: failed to save worst PF checkpoint: {_e_svw}")
                                _write_front_csv(worst_pf_points, per_variant_dir / f"{variant.name}_worst_pareto.csv")

                            uc = _update_front(cvar_pf_points, cvar_mk, cvar_en, iteration, int(global_step), ck_cvar)
                            if uc:
                                try:
                                    torch.save(agent.state_dict(), per_variant_dir / ck_cvar)
                                except Exception as _e_svc:
                                    print(f"[ablation+traj] Warning: failed to save CVaR PF checkpoint: {_e_svc}")
                                _write_front_csv(cvar_pf_points, per_variant_dir / f"{variant.name}_cvar_pareto.csv")

                            if len(mean_pf_points) > 0:
                                hv_total, contribs = _hv_and_contrib(mean_pf_points)
                                try:
                                    idx_hv = int(np.argmax(np.asarray(contribs) if len(contribs) > 0 else np.asarray([0.0])))
                                except Exception:
                                    idx_hv = 0
                                mk_hv, en_hv, it_hv, st_hv, ck_hv = mean_pf_points[idx_hv]
                                path_hv = per_variant_dir / f"{variant.name}_mean_pf_hvbest.csv"
                                new_file_hv = not path_hv.exists()
                                with path_hv.open("a", newline="") as f:
                                    w = _csv.writer(f)
                                    if new_file_hv:
                                        w.writerow(["iteration", "global_step", "checkpoint", "makespan", "energy", "hv_total", "hv_contrib"])
                                    w.writerow([int(it_hv), int(st_hv), ck_hv, float(mk_hv), float(en_hv), float(hv_total), float(contribs[idx_hv] if len(contribs)>0 else hv_total)])

                                idx_knee, dist_knee = _knee_index(mean_pf_points)
                                if idx_knee >= 0:
                                    mk_k, en_k, it_k, st_k, ck_k = mean_pf_points[idx_knee]
                                    path_knee = per_variant_dir / f"{variant.name}_mean_pf_knee.csv"
                                    new_file_k = not path_knee.exists()
                                    with path_knee.open("a", newline="") as f:
                                        w = _csv.writer(f)
                                        if new_file_k:
                                            w.writerow(["iteration", "global_step", "checkpoint", "makespan", "energy", "knee_distance"])
                                        w.writerow([int(it_k), int(st_k), ck_k, float(mk_k), float(en_k), float(dist_knee)])

        # Ensure final parameter snapshot at the last iteration (explicitly capture final state)
        if traj_collector is not None:
            try:
                named_params = dict(agent.actor.named_parameters())
                traj_collector.collect(named_params, step=num_iterations)
                print(f"[traj][{run_tag}] collected final snapshot at iter {num_iterations}")
            except Exception as e:
                print(f"[traj][{run_tag}] warn: failed to collect final snapshot: {e}")

        # After training: offline plots
        if getattr(args, "offline_plots_after_training", True):
            _offline_plot_per_variant_metrics(per_variant_dir, variant.name)

        # Trajectory visualization
        if traj_collector is not None:
            try:
                # Always save the base trajectory plots
                visualize_trajectory(
                    actor_model=agent.actor,
                    collector=traj_collector,
                    variant_name=variant.name,
                    log_dir=per_variant_dir,
                    config=args.trajectory,
                    eval_function=None,
                )

                # Optional: loss landscapes for makespan and active energy
                if bool(getattr(args.trajectory, 'plot_landscape', False)):
                    # Define eval fns that read current actor params (visualizer mutates them)
                    def _eval_makespan():
                        try:
                            mk, eobs, etot, _m = AG._test_agent(agent, args)
                            return float(mk)
                        except Exception as _e:
                            print(f"[landscape] makespan eval failed: {_e}")
                            return float('inf')

                    def _eval_active_energy():
                        try:
                            _mk, _eobs, _etot, m = AG._test_agent(agent, args)
                            ae = m.get('avg_active_energy')
                            if ae is None:
                                return float('inf')
                            return float(ae)
                        except Exception as _e:
                            print(f"[landscape] active energy eval failed: {_e}")
                            return float('inf')

                    # Render two separate landscapes side-by-side in output dir
                    try:
                        visualize_trajectory(
                            actor_model=agent.actor,
                            collector=traj_collector,
                            variant_name=variant.name,
                            log_dir=per_variant_dir,
                            config=args.trajectory,
                            eval_function=_eval_makespan,
                            landscape_filename="trajectory_landscape_makespan.png",
                        )
                    except Exception as _e1:
                        print(f"[ablation+traj] makespan landscape failed: {_e1}")

                    try:
                        visualize_trajectory(
                            actor_model=agent.actor,
                            collector=traj_collector,
                            variant_name=variant.name,
                            log_dir=per_variant_dir,
                            config=args.trajectory,
                            eval_function=_eval_active_energy,
                            landscape_filename="trajectory_landscape_active_energy.png",
                        )
                    except Exception as _e2:
                        print(f"[ablation+traj] active energy landscape failed: {_e2}")
            except Exception as e:
                print(f"[ablation+traj] trajectory visualization failed: {e}")

    finally:
        try:
            envs.close()
        except Exception:
            pass
        try:
            writer.close()
        except Exception:
            pass
        try:
            pbar.close()
        except Exception:
            pass


def main(args: Args) -> None:
    # Device selection
    device = AG._pick_device(args.device)
    if (args.nn_device or "same").lower() == "same":
        nn_device = device
    else:
        nn_device = AG._pick_device(args.nn_device)

    # Output directories
    base_out = Path(args.output_dir) / args.exp_name
    ablation_dir = base_out / args.ablation_subdir
    per_variant_root = ablation_dir / "per_variant"

    # Reuse ablation_gnn variants factory by training only requested variant
    variants: list[AG.AblationVariant] = []
    if args.train_only_variant:
        variants = [AG.AblationVariant(name=args.train_only_variant)]
    else:
        # Fallback: baseline-only if unspecified
        variants = [AG.AblationVariant(name="baseline")]

    for variant in variants:
        # Inject low-pass options if present on args (keeping compatibility)
        try:
            variant.lowpass_reg_lambda = float(getattr(args, 'variant_lowpass_reg_lambda', 0.0))
            variant.cache_lowpass_from_forward = bool(getattr(args, 'variant_cache_lowpass_from_forward', False))
        except Exception:
            pass

        per_variant_dir = per_variant_root / variant.name
        _train_one_variant_with_traj(args, variant, device, per_variant_dir)


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
