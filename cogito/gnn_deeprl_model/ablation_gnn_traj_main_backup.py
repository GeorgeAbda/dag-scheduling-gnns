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
import random

# Import base ablation components
from cogito.gnn_deeprl_model import ablation_gnn as AG
from cogito.dataset_generator.gen_dataset import DatasetArgs
from cogito.gnn_deeprl_model.actor_trajectory_viz import TrajectoryCollector, TrajectoryVisualizer
from cogito.dataset_generator.core.gen_dataset import generate_dataset
from cogito.dataset_generator.core.models import Dataset
from cogito.gnn_deeprl_model.core.env.gym_env import CloudSchedulingGymEnvironment
from cogito.gnn_deeprl_model.agents.gin_agent.wrapper import GinAgentWrapper
from gymnasium.wrappers import RecordEpisodeStatistics
from cogito.config.settings import HOST_SPECS_PATH

# Seed file reader helper
def read_seed_file(path: str) -> list[int]:
    try:
        with open(path, 'r') as f:
            content = f.read().strip()
            if content.startswith('['):
                import json
                return [int(x) for x in json.loads(content)]
            else:
                return [int(line.strip()) for line in content.split('\n') if line.strip()]
    except Exception as e:
        print(f"Warning: failed to read seed file {path}: {e}")
        return []
from cogito.dataset_generator.core.gen_vm import generate_hosts, generate_vms, allocate_vms

@dataclass
class Args:
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
    log_loss_every: int = 1
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
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None

    test_every_iters: int = 5
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
    # Minimum global training step before best-return tracking is enabled
    best_return_min_step: int = 0
    # Decision boundary visualization: plot every N iterations (0 disables)
    decision_boundary_every: int = 0
    decision_boundary_seed: int = 99999
    decision_boundary_grid_res: int = 200
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

    # Actor trajectory visualization
    trajectory_enabled: bool = False
    trajectory_collect_every: int = 10  # Collect actor params every N iterations
    trajectory_method: str = "svd"  # "svd", "pca", or "random"




def _log_scalar_csv(per_variant_dir: Path, variant: str, global_step: int, metric_name: str, value: float) -> None:
    path = per_variant_dir / f"{variant}_{metric_name}.csv"
    new_file = not path.exists()
    with path.open("a", newline="") as f:
        w = _csv.writer(f)
        if new_file:
            w.writerow(["Wall time", "Step", "Value"])
        w.writerow([time.time(), int(global_step), float(value)])


def _pca_2d(X: np.ndarray):
    """Simple PCA to 2D for decision boundary visualization."""
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comp = Vt[:2].T  # (D,2)
    X2 = Xc @ comp   # (N,2)
    return X2.astype(np.float64), mu.squeeze().astype(np.float64), comp.astype(np.float64)


def _agent_query_grid(agent, X2: np.ndarray, mu: np.ndarray, comp: np.ndarray, num_vms: int, device, grid_res: int = 200, pad: float = 0.05):
    """Query agent directly on inverse-transformed PCA grid points."""
    xmin, ymin = X2.min(axis=0)
    xmax, ymax = X2.max(axis=0)
    dx, dy = xmax - xmin, ymax - ymin
    xmin -= pad * dx
    xmax += pad * dx
    ymin -= pad * dy
    ymax += pad * dy
    xs = np.linspace(xmin, xmax, grid_res)
    ys = np.linspace(ymin, ymax, grid_res)
    XX, YY = np.meshgrid(xs, ys)
    grid_2d = np.stack([XX.ravel(), YY.ravel()], axis=1)  # (grid_res^2, 2)
    
    # Inverse transform: X_reconstructed = X2 @ comp.T + mu
    # grid_2d is in PCA space, transform back to original space
    grid_high_d = grid_2d @ comp.T + mu  # (grid_res^2, D)
    
    # Query agent for each grid point
    labels = np.empty(grid_2d.shape[0], dtype=int)
    batch_size = 256  # Process in batches to avoid memory issues
    for i in range(0, grid_high_d.shape[0], batch_size):
        batch = grid_high_d[i:i+batch_size]
        batch_tensor = torch.from_numpy(batch.astype(np.float32)).to(device)
        with torch.no_grad():
            actions, _, _, _ = agent.get_action_and_value(batch_tensor)
        vm_choices = (actions.cpu().numpy() % num_vms).astype(int)
        labels[i:i+batch_size] = vm_choices
    
    Z = labels.reshape(grid_res, grid_res)
    return XX, YY, Z


def _plot_decision_boundary_first_state(agent: AG.AblationGinAgent, args: Args, iteration: int, per_variant_dir: Path, variant_name: str, dataset_cfg: DatasetArgs | None = None):
    """Collect first-state decision boundaries by running a single episode and plotting VM-task pair actions in 2D PCA space."""
    print(f"[decision_boundary][DEBUG] _plot_decision_boundary_first_state called for iteration {iteration}")
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        print(f"[decision_boundary][DEBUG] matplotlib imported successfully")
    except Exception as e:
        print(f"[decision_boundary] Skipping: matplotlib not available: {e}")
        return

    try:
        # Use the same dataset configuration as training
        db_seed = int(getattr(args, 'decision_boundary_seed', 99999))
        grid_res = int(getattr(args, 'decision_boundary_grid_res', 200))
        print(f"[decision_boundary][DEBUG] Config: db_seed={db_seed}, grid_res={grid_res}")
        
        # Use provided dataset config (from wide/longcp) or fall back to args.dataset
        ds_cfg = dataset_cfg if dataset_cfg is not None else args.dataset
        gnp_p_val = getattr(ds_cfg, 'gnp_p', None)
        req_div_val = getattr(ds_cfg, 'req_divisor', None)
        
        print(f"[decision_boundary][DEBUG] Using training dataset config: "
              f"host_count={ds_cfg.host_count}, vm_count={ds_cfg.vm_count}, "
              f"dag_method={ds_cfg.dag_method}, gnp_p={gnp_p_val}, req_divisor={req_div_val}")
        
        # Build kwargs for generate_dataset, only include optional params if they're not None
        gen_kwargs = {
            'seed': db_seed,
            'host_count': int(ds_cfg.host_count),
            'vm_count': int(ds_cfg.vm_count),
            'max_memory_gb': int(ds_cfg.max_memory_gb),
            'min_cpu_speed_mips': int(ds_cfg.min_cpu_speed),
            'max_cpu_speed_mips': int(ds_cfg.max_cpu_speed),
            'workflow_count': int(ds_cfg.workflow_count),
            'dag_method': str(ds_cfg.dag_method),
            'gnp_min_n': int(ds_cfg.gnp_min_n),
            'gnp_max_n': int(ds_cfg.gnp_max_n),
            'task_length_dist': str(ds_cfg.task_length_dist),
            'min_task_length': int(ds_cfg.min_task_length),
            'max_task_length': int(ds_cfg.max_task_length),
            'task_arrival': str(ds_cfg.task_arrival),
            'arrival_rate': float(ds_cfg.arrival_rate),
            'vm_rng_seed': 0,
        }
        
        # Only add optional parameters if they're not None
        if gnp_p_val is not None:
            gen_kwargs['gnp_p'] = gnp_p_val
        if req_div_val is not None:
            gen_kwargs['req_divisor'] = int(req_div_val)
        
        dataset = generate_dataset(**gen_kwargs)

        print(f"[decision_boundary][DEBUG] Dataset generated successfully")
        env = CloudSchedulingGymEnvironment(dataset=dataset, collect_timelines=False, compute_metrics=False, profile=False)
        env = GinAgentWrapper(env)
        obs, _ = env.reset(seed=db_seed)
        print(f"[decision_boundary][DEBUG] Environment created and reset")

        num_vms = len(env.prev_obs.vm_observations)
        num_tasks = len(env.prev_obs.task_observations)
        num_actions = num_vms * num_tasks
        print(f"[decision_boundary][DEBUG] First state: {num_vms} VMs, {num_tasks} tasks, {num_actions} possible actions")

        # Run a full episode to get metrics for CSV logging and VM CPU usage profile
        episode_makespan = 0.0
        episode_active_energy = 0.0
        episode_total_return = 0.0
        episode_makespan_return = 0.0
        episode_active_energy_return = 0.0
        steps = 0
        vm_usage_rows: list[list[float]] = []  # per-step VM used CPU fraction
        
        while True:
            steps += 1
            # Snapshot VM CPU usage before taking the action
            try:
                vm_obs = getattr(env, "prev_obs", None)
                if vm_obs is not None:
                    vm_list = getattr(vm_obs, "vm_observations", [])
                    row = [float(getattr(v, "used_cpu_fraction_cores", 0.0)) for v in vm_list]
                    # Pad/truncate to num_vms for safety
                    if len(row) < num_vms:
                        row = row + [0.0] * (num_vms - len(row))
                    elif len(row) > num_vms:
                        row = row[:num_vms]
                    vm_usage_rows.append(row)
            except Exception:
                pass

            obs_tensor = torch.from_numpy(obs.astype(np.float32).reshape(1, -1)).to(agent.device)
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(obs_tensor)
            vm_action = int(action.item())
            obs, reward, terminated, truncated, info = env.step(vm_action)
            episode_total_return += reward
            if terminated or truncated:
                if hasattr(env, 'prev_obs') and env.prev_obs is not None:
                    episode_makespan = float(env.prev_obs.makespan())
                if isinstance(info, dict):
                    episode_active_energy = float(info.get('total_energy_active', 0.0))
                    episode_makespan_return = float(info.get('episode', {}).get('makespan_return', 0.0))
                    episode_active_energy_return = float(info.get('episode', {}).get('active_energy_return', 0.0))
                break

        print(f"[decision_boundary][DEBUG] Episode completed: {steps} steps")
        print(f"[decision_boundary][DEBUG] Metrics: makespan={episode_makespan:.2f}, active_energy={episode_active_energy:.2f}")
        print(f"[decision_boundary][DEBUG] Returns: total={episode_total_return:.4f}, makespan_r={episode_makespan_return:.4f}, active_energy_r={episode_active_energy_return:.4f}")

        # Convert VM usage log to array (steps x num_vms) and save CSV + heatmap
        try:
            if vm_usage_rows:
                vm_usage = np.asarray(vm_usage_rows, dtype=float)
                # Ensure correct shape
                if vm_usage.ndim == 1:
                    vm_usage = vm_usage.reshape(-1, num_vms)
                csv_vm_path = per_variant_dir / f"{variant_name}_vm_usage_iter{iteration:05d}.csv"
                import csv as _csv
                with csv_vm_path.open("w", newline="") as f_vm:
                    w_vm = _csv.writer(f_vm)
                    header = ["Step"] + [f"VM_{i}_used_cpu_frac" for i in range(num_vms)]
                    w_vm.writerow(header)
                    for t_idx, row in enumerate(vm_usage):
                        w_vm.writerow([t_idx] + [float(x) for x in row])
                print(f"[decision_boundary][DEBUG] Logged VM CPU usage to {csv_vm_path}")

                # Heatmap: time (x) vs VM (y)
                import matplotlib.pyplot as plt
                vmax_val = float(np.max(vm_usage)) if vm_usage.size > 0 else 0.0
                if vmax_val <= 0.0:
                    vmax_val = 1.0
                plt.figure(figsize=(8, 4))
                plt.imshow(
                    vm_usage.T,
                    aspect="auto",
                    origin="lower",
                    interpolation="nearest",
                    extent=[0, vm_usage.shape[0], 0, num_vms],
                    vmin=0.0,
                    vmax=vmax_val,
                    cmap="viridis",
                )
                plt.colorbar(label="Used CPU Fraction")
                plt.xlabel("Environment Step")
                plt.ylabel("VM ID")
                plt.title(f"{variant_name} - Iteration {iteration}\nPer-VM CPU Usage During Episode")
                out_vm_png = per_variant_dir / f"{variant_name}_vm_usage_iter{iteration:05d}.png"
                plt.tight_layout()
                plt.savefig(str(out_vm_png), dpi=200)
                plt.close()
                print(f"[decision_boundary][DEBUG] Saved VM CPU usage heatmap to {out_vm_png}")
        except Exception as _e_vm:
            print(f"[decision_boundary][DEBUG] Failed to log VM CPU usage at iteration {iteration}: {_e_vm}")
        
        # Reset to get the first state for perturbation
        obs, _ = env.reset(seed=db_seed)
        first_obs = obs.copy()
        print(f"[decision_boundary][DEBUG] Got first state, obs.shape={first_obs.shape}")
        
        # Parse observation structure to find task CPU and memory demand locations
        # Structure: [num_tasks, num_vms, num_task_deps, num_compatibilities] + task features + vm features + edges
        header_size = 4
        num_tasks_in_obs = int(first_obs[0])
        num_vms_in_obs = int(first_obs[1])
        
        # Offsets in the flattened observation array
        # After header: task_scheduled, task_ready, task_length, task_completion_time, task_memory_req_mb, task_cpu_req_cores
        task_scheduled_start = header_size
        task_ready_start = task_scheduled_start + num_tasks_in_obs
        task_length_start = task_ready_start + num_tasks_in_obs
        task_completion_start = task_length_start + num_tasks_in_obs
        task_memory_start = task_completion_start + num_tasks_in_obs
        task_cpu_start = task_memory_start + num_tasks_in_obs
        
        print(f"[decision_boundary][DEBUG] Observation structure: num_tasks={num_tasks_in_obs}, num_vms={num_vms_in_obs}")
        print(f"[decision_boundary][DEBUG] Task memory range: [{task_memory_start}:{task_memory_start + num_tasks_in_obs}]")
        print(f"[decision_boundary][DEBUG] Task CPU range: [{task_cpu_start}:{task_cpu_start + num_tasks_in_obs}]")
        
        # Generate variations by perturbing only task CPU and memory demands
        print(f"[decision_boundary][DEBUG] Generating state variations by perturbing task resource demands...")
        num_samples = 2000
        X = [first_obs]
        y_agent = []
        
        # Get agent's action for the original first state
        obs_tensor = torch.from_numpy(first_obs.astype(np.float32).reshape(1, -1)).to(agent.device)
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(obs_tensor)
        vm_choice = int(action.item()) % num_vms
        y_agent.append(vm_choice)
        
        # Get original task demands for reference
        orig_memory = first_obs[task_memory_start:task_memory_start + num_tasks_in_obs].copy()
        orig_cpu = first_obs[task_cpu_start:task_cpu_start + num_tasks_in_obs].copy()
        print(f"[decision_boundary][DEBUG] Original memory demands: min={orig_memory.min():.1f}, max={orig_memory.max():.1f}, mean={orig_memory.mean():.1f}")
        print(f"[decision_boundary][DEBUG] Original CPU demands: min={orig_cpu.min():.2f}, max={orig_cpu.max():.2f}, mean={orig_cpu.mean():.2f}")
        
        # Create a 3D grid in INTERPRETABLE space: (avg_task_cpu, avg_task_length, avg_task_memory)
        # Show multiple 2D slices (CPU vs Length) for different memory levels
        print(f"[decision_boundary][DEBUG] Creating interpretable 3D grid: avg_task_cpu vs avg_task_length vs avg_task_memory...")
        
        # Define ranges for average task CPU, length, and memory
        cpu_min, cpu_max = 0.1, 2.0  # Average CPU per task (cores)
        
        # Get task length range from original state
        task_length_start = task_ready_start + num_tasks_in_obs  # After task_ready
        orig_task_lengths = first_obs[task_length_start:task_length_start + num_tasks_in_obs].copy()
        length_min = max(500, orig_task_lengths.min() * 0.5)
        length_max = min(100_000, orig_task_lengths.max() * 1.5)
        
        # Create 3 memory levels to visualize
        num_memory_levels = 3
        mem_vals = np.linspace(100, 3000, num_memory_levels)  # Low, Medium, High memory
        print(f"[decision_boundary][DEBUG] Memory levels: {mem_vals}")
        
        cpu_vals = np.linspace(cpu_min, cpu_max, grid_res)
        length_vals = np.linspace(length_min, length_max, grid_res)
        
        # Store results for each memory level
        Z_all = []
        
        for mem_idx, avg_mem in enumerate(mem_vals):
            print(f"[decision_boundary][DEBUG] Processing memory level {mem_idx+1}/{num_memory_levels}: {avg_mem:.0f} MB")
            XX, YY = np.meshgrid(cpu_vals, length_vals)
            Z_labels = np.empty((grid_res, grid_res), dtype=int)
            batch_size = 256
            grid_points = np.stack([XX.ravel(), YY.ravel()], axis=1)  # (grid_res^2, 2)
            
            for i in range(0, grid_points.shape[0], batch_size):
                batch_points = grid_points[i:i+batch_size]
                batch_states = np.tile(first_obs, (batch_points.shape[0], 1))
                
                # Set all tasks to have the same CPU, length, and memory
                for j, (avg_cpu, avg_length) in enumerate(batch_points):
                    batch_states[j, task_cpu_start:task_cpu_start + num_tasks_in_obs] = avg_cpu
                    batch_states[j, task_length_start:task_length_start + num_tasks_in_obs] = avg_length
                    batch_states[j, task_memory_start:task_memory_start + num_tasks_in_obs] = avg_mem
                
                # Query agent
                batch_tensor = torch.from_numpy(batch_states.astype(np.float32)).to(agent.device)
                with torch.no_grad():
                    actions, _, _, _ = agent.get_action_and_value(batch_tensor)
                vm_choices = (actions.cpu().numpy() % num_vms).astype(int)
                
                # Place in grid
                for j, vm_choice in enumerate(vm_choices):
                    grid_idx = i + j
                    row = grid_idx // grid_res
                    col = grid_idx % grid_res
                    Z_labels[row, col] = vm_choice
            
            Z_all.append(Z_labels)
        
        print(f"[decision_boundary][DEBUG] Decision grids computed for {num_memory_levels} memory levels")

        # K is now the number of unique VMs (should be num_vms)
        K = num_vms
        print(f"[decision_boundary][DEBUG] Number of VMs: {K}")
        
        # Generate distinct colors for each VM
        import matplotlib.cm as cm
        # Use tab10 for up to 10 VMs, tab20 for up to 20, otherwise hsv
        if K <= 10:
            cmap = cm.get_cmap('tab10', K)
        elif K <= 20:
            cmap = cm.get_cmap('tab20', K)
        else:
            cmap = cm.get_cmap('hsv', K)
        colors = [cmap(i) for i in range(K)]
        cmap_discrete = ListedColormap(colors)

        # Create subplots for each memory level
        fig, axes = plt.subplots(1, num_memory_levels, figsize=(6*num_memory_levels, 5))
        if num_memory_levels == 1:
            axes = [axes]
        
        for mem_idx, (ax, Z, avg_mem) in enumerate(zip(axes, Z_all, mem_vals)):
            # Plot decision regions
            XX, YY = np.meshgrid(cpu_vals, length_vals)
            contour = ax.contourf(XX, YY, Z, levels=np.arange(K+1)-0.5, cmap=cmap_discrete, alpha=0.8, antialiased=True)
            
            ax.set_xlabel("Avg Task CPU (cores)", fontsize=10)
            ax.set_ylabel("Avg Task Length (MI)", fontsize=10)
            ax.set_title(f"Memory = {avg_mem:.0f} MB", fontsize=11)
            ax.grid(True, alpha=0.3)
        
        # Add a single colorbar for all subplots
        fig.subplots_adjust(right=0.92)
        cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(contour, cax=cbar_ax, ticks=np.arange(K))
        cbar.ax.set_yticklabels([f'VM {i}' for i in range(K)])
        cbar.set_label('VM Choice', fontsize=10)
        
        fig.suptitle(f"{variant_name} - Iteration {iteration}\nVM Choice by Workload Characteristics", fontsize=13, y=0.98)
        out_png = per_variant_dir / f"{variant_name}_decision_boundary_iter{iteration:05d}.png"
        plt.tight_layout(rect=[0, 0, 0.93, 0.96])
        plt.savefig(str(out_png), bbox_inches="tight", dpi=200)
        plt.close()
        print(f"[decision_boundary] Saved {out_png} | memory_levels={num_memory_levels} | VMs={K}")
        
        # Log metrics to CSV
        csv_path = per_variant_dir / f"{variant_name}_decision_boundary_metrics.csv"
        import csv
        import os
        file_exists = os.path.exists(csv_path)
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['Iteration', 'Makespan', 'Active_Energy', 'Makespan_Return', 'Active_Energy_Return', 'Total_Return'])
            writer.writerow([iteration, episode_makespan, episode_active_energy, episode_makespan_return, episode_active_energy_return, episode_total_return])
        print(f"[decision_boundary] Logged metrics to {csv_path}")
        
        # Close environment
        env.close()
    except Exception as e:
        print(f"[decision_boundary] Failed to plot at iteration {iteration}: {e}")
        import traceback
        traceback.print_exc()


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

    # Global determinism for evaluation: seed Python, NumPy, and PyTorch RNGs
    eval_seed = int(getattr(args, 'seed', 12345))
    try:
        random.seed(eval_seed)
        np.random.seed(eval_seed)
        torch.manual_seed(eval_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(eval_seed)
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    except Exception as _e_seed:
        print(f"[ablation+traj] Warning: failed to set deterministic seed: {_e_seed}")

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

    def _compute_optimal_req_divisor(dataset_cfg: dict, seed: int) -> int:
        """Compute a conservative req_divisor so that, on the smallest VM, repeating
        the per-task demand (CPU cores and memory) for all tasks in a job does not
        exceed that VM's capacity.
        """
        host_count = int(dataset_cfg.get("host_count", 4))
        vm_count = int(dataset_cfg.get("vm_count", 10))
        max_memory_gb = int(dataset_cfg.get("max_memory_gb", 10))
        min_cpu_speed = int(dataset_cfg.get("min_cpu_speed", 500))
        max_cpu_speed = int(dataset_cfg.get("max_cpu_speed", 5000))
        n_tasks = int(dataset_cfg.get("gnp_max_n", 40))
        if n_tasks <= 0:
            n_tasks = 1
        rng = np.random.RandomState(int(seed))
        hosts = generate_hosts(n=host_count, rng=rng)
        vms = generate_vms(
            n=vm_count,
            max_memory_gb=max_memory_gb,
            min_cpu_speed_mips=min_cpu_speed,
            max_cpu_speed_mips=max_cpu_speed,
            rng=rng,
        )
        allocate_vms(vms, hosts, rng)
        if not vms:
            return 1
        mem_caps = [int(getattr(vm, "memory_mb", 0)) for vm in vms]
        core_caps = [int(max(1, getattr(vm, "cpu_cores", 1))) for vm in vms]
        min_mem = max(1, min(mem_caps))
        max_mem = max(mem_caps)
        min_cores = max(1, min(core_caps))
        max_cores = max(core_caps)
        max_safe_mem_per_task = max(1024, min_mem // n_tasks)
        max_safe_cores_per_task = max(1, min_cores // n_tasks)
        req_div_mem = max(1, max_mem // max_safe_mem_per_task)
        req_div_core = max(1, max_cores // max_safe_cores_per_task)
        print(f"[{dataset_cfg.get('style', 'unknown')}] req_div_mem={req_div_mem}, req_div_core={req_div_core}, max_mem={max_mem}, max_cores={max_cores}, n_tasks={n_tasks}")
        return int(max(req_div_mem, req_div_core))

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
        # Apply global req_divisor override if requested at top level, otherwise compute optimal
        try:
            req_div = getattr(args, 'dataset_req_divisor', None)
        except Exception as e:
            req_div = None
            print(f"Warning: Could not get dataset_req_divisor from args: {e}")
        
        if req_div is not None:
            # Global override provided
            ds_args = _dc_replace(ds_args, req_divisor=int(req_div))
        else:
            # Compute optimal req_divisor for this domain
            computed_req_div = _compute_optimal_req_divisor(ds, args.seed)
            ds_args = _dc_replace(ds_args, req_divisor=int(computed_req_div))
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
        envs = gym.vector.AsyncVectorEnv([_make_env_dataset for _ in range(int(args.num_envs))])
    else:
        envs = gym.vector.AsyncVectorEnv([AG._make_env_thunk(i, envs_args[i]) for i in range(int(args.num_envs))])
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
    traj_collector: TrajectoryCollector | None = None
    if args.trajectory_enabled:
        traj_collector = TrajectoryCollector(collect_every=args.trajectory_collect_every)
        print(f"[trajectory] Enabled: collecting every {args.trajectory_collect_every} iterations, method={args.trajectory_method}")

    # Best-model tracking based on episodic return (higher is better)
    best_return = float("-inf")
    best_return_state = None
    best_return_step = -1

    obs = torch.zeros((args.num_steps, args.num_envs) + obs_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + act_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards_energy = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards_makespan = torch.zeros((args.num_steps, args.num_envs)).to(device)
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
            # If no domain configs, return the first available or None
            if ds_wide is not None:
                return ds_wide
            if ds_long is not None:
                return ds_long
            return None
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
        # Final fallback: return whichever domain config is available
        if ds_wide is not None:
            return ds_wide
        if ds_long is not None:
            return ds_long
        return None

    # Single-episode eval with active energy only
    def _eval_one_episode(agent: AG.Agent, args: Args, seed: int, ds_override: DatasetArgs | None = None) -> tuple[float, float]:
        # Build a temp Args with dataset override if provided
        if ds_override is not None:
            tmp_args = _dc_replace(args)
            tmp_args.dataset = _dc_replace(args.dataset,
                                           style=ds_override.style,
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
                                           arrival_rate=ds_override.arrival_rate,
                                           req_divisor=ds_override.req_divisor)
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

    # Plot initial decision boundary (iteration 0, before any training)
    db_every = int(getattr(args, 'decision_boundary_every', 0))
    if db_every > 0:
        print(f"[decision_boundary][DEBUG] Plotting initial (untrained) decision boundary at iteration 0...")
        # Use the first available domain config (prefer wide, then longcp, then default)
        db_dataset_cfg = ds_wide if ds_wide is not None else (ds_long if ds_long is not None else None)
        _plot_decision_boundary_first_state(agent, args, 0, per_variant_dir, variant.name, dataset_cfg=db_dataset_cfg)

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
                try:
                    dbg_e = infos.get("dbg_energy_reward", None)
                    dbg_m = infos.get("dbg_makespan_reward", None)
                except Exception:
                    dbg_e, dbg_m = None, None
                if dbg_e is not None:
                    try:
                        rewards_energy[step] = torch.as_tensor(dbg_e, dtype=torch.float32, device=device).view(-1)
                    except Exception:
                        pass
                if dbg_m is not None:
                    try:
                        rewards_makespan[step] = torch.as_tensor(dbg_m, dtype=torch.float32, device=device).view(-1)
                    except Exception:
                        pass
                next_obs_tensor, next_done_tensor = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

                if isinstance(infos, dict) and "final_info" in infos:
                    for info in infos["final_info"]:
                        if info and "episode" in info:
                            try:
                                pbar.update(global_step - pbar.n)
                            except Exception:
                                pass
                            ep_ret = None
                            try:
                                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                                # Log decomposed episodic returns from the GinAgentWrapper if available
                                try:
                                    if "active_energy_return" in info:
                                        writer.add_scalar(
                                            "charts/episodic_active_energy_return",
                                            float(info["active_energy_return"]),
                                            global_step,
                                        )
                                except Exception:
                                    pass
                                try:
                                    if "makespan_return" in info:
                                        writer.add_scalar(
                                            "charts/episodic_makespan_return",
                                            float(info["makespan_return"]),
                                            global_step,
                                        )
                                except Exception:
                                    pass
                                ep_ret = float(info["episode"]["r"])
                            except Exception:
                                ep_ret = None
                            # Update best-return tracker (maximize episodic return) after warmup
                            min_step = int(getattr(args, "best_return_min_step", 0))
                            if ep_ret is not None and global_step >= min_step and ep_ret > best_return:
                                best_return = ep_ret
                                try:
                                    best_return_state = agent.state_dict().copy()
                                except Exception:
                                    best_return_state = agent.state_dict()
                                best_return_step = int(global_step)

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

            def _discounted_sum(rew: torch.Tensor, gamma: float) -> torch.Tensor:
                out = torch.zeros_like(rew)
                running = torch.zeros(rew.shape[1], device=rew.device)
                for t in reversed(range(rew.shape[0])):
                    running = rew[t] + gamma * running
                    out[t] = running
                return out

            returns_energy = _discounted_sum(rewards_energy, args.gamma)
            returns_makespan = _discounted_sum(rewards_makespan, args.gamma)

            b_obs = obs.reshape((-1,) + obs_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + act_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_returns_energy = returns_energy.reshape(-1)
            b_returns_makespan = returns_makespan.reshape(-1)
            b_values = values.reshape(-1)

            b_inds = np.arange(batch_size)
            logged_grad_norms_for_iter = False
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

                    mb_adv_energy = b_returns_energy[mb_inds]
                    mb_adv_makespan = b_returns_makespan[mb_inds]
                    if args.norm_adv:
                        if mb_adv_energy.numel() > 1:
                            mb_adv_energy = (mb_adv_energy - mb_adv_energy.mean()) / (mb_adv_energy.std() + 1e-8)
                        if mb_adv_makespan.numel() > 1:
                            mb_adv_makespan = (mb_adv_makespan - mb_adv_makespan.mean()) / (mb_adv_makespan.std() + 1e-8)

                    pg_loss_energy = torch.max(
                        -mb_adv_energy * ratio,
                        -mb_adv_energy * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef),
                    ).mean()
                    pg_loss_makespan = torch.max(
                        -mb_adv_makespan * ratio,
                        -mb_adv_makespan * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef),
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

                    if not logged_grad_norms_for_iter and start == 0:
                        optimizer.zero_grad()
                        pg_loss_energy.backward(retain_graph=True)
                        s_e = 0.0
                        for p in agent.parameters():
                            if p.grad is not None:
                                g = p.grad.detach()
                                s_e += float(torch.sum(g * g).item())
                        grad_energy_norm = float(s_e ** 0.5)

                        optimizer.zero_grad()
                        pg_loss_makespan.backward(retain_graph=True)
                        s_m = 0.0
                        for p in agent.parameters():
                            if p.grad is not None:
                                g = p.grad.detach()
                                s_m += float(torch.sum(g * g).item())
                        grad_mk_norm = float(s_m ** 0.5)

                        try:
                            writer.add_scalar("diagnostics/grad_norm_energy", grad_energy_norm, int(global_step))
                            writer.add_scalar("diagnostics/grad_norm_makespan", grad_mk_norm, int(global_step))
                        except Exception:
                            pass
                        try:
                            _log_scalar_csv(per_variant_dir, variant.name, int(global_step), "grad_norm_energy", grad_energy_norm)
                            _log_scalar_csv(per_variant_dir, variant.name, int(global_step), "grad_norm_makespan", grad_mk_norm)
                        except Exception:
                            pass
                        logged_grad_norms_for_iter = True

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
                        
                        # TensorBoard logging for entropy and losses
                        try:
                            writer.add_scalar("losses/policy_loss", pg_val, global_step)
                            writer.add_scalar("losses/value_loss", v_val, global_step)
                            writer.add_scalar("losses/entropy", ent_val, global_step)
                            writer.add_scalar("losses/approx_kl", kl_val, global_step)
                        except Exception:
                            pass

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
                if traj_collector is not None and (iteration % max(1, int(args.trajectory_collect_every)) == 0):
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

            # Decision boundary visualization at initialization (iter 0) and every K iterations
            db_every = int(getattr(args, 'decision_boundary_every', 0))
            if db_every > 0:
                should_plot = (iteration == 0 or iteration % db_every == 0 or iteration == num_iterations)
                print(f"[decision_boundary][DEBUG] iteration={iteration}, db_every={db_every}, should_plot={should_plot}")
                if should_plot:
                    print(f"[decision_boundary][DEBUG] Attempting to plot decision boundary at iteration {iteration}...")
                    # Use the first available domain config (prefer wide, then longcp, then default)
                    db_dataset_cfg = ds_wide if ds_wide is not None else (ds_long if ds_long is not None else None)
                    _plot_decision_boundary_first_state(agent, args, iteration, per_variant_dir, variant.name, dataset_cfg=db_dataset_cfg)
            else:
                if iteration == 0:
                    print(f"[decision_boundary][DEBUG] Decision boundary plotting DISABLED (decision_boundary_every={db_every})")

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
                if len(traj_collector.get_parameters()) >= 2:
                    print(f"[ablation+traj] Generating trajectory visualization for {variant.name}...")
                    print(f"[ablation+traj] Collected {len(traj_collector.get_parameters())} snapshots")
                    
                    # Create visualizer
                    visualizer = TrajectoryVisualizer(
                        optimized_parameters=list(agent.actor.named_parameters()),
                        intermediate_parameters=traj_collector.get_parameter_snapshots(),
                        method=args.trajectory_method,
                        device=device
                    )
                    
                    # Plot simple trajectory
                    traj_png = per_variant_dir / f"{variant.name}_actor_trajectory.png"
                    visualizer.plot_trajectory(
                        save_path=str(traj_png),
                        title=f"{variant.name} - Actor Learning Trajectory",
                        show_arrows=True
                    )
                    print(f"[ablation+traj] Saved trajectory plot: {traj_png}")
                    
                    # Plot interactive version
                    traj_html = per_variant_dir / f"{variant.name}_actor_trajectory_interactive.html"
                    visualizer.plot_trajectory_interactive(
                        save_path=str(traj_html),
                        title=f"{variant.name} - Actor Learning Trajectory (Interactive)"
                    )
                    print(f"[ablation+traj] Saved interactive trajectory: {traj_html}")

                    # Overlay Active Energy landscape over parameter space
                    def _eval_active_energy() -> float:
                        # Always evaluate on the available domain config (prefer wide)
                        _ds = ds_wide if ds_wide is not None else ds_long
                        _seed = int(eval_seed)
                        _mk, _ae = _eval_one_episode(agent, args, _seed, _ds)
                        return float(_ae)

                    landscape_png = per_variant_dir / f"{variant.name}_actor_landscape_active_energy.png"
                    try:
                        visualizer.plot_trajectory_with_landscape(
                            actor_model=agent.actor,
                            eval_function=_eval_active_energy,
                            save_path=str(landscape_png),
                            title=f"{variant.name} - Active Energy Reward Landscape",
                            grid_points=20,
                            colorbar_label="Active Energy Reward"
                        )
                        print(f"[ablation+traj] Saved active-energy landscape: {landscape_png}")
                    except Exception as _e_l:
                        print(f"[ablation+traj] Active-energy landscape failed: {_e_l}")

                    # Overlay Makespan landscape over parameter space
                    def _eval_makespan() -> float:
                        _ds = ds_wide if ds_wide is not None else ds_long
                        _seed = int(eval_seed)
                        _mk, _ae = _eval_one_episode(agent, args, _seed, _ds)
                        return float(_mk)

                    landscape_mk_png = per_variant_dir / f"{variant.name}_actor_landscape_makespan.png"
                    try:
                        visualizer.plot_trajectory_with_landscape(
                            actor_model=agent.actor,
                            eval_function=_eval_makespan,
                            save_path=str(landscape_mk_png),
                            title=f"{variant.name} - Makespan Reward Landscape",
                            grid_points=20,
                            colorbar_label="Makespan Reward"
                        )
                        print(f"[ablation+traj] Saved makespan landscape: {landscape_mk_png}")
                    except Exception as _e_l2:
                        print(f"[ablation+traj] Makespan landscape failed: {_e_l2}")
                else:
                    print(f"[ablation+traj] Skipping trajectory visualization: only {len(traj_collector.get_parameters())} snapshots collected (need >=2)")
            except Exception as e:
                print(f"[ablation+traj] trajectory visualization failed: {e}")
                import traceback
                traceback.print_exc()

    finally:
        # Save best-return checkpoint (if any) before shutting down writer/envs
        try:
            if best_return_state is not None:
                ckpt_best_path = per_variant_dir / f"{variant.name}_best_return.pt"
                torch.save({"state_dict": best_return_state, "best_return": float(best_return), "global_step": int(best_return_step)}, ckpt_best_path)
                print(f"[ablation+traj] Saved best-return model: return={best_return:.4f} step={best_return_step} -> {ckpt_best_path}")
        except Exception as _e_best:
            print(f"[ablation+traj] Warning: failed to save best-return checkpoint: {_e_best}")
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
