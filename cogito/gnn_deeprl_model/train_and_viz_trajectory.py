"""
Train a single RL architecture and generate an actor learning trajectory visualization.

This script integrates the lightweight utilities in:
- scheduler/rl_model/actor_trajectory_viz.py
- scheduler/rl_model/ablation_gnn_with_trajectory.py

It runs a compact PPO loop (adapted from ablation_gnn.py) for a single AblationVariant,
collects actor parameter snapshots during training, and produces:
- A 2D trajectory plot (and optional interactive HTML)
- Optional loss landscape contour around the trajectory (slow; off by default)

Usage example:

  python -m scheduler.rl_model.train_and_viz_trajectory \
    --exp_name demo_traj \
    --variant.graph_type gin --variant.gin_num_layers 2 \
    --total_timesteps 50000 --num_envs 4 --num_steps 128 \
    --trajectory.enabled True --trajectory.collect_every 25 --trajectory.method svd \
    --trajectory.plot_landscape False

Outputs are saved under logs/<exp_name>/trajectories/<variant.name>/
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.nn.functional import softmax
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import tyro
import gymnasium as gym

import sys
import os, torch
# Local project imports
from cogito.dataset_generator.gen_dataset import DatasetArgs
from cogito.gnn_deeprl_model.core.env.gym_env import CloudSchedulingGymEnvironment
from cogito.gnn_deeprl_model.agents.gin_agent.wrapper import GinAgentWrapper
from cogito.gnn_deeprl_model.agents.agent import Agent

from cogito.gnn_deeprl_model.ablation_gnn import (
    AblationVariant,
    AblationGinAgent,
)
from cogito.gnn_deeprl_model.ablation_gnn_with_trajectory import (
    TrajectoryConfig,
    integrate_trajectory_collection,
    visualize_trajectory,
)


# ------------------------------ CLI args ------------------------------

@dataclass
class RunArgs:
    # Experiment basics
    exp_name: str = "traj_run"
    output_dir: str = "logs"
    seed: int = 12345
    device: str = "auto"  # cpu|cuda|mps|auto
    nn_device: str = "same"  # same|cpu|cuda|mps
    torch_deterministic: bool = True
    no_tensorboard: bool = True

    # PPO training
    total_timesteps: int = 2000000
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
    target_kl: float = 0.02

    # Evaluation pass used for (optional) landscape
    test_iterations: int = 4

    # Variant (architecture) selection
    variant: AblationVariant = field(
        default_factory=lambda: AblationVariant(name="gin_l2", graph_type="gin", gin_num_layers=2)
    )

    # Dataset
    dataset: DatasetArgs = field(
        default_factory=lambda: DatasetArgs(
            host_count=4,
            vm_count=10,
            workflow_count=10,
            gnp_min_n=12,
            gnp_max_n=16,
            max_memory_gb=10,
            min_cpu_speed=500,
            max_cpu_speed=10000,
            min_task_length=500,
            max_task_length=1000,
            task_arrival="static",
            dag_method="linear",
        )
    )

    # Trajectory viz
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)


# ------------------------------ Helpers ------------------------------

def _pick_device(choice: str) -> torch.device:
    c = (choice or "auto").lower()
    if c == "cpu":
        return torch.device("cpu")
    if c == "mps":
        return torch.device("mps") if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else torch.device("cpu")
    if c == "cuda":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _available_cpus() -> int:
    """Best-effort count of available logical CPUs, respecting CPU affinity/cgroups when possible."""
    # Try Linux sched_getaffinity (affinity-aware)
    try:
        if hasattr(os, "sched_getaffinity"):
            return len(os.sched_getaffinity(0))  # type: ignore[arg-type]
    except Exception:
        pass
    # Try psutil if present
    try:
        import psutil  # type: ignore
        n = psutil.cpu_count(logical=True)
        if n is not None:
            return int(n)
    except Exception:
        pass
    # Fallback
    return int(os.cpu_count() or 1)


def _make_env(args: RunArgs) -> gym.Env:
    env = CloudSchedulingGymEnvironment(
        dataset_args=args.dataset,
        collect_timelines=False,
        compute_metrics=False,
        profile=False,
    )
    env = GinAgentWrapper(env, use_lagrangian=False, constrained_mode=False)
    from gymnasium.wrappers import RecordEpisodeStatistics
    return RecordEpisodeStatistics(env)


def _make_test_env(args: RunArgs) -> gym.Env:
    env = CloudSchedulingGymEnvironment(
        dataset_args=args.dataset,
        collect_timelines=False,
        compute_metrics=True,
        profile=False,
    )
    return GinAgentWrapper(env, use_lagrangian=False, constrained_mode=False)


@torch.inference_mode()
def _test_makespan_plus_active(agent: AblationGinAgent, args: RunArgs) -> float:
    """Deterministic evaluation returning balanced normalized sum:
    L = (makespan / baseline_makespan) + (active_energy / baseline_active_energy)

    Baselines are computed via the wrapper's greedy fastest baseline at reset when available.
    If active split is unavailable, total energy is used as a fallback for baseline and/or episode.
    """
    total = 0.0
    for s in range(max(1, int(args.test_iterations))):
        env = _make_test_env(args)
        next_obs, _ = env.reset(seed=args.seed + 10 + s)
        # Compute greedy baseline once per episode (if available)
        baseline_ms: float | None = None
        baseline_energy: float | None = None
        try:
            if hasattr(env, '_compute_fastest_baseline') and getattr(env, 'initial_obs', None) is not None:
                bm, be = env._compute_fastest_baseline(env.initial_obs)  # type: ignore[attr-defined]
                baseline_ms = float(bm)
                baseline_energy = float(be)
        except Exception:
            baseline_ms, baseline_energy = None, None
        final_info: dict | None = None
        while True:
            obs_tensor = torch.from_numpy(np.asarray(next_obs, dtype=np.float32).reshape(1, -1))
            action, _, _, _ = agent.get_action_and_value(obs_tensor, deterministic=True)
            next_obs, _, terminated, truncated, info = env.step(int(action.item()))
            if terminated or truncated:
                final_info = info
                break
        # Extract episode metrics
        eps = 1e-9
        if isinstance(final_info, dict):
            mk = float(final_info.get("makespan", env.prev_obs.makespan()))
            ae_val = final_info.get("total_energy_active", None)
            if ae_val is None:
                ae = float(final_info.get("total_energy", env.prev_obs.energy_consumption()))
            else:
                ae = float(ae_val)
        else:
            mk = float(env.prev_obs.makespan())
            ae = float(env.prev_obs.energy_consumption())

        # Safe baselines; if missing, fall back to per-episode unnormalized sum
        if (baseline_ms is None or baseline_ms <= eps) or (baseline_energy is None or baseline_energy <= eps):
            loss = mk + ae
        else:
            loss = (mk / max(baseline_ms, eps)) + (ae / max(baseline_energy, eps))
        total += loss
        env.close()
    return total / float(max(1, int(args.test_iterations)))


# ------------------------------ Training + Trajectory ------------------------------

def train_and_visualize(args: RunArgs) -> None:
    # Repro + devices
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = bool(args.torch_deterministic)
    torch.set_num_threads(1)            # intra-op (already 1 via env, but explicit)
    torch.set_num_interop_threads(1)    # inter-op
    device = _pick_device(args.device)
    nn_device = _pick_device(args.nn_device if args.nn_device != "same" else args.device)

    # Paths
    run_root = Path(args.output_dir) / args.exp_name
    per_variant_dir = run_root / "ablation" / "per_variant" / args.variant.name
    per_variant_dir.mkdir(parents=True, exist_ok=True)

    print("Threads:", {
        "OMP": os.getenv("OMP_NUM_THREADS"),
        "MKL": os.getenv("MKL_NUM_THREADS"),
        "OPENBLAS": os.getenv("OPENBLAS_NUM_THREADS"),
        "NUMEXPR": os.getenv("NUMEXPR_NUM_THREADS"),
        "TORCH_NUM_THREADS": os.getenv("TORCH_NUM_THREADS"),
        "torch.get_num_threads()": torch.get_num_threads(),
        "torch.get_num_interop_threads()": torch.get_num_interop_threads(),
    })
    # Env vector
    # Ensure sufficient CPUs are available on this node/session
    min_cpus_required = 8
    avail_cpus = _available_cpus()
    if avail_cpus < min_cpus_required:
        print(f"ERROR: Only {avail_cpus} CPUs available; need at least {min_cpus_required} to run. Exiting.")
        sys.exit(1)
    print(f"!!!! {avail_cpus} CPUs available; need at least {min_cpus_required} to run. working.")
    def _thunk():
        return _make_env(args)
    envs = gym.vector.AsyncVectorEnv([_thunk for _ in range(int(args.num_envs))])
    obs_space = envs.single_observation_space
    act_space = envs.single_action_space
    assert isinstance(act_space, gym.spaces.Discrete)

    # Agent
    agent = AblationGinAgent(nn_device, args.variant)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Storage
    batch_size = int(args.num_envs * args.num_steps)
    minibatch_size = max(1, int(batch_size // max(1, args.num_minibatches)))
    num_iterations = max(1, int(args.total_timesteps // max(1, batch_size)))

    obs = torch.zeros((args.num_steps, args.num_envs) + obs_space.shape, device=nn_device)
    actions = torch.zeros((args.num_steps, args.num_envs) + act_space.shape, device=nn_device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=nn_device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=nn_device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=nn_device)
    values = torch.zeros((args.num_steps, args.num_envs), device=nn_device)
    print(f'Total timesteps: {args.total_timesteps}')
    print(f'Num envs: {args.num_envs}')
    print(f'Num steps: {args.num_steps}')
    print(f'Num minibatches: {args.num_minibatches}')
    print(f'Num updates: {num_iterations}')
    # Trajectory collector
    collector = integrate_trajectory_collection(
        actor_model=agent.actor,
        variant_name=args.variant.name,
        log_dir=run_root,
        config=args.trajectory,
    )
    print(f'integrating trajectory collection: {collector is not None}')

    # TB (optional)
    writer: Optional[SummaryWriter] = None
    if not args.no_tensorboard:
        tb_dir = run_root / "tb" / args.variant.name
        tb_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tb_dir))

    # Reset envs
    global_step = 0
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=nn_device)
    next_done_tensor = torch.zeros(args.num_envs, device=nn_device)

    pbar = tqdm(total=args.total_timesteps, desc="Train+Traj")

    # Save initial checkpoint
    try:
        torch.save(agent.state_dict(), per_variant_dir / f"{args.variant.name}_iter00000.pt")
    except Exception:
        pass
    
    for iteration in range(1, num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        # Rollout
        for step in range(args.num_steps):
            # print(f'step: {step}')
            global_step += args.num_envs
            obs[step] = next_obs_tensor
            dones[step] = next_done_tensor
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs_tensor)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(action.detach().cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward, dtype=torch.float32, device=nn_device).view(-1)
            next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=nn_device)
            next_done_tensor = torch.tensor(next_done, dtype=torch.float32, device=nn_device)

            if "final_info" in infos:
                pbar.update(global_step - pbar.n)

        # GAE
        with torch.no_grad():
            next_value = agent.get_value(next_obs_tensor).reshape(1, -1)
            advantages = torch.zeros_like(rewards, device=nn_device)
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

        b_obs = obs.reshape((-1,) + obs.shape[2:])
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + actions.shape[2:])
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(batch_size)
        # PPO update
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            # print(f'update epoch: {epoch}')
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]
                # print(f'start: {start}, end: {end}')
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                mb_adv = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef),
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

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

                # Trajectory collection (lightweight)
                if collector is not None and (iteration % max(1, args.trajectory.collect_every) == 0) and start == 0 and epoch == 0:
                    try:
                        # print(f'Collecting trajectory at iteration {iteration}')
                        # Use only learnable parameters with stable ordering
                        named_params = dict(agent.actor.named_parameters())
                        collector.collect(named_params, loss=float(pg_loss.detach().cpu().item()), step=iteration)
                        print(f'Collected trajectory at iteration {iteration}')
                    except Exception:
                        print(f'Failed to collect trajectory at iteration {iteration}')
                        pass

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

        # Periodic checkpoint (optional)
        if iteration % max(1, num_iterations // 5) == 0:
            try:
                torch.save(agent.state_dict(), per_variant_dir / f"{args.variant.name}_iter{iteration:05d}.pt")
            except Exception:
                pass

    pbar.update(args.total_timesteps - pbar.n)
    pbar.close()

    # Final checkpoint
    try:
        torch.save(agent.state_dict(), per_variant_dir / f"{args.variant.name}_final.pt")
    except Exception:
        pass

    # Visualization
    # Evaluation-mode loss landscape only: use configured args.test_iterations
    def _eval_actor_loss_eval_like() -> float:
        # Respect current configuration of args.test_iterations
        return _test_makespan_plus_active(agent, args)

    # Always render trajectory plots; landscapes only if enabled
    if not args.trajectory.plot_landscape:
        visualize_trajectory(
            actor_model=agent.actor,
            collector=collector,
            variant_name=args.variant.name,
            log_dir=run_root,
            config=args.trajectory,
            eval_function=None,
        )
    else:
        # Evaluation-only landscape
        visualize_trajectory(
            actor_model=agent.actor,
            collector=collector,
            variant_name=args.variant.name,
            log_dir=run_root,
            config=args.trajectory,
            eval_function=_eval_actor_loss_eval_like,
            landscape_filename="trajectory_landscape_eval.png",
        )


# ------------------------------ Main ------------------------------

def main(args: RunArgs):
    train_and_visualize(args)


if __name__ == "__main__":
    main(tyro.cli(RunArgs))
