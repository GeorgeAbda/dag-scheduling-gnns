"""
Joint domain-randomized training: each parallel environment uses a different configuration,
and training proceeds with mixed rollouts from diverse workloads in every PPO update.

Also performs evaluation on a grid of configurations at configured intervals and
generates plots over evaluation checkpoints for:
- Makespan
- Active energy
- Idle energy
- Active+Idle energy (sum)

Usage examples:
  python -m scheduler.rl_model.train_domain_randomized \
    --num_envs 8 \
    --total_timesteps 500000 \
    --test_every_iters 5 \
    --host_range 2 6 \
    --vm_range 4 16 \
    --task_counts 6 8 10 12 16 \
    --length_dists normal uniform left_skewed right_skewed \
    --eval_task_counts 6 8 10 12 16 \
    --eval_vm_counts 4 8 12 16
"""
from __future__ import annotations

import logging
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.wrappers import RecordEpisodeStatistics
from torch.utils.tensorboard import SummaryWriter
import tyro
from tqdm import tqdm

# Add grandparent to path (mirror of train.py)
import sys
grandparent_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
if grandparent_dir not in sys.path:
    sys.path.insert(0, grandparent_dir)

from cogito.config.settings import MIN_TESTING_DS_SEED
from cogito.dataset_generator.gen_dataset import DatasetArgs
from cogito.gnn_deeprl_model.agents.agent import Agent
from cogito.gnn_deeprl_model.agents.gin_agent.agent import GinAgent
from cogito.gnn_deeprl_model.agents.gin_agent.wrapper import GinAgentWrapper
from cogito.gnn_deeprl_model.core.env.gym_env import CloudSchedulingGymEnvironment


@dataclass
class DRArgs:
    # Base experiment args (subset from train.py for brevity; can be extended as needed)
    exp_name: str = "dr_mixed"
    seed: int = 1
    output_dir: str = "logs"
    torch_deterministic: bool = True
    device: str = "cpu"
    torch_num_threads: int | None = None
    capture_video: bool = False

    total_timesteps: int = 500_000
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

    # Evaluation cadence
    test_every_iters: int = 5
    test_iterations: int = 3

    # Domain randomization ranges (used to sample per-env configs)
    host_range: Tuple[int, int] = (4, 10)
    vm_range: Tuple[int, int] = (10, 16)
    task_counts: List[int] = field(default_factory=lambda: [6, 8, 10, 12, 16])
    length_dists: List[str] = field(default_factory=lambda: ["normal", "uniform", "left_skewed", "right_skewed"])
    # Other dataset defaults (taken as base when not randomized)
    dataset: DatasetArgs = field(
        default_factory=lambda: DatasetArgs(
            host_count=4,
            vm_count=10,
            workflow_count=10,
            gnp_min_n=10,
            gnp_max_n=10,
            max_memory_gb=10,
            min_cpu_speed=500,
            max_cpu_speed=5000,
            min_task_length=500,
            max_task_length=100_000,
            task_arrival="static",
            dag_method="gnp",
        )
    )

    # Evaluation grid for plotting at checkpoints
    eval_task_counts: List[int] = field(default_factory=lambda: [6, 8, 10, 12, 16])
    eval_vm_counts: List[int] = field(default_factory=lambda: [4, 8, 12, 16])

    # runtime filled
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0
    run_name: str = ""


# -----------------------------
# Environment helpers
# -----------------------------

def _sample_dataset_args(rng: random.Random, base: DatasetArgs, args: DRArgs) -> DatasetArgs:
    h_lo, h_hi = args.host_range
    v_lo, v_hi = args.vm_range
    host_count = rng.randint(min(h_lo, h_hi), max(h_lo, h_hi))
    vm_count = rng.randint(min(v_lo, v_hi), max(v_lo, v_hi))
    tc = rng.choice(args.task_counts)
    dist = rng.choice(args.length_dists) if args.length_dists else base.task_length_dist
    da = DatasetArgs(
        seed=base.seed,
        host_count=host_count,
        vm_count=vm_count,
        max_memory_gb=base.max_memory_gb,
        min_cpu_speed=base.min_cpu_speed,
        max_cpu_speed=base.max_cpu_speed,
        workflow_count=base.workflow_count,
        dag_method="gnp",
        gnp_min_n=tc,
        gnp_max_n=tc,
        task_length_dist=dist,
        min_task_length=base.min_task_length,
        max_task_length=base.max_task_length,
        task_arrival=base.task_arrival,
        arrival_rate=base.arrival_rate,
    )
    return da


def make_env_dr(idx: int, args: DRArgs, rng_seed: int):
    rng = random.Random(rng_seed)
    dataset_args = _sample_dataset_args(rng, args.dataset, args)
    env: gym.Env = CloudSchedulingGymEnvironment(
        dataset_args=dataset_args,
        collect_timelines=False,
        compute_metrics=False,
        profile=False,
    )
    env = GinAgentWrapper(env)
    return RecordEpisodeStatistics(env)


def make_env_thunk_dr(idx: int, args: DRArgs):
    def _thunk():
        return make_env_dr(idx, args, args.seed + idx)
    return _thunk


def make_test_env(args: DRArgs, dataset_args: DatasetArgs):
    env: gym.Env = CloudSchedulingGymEnvironment(
        dataset_args=dataset_args,
        collect_timelines=False,
        compute_metrics=True,
        profile=False,
    )
    return GinAgentWrapper(env)


def make_agent(device: torch.device) -> Agent:
    return GinAgent(device)


# -----------------------------
# Evaluation utilities
# -----------------------------

def eval_on_grid(agent: Agent, device: torch.device, args: DRArgs, writer: SummaryWriter, global_step: int, checkpoint_idx: int, out_dir: Path):
    # Build evaluation grid
    eval_configs: list[DatasetArgs] = []
    for tc in args.eval_task_counts:
        for vmc in args.eval_vm_counts:
            da = DatasetArgs(**vars(args.dataset))
            da.gnp_min_n = tc
            da.gnp_max_n = tc
            da.vm_count = vmc
            eval_configs.append(da)

    # Iterate and collect metrics
    import csv as _csv
    csv_path = out_dir / "eval_grid_metrics.csv"
    header_needed = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8") as f:
        if header_needed:
            f.write("checkpoint,global_step,config_idx,tasks,vms,makespan,energy_total,energy_active,energy_idle\n")
        for i, da in enumerate(eval_configs):
            env = make_test_env(args, da)
            total_mk = 0.0
            total_energy_total = 0.0
            total_active = 0.0
            total_idle = 0.0
            for s in range(args.test_iterations):
                next_obs, _ = env.reset(seed=MIN_TESTING_DS_SEED + s)
                while True:
                    with torch.no_grad():
                        obs_tensor = torch.from_numpy(np.asarray(next_obs, dtype=np.float32).reshape(1, -1)).to(device)
                        action, _, _, _ = agent.get_action_and_value(obs_tensor)
                    next_obs, _, terminated, truncated, info = env.step(int(action.item()))
                    if terminated or truncated:
                        # try to extract detailed energy breakdown
                        if isinstance(info, dict):
                            total_energy_total += float(info.get("total_energy", 0.0))
                            total_active += float(info.get("total_energy_active", 0.0))
                            total_idle += float(info.get("total_energy_idle", 0.0))
                        # fallback to observation estimates
                        if hasattr(env, "prev_obs") and env.prev_obs is not None:
                            total_mk += float(env.prev_obs.makespan())
                            if info.get("total_energy") is None:
                                total_energy_total += float(env.prev_obs.energy_consumption())
                        break
            env.close()
            n = max(1, args.test_iterations)
            avg_mk = total_mk / n
            avg_total = total_energy_total / n
            avg_act = total_active / n
            avg_idle = total_idle / n
            # TB scalar tags per-config index
            writer.add_scalar(f"grid/T{da.gnp_min_n}_V{da.vm_count}/makespan", avg_mk, global_step)
            writer.add_scalar(f"grid/T{da.gnp_min_n}_V{da.vm_count}/energy_total", avg_total, global_step)
            if avg_act > 0 or avg_idle > 0:
                writer.add_scalar(f"grid/T{da.gnp_min_n}_V{da.vm_count}/energy_active", avg_act, global_step)
                writer.add_scalar(f"grid/T{da.gnp_min_n}_V{da.vm_count}/energy_idle", avg_idle, global_step)
                writer.add_scalar(f"grid/T{da.gnp_min_n}_V{da.vm_count}/energy_active_plus_idle", avg_act + avg_idle, global_step)
            # append CSV row
            f.write(f"{checkpoint_idx},{global_step},{i},{da.gnp_min_n},{da.vm_count},{avg_mk},{avg_total},{avg_act},{avg_idle}\n")

    # Create simple line plots over checkpoints for each config
    import csv as _csv
    recs = []
    with csv_path.open("r", encoding="utf-8") as f:
        rdr = _csv.DictReader(f)
        for r in rdr:
            recs.append(r)

    # Build series per config label
    series = {}
    for r in recs:
        key = f"T{r['tasks']}-V{r['vms']}"
        s = int(r["checkpoint"])
        series.setdefault(key, {"s": [], "mk": [], "et": [], "ea": [], "ei": []})
        series[key]["s"].append(s)
        series[key]["mk"].append(float(r["makespan"]))
        series[key]["et"].append(float(r["energy_total"]))
        series[key]["ea"].append(float(r["energy_active"]))
        series[key]["ei"].append(float(r["energy_idle"]))

    def _plot(metric_key: str, ylabel: str, fname: str):
        plt.figure(figsize=(9, 6))
        for label, d in series.items():
            xs = d["s"]
            ys = d[metric_key]
            order = np.argsort(xs)
            xs = np.array(xs)[order]
            ys = np.array(ys)[order]
            plt.plot(xs, ys, marker="o", label=label)
        plt.xlabel("Evaluation checkpoint")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} across configs over checkpoints")
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=200)
        plt.close()

    _plot("mk", "Makespan", "grid_makespan_over_time.png")
    _plot("et", "Total Energy", "grid_energy_total_over_time.png")
    _plot("ea", "Active Energy", "grid_energy_active_over_time.png")
    _plot("ei", "Idle Energy", "grid_energy_idle_over_time.png")


# -----------------------------
# Training loop (PPO) – adapted from train.py
# -----------------------------

def train(args: DRArgs):
    args.run_name = f"{int(time.time())}_{args.exp_name}"

    # Logging
    writer = SummaryWriter(f"{args.output_dir}/{args.run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Seeding and device
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    if args.torch_num_threads:
        try:
            torch.set_num_threads(int(args.torch_num_threads))
        except Exception:
            pass

    def pick_device(choice: str) -> torch.device:
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

    device = pick_device(args.device)

    # Derived sizes
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = max(1, int(args.batch_size // max(1, args.num_minibatches)))
    args.num_iterations = max(1, int(args.total_timesteps // max(1, args.batch_size)))

    # Vectorized envs with per-env randomized configs
    envs = gym.vector.AsyncVectorEnv([make_env_thunk_dr(i, args) for i in range(args.num_envs)])
    obs_space = envs.single_observation_space
    act_space = envs.single_action_space
    assert isinstance(act_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = make_agent(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Storage
    obs = torch.zeros((args.num_steps, args.num_envs) + obs_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + act_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs_tensor = torch.Tensor(next_obs).to(device)
    next_done_tensor = torch.zeros(args.num_envs).to(device)

    eval_checkpoint = 0
    pbar = tqdm(total=args.total_timesteps)
    last_model_save = 0

    try:
        for iteration in range(1, args.num_iterations + 1):
            if args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / args.num_iterations
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
                rewards[step] = torch.Tensor(reward).to(device).view(-1)
                next_obs_tensor, next_done_tensor = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

                if "final_info" in infos:
                    for info in infos["final_info"]:
                        if info and "episode" in info:
                            writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                            writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                            # Keep the progress bar aligned with total timesteps progressed
                            if pbar is not None:
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

            # flatten batch
            b_obs = obs.reshape((-1,) + obs_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + act_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # optimize policy
            b_inds = np.arange(args.batch_size)
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()
                    with torch.no_grad():
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                    mb_adv = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                    pg_loss = torch.max(-mb_adv * ratio, -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)).mean()
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
                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

            # scalars
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

            # Evaluation on grid at cadence
            if args.test_every_iters > 0 and (iteration % args.test_every_iters == 0 or iteration == args.num_iterations):
                eval_checkpoint += 1
                out_dir = Path(args.output_dir) / args.run_name
                out_dir.mkdir(parents=True, exist_ok=True)
                eval_on_grid(agent, device, args, writer, global_step, eval_checkpoint, out_dir)

            # Periodic checkpoint save
            if (global_step - last_model_save) >= 10_000:
                ckpt_path = f"{args.output_dir}/{args.run_name}/model_{global_step}.pt"
                torch.save(agent.state_dict(), ckpt_path)
                last_model_save = global_step
    finally:
        # Ensure final model is saved even if interrupted
        try:
            torch.save(agent.state_dict(), f"{args.output_dir}/{args.run_name}/model.pt")
        except Exception:
            pass
        try:
            envs.close()
        except Exception:
            pass
        try:
            writer.close()
        except Exception:
            pass
        if pbar is not None:
            try:
                pbar.close()
            except Exception:
                pass


if __name__ == "__main__":
    # Configure logging
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")
    train(tyro.cli(DRArgs))
