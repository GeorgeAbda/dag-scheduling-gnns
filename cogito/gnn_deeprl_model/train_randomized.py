"""
Single-policy training with heavy domain randomization across stages.

This script runs multiple short training stages. At each stage, it samples a new
DatasetArgs configuration (e.g., host_count, vm_count, task counts, and length distribution),
trains for a fixed number of timesteps, and warm-starts the next stage from the
previous model. After each stage, it evaluates the current model on a fixed suite of
configurations and plots makespan and energy trends over stages.

Usage examples:
  python -m scheduler.rl_model.train_randomized \
    --stages 6 \
    --stage_timesteps 150000 \
    --host_range 2 6 \
    --vm_range 4 16 \
    --task_counts 6 8 10 12 16 \
    --length_dists normal uniform left_skewed right_skewed \
    --num_envs 8

Notes:
  - Each stage uses a single (randomized) dataset configuration for all parallel envs.
    This approximates domain randomization over training. For even more variety,
    increase `stages` or widen the sampling ranges.
  - After each stage, we evaluate on a fixed evaluation grid and save CSV/plots into
    the run directory under logs/.
"""
from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tyro
import torch

from cogito.dataset_generator.gen_dataset import DatasetArgs
from cogito.gnn_deeprl_model.train import Args as BaseArgs, train as base_train, test_agent
from cogito.gnn_deeprl_model.agents.gin_agent.agent import GinAgent


@dataclass
class RandArgs(BaseArgs):
    # Staging (how many random configs to train sequentially)
    stages: int = 6
    stage_timesteps: int = 150_000

    # Sampling ranges for dataset parameters
    host_range: Tuple[int, int] = (2, 6)
    vm_range: Tuple[int, int] = (4, 16)
    # Candidate per-workflow task counts to sample per stage (fixed per stage)
    task_counts: List[int] = field(default_factory=lambda: [6, 8, 10, 12, 16])
    # Candidate task length distributions
    length_dists: List[str] = field(default_factory=lambda: ["normal", "uniform", "left_skewed", "right_skewed"])

    # Evaluation grid (fixed) used after each stage
    eval_task_counts: List[int] = field(default_factory=lambda: [6, 8, 10, 12, 16])
    eval_vm_counts: List[int] = field(default_factory=lambda: [4, 8, 12, 16])


def _find_latest_run_dir(logs_dir: Path, exp_suffix: str) -> str | None:
    if not logs_dir.exists():
        return None
    candidates = []
    for p in logs_dir.iterdir():
        if p.is_dir() and p.name.endswith(f"_{exp_suffix}"):
            parts = p.name.split("_", 1)
            try:
                ts = int(parts[0])
            except Exception:
                ts = 0
            candidates.append((ts, p.name))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def _sample_dataset_args(rng: random.Random, base: DatasetArgs, args: RandArgs) -> DatasetArgs:
    host_lo, host_hi = args.host_range
    vm_lo, vm_hi = args.vm_range
    host_count = rng.randint(min(host_lo, host_hi), max(host_lo, host_hi))
    vm_count = rng.randint(min(vm_lo, vm_hi), max(vm_lo, vm_hi))
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


def _eval_and_plot(agent: GinAgent, device: torch.device, base_args: RandArgs, stage_idx: int, out_dir: Path):
    # Build evaluation grid
    eval_configs: list[DatasetArgs] = []
    for tc in base_args.eval_task_counts:
        for vmc in base_args.eval_vm_counts:
            da = DatasetArgs(**vars(base_args.dataset))
            da.gnp_min_n = tc
            da.gnp_max_n = tc
            da.vm_count = vmc
            eval_configs.append(da)

    # Collect metrics
    rows = []
    for i, da in enumerate(eval_configs):
        tmp_args = RandArgs(**vars(base_args))
        tmp_args.dataset = da
        avg_mk, avg_energy_obs, avg_total_energy, _m = test_agent(agent, tmp_args)
        rows.append({
            "stage": stage_idx,
            "config_idx": i,
            "tasks": da.gnp_min_n,
            "vms": da.vm_count,
            "makespan": float(avg_mk),
            "energy": float(avg_total_energy if avg_total_energy else avg_energy_obs),
        })

    # Append to CSV
    csv_path = out_dir / "eval_metrics.csv"
    header_needed = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8") as f:
        if header_needed:
            f.write("stage,config_idx,tasks,vms,makespan,energy\n")
        for r in rows:
            f.write(f"{r['stage']},{r['config_idx']},{r['tasks']},{r['vms']},{r['makespan']},{r['energy']}\n")

    # Plot per-config trends over stages (re-read full CSV)
    import csv as _csv
    stages, makespans, energies, labels = [], {}, {}, {}
    with csv_path.open("r", encoding="utf-8") as f:
        rdr = _csv.DictReader(f)
        for rec in rdr:
            s = int(rec["stage"])
            ci = int(rec["config_idx"])
            mk = float(rec["makespan"])
            en = float(rec["energy"])
            t = int(rec["tasks"])
            v = int(rec["vms"])
            label = f"T{t}-V{v}"
            labels[ci] = label
            if ci not in makespans:
                makespans[ci] = {}
                energies[ci] = {}
            makespans[ci][s] = mk
            energies[ci][s] = en
            stages.append(s)
    stages = sorted(set(stages))

    def _plot(metric_dict: dict[int, dict[int, float]], title: str, fname: str):
        plt.figure(figsize=(8, 5))
        for ci, series in metric_dict.items():
            xs = sorted(series.keys())
            ys = [series[x] for x in xs]
            plt.plot(xs, ys, marker="o", label=labels.get(ci, str(ci)))
        plt.xlabel("Stage")
        plt.ylabel(title)
        plt.title(title + " over stages (eval grid)")
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=200)
        plt.close()

    _plot(makespans, "Makespan", "makespan_over_stages.png")
    _plot(energies, "Energy", "energy_over_stages.png")


def main(args: RandArgs):
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=getattr(logging, level_name, logging.INFO), format="%(asctime)s %(levelname)s %(message)s")

    base_exp = args.exp_name if args.exp_name and args.exp_name != "test" else "rand_dr"
    logs_dir = Path("logs")
    rng = random.Random(args.seed)
    prev_run_dir: str | None = None

    # Split timesteps by stage
    assert args.stage_timesteps > 0, "stage_timesteps must be > 0"

    for s in range(1, int(args.stages) + 1):
        # Sample a dataset config for this stage
        stage_da = _sample_dataset_args(rng, args.dataset, args)

        # Prepare stage args
        stage_args = RandArgs(**vars(args))
        stage_args.dataset = stage_da
        stage_args.total_timesteps = int(args.stage_timesteps)
        stage_args.exp_name = f"{base_exp}_stage{s}of{args.stages}_T{stage_da.gnp_min_n}_V{stage_da.vm_count}_{stage_da.task_length_dist}"
        stage_args.load_model_dir = prev_run_dir

        # Train
        base_train(stage_args)

        # Find latest run dir for this stage
        latest_dir = _find_latest_run_dir(logs_dir, stage_args.exp_name)
        if latest_dir is not None:
            prev_run_dir = latest_dir
        else:
            logging.warning("Could not locate run dir for stage '%s' under '%s'", stage_args.exp_name, logs_dir)

        # Evaluate and plot trends so far
        # Load model into agent
        device = torch.device("cpu")
        agent = GinAgent(device)
        try:
            model_path = Path("scheduler/rl_model/logs") / prev_run_dir / "model.pt"
            agent.load_state_dict(torch.load(str(model_path), weights_only=True))
        except Exception as e:
            logging.warning("Failed to load model for eval: %s", e)
        out_dir = Path("logs") / prev_run_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        _eval_and_plot(agent, device, stage_args, s, out_dir)


if __name__ == "__main__":
    main(tyro.cli(RandArgs))
