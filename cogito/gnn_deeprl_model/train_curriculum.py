"""
Curriculum training script with two phases:

- Stage 1 (size curriculum): train sequentially on fixed per-workflow task counts
  (default: 6, 8, 10). Weights are warm-started from previous stage.
- Stage 2 (randomization): continue training across several stages while randomizing
  VM count, host count, and task length distribution per stage.

After each stage, the script evaluates on a fixed configuration grid and plots
makespan and energy trends per configuration across stages.

Usage examples:
  python -m scheduler.rl_model.train_curriculum \
    --stage1_task_counts 6 8 10 \
    --stage1_stage_timesteps 150000 \
    --stage2_stages 4 \
    --stage2_stage_timesteps 200000 \
    --stage2_host_range 2 6 \
    --stage2_vm_range 4 16 \
    --stage2_length_dists normal uniform left_skewed right_skewed \
    --num_envs 8
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
class CurrArgs(BaseArgs):
    # Stage 1: curriculum over per-workflow task counts
    stage1_task_counts: List[int] = field(default_factory=lambda: [6, 8, 10])
    stage1_stage_timesteps: int = 150_000

    # Stage 2: randomization over VM/host counts and length distribution
    stage2_stages: int = 4
    stage2_stage_timesteps: int = 200_000
    stage2_host_range: Tuple[int, int] = (2, 6)
    stage2_vm_range: Tuple[int, int] = (4, 16)
    stage2_length_dists: List[str] = field(default_factory=lambda: ["normal", "uniform", "left_skewed", "right_skewed"])

    # Evaluation grid
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


def _eval_and_plot(agent: GinAgent, device: torch.device, base_args: CurrArgs, stage_idx: int, out_dir: Path):
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
        tmp_args = CurrArgs(**vars(base_args))
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


def main(args: CurrArgs):
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=getattr(logging, level_name, logging.INFO), format="%(asctime)s %(levelname)s %(message)s")

    base_exp = args.exp_name if args.exp_name and args.exp_name != "test" else "curriculum"
    logs_dir = Path("logs")
    rng = random.Random(args.seed)
    prev_run_dir: str | None = None
    stage_counter = 0

    # Stage 1: curriculum over task counts
    for tc in args.stage1_task_counts:
        stage_counter += 1
        stage_args = CurrArgs(**vars(args))
        stage_args.dataset.gnp_min_n = tc
        stage_args.dataset.gnp_max_n = tc
        stage_args.total_timesteps = int(args.stage1_stage_timesteps)
        stage_args.exp_name = f"{base_exp}_s1_tc{tc}_stage{stage_counter}"
        stage_args.load_model_dir = prev_run_dir

        base_train(stage_args)
        latest_dir = _find_latest_run_dir(logs_dir, stage_args.exp_name)
        if latest_dir is not None:
            prev_run_dir = latest_dir
        else:
            logging.warning("Could not locate run dir for stage '%s' under '%s'", stage_args.exp_name, logs_dir)

        # Evaluate after each stage 1 step
        device = torch.device("cpu")
        agent = GinAgent(device)
        try:
            model_path = Path("scheduler/rl_model/logs") / prev_run_dir / "model.pt"
            agent.load_state_dict(torch.load(str(model_path), weights_only=True))
        except Exception as e:
            logging.warning("Failed to load model for eval: %s", e)
        out_dir = Path("logs") / prev_run_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        _eval_and_plot(agent, device, stage_args, stage_counter, out_dir)

    # Stage 2: randomization across several stages
    h_lo, h_hi = args.stage2_host_range
    v_lo, v_hi = args.stage2_vm_range
    length_dists = list(args.stage2_length_dists)

    for s in range(1, int(args.stage2_stages) + 1):
        stage_counter += 1
        host_count = rng.randint(min(h_lo, h_hi), max(h_lo, h_hi))
        vm_count = rng.randint(min(v_lo, v_hi), max(v_lo, v_hi))
        dist = rng.choice(length_dists) if length_dists else args.dataset.task_length_dist
        # Keep workflow_count consistent; randomize per-workflow task count around prior curriculum range
        tc = rng.choice(args.stage1_task_counts)

        stage_da = DatasetArgs(**vars(args.dataset))
        stage_da.host_count = host_count
        stage_da.vm_count = vm_count
        stage_da.gnp_min_n = tc
        stage_da.gnp_max_n = tc
        stage_da.task_length_dist = dist
        stage_da.dag_method = "gnp"

        stage_args = CurrArgs(**vars(args))
        stage_args.dataset = stage_da
        stage_args.total_timesteps = int(args.stage2_stage_timesteps)
        stage_args.exp_name = f"{base_exp}_s2_T{tc}_H{host_count}_V{vm_count}_{dist}_stage{stage_counter}"
        stage_args.load_model_dir = prev_run_dir

        base_train(stage_args)
        latest_dir = _find_latest_run_dir(logs_dir, stage_args.exp_name)
        if latest_dir is not None:
            prev_run_dir = latest_dir
        else:
            logging.warning("Could not locate run dir for stage '%s' under '%s'", stage_args.exp_name, logs_dir)

        # Evaluate and plot
        device = torch.device("cpu")
        agent = GinAgent(device)
        try:
            model_path = Path("scheduler/rl_model/logs") / prev_run_dir / "model.pt"
            agent.load_state_dict(torch.load(str(model_path), weights_only=True))
        except Exception as e:
            logging.warning("Failed to load model for eval: %s", e)
        out_dir = Path("logs") / prev_run_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        _eval_and_plot(agent, device, stage_args, stage_counter, out_dir)


if __name__ == "__main__":
    main(tyro.cli(CurrArgs))
