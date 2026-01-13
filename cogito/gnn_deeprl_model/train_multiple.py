"""
Train the RL agent on several workflows, each with exactly 10 tasks (jobs).

This script reuses the core training loop in `scheduler/rl_model/train.py` and
forces the dataset generator to create workflows with 10 tasks by overriding
`dataset.gnp_min_n` and `dataset.gnp_max_n` to 10 before training starts.

Usage examples:
  - Default settings (several workflows with 10 jobs each):
      python -m scheduler.rl_model.train_multiple

  - Customize number of workflows and parallel envs:
      python -m scheduler.rl_model.train_multiple --dataset.workflow_count 8 --num_envs 8

  - Customize hosts/VMs while keeping 10-job workflows:
      python -m scheduler.rl_model.train_multiple --dataset.host_count 4 --dataset.vm_count 10

Notes:
  - You can configure any other hyperparameters or dataset fields like normal; this
    script only pins the per-workflow task count to 10.
  - For reproducibility, use --seed and consider setting LOG_LEVEL=INFO or DEBUG.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import tyro

# Reuse the existing training entrypoints
from cogito.gnn_deeprl_model.train import Args as BaseArgs, train as base_train


@dataclass
class MultiArgs(BaseArgs):
    """Extended args to support training over multiple task counts.

    Provide a list of task counts in `task_counts`. If more than one value is provided,
    the script will perform curriculum-style sequential training: it will train on the
    first task count, then continue training on the next, reusing the weights from the
    previous stage, and so on.
    """
    task_counts: List[int] = field(default_factory=lambda: [10])
    """List of per-workflow task counts to train on sequentially (e.g., 8 10 12 16)."""

    stage_timesteps: int | None = None
    """Optional; if set, overrides total_timesteps for each stage individually."""


def _find_latest_run_dir(logs_dir: Path, exp_name_suffix: str) -> str | None:
    """Return the most recent run directory name matching the given exp_name suffix.

    The base trainer names runs as `<timestamp>_<exp_name>`. We search under logs/ for
    directories ending with `exp_name_suffix` and return the latest by timestamp prefix.
    """
    if not logs_dir.exists():
        return None
    candidates = []
    for p in logs_dir.iterdir():
        if not p.is_dir():
            continue
        name = p.name
        if name.endswith(f"_{exp_name_suffix}"):
            # Parse leading timestamp if present
            parts = name.split("_", 1)
            try:
                ts = int(parts[0])
            except Exception:
                ts = 0
            candidates.append((ts, name))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def main(args: MultiArgs):
    # Configure logging so env debug statements are visible during training
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")

    # Ensure we are using G(n,p) DAG generator (unless user explicitly set something else)
    if not args.dataset.dag_method:
        args.dataset.dag_method = "gnp"

    # Normalize and sort unique task counts to improve reproducibility
    task_counts = list(dict.fromkeys(int(x) for x in args.task_counts if int(x) > 0))
    if not task_counts:
        task_counts = [10]

    logs_dir = Path("logs")

    # Single-stage training (backwards compatible): exactly one task count
    if len(task_counts) == 1:
        tc = task_counts[0]
        args.dataset.gnp_min_n = tc
        args.dataset.gnp_max_n = tc
        if not args.exp_name or args.exp_name == "test":
            args.exp_name = f"multi_tc{tc}"
        base_train(args)
        return

    # Multi-stage curriculum training over multiple task counts
    base_exp = args.exp_name if args.exp_name and args.exp_name != "test" else "multi_varied_tasks"
    prev_run_dir: str | None = None

    for idx, tc in enumerate(task_counts):
        stage_args = MultiArgs(**vars(args))  # shallow copy of CLI args
        stage_args.dataset.gnp_min_n = tc
        stage_args.dataset.gnp_max_n = tc

        # Divide total timesteps across stages if stage_timesteps not explicitly set
        if stage_args.stage_timesteps is not None and stage_args.stage_timesteps > 0:
            stage_args.total_timesteps = int(stage_args.stage_timesteps)
        else:
            # Even split across stages
            stages = max(1, len(task_counts))
            stage_args.total_timesteps = max(1, int(args.total_timesteps // stages))

        # Name this stage for easier log discovery
        stage_exp_name = f"{base_exp}_tc{tc}_stage{idx+1}of{len(task_counts)}"
        stage_args.exp_name = stage_exp_name

        # If there is a previous run, attempt to warm start from its final model
        if prev_run_dir is not None:
            stage_args.load_model_dir = prev_run_dir
        else:
            stage_args.load_model_dir = None

        # Run training for this stage
        base_train(stage_args)

        # Discover the run directory we just created to feed into next stage
        latest_dir = _find_latest_run_dir(logs_dir, stage_exp_name)
        if latest_dir is not None:
            prev_run_dir = latest_dir
        else:
            # If not found, keep previous (if any); otherwise next stage will start cold
            logging.warning("Could not locate run dir for stage '%s' under '%s'", stage_exp_name, logs_dir)


if __name__ == "__main__":
    # Expose the extended CLI; users can still pass all BaseArgs fields, plus task_counts and stage_timesteps
    main(tyro.cli(MultiArgs))
