"""
Orchestrate two joint domain-randomized trainings (GNP and LINEAR), save models,
then compare their action-score correlations on held-out linear and gnp workflows.

This script programmatically calls:
- scheduler.rl_model.train_domain_randomized.train(DRArgs) twice
  1) with dataset.dag_method = "gnp"  (exp_name = dr_gnp)
  2) with dataset.dag_method = "linear" (exp_name = dr_linear)
- scheduler.viz_results.decision_boundaries.score_correlation_agents.main(Args)
  to compute and plot score correlations for:
  A) new linear workflows
  B) new gnp workflows

Usage (mirrors your requested CLI settings):
  python -m scheduler.rl_model.run_gnp_vs_linear \
    --num_envs 8 \
    --total_timesteps 500000 \
    --test_every_iters 5 \
    --host_range 4 6 \
    --vm_range 10 16 \
    --task_counts 6 8 10 12 16 \
    --length_dists normal uniform left_skewed right_skewed \
    --eval_task_counts 6 8 10 12 16 \
    --eval_vm_counts 4 8 12 16

Outputs:
- Trained checkpoints under logs/<timestamp>_dr_gnp and logs/<timestamp>_dr_linear (final model.pt + checkpoints)
- Correlation plots under logs/<gnp_run>_vs_<linear_run>/score_correlation.pdf for both linear and gnp datasets
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import tyro

from cogito.gnn_deeprl_model.train_domain_randomized import DRArgs, train as dr_train
from cogito.viz_results.decision_boundaries.score_correlation_agents import Args as CorrArgs, main as corr_main
from cogito.dataset_generator.gen_dataset import DatasetArgs


@dataclass
class OrchestrateArgs:
    # mirror DRArgs fields that you want to control from the CLI
    exp_name_gnp: str = "dr_gnp"
    exp_name_linear: str = "dr_linear"
    seed: int = 1
    output_dir: str = "logs"
    device: str = "cpu"

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

    test_every_iters: int = 5
    test_iterations: int = 3

    host_range: Tuple[int, int] = (4, 6)
    vm_range: Tuple[int, int] = (10, 16)
    task_counts: List[int] = field(default_factory=lambda: [6, 8, 10, 12, 16])
    length_dists: List[str] = field(default_factory=lambda: ["normal", "uniform", "left_skewed", "right_skewed"])

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


def _build_base_dr_args(or_args: OrchestrateArgs) -> DRArgs:
    return DRArgs(
        exp_name="placeholder",
        seed=or_args.seed,
        output_dir=or_args.output_dir,
        device=or_args.device,
        total_timesteps=or_args.total_timesteps,
        learning_rate=or_args.learning_rate,
        num_envs=or_args.num_envs,
        num_steps=or_args.num_steps,
        anneal_lr=or_args.anneal_lr,
        gamma=or_args.gamma,
        gae_lambda=or_args.gae_lambda,
        num_minibatches=or_args.num_minibatches,
        update_epochs=or_args.update_epochs,
        norm_adv=or_args.norm_adv,
        clip_coef=or_args.clip_coef,
        clip_vloss=or_args.clip_vloss,
        ent_coef=or_args.ent_coef,
        vf_coef=or_args.vf_coef,
        max_grad_norm=or_args.max_grad_norm,
        target_kl=or_args.target_kl,
        test_every_iters=or_args.test_every_iters,
        test_iterations=or_args.test_iterations,
        host_range=or_args.host_range,
        vm_range=or_args.vm_range,
        task_counts=list(or_args.task_counts),
        length_dists=list(or_args.length_dists),
        dataset=DatasetArgs(
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
            dag_method="gnp",  # will override per run
        ),
        eval_task_counts=list(or_args.eval_task_counts),
        eval_vm_counts=list(or_args.eval_vm_counts),
    )


def main(args: OrchestrateArgs):
    logs_dir = Path(args.output_dir)

    # 1) Train GNP randomized agent
    dr_args_gnp = _build_base_dr_args(args)
    dr_args_gnp.exp_name = args.exp_name_gnp
    dr_args_gnp.dataset.dag_method = "gnp"
    dr_train(dr_args_gnp)
    gnp_dir = _find_latest_run_dir(logs_dir, args.exp_name_gnp)
    if gnp_dir is None:
        raise RuntimeError("Could not locate GNP run directory after training")

    # 2) Train LINEAR randomized agent
    dr_args_lin = _build_base_dr_args(args)
    dr_args_lin.exp_name = args.exp_name_linear
    dr_args_lin.dataset.dag_method = "linear"
    dr_train(dr_args_lin)
    lin_dir = _find_latest_run_dir(logs_dir, args.exp_name_linear)
    if lin_dir is None:
        raise RuntimeError("Could not locate LINEAR run directory after training")

    # 3) Correlation on held-out linear workflows
    out_dir = logs_dir / f"{gnp_dir}_vs_{lin_dir}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    corr_linear = CorrArgs(
        model_a_dir=gnp_dir,
        model_b_dir=lin_dir,
        model_a_filename="model.pt",
        model_b_filename="model.pt",
        dataset=DatasetArgs(
            host_count=4,
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
        ),
        out_path=str(out_dir / "score_correlation_linear_workflows.pdf"),
    )
    corr_main(corr_linear)

    # 4) Correlation on held-out gnp workflows
    corr_gnp = CorrArgs(
        model_a_dir=gnp_dir,
        model_b_dir=lin_dir,
        model_a_filename="model.pt",
        model_b_filename="model.pt",
        dataset=DatasetArgs(
            host_count=4,
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
        ),
        out_path=str(out_dir / "score_correlation_gnp_workflows.pdf"),
    )
    corr_main(corr_gnp)

    print("\nCompleted training and correlation analyses.")
    print(f"GNP run dir: {gnp_dir}")
    print(f"LINEAR run dir: {lin_dir}")


if __name__ == "__main__":
    main(tyro.cli(OrchestrateArgs))
