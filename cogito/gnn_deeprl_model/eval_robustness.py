import os
import sys
import math
import csv
from dataclasses import dataclass
from typing import Sequence, Optional

import numpy as np
import torch
import tyro

# Ensure project root on path (same pattern as train.py)
grandparent_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if grandparent_dir not in sys.path:
    sys.path.insert(0, grandparent_dir)

from cogito.dataset_generator.core.gen_dataset import generate_dataset
from cogito.dataset_generator.gen_dataset import DatasetArgs
from cogito.gnn_deeprl_model.core.env.gym_env import CloudSchedulingGymEnvironment
from cogito.gnn_deeprl_model.agents.gin_agent.agent import GinAgent
from cogito.gnn_deeprl_model.agents.gin_agent.wrapper import GinAgentWrapper


@dataclass
class EvalArgs:
    model_path: str
    """Path to a trained model .pt file (GinAgent state_dict)."""

    out_csv: str = "logs/robustness_eval.csv"
    """Path to write CSV results."""

    seeds: int = 5
    """Number of seeds per config."""

    noise_levels: Sequence[float] = (0.0, 0.1, 0.2, 0.4)
    """Task length multiplicative noise sigma values (lognormal std in log-space)."""

    # Base dataset generation defaults
    host_count: int = 2
    vm_count: int = 2
    max_memory_gb: int = 10
    min_cpu_speed: int = 500
    max_cpu_speed: int = 5000
    workflow_count: int = 3
    task_length_dist: str = "normal"
    min_task_length: int = 500
    max_task_length: int = 100_000
    task_arrival: str = "static"
    arrival_rate: float = 3.0

    # Families to evaluate: a small set mapped to DatasetArgs variations
    families: Sequence[str] = ("gnp_small", "gnp_medium", "linear", "pegasus")

    device: str = "cpu"
    # Visualization / tracing controls
    collect_timelines: bool = True
    """If true, environment collects per-VM timelines and capacities at episode end (slight overhead)."""
    save_first_episode_heatmap: bool = True
    """If true, save a utilization heatmap PNG for the very first evaluated episode."""


def load_agent(model_path: str, device: torch.device) -> GinAgent:
    agent = GinAgent(device)
    state = torch.load(model_path, map_location=device)
    agent.load_state_dict(state)
    agent.eval()
    return agent


def make_family_args(base: EvalArgs, family: str, seed: int) -> DatasetArgs:
    # Configure DAG family via DatasetArgs knobs available
    if family == "gnp_small":
        gnp_min_n, gnp_max_n, dag_method = 8, 16, "gnp"
    elif family == "gnp_medium":
        gnp_min_n, gnp_max_n, dag_method = 20, 40, "gnp"
    elif family == "linear":
        # Narrow range to keep linear chains at manageable sizes
        gnp_min_n, gnp_max_n, dag_method = 12, 24, "linear"
    elif family == "pegasus":
        # Size controlled by underlying files; this uses real motifs
        gnp_min_n, gnp_max_n, dag_method = 1, 1, "pegasus"
    else:
        raise ValueError(f"Unknown family: {family}")

    return DatasetArgs(
        seed=seed,
        host_count=base.host_count,
        vm_count=base.vm_count,
        max_memory_gb=base.max_memory_gb,
        min_cpu_speed=base.min_cpu_speed,
        max_cpu_speed=base.max_cpu_speed,
        workflow_count=base.workflow_count,
        dag_method=dag_method,
        gnp_min_n=gnp_min_n,
        gnp_max_n=gnp_max_n,
        task_length_dist=base.task_length_dist,
        min_task_length=base.min_task_length,
        max_task_length=base.max_task_length,
        task_arrival=base.task_arrival,
        arrival_rate=base.arrival_rate,
    )


def apply_task_length_noise(dataset, sigma: float, rng: np.random.RandomState):
    """Apply multiplicative lognormal noise with log-std sigma to each task length."""
    if sigma <= 0:
        return dataset
    for wf in dataset.workflows:
        for t in wf.tasks:
            # lognormal: exp(N(0, sigma^2))
            noise = rng.lognormal(mean=0.0, sigma=sigma)
            new_len = max(1, int(round(t.length * noise)))
            t.length = new_len
    return dataset


def run_episode(agent: GinAgent, dataset, seed: int, return_trace: bool = False, collect_timelines: bool = False) -> tuple[float, float, float, float, dict, Optional[dict]]:
    env = CloudSchedulingGymEnvironment(dataset=dataset, collect_timelines=collect_timelines, compute_metrics=True)
    env = GinAgentWrapper(env)
    obs, _ = env.reset(seed=seed)
    final_info = None
    while True:
        obs_tensor = torch.from_numpy(obs.astype(np.float32).reshape(1, -1))
        action, _, _, _ = agent.get_action_and_value(obs_tensor)
        vm_action = int(action.item())
        obs, _, terminated, truncated, info = env.step(vm_action)
        if terminated or truncated:
            final_info = info
            break
    assert env.prev_obs is not None
    makespan = env.prev_obs.makespan()
    total_energy = float(final_info.get("total_energy", env.prev_obs.energy_consumption())) if isinstance(final_info, dict) else env.prev_obs.energy_consumption()
    total_energy_active = float(final_info.get("total_energy_active", 0.0)) if isinstance(final_info, dict) else 0.0
    total_energy_idle = float(final_info.get("total_energy_idle", 0.0)) if isinstance(final_info, dict) else 0.0
    metrics = {}
    if isinstance(final_info, dict):
        for k in [
            "bottleneck_steps",
            "decision_steps",
            "sum_ready_tasks",
            "sum_bottleneck_ready_tasks",
            "cumulative_wait_time",
        ]:
            if k in final_info:
                metrics[k] = final_info[k]
    # Collect trace if requested
    trace = None
    if return_trace and isinstance(final_info, dict):
        trace = {
            "vm_total_mem": final_info.get("vm_total_mem", []),
            "vm_total_cores": final_info.get("vm_total_cores", []),
            "vm_timelines": final_info.get("vm_timelines", []),
            # Assignability time series (optional)
            "timeline_t": final_info.get("timeline_t", []),
            "timeline_ready": final_info.get("timeline_ready", []),
            "timeline_schedulable": final_info.get("timeline_schedulable", []),
        }
    env.close()
    return makespan, total_energy, total_energy_active, total_energy_idle, metrics, trace


def evaluate(args: EvalArgs):
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    device = torch.device(args.device)
    agent = load_agent(args.model_path, device)

    rows = [("family", "sigma", "seed", "makespan", "total_energy", "active_energy", "idle_energy",
             "bottleneck_steps", "decision_steps", "sum_ready_tasks", "sum_bottleneck_ready_tasks", "cumulative_wait_time")]
    first_episode_trace: Optional[dict] = None
    base_seed = 12345
    for family in args.families:
        for sigma in args.noise_levels:
            for s in range(args.seeds):
                seed = base_seed + s
                fam_args = make_family_args(args, family, seed)
                # Generate dataset for this seed/family
                dataset = generate_dataset(
                    seed=fam_args.seed,
                    host_count=fam_args.host_count,
                    vm_count=fam_args.vm_count,
                    max_memory_gb=fam_args.max_memory_gb,
                    min_cpu_speed_mips=fam_args.min_cpu_speed,
                    max_cpu_speed_mips=fam_args.max_cpu_speed,
                    workflow_count=fam_args.workflow_count,
                    dag_method=fam_args.dag_method,
                    gnp_min_n=fam_args.gnp_min_n,
                    gnp_max_n=fam_args.gnp_max_n,
                    task_length_dist=fam_args.task_length_dist,
                    min_task_length=fam_args.min_task_length,
                    max_task_length=fam_args.max_task_length,
                    task_arrival=fam_args.task_arrival,
                    arrival_rate=fam_args.arrival_rate,
                    vm_rng_seed=0,  # keep VM configs fixed across seeds for fair comparisons
                )
                # Apply multiplicative noise
                rng = np.random.RandomState(seed * 7919 + 17)
                dataset = apply_task_length_noise(dataset, sigma, rng)

                # Run one episode (dataset contains multiple workflows)
                # Capture the first episode's detailed trace for heatmap plotting
                want_trace = first_episode_trace is None
                mk, en, en_act, en_idle, m, trace = run_episode(
                    agent, dataset, seed=seed, return_trace=want_trace, collect_timelines=args.collect_timelines
                )
                if want_trace and trace is not None:
                    first_episode_trace = trace
                rows.append((family, sigma, seed, mk, en, en_act, en_idle,
                             int(m.get("bottleneck_steps", 0)), int(m.get("decision_steps", 0)),
                             int(m.get("sum_ready_tasks", 0)), int(m.get("sum_bottleneck_ready_tasks", 0)),
                             float(m.get("cumulative_wait_time", 0.0))))
                print(f"[EVAL] family={family} sigma={sigma} seed={seed} -> makespan={mk:.3f} energy={en:.3f} (active={en_act:.3f}, idle={en_idle:.3f}) "
                      f"bneck_steps={m.get('bottleneck_steps', 0)}/{m.get('decision_steps', 0)} ready_blocked={m.get('sum_bottleneck_ready_tasks', 0)}/{m.get('sum_ready_tasks', 0)}")

    # Write CSV
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"Wrote results to {args.out_csv}")

    # Optional plotting if matplotlib is available
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns
        df = pd.DataFrame(rows[1:], columns=rows[0])
        # Aggregate by family and sigma
        agg = df.groupby(["family", "sigma"], as_index=False).agg({
            "makespan": "mean",
            "total_energy": "mean",
            "active_energy": "mean",
            "idle_energy": "mean",
            "bottleneck_steps": "mean",
            "decision_steps": "mean",
            "sum_ready_tasks": "mean",
            "sum_bottleneck_ready_tasks": "mean",
            "cumulative_wait_time": "mean",
        })
        # Plot degradation curves per family
        for metric in ["makespan", "total_energy"]:
            plt.figure()
            for fam in args.families:
                sub = agg[agg["family"] == fam].sort_values("sigma")
                plt.plot(sub["sigma"], sub[metric], marker="o", label=fam)
            plt.xlabel("Task length noise sigma")
            plt.ylabel(metric.replace("_", " ").title())
            plt.title(f"Robustness vs noise: {metric}")
            plt.legend()
            out_png = os.path.splitext(args.out_csv)[0] + f"_{metric}.png"
            plt.savefig(out_png, bbox_inches="tight")
            print(f"Saved plot: {out_png}")
        # Plot active energy fraction vs noise per family
        plt.figure()
        for fam in args.families:
            sub = agg[agg["family"] == fam].sort_values("sigma")
            frac = sub["active_energy"] / np.maximum(1e-9, (sub["active_energy"] + sub["idle_energy"]))
            plt.plot(sub["sigma"], frac, marker="o", label=fam)
        plt.xlabel("Task length noise sigma")
        plt.ylabel("Active energy fraction")
        plt.title("Active vs Idle Energy Fraction across noise")
        plt.ylim(0.0, 1.0)
        plt.legend()
        out_png = os.path.splitext(args.out_csv)[0] + "_active_fraction.png"
        plt.savefig(out_png, bbox_inches="tight")
        print(f"Saved plot: {out_png}")
        # Plot bottleneck ratios vs noise per family
        plt.figure()
        for fam in args.families:
            sub = agg[agg["family"] == fam].sort_values("sigma")
            ratio_steps = sub["bottleneck_steps"] / np.maximum(1e-9, sub["decision_steps"])
            ratio_ready = sub["sum_bottleneck_ready_tasks"] / np.maximum(1e-9, sub["sum_ready_tasks"])
            plt.plot(sub["sigma"], ratio_steps, marker="o", label=f"{fam} (steps)")
            plt.plot(sub["sigma"], ratio_ready, marker="x", linestyle="--", label=f"{fam} (ready)")
        plt.xlabel("Task length noise sigma")
        plt.ylabel("Bottleneck ratio")
        plt.title("Resource bottlenecks vs noise")
        plt.ylim(0.0, 1.0)
        plt.legend()
        out_png = os.path.splitext(args.out_csv)[0] + "_bottleneck_ratios.png"
        plt.savefig(out_png, bbox_inches="tight")
        print(f"Saved plot: {out_png}")

        # First-episode utilization heatmap (max of mem/core utilization per VM over time)
        if args.save_first_episode_heatmap and first_episode_trace is not None:
            try:
                vm_timelines = first_episode_trace.get("vm_timelines", [])
                vm_total_mem = first_episode_trace.get("vm_total_mem", [])
                vm_total_cores = first_episode_trace.get("vm_total_cores", [])
                # Determine makespan from timelines
                makespan_est = 0.0
                for vm_list in vm_timelines:
                    for seg in vm_list:
                        makespan_est = max(makespan_est, float(seg.get("t_end", 0.0)))
                if makespan_est <= 0.0:
                    makespan_est = 1.0
                V = len(vm_timelines)
                T = 200
                ts = np.linspace(0.0, makespan_est, T)
                mem_util = np.zeros((V, T), dtype=float)
                core_util = np.zeros((V, T), dtype=float)
                for v in range(V):
                    cap_m = max(1e-9, float(vm_total_mem[v] if v < len(vm_total_mem) else 1.0))
                    cap_c = max(1.0, float(vm_total_cores[v] if v < len(vm_total_cores) else 1.0))
                    for seg in vm_timelines[v]:
                        t0 = float(seg.get("t_start", 0.0))
                        t1 = float(seg.get("t_end", 0.0))
                        m = float(seg.get("mem", 0.0))
                        c = float(seg.get("cores", 0.0))
                        if t1 <= t0:
                            continue
                        mask = (ts >= t0) & (ts <= t1)
                        mem_util[v, mask] += m / cap_m
                        core_util[v, mask] += c / cap_c
                util = np.clip(np.maximum(mem_util, core_util), 0.0, 1.0)

                fig = plt.figure(figsize=(10, max(3, V * 0.6)))
                sns.heatmap(util, cmap="YlOrRd", cbar=True, vmin=0.0, vmax=1.0,
                            xticklabels=False, yticklabels=[f"VM {i}" for i in range(V)])
                plt.title("First Episode VM Utilization (max of mem/core)")
                plt.xlabel("Time →")
                plt.ylabel("VMs")
                thr = 0.95
                ys, xs = np.where(util >= thr)
                plt.scatter(xs + 0.5, ys + 0.5, s=6, c="cyan", marker="s", alpha=0.6)
                out_png = os.path.splitext(args.out_csv)[0] + "_first_episode_util_heatmap.png"
                plt.savefig(out_png, bbox_inches="tight", dpi=200)
                plt.close(fig)
                print(f"Saved plot: {out_png}")
            except Exception as e:
                print(f"Heatmap plotting skipped: {e}")

        # First-episode assignability gap histogram (if series available)
        if args.save_first_episode_heatmap and first_episode_trace is not None:
            try:
                t = first_episode_trace.get("timeline_t", None)
                r = first_episode_trace.get("timeline_ready", None)
                s = first_episode_trace.get("timeline_schedulable", None)
                if t is not None and r is not None and s is not None and len(t) == len(r) == len(s) and len(t) > 0:
                    r_np = np.array(r, dtype=int)
                    s_np = np.array(s, dtype=int)
                    gap = np.maximum(0, r_np - s_np)

                    # Publication-quality histogram with KDE and annotated stats
                    sns.set_theme(context="paper", style="whitegrid", font_scale=1.2)
                    fig, ax = plt.subplots(figsize=(6.5, 4.0))
                    bins = min(40, max(8, int(gap.max()) + 1))
                    sns.histplot(gap, bins=bins, kde=True, stat="density", color="#4C72B0", edgecolor="white", linewidth=0.8, ax=ax)
                    mean_v = float(np.mean(gap))
                    med_v = float(np.median(gap))
                    p95_v = float(np.percentile(gap, 95))
                    ax.axvline(mean_v, color="#DD8452", linestyle="-", linewidth=1.8, label=f"Mean = {mean_v:.2f}")
                    ax.axvline(med_v, color="#55A868", linestyle="--", linewidth=1.8, label=f"Median = {med_v:.2f}")
                    ax.set_xlabel("Assignability gap (R − S)")
                    ax.set_ylabel("Density")
                    ax.legend(frameon=True)
                    # Annotate N and 95th percentile
                    txt = f"N = {len(gap)}\n95th = {p95_v:.2f}"
                    ax.text(0.98, 0.95, txt, transform=ax.transAxes, ha="right", va="top",
                            fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#BBBBBB", alpha=0.9))
                    sns.despine()
                    out_png = os.path.splitext(args.out_csv)[0] + "_first_episode_assign_gap_hist.png"
                    plt.tight_layout()
                    plt.savefig(out_png, bbox_inches="tight", dpi=300)
                    plt.close(fig)
                    print(f"Saved plot: {out_png}")
            except Exception as e:
                print(f"Assignability plotting skipped: {e}")
    except Exception as e:
        print(f"Plotting skipped ({e}). CSV is available.")


if __name__ == "__main__":
    evaluate(tyro.cli(EvalArgs))
