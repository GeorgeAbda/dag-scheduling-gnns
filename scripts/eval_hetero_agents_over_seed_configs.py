#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Tuple, Any, Sequence

import os
import sys
import platform
import numpy as np
import torch
import tyro
from tqdm import tqdm
import random

# Ensure project root (one level up from scripts/) is on sys.path so that 'scheduler' is importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cogito.dataset_generator.core.gen_vm as gen_vm
from cogito.dataset_generator.gen_dataset import DatasetArgs
from cogito.dataset_generator.core.gen_vm import generate_hosts, generate_vms, allocate_vms
from cogito.gnn_deeprl_model.ablation_gnn import AblationGinAgent, AblationVariant, _pick_device
from cogito.gnn_deeprl_model.core.env.gym_env import CloudSchedulingGymEnvironment
from cogito.gnn_deeprl_model.agents.gin_agent.wrapper import GinAgentWrapper


def _compute_optimal_req_divisor(dataset_cfg: dict, seed: int) -> int:
    """Compute a conservative req_divisor so that, on the smallest VM, repeating
    the per-task demand (CPU cores and memory) for all tasks in a job does not
    exceed that VM's capacity.

    We approximate the number of tasks per workflow using gnp_max_n from the
    config and use the host/vm generation code so that capacities match the
    scheduler's dataset generator.
    """\

    # Basic dataset parameters
    host_count = int(dataset_cfg.get("host_count", 10))
    vm_count = int(dataset_cfg.get("vm_count", 10))
    max_memory_gb = int(dataset_cfg.get("max_memory_gb", 10))
    min_cpu_speed = int(dataset_cfg.get("min_cpu_speed", 500))
    max_cpu_speed = int(dataset_cfg.get("max_cpu_speed", 5000))

    # Upper bound on tasks per workflow ("job"): gnp_max_n
    n_tasks = int(dataset_cfg.get("gnp_max_n", 40))
    if n_tasks <= 0:
        n_tasks = 1

    # Recreate hosts/VMs using the same logic as the dataset generator so that
    # capacities are realistic for this config. We then allocate VMs to hosts so
    # VM capacities mirror host capacities, matching fixed-dataset behavior.
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

    # Safe per-task demand on the smallest VM: if every task had this demand and
    # all tasks were hypothetically placed on that VM, we would not exceed its
    # capacity.
    max_safe_mem_per_task = max(1024, min_mem // n_tasks)
    max_safe_cores_per_task = max(1, min_cores // n_tasks)

    # Translate the safe per-task demand into a divisor relative to the maximum
    # VM capacity, matching the way generate_dataset sets max_req_*.
    req_div_mem = max(1, max_mem // max_safe_mem_per_task)
    req_div_core = max(1, max_cores // max_safe_cores_per_task)
    
    print(f"req_div_mem={req_div_mem}, req_div_core={req_div_core}, max_mem={max_mem}, max_cores={max_cores}, n_tasks={n_tasks}")
    return int(max(req_div_mem, req_div_core))


def _system_metadata(device: torch.device) -> Dict[str, Any]:
    """Collect basic system/runtime metadata for reproducibility."""
    info: Dict[str, Any] = {}
    try:
        info["hostname"] = getattr(os, "uname", lambda: None)().nodename if hasattr(os, "uname") else None
    except Exception:
        info["hostname"] = None
    try:
        info["platform"] = platform.platform()
    except Exception:
        info["platform"] = None
    try:
        info["python_version"] = sys.version
    except Exception:
        info["python_version"] = None
    try:
        info["torch_version"] = torch.__version__
    except Exception:
        info["torch_version"] = None
    try:
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    except Exception:
        info["cuda_available"] = None
        info["cuda_device_count"] = None
    try:
        info["device_type"] = str(device.type)
    except Exception:
        info["device_type"] = None
    try:
        info["torch_num_threads"] = int(torch.get_num_threads())
    except Exception:
        info["torch_num_threads"] = None
    return info


def _dataset_metadata(dataset_cfg: dict, seeds: Sequence[int], domain: str) -> Dict[str, Any]:
    """Summarize high-level dataset/job configuration for one domain."""
    return {
        "domain": domain,
        "style": str(dataset_cfg.get("style", "unknown")),
        "host_count": int(dataset_cfg.get("host_count", 10)),
        "vm_count": int(dataset_cfg.get("vm_count", 10)),
        "workflow_count": int(dataset_cfg.get("workflow_count", 1)),
        "gnp_min_n": int(dataset_cfg.get("gnp_min_n", 0)),
        "gnp_max_n": int(dataset_cfg.get("gnp_max_n", 0)),
        "task_arrival": str(dataset_cfg.get("task_arrival", "static")),
        "dag_method": str(dataset_cfg.get("dag_method", "gnp")),
        "seeds": [int(s) for s in seeds],
    }


@dataclass
class Args:
    longcp_config: str = "data/rl_configs/train_long_cp_p08_seeds.json"
    wide_config: str = "data/rl_configs/train_wide_p005_seeds.json"

    longcp_ckpt: str = ""
    wide_ckpt: str = ""
    mixed_ckpt: str = ""

    device: str = "cpu"
    dataset_req_divisor: int | None = None

    # Optional: override the default host specs JSON path (data/host_specs.json)
    # used by generate_hosts via HOST_SPECS_PATH. When set, this path is applied
    # before any datasets are constructed.
    host_specs_path: str | None = None

    # Number of evaluation rollouts per seed (job) when estimating performance
    eval_repeats_per_seed: int = 5
    
    # Random seed for evaluation (for reproducibility and regime-specific randomness)
    eval_seed: int = 12345

    out_csv: str = "logs/hetero_eval_over_seeds.csv"

    # Plot decision regions during evaluation (CPU x Length per memory level)
    plot_decision_regions: bool = False
    plot_grid_res: int = 30
    plot_outdir: str = "logs/eval_action_regions"
    plot_n_seeds_per_domain: int = 1


def _dataset_args_from_cfg(dataset_cfg: dict, seed: int, override_req_divisor: int | None = None) -> DatasetArgs:
    # Use override if provided, otherwise compute optimal
    if override_req_divisor is not None:
        req_div = override_req_divisor
    else:
        req_div = _compute_optimal_req_divisor(dataset_cfg, seed)

    return DatasetArgs(
        seed=int(seed),
        host_count=int(dataset_cfg.get("host_count", 10)),
        vm_count=int(dataset_cfg.get("vm_count", 10)),
        max_memory_gb=int(dataset_cfg.get("max_memory_gb", 10)),
        min_cpu_speed=int(dataset_cfg.get("min_cpu_speed", 500)),
        max_cpu_speed=int(dataset_cfg.get("max_cpu_speed", 5000)),
        workflow_count=int(dataset_cfg.get("workflow_count", 1)),
        dag_method=str(dataset_cfg.get("dag_method", "gnp")),
        gnp_min_n=int(dataset_cfg.get("gnp_min_n", 10)),
        gnp_max_n=int(dataset_cfg.get("gnp_max_n", 40)),
        task_length_dist=str(dataset_cfg.get("task_length_dist", "normal")),
        min_task_length=int(dataset_cfg.get("min_task_length", 500)),
        max_task_length=int(dataset_cfg.get("max_task_length", 100_000)),
        task_arrival=str(dataset_cfg.get("task_arrival", "static")),
        arrival_rate=float(dataset_cfg.get("arrival_rate", 3.0)),
        style=str(dataset_cfg.get("style", "generic")),
        gnp_p=dataset_cfg.get("gnp_p", None),
        req_divisor=int(req_div),
    )


def _build_fixed_dataset(dataset_cfg: dict, seed: int, override_req_divisor: int | None = None) -> Any:
    """Build a deterministic Dataset object for a given config+seed using the same code path as the env.
    We call CloudSchedulingGymEnvironment.gen_dataset so all style/req_divisor/gnp_p logic is identical.
    """
    ds_args = _dataset_args_from_cfg(dataset_cfg, seed, override_req_divisor)
    return CloudSchedulingGymEnvironment.gen_dataset(seed, ds_args)


def _print_dataset_summary(ds: Any, domain: str, seed: int) -> None:
    try:
        vms = getattr(ds, "vms", [])
        ws = getattr(ds, "workflows", [])
        speeds = sorted({int(getattr(v, "cpu_speed_mips", 0)) for v in vms})
        cores = sorted({int(getattr(v, "cpu_cores", 0)) for v in vms})
        mems = sorted({int(getattr(v, "memory_mb", 0)) for v in vms})
        print(f"[eval][{domain}][seed={seed}] VMs: count={len(vms)} cpu_speed_mips_unique={speeds} cores_unique={cores} mem_mb_unique={mems}")

        all_tasks = []
        edges_total = 0
        for wf in ws or []:
            ts = getattr(wf, "tasks", [])
            all_tasks.extend(ts)
            for t in ts:
                edges_total += len(getattr(t, "child_ids", []) or [])
        if all_tasks:
            mem_arr = np.array([int(getattr(t, "req_memory_mb", 0)) for t in all_tasks], dtype=np.int64)
            core_arr = np.array([int(getattr(t, "req_cpu_cores", 0)) for t in all_tasks], dtype=np.int64)
            print(
                f"[eval][{domain}][seed={seed}] Jobs: workflows={len(ws)} tasks_total={len(all_tasks)} edges_total={int(edges_total)} "
                f"req_mem_mb[min,mean,max]=[{mem_arr.min()},{float(mem_arr.mean()):.1f},{mem_arr.max()}] "
                f"req_cpu_cores[min,mean,max]=[{core_arr.min()},{float(core_arr.mean()):.1f},{core_arr.max()}]"
            )
        else:
            print(f"[eval][{domain}][seed={seed}] Jobs: workflows=0 tasks_total=0 edges_total=0")
    except Exception as e:
        try:
            print(f"[eval][{domain}][seed={seed}] Dataset summary unavailable: {e}")
        except Exception:
            pass


def _load_hetero_agent(ckpt: Path, device: torch.device) -> AblationGinAgent:
    """Load hetero agent with auto-detected architecture from checkpoint."""
    var = AblationVariant(name="hetero", graph_type="hetero", use_task_dependencies=True)
    state = torch.load(str(ckpt), map_location=device)
    
    # Auto-detect hidden_dim from first encoder layer shape
    # actor.network.task_encoder.0.weight has shape [hidden_dim, input_features]
    hidden_dim = 64  # default
    embedding_dim = 32  # default
    
    if "actor.network.task_encoder.0.weight" in state:
        hidden_dim = state["actor.network.task_encoder.0.weight"].shape[0]
    
    # Auto-detect embedding_dim from final encoder layer output
    # actor.network.task_encoder.6.weight has shape [embedding_dim, hidden_dim]
    if "actor.network.task_encoder.6.weight" in state:
        embedding_dim = state["actor.network.task_encoder.6.weight"].shape[0]
    
    print(f"[load_ckpt] {ckpt.name}: detected hidden_dim={hidden_dim}, embedding_dim={embedding_dim}")
    
    agent = AblationGinAgent(device, var, hidden_dim=hidden_dim, embedding_dim=embedding_dim)
    agent.load_state_dict(state, strict=False)
    agent.eval()
    return agent


def _eval_agent_on_seeds(
    agent: AblationGinAgent,
    agent_train_domain: str,
    eval_domain: str,
    seeds: Sequence[int],
    dataset_cfg: dict,
    device: torch.device,
    fixed_datasets: Dict[int, Any] | None = None,
    override_req_divisor: int | None = None,
    repeats_per_seed: int = 1,
) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    rows: List[Dict[str, float]] = []

    total_makespan = 0.0
    total_energy = 0.0
    total_energy_active = 0.0
    total_energy_idle = 0.0
    total_entropy = 0.0

    for s in tqdm(seeds, desc=f"{agent_train_domain}->{eval_domain}"):
        s_int = int(s)

        # Per-seed accumulators (to average over multiple rollouts for this job)
        seed_mk = 0.0
        seed_en_total = 0.0
        seed_en_active = 0.0
        seed_en_idle = 0.0
        seed_entropy = 0.0

        n_rep = max(1, int(repeats_per_seed))

        for r in range(n_rep):
            if fixed_datasets is not None and s_int in fixed_datasets:
                base_env = CloudSchedulingGymEnvironment(
                    dataset=fixed_datasets[s_int],
                    collect_timelines=False,
                    compute_metrics=True,
                    profile=False,
                    fixed_env_seed=True,
                )
            else:
                ds_args = _dataset_args_from_cfg(dataset_cfg, seed=s_int, override_req_divisor=override_req_divisor)
                base_env = CloudSchedulingGymEnvironment(
                    dataset_args=ds_args,
                    collect_timelines=False,
                    compute_metrics=True,
                    profile=False,
                    fixed_env_seed=True,
                )

            env = GinAgentWrapper(base_env)

            # Use the numeric seed for a fixed job/environment across repeats;
            # repeated evaluations differ only due to the stochastic policy.
            env_seed = s_int
            obs_np, _ = env.reset(seed=env_seed)
            done = False
            final_info: dict | None = None

            # Per-episode entropy accumulator (mean entropy over timesteps)
            ep_entropy_sum = 0.0
            ep_steps = 0

            while not done:
                obs_t = torch.from_numpy(np.asarray(obs_np, dtype=np.float32)).unsqueeze(0).to(agent.device)
                with torch.no_grad():
                    # Stochastic evaluation: sample actions from the policy distribution
                    action, _, entropy, _ = agent.get_action_and_value(obs_t)
                # entropy is a tensor of shape [1]; take the scalar value
                try:
                    ep_entropy_sum += float(entropy.mean().item())
                except Exception:
                    pass
                ep_steps += 1

                obs_np, _, terminated, truncated, info = env.step(int(action.item()))
                if terminated or truncated:
                    final_info = info
                    done = True

            assert env.prev_obs is not None
            mk = float(env.prev_obs.makespan())
            # Use the observation-based energy estimate as the primary metric,
            # matching the agent's training objective (same as in
            # eval_heuristics_on_seeds.py).
            en_obs = float(env.prev_obs.energy_consumption())

            # For backwards compatibility, we keep both energy_total and
            # energy_active columns, but set them both to obs.energy_consumption().
            # energy_idle is kept at 0.0.
            en_total = en_obs
            en_active = en_obs
            en_idle = 0.0

            seed_mk += mk
            seed_en_total += en_total
            seed_en_active += en_active
            seed_en_idle += en_idle

            # Mean entropy for this episode
            if ep_steps > 0:
                seed_entropy += ep_entropy_sum / float(ep_steps)

            env.close()

        # Average over repeats for this seed
        mk_mean = seed_mk / float(n_rep)
        en_total_mean = seed_en_total / float(n_rep)
        en_active_mean = seed_en_active / float(n_rep)
        en_idle_mean = seed_en_idle / float(n_rep)
        entropy_mean = seed_entropy / float(n_rep) if n_rep > 0 else 0.0

        total_makespan += mk_mean
        total_energy += en_total_mean
        total_energy_active += en_active_mean
        total_energy_idle += en_idle_mean
        total_entropy += entropy_mean

        rows.append(
            {
                "agent_train_domain": agent_train_domain,
                "eval_domain": eval_domain,
                "seed": int(s),
                "makespan": mk_mean,
                "energy_total": en_total_mean,
                "energy_active": en_active_mean,
                "energy_idle": en_idle_mean,
                "entropy": entropy_mean,
            }
        )

    n = max(1, len(seeds))
    summary = {
        "agent_train_domain": agent_train_domain,
        "eval_domain": eval_domain,
        "seeds": float(len(seeds)),
        "mean_makespan": total_makespan / n,
        "mean_energy_total": total_energy / n,
        "mean_energy_active": (total_energy_active / n) if total_energy_active > 0.0 else 0.0,
        "mean_energy_idle": (total_energy_idle / n) if total_energy_idle > 0.0 else 0.0,
        "mean_entropy": (total_entropy / n) if total_entropy > 0.0 else 0.0,
    }
    return rows, summary


def _plot_action_regions_single_task(
    agent: AblationGinAgent,
    dataset: Any,
    device: torch.device,
    out_dir: Path,
    agent_label: str,
    domain_label: str,
    seed: int,
    grid_res: int = 30,
) -> None:
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        base_env = CloudSchedulingGymEnvironment(
            dataset=dataset,
            collect_timelines=False,
            compute_metrics=False,
            profile=False,
            fixed_env_seed=True,
        )
        env = GinAgentWrapper(base_env)
        obs, _ = env.reset(seed=int(seed))

        num_tasks = int(obs[0])
        num_vms = int(obs[1])
        header = 4
        t_sched_start = header
        t_ready_start = t_sched_start + num_tasks
        t_length_start = t_ready_start + num_tasks
        t_completion_start = t_length_start + num_tasks
        t_mem_start = t_completion_start + num_tasks
        t_cpu_start = t_mem_start + num_tasks

        t_sched = obs[t_sched_start:t_sched_start + num_tasks]
        t_ready = obs[t_ready_start:t_ready_start + num_tasks]
        t_len = obs[t_length_start:t_length_start + num_tasks]
        t_mem = obs[t_mem_start:t_mem_start + num_tasks]
        t_cpu = obs[t_cpu_start:t_cpu_start + num_tasks]

        target_idx = None
        for i in range(num_tasks):
            if int(t_ready[i]) == 1 and int(t_sched[i]) == 0:
                target_idx = i
                break
        if target_idx is None:
            target_idx = 0

        cpu_min = 0.1
        cpu_max = float(max(1.0, float(np.max(t_cpu)) if t_cpu.size > 0 else 1.0))
        length_min = float(max(100.0, float(np.min(t_len)) * 0.5 if t_len.size > 0 else 500.0))
        length_max = float(min(100000.0, float(np.max(t_len)) * 1.5 if t_len.size > 0 else 10000.0))

        xs = np.linspace(cpu_min, cpu_max, int(grid_res))
        ys = np.linspace(length_min, length_max, int(grid_res))
        XX, YY = np.meshgrid(xs, ys)

        grid = np.stack([XX.ravel(), YY.ravel()], axis=1)
        labels = np.empty(grid.shape[0], dtype=int)

        batch = 256
        for i in range(0, grid.shape[0], batch):
            pts = grid[i:i + batch]
            states = np.tile(obs, (pts.shape[0], 1))
            states[:, t_cpu_start + target_idx] = pts[:, 0]
            states[:, t_length_start + target_idx] = pts[:, 1]
            st = torch.from_numpy(states.astype(np.float32)).to(device)
            with torch.no_grad():
                # Use argmax for deterministic action selection
                acts, logits, _, _ = agent.get_action_and_value(st)
                acts = logits.argmax(dim=-1)
            vms = (acts.cpu().numpy().astype(int) % num_vms).astype(int)
            labels[i:i + batch] = vms

        Z = labels.reshape(int(grid_res), int(grid_res))

        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        import matplotlib.cm as cm

        K = num_vms
        cmap = cm.get_cmap('tab10', K) if K <= 10 else (cm.get_cmap('tab20', K) if K <= 20 else cm.get_cmap('hsv', K))
        colors = [cmap(i) for i in range(K)]
        cmap_discrete = ListedColormap(colors)

        plt.figure(figsize=(7, 5))
        plt.contourf(XX, YY, Z, levels=np.arange(K + 1) - 0.5, cmap=cmap_discrete, alpha=0.85)
        cbar = plt.colorbar(ticks=np.arange(K))
        cbar.ax.set_yticklabels([f"VM {i}" for i in range(K)])
        cbar.set_label('VM Choice')
        plt.xlabel('Task CPU (cores)')
        plt.ylabel('Task Length (MI)')
        plt.title(f"{agent_label} on {domain_label} | seed={int(seed)} | task={int(target_idx)}")
        out_png = out_dir / f"{agent_label}_{domain_label}_seed{int(seed)}_cpu_len.png"
        plt.tight_layout()
        plt.savefig(str(out_png), dpi=200)
        plt.close()

        counts = np.bincount(labels, minlength=num_vms)
        shares = counts / float(labels.size)
        area_csv = out_dir / "area_share.csv"
        import csv
        exists = area_csv.exists()
        with area_csv.open('a', newline='') as f:
            w = csv.writer(f)
            if not exists:
                w.writerow(['agent', 'domain', 'seed', 'vm', 'area_share'])
            for vm_id in range(num_vms):
                w.writerow([agent_label, domain_label, int(seed), int(vm_id), float(shares[vm_id])])

        env.close()
    except Exception as e:
        try:
            print(f"[eval][plot] failed for {agent_label} {domain_label} seed={seed}: {e}")
        except Exception:
            pass


def main(a: Args) -> None:
    # If a custom host specs path is provided, override the path used by the
    # dataset generator before any datasets are built.
    try:
        if getattr(a, "host_specs_path", None):
            # Set BOTH the module variable AND the environment variable
            # generate_hosts() reads from os.environ, not the module variable
            host_path = str(Path(str(a.host_specs_path)).resolve())
            gen_vm.HOST_SPECS_PATH = Path(host_path)
            os.environ["HOST_SPECS_PATH"] = host_path
            print(f"[eval] Overriding HOST_SPECS_PATH to: {host_path}")
    except Exception as e:
        print(f"[eval] Warning: failed to apply host_specs_path override: {e}")

    # Resolve device (cpu/mps/cuda/auto) using the same helper as ablation_gnn.
    device = _pick_device(a.device)

    # Global determinism: seed Python, NumPy, and PyTorch RNGs.
    # We use the eval_seed from args so different regimes can have different random states.
    eval_seed = a.eval_seed
    try:
        random.seed(eval_seed)
    except Exception:
        pass
    try:
        np.random.seed(eval_seed)
    except Exception:
        pass
    try:
        torch.manual_seed(eval_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(eval_seed)
        # Prefer deterministic algorithms when available
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    except Exception:
        pass

    cfg_long = json.loads(Path(a.longcp_config).read_text())
    cfg_wide = json.loads(Path(a.wide_config).read_text())

    # Support both old format (train.seeds) and new format (all_seeds)
    if "train" in cfg_long:
        tr_long = cfg_long.get("train", {})
        seeds_long: List[int] = [int(s) for s in tr_long.get("seeds", [])]
        ds_long = dict(tr_long.get("dataset", {}))
    else:
        tr_long = {}
        seeds_long: List[int] = [int(s) for s in cfg_long.get("all_seeds", [])]
        ds_long = dict(cfg_long.get("dataset", {}))

    if "train" in cfg_wide:
        tr_wide = cfg_wide.get("train", {})
        seeds_wide: List[int] = [int(s) for s in tr_wide.get("seeds", [])]
        ds_wide = dict(tr_wide.get("dataset", {}))
    else:
        tr_wide = {}
        seeds_wide: List[int] = [int(s) for s in cfg_wide.get("all_seeds", [])]
        ds_wide = dict(cfg_wide.get("dataset", {}))

    long_ckpt = Path(a.longcp_ckpt)
    wide_ckpt = Path(a.wide_ckpt)
    mixed_ckpt = Path(a.mixed_ckpt) if a.mixed_ckpt else None

    if not long_ckpt.exists():
        raise SystemExit(f"Long_cp checkpoint not found: {long_ckpt}")
    if not wide_ckpt.exists():
        raise SystemExit(f"Wide checkpoint not found: {wide_ckpt}")
    if mixed_ckpt is not None and (not mixed_ckpt.exists()):
        raise SystemExit(f"Mixed checkpoint not found: {mixed_ckpt}")

    agent_long = _load_hetero_agent(long_ckpt, device)
    agent_wide = _load_hetero_agent(wide_ckpt, device)
    agent_mixed: AblationGinAgent | None = None
    if mixed_ckpt is not None:
        agent_mixed = _load_hetero_agent(mixed_ckpt, device)

    # Pre-generate fixed datasets per domain and seed so both agents see identical jobs
    fixed_long: Dict[int, Any] = {int(s): _build_fixed_dataset(ds_long, int(s), a.dataset_req_divisor) for s in seeds_long}
    fixed_wide: Dict[int, Any] = {int(s): _build_fixed_dataset(ds_wide, int(s), a.dataset_req_divisor) for s in seeds_wide}

    # Print summary once at the start for a representative seed per domain (first seed)
    if seeds_long:
        s0 = int(seeds_long[0])
        if s0 in fixed_long:
            _print_dataset_summary(fixed_long[s0], "long_cp", s0)
    if seeds_wide:
        s0 = int(seeds_wide[0])
        if s0 in fixed_wide:
            _print_dataset_summary(fixed_wide[s0], "wide", s0)

    if a.plot_decision_regions:
        out_root = Path(a.plot_outdir)
        try:
            sel_long = [int(s) for s in seeds_long[: max(0, int(a.plot_n_seeds_per_domain))]]
            sel_wide = [int(s) for s in seeds_wide[: max(0, int(a.plot_n_seeds_per_domain))]]
        except Exception:
            sel_long, sel_wide = [], []
        for s in sel_long:
            if s in fixed_long:
                _plot_action_regions_single_task(agent_long, fixed_long[s], device, out_root / "long_cp", "long_cp", "long_cp", s, grid_res=int(a.plot_grid_res))
                _plot_action_regions_single_task(agent_wide, fixed_long[s], device, out_root / "wide_on_longcp", "wide", "long_cp", s, grid_res=int(a.plot_grid_res))
        for s in sel_wide:
            if s in fixed_wide:
                _plot_action_regions_single_task(agent_wide, fixed_wide[s], device, out_root / "wide", "wide", "wide", s, grid_res=int(a.plot_grid_res))
                _plot_action_regions_single_task(agent_long, fixed_wide[s], device, out_root / "longcp_on_wide", "long_cp", "wide", s, grid_res=int(a.plot_grid_res))

    all_rows: List[Dict[str, float]] = []
    summaries: List[Dict[str, float]] = []

    # long_cp agent on long_cp and wide seeds
    rows, summ = _eval_agent_on_seeds(
        agent_long,
        "long_cp",
        "long_cp",
        seeds_long,
        ds_long,
        device,
        fixed_datasets=fixed_long,
        override_req_divisor=a.dataset_req_divisor,
        repeats_per_seed=a.eval_repeats_per_seed,
    )
    all_rows.extend(rows)
    summaries.append(summ)

    rows, summ = _eval_agent_on_seeds(
        agent_long,
        "long_cp",
        "wide",
        seeds_wide,
        ds_wide,
        device,
        fixed_datasets=fixed_wide,
        override_req_divisor=a.dataset_req_divisor,
        repeats_per_seed=a.eval_repeats_per_seed,
    )
    all_rows.extend(rows)
    summaries.append(summ)

    # wide agent on long_cp and wide seeds
    rows, summ = _eval_agent_on_seeds(
        agent_wide,
        "wide",
        "long_cp",
        seeds_long,
        ds_long,
        device,
        fixed_datasets=fixed_long,
        override_req_divisor=a.dataset_req_divisor,
        repeats_per_seed=a.eval_repeats_per_seed,
    )
    all_rows.extend(rows)
    summaries.append(summ)

    rows, summ = _eval_agent_on_seeds(
        agent_wide,
        "wide",
        "wide",
        seeds_wide,
        ds_wide,
        device,
        fixed_datasets=fixed_wide,
        override_req_divisor=a.dataset_req_divisor,
        repeats_per_seed=a.eval_repeats_per_seed,
    )
    all_rows.extend(rows)
    summaries.append(summ)

    # Optional: mixed agent (trained on mixed domain) on long_cp and wide seeds
    if agent_mixed is not None:
        rows, summ = _eval_agent_on_seeds(
            agent_mixed,
            "mixed",
            "long_cp",
            seeds_long,
            ds_long,
            device,
            fixed_datasets=fixed_long,
            override_req_divisor=a.dataset_req_divisor,
            repeats_per_seed=a.eval_repeats_per_seed,
        )
        all_rows.extend(rows)
        summaries.append(summ)

        rows, summ = _eval_agent_on_seeds(
            agent_mixed,
            "mixed",
            "wide",
            seeds_wide,
            ds_wide,
            device,
            fixed_datasets=fixed_wide,
            override_req_divisor=a.dataset_req_divisor,
            repeats_per_seed=a.eval_repeats_per_seed,
        )
        all_rows.extend(rows)
        summaries.append(summ)

    out_path = Path(a.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Write per-seed rows
    import csv as _csv

    fieldnames = [
        "agent_train_domain",
        "eval_domain",
        "seed",
        "makespan",
        "energy_total",
        "energy_active",
        "energy_idle",
        "entropy",
    ]

    with out_path.open("w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    # Also write a companion summary CSV
    summary_path = out_path.with_suffix(".summary.csv")
    summary_fields = [
        "agent_train_domain",
        "eval_domain",
        "seeds",
        "mean_makespan",
        "mean_energy_total",
        "mean_energy_active",
        "mean_energy_idle",
        "mean_entropy",
    ]
    with summary_path.open("w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        for s in summaries:
            w.writerow(s)

    # Metadata JSON with configs, checkpoints, seeds, and system/runtime info
    meta: Dict[str, Any] = {
        "run_type": "hetero_eval_over_seed_configs",
        "args": asdict(a),
        "system": _system_metadata(device),
        "configs": {
            "longcp_config_path": str(a.longcp_config),
            "wide_config_path": str(a.wide_config),
            "longcp_train": tr_long,
            "wide_train": tr_wide,
        },
        "datasets": {
            "long_cp": _dataset_metadata(ds_long, seeds_long, "long_cp"),
            "wide": _dataset_metadata(ds_wide, seeds_wide, "wide"),
        },
        "checkpoints": {
            "longcp_ckpt": str(long_ckpt),
            "wide_ckpt": str(wide_ckpt),
            "mixed_ckpt": str(mixed_ckpt) if mixed_ckpt is not None else None,
        },
        "outputs": {
            "per_seed_csv": str(out_path),
            "summary_csv": str(summary_path),
        },
    }
    try:
        meta_path = out_path.with_suffix(".meta.json")
        with meta_path.open("w") as f:
            json.dump(meta, f, indent=2)
        print(f"Wrote metadata to: {meta_path}")
    except Exception as e:
        print(f"Failed to write metadata JSON: {e}")

    print(f"Wrote per-seed metrics to: {out_path}")
    print(f"Wrote summary metrics to: {summary_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
