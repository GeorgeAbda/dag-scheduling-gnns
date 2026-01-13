import time
import argparse
from typing import List

import numpy as np

from cogito.gnn_deeprl_model.core.env.gym_env import CloudSchedulingGymEnvironment
from cogito.gnn_deeprl_model.core.env.action import EnvAction
from cogito.dataset_generator.gen_dataset import DatasetArgs


def pick_ready_task(env: CloudSchedulingGymEnvironment) -> int | None:
    # Prefer using cached masks if available
    if getattr(env, "_is_ready_mask", None) is not None and getattr(env, "_is_assigned_mask", None) is not None:
        mask = env._is_ready_mask & (~env._is_assigned_mask)
        if mask.size > 0:
            mask[0] = False
            mask[len(mask) - 1] = False
            ready_list = np.nonzero(mask)[0].tolist()
            return ready_list[0] if ready_list else None
    # Fallback
    ready: List[int] = []
    for tid, ts in enumerate(env.state.task_states):
        if tid == 0 or tid == len(env.state.task_states) - 1:
            continue
        if ts.is_ready and ts.assigned_vm_id is None:
            ready.append(tid)
    return ready[0] if ready else None


def pick_compatible_vm(env: CloudSchedulingGymEnvironment, t_id: int) -> int:
    # Use cached compat matrix if available
    if getattr(env, "_compat_bool", None) is not None:
        idxs = np.nonzero(env._compat_bool[t_id])[0].tolist()
        return idxs[0]
    # Fallback to compatibilities list
    for (tt, vv) in env.state.static_state.compatibilities:
        if tt == t_id:
            return vv
    return 0


def run_episode(env: CloudSchedulingGymEnvironment, seed: int = 123):
    env.reset_timers()
    obs, _ = env.reset(seed=seed)
    steps = 0
    while True:
        t_id = pick_ready_task(env)
        if t_id is None:
            # No ready tasks left; should only happen near episode end
            # Force end if dummy end is ready
            if env.state.task_states[-1].assigned_vm_id is not None:
                break
            # Otherwise, pick any unscheduled task compatible with any VM (fallback)
            for tid, ts in enumerate(env.state.task_states):
                if ts.assigned_vm_id is None and tid not in (0, len(env.state.task_states) - 1):
                    t_id = tid
                    break
            if t_id is None:
                break
        vm_id = pick_compatible_vm(env, t_id)
        obs, reward, terminated, truncated, info = env.step(EnvAction(task_id=t_id, vm_id=vm_id))
        steps += 1
        if terminated or truncated:
            break
    timers = env.get_timers()
    return timers, steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--start-seed", type=int, default=123)
    parser.add_argument("--feasibility-mode", type=str, default="optimized", choices=["optimized", "legacy"])
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    # Small dataset for quick profiling
    ds_args = DatasetArgs(
        host_count=3,
        vm_count=6,
        workflow_count=3,
        gnp_min_n=10,
        gnp_max_n=10,
        max_memory_gb=8,
        min_cpu_speed=500,
        max_cpu_speed=3000,
        min_task_length=500,
        max_task_length=5000,
        task_arrival="static",
        dag_method="gnp",
    )

    env = CloudSchedulingGymEnvironment(
        dataset_args=ds_args,
        collect_timelines=False,
        compute_metrics=True,
        profile=True,
        feasibility_mode=args.feasibility_mode,
    )

    episodes = args.episodes
    start_seed = args.start_seed
    t_wall_start = time.perf_counter()
    agg = {"ready": 0.0, "feasible": 0.0, "end_energy": 0.0, "end_cp": 0.0}
    total_steps = 0
    for i in range(episodes):
        timers, steps = run_episode(env, seed=start_seed + i)
        if not args.quiet:
            print(f"Episode {i+1} timers: {timers}, steps: {steps}")
        for k in agg:
            agg[k] += float(timers.get(k, 0.0))
        total_steps += steps
    wall = time.perf_counter() - t_wall_start
    print("Totals:", agg)
    print(f"Total steps: {total_steps}")
    sps = (total_steps / wall) if wall > 0 else 0.0
    print(f"Wall-clock seconds: {wall:.6f} (mode={args.feasibility_mode}, episodes={episodes}), steps/sec: {sps:.2f}")


if __name__ == "__main__":
    main()
