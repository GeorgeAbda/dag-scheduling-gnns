from dataclasses import dataclass

import tyro
import json

from cogito.dataset_generator.core.gen_dataset import (
    generate_dataset,
    generate_dataset_long_cp_queue_free,
    generate_dataset_wide_queue_free,
)


@dataclass
class DatasetArgs:
    seed: int = 12345
    """random seed"""
    host_count: int = 2
    """number of hosts"""
    vm_count: int = 1
    """number of VMs"""
    max_memory_gb: int = 10
    """maximum amount of RAM for a VM (in GB)"""
    min_cpu_speed: int = 500
    """minimum CPU speed in MIPS"""
    max_cpu_speed: int = 5000
    """maximum CPU speed in MIPS"""
    workflow_count: int = 3
    """number of workflows"""
    dag_method: str = "gnp"
    """DAG generation method (pegasus, gnp)"""
    gnp_min_n: int = 1
    """minimum number of tasks per workflow (for G(n,p) method)"""
    gnp_max_n: int = 10
    """maximum number of tasks per workflow (for G(n,p) method)"""
    task_length_dist: str = "normal"
    """task length distribution (normal, uniform, left_skewed, right_skewed)"""
    min_task_length: int = 500
    """minimum task length"""
    max_task_length: int = 100_000
    """maximum task length"""
    task_arrival: str = "dynamic"
    """task arrival mode (static, dynamic)"""
    arrival_rate: float = 3
    """arrival rate of workflows/second (for dynamic arrival)"""
    style: str = "generic"
    """dataset style: generic | long_cp | wide (queue-free enforced for long_cp/wide)"""
    gnp_p: float | None = None
    """optional fixed p for G(n,p); if set with style=long_cp|wide, it pins the style to this p (p_range=(p,p))."""
    req_divisor: int | None = None
    """Optional: when set, can be used by callers to pass a specific req_divisor into generic generators."""


def main(args: DatasetArgs):
    if args.style == "long_cp":
        print(f"Generating long_cp dataset with seed {args.seed}")
        dataset = generate_dataset_long_cp_queue_free(
            seed=args.seed,
            host_count=args.host_count,
            vm_count=args.vm_count,
            max_memory_gb=args.max_memory_gb,
            min_cpu_speed_mips=args.min_cpu_speed,
            max_cpu_speed_mips=args.max_cpu_speed,
            workflow_count=args.workflow_count,
            gnp_min_n=args.gnp_min_n,
            gnp_max_n=args.gnp_max_n,
            task_length_dist=args.task_length_dist,
            min_task_length=args.min_task_length,
            max_task_length=args.max_task_length,
            task_arrival=args.task_arrival,
            arrival_rate=args.arrival_rate,
            vm_rng_seed=0, 
            req_divisor=args.req_divisor
        )
    elif args.style == "wide":
        print(f"Generating wide dataset with seed {args.seed}")

        dataset = generate_dataset_wide_queue_free(
            seed=args.seed,
            host_count=args.host_count,
            vm_count=args.vm_count,
            max_memory_gb=args.max_memory_gb,
            min_cpu_speed_mips=args.min_cpu_speed,
            max_cpu_speed_mips=args.max_cpu_speed,
            workflow_count=args.workflow_count,
            gnp_min_n=args.gnp_min_n,
            gnp_max_n=args.gnp_max_n,
            task_length_dist=args.task_length_dist,
            min_task_length=args.min_task_length,
            max_task_length=args.max_task_length,
            task_arrival=args.task_arrival,
            arrival_rate=args.arrival_rate,
            vm_rng_seed=0,
            req_divisor=args.req_divisor
        )
    else:

        print(f"Generating default dataset with seed {args.seed}")
        dataset = generate_dataset(
            seed=args.seed,
            host_count=args.host_count,
            vm_count=args.vm_count,
            max_memory_gb=args.max_memory_gb,
            min_cpu_speed_mips=args.min_cpu_speed,
            max_cpu_speed_mips=args.max_cpu_speed,
            workflow_count=args.workflow_count,
            dag_method=args.dag_method,
            gnp_min_n=args.gnp_min_n,
            gnp_max_n=args.gnp_max_n,
            task_length_dist=args.task_length_dist,
            min_task_length=args.min_task_length,
            max_task_length=args.max_task_length,
            task_arrival=args.task_arrival,
            arrival_rate=args.arrival_rate,
            req_divisor=args.req_divisor
        )

    json_data = json.dumps(dataset.to_json())
    print(json_data)


if __name__ == "__main__":
    main(tyro.cli(DatasetArgs))
