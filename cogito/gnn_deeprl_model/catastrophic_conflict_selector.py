"""
Catastrophic Conflict Selector: Find MDP2 that creates maximum gradient conflict
to prove that strong conflict prevents learning.

This goes beyond the basic adversarial selector by:
1. Searching for cos(g1, g2) < -0.8 (catastrophic conflict)
2. Verifying conflict persists across multiple policy checkpoints
3. Testing if conflict increases with training (divergence indicator)
"""
import numpy as np
import torch
from typing import Tuple
from dataclasses import dataclass

from copy import deepcopy
from cogito.dataset_generator.gen_dataset import DatasetArgs
from cogito.gnn_deeprl_model.adversarial_mdp_selector import (
    AdversarialMDPConfig,
    compute_actor_gradient,
    generate_diverse_candidate,
)
from cogito.gnn_deeprl_model.core.env.gym_env import CloudSchedulingGymEnvironment
from cogito.gnn_deeprl_model.agents.gin_agent.wrapper import GinAgentWrapper


def generate_bottleneck_candidate(
    base_dataset_args: DatasetArgs,
    candidate_idx: int,
    config: 'CatastrophicConflictConfig',
    rng: np.random.RandomState,
) -> Tuple[DatasetArgs, int, float]:
    """
    Generate a candidate MDP with bottleneck (non-queue-free regime).
    
    Bottleneck is created by:
    1. Keeping VM/Host ratio fixed at 1.0
    2. Varying resource pressure (tasks demand more CPU/memory)
    3. Varying DAG structure (style, edge prob, task count)
    
    Returns:
        (dataset_args, seed, resource_pressure)
    """
    seed = 200000 + candidate_idx
    
    # Copy base args
    args = deepcopy(base_dataset_args)
    
    # Vary DAG structure (like adversarial selector)
    styles = ["long_cp", "wide", "generic"]
    style = rng.choice(styles)
    args.style = style
    
    # Vary edge probability
    gnp_p = rng.uniform(0.05, 0.95)
    args.gnp_p = gnp_p
    
    # Vary task count
    min_n = rng.randint(6, 20)
    max_n = min_n + rng.randint(2, 8)
    args.gnp_min_n = min_n
    args.gnp_max_n = max_n
    
    # Create bottleneck by varying resource demand (not VM/Host ratio)
    if config.include_bottlenecks:
        # Keep VM/Host ratio fixed at 1.0 (same as base)
        base_hosts = getattr(base_dataset_args, 'host_count', 10)
        base_vms = getattr(base_dataset_args, 'vm_count', 10)
        args.host_count = base_hosts
        args.vm_count = base_vms  # Keep 1:1 ratio
        
        # Vary resource pressure (task demands relative to VM capacity)
        # Higher pressure = tasks demand more CPU/memory = bottleneck through contention
        resource_pressure = rng.uniform(*config.resource_pressure_range)
        
        # Increase task resource requirements (CPU time)
        if hasattr(args, 'min_task_length') and hasattr(args, 'max_task_length'):
            base_min = getattr(base_dataset_args, 'min_task_length', 500)
            base_max = getattr(base_dataset_args, 'max_task_length', 100000)
            args.min_task_length = int(base_min * resource_pressure)
            args.max_task_length = int(base_max * resource_pressure)
        
        # Increase memory requirements
        if hasattr(args, 'max_memory_gb'):
            base_mem = getattr(base_dataset_args, 'max_memory_gb', 128)
            args.max_memory_gb = int(base_mem * resource_pressure)
        
        # Remove req_divisor for high resource pressure (bottleneck through demand)
        # req_divisor scales down task requirements to ensure queue-free
        # When resource_pressure > 1.0, tasks demand more than VM capacity → bottleneck
        if resource_pressure > 1.0:
            # High demand regime: remove divisor to create real contention
            if hasattr(args, 'req_divisor'):
                delattr(args, 'req_divisor')
        else:
            # Low demand regime: keep divisor to maintain queue-free property
            if hasattr(base_dataset_args, 'req_divisor'):
                args.req_divisor = base_dataset_args.req_divisor
    else:
        resource_pressure = 1.0  # Default if bottlenecks disabled
    
    return args, seed, resource_pressure


@dataclass
class CatastrophicConflictConfig:
    """Configuration for catastrophic conflict search."""
    num_candidates: int = 1000  # Large search space
    rollout_steps: int = 256  # Longer rollouts for stable gradients
    conflict_threshold: float = -0.8  # Minimum conflict (more negative = stronger)
    verify_persistence: bool = True  # Verify conflict persists after training
    verification_steps: int = 1000  # Steps to train before re-checking conflict
    
    # Bottleneck/queue regime parameters
    include_bottlenecks: bool = True  # Include non-queue-free regimes
    vm_host_ratio_range: tuple = (1.0, 1.0)  # VM/Host ratio (fixed at 1.0)
    resource_pressure_range: tuple = (0.3, 2.0)  # Resource demand multiplier (>1.0 = bottleneck)


def find_catastrophic_conflict_mdp(
    agent,
    mdp1_dataset_args: DatasetArgs,
    mdp1_seed: int,
    device: torch.device,
    alpha_makespan: float = 0.5,
    alpha_energy: float = 0.5,
    config: CatastrophicConflictConfig = None,
) -> Tuple[DatasetArgs, int, float, dict]:
    """
    Find MDP2 that creates catastrophic gradient conflict with MDP1.
    
    Returns:
        (mdp2_args, mdp2_seed, conflict_score, diagnostics)
    """
    if config is None:
        config = CatastrophicConflictConfig()
    
    # Use adversarial config for diverse candidate generation
    adv_config = AdversarialMDPConfig(
        num_candidates=config.num_candidates,
        rollout_steps=config.rollout_steps,
        seed_pool_start=200000,
        conflict_metric="cosine",
        vary_structure=True,
        gnp_p_range=(0.05, 0.95),  # Wider range
        task_count_range=(6, 20),  # Wider range
        style_pool=["long_cp", "wide", "generic"],
    )
    
    # Compute reference gradient from MDP1
    env1 = CloudSchedulingGymEnvironment(
        dataset_args=mdp1_dataset_args,
        collect_timelines=False,
        compute_metrics=False,
    )
    env1 = GinAgentWrapper(env1)
    env1.reset(seed=mdp1_seed)
    
    grad_mdp1 = compute_actor_gradient(
        agent, env1, config.rollout_steps, device, alpha_makespan, alpha_energy
    )
    env1.close()
    
    norm_mdp1 = grad_mdp1.norm().item()
    if norm_mdp1 < 1e-9:
        print("[CatastrophicConflict] Warning: MDP1 gradient is near-zero")
        return mdp1_dataset_args, mdp1_seed, 0.0, {}
    
    # Search for catastrophic conflict
    best_args = mdp1_dataset_args
    best_seed = adv_config.seed_pool_start
    best_conflict = float("inf")
    
    catastrophic_candidates = []  # Store all candidates below threshold
    
    rng = np.random.RandomState(42)
    
    print(f"[CatastrophicConflict] Searching {config.num_candidates} candidates")
    print(f"  Target: cos < {config.conflict_threshold} (catastrophic conflict)")
    print(f"  Bottleneck regimes: {'ENABLED' if config.include_bottlenecks else 'DISABLED'}")
    if config.include_bottlenecks:
        print(f"    VM/Host ratio: FIXED at 1.0")
        print(f"    Resource pressure: {config.resource_pressure_range[0]:.1f}x - {config.resource_pressure_range[1]:.1f}x (>1.0 = bottleneck)")
    print(f"  MDP1: style={mdp1_dataset_args.style}, p={getattr(mdp1_dataset_args, 'gnp_p', 'N/A')}, "
          f"n={mdp1_dataset_args.gnp_min_n}, seed={mdp1_seed}")
    print()
    
    for cand_idx in range(config.num_candidates):
        if cand_idx % 100 == 0 and cand_idx > 0:
            print(f"  Progress: {cand_idx}/{config.num_candidates} candidates searched, "
                  f"best conflict: {best_conflict:.4f}")
        
        # Generate diverse candidate (with or without bottlenecks)
        if config.include_bottlenecks:
            mdp2_dataset_args, cand_seed, resource_pressure = generate_bottleneck_candidate(
                mdp1_dataset_args, cand_idx, config, rng
            )
        else:
            mdp2_dataset_args, cand_seed = generate_diverse_candidate(
                mdp1_dataset_args, cand_idx, adv_config, rng
            )
            resource_pressure = 1.0  # Default
        
        env2 = CloudSchedulingGymEnvironment(
            dataset_args=mdp2_dataset_args,
            collect_timelines=False,
            compute_metrics=False,
        )
        env2 = GinAgentWrapper(env2)
        env2.reset(seed=cand_seed)
        
        try:
            grad_mdp2 = compute_actor_gradient(
                agent, env2, config.rollout_steps, device, alpha_makespan, alpha_energy
            )
            env2.close()
            
            norm_mdp2 = grad_mdp2.norm().item()
            if norm_mdp2 < 1e-9:
                continue
            
            # Compute conflict
            cosine_sim = torch.dot(grad_mdp1, grad_mdp2).item() / (norm_mdp1 * norm_mdp2 + 1e-9)
            
            # Store catastrophic candidates
            if cosine_sim < config.conflict_threshold:
                catastrophic_candidates.append({
                    'args': mdp2_dataset_args,
                    'seed': cand_seed,
                    'conflict': cosine_sim,
                    'style': mdp2_dataset_args.style,
                    'p': getattr(mdp2_dataset_args, 'gnp_p', 0.0),
                    'n': mdp2_dataset_args.gnp_min_n,
                    'resource_pressure': resource_pressure,
                    'is_bottleneck': resource_pressure > 1.0,
                })
            
            if cosine_sim < best_conflict:
                best_conflict = cosine_sim
                best_seed = cand_seed
                best_args = mdp2_dataset_args
                best_resource_pressure = resource_pressure
                
                if cosine_sim < config.conflict_threshold:
                    has_divisor = hasattr(mdp2_dataset_args, 'req_divisor')
                    regime = "BOTTLENECK" if resource_pressure > 1.0 else "queue-free"
                    pressure_str = f", pressure={resource_pressure:.2f}x" if config.include_bottlenecks else ""
                    divisor_str = "" if has_divisor else ", no-divisor"
                    print(f"  → CATASTROPHIC [{regime}]: style={mdp2_dataset_args.style}, "
                          f"p={getattr(mdp2_dataset_args, 'gnp_p', 'N/A'):.2f}, "
                          f"n={mdp2_dataset_args.gnp_min_n}{pressure_str}{divisor_str}, "
                          f"conflict={cosine_sim:.4f}")
        
        except Exception as e:
            continue
    
    print()
    print(f"[CatastrophicConflict] Search complete!")
    print(f"  Best conflict: {best_conflict:.4f}")
    print(f"  Catastrophic candidates found: {len(catastrophic_candidates)}")
    
    # Diagnostics
    diagnostics = {
        'best_conflict': best_conflict,
        'catastrophic_count': len(catastrophic_candidates),
        'is_catastrophic': best_conflict < config.conflict_threshold,
        'mdp1_grad_norm': norm_mdp1,
        'all_catastrophic': catastrophic_candidates,
    }
    
    if best_conflict < config.conflict_threshold:
        print(f"  ✓ CATASTROPHIC CONFLICT ACHIEVED (cos={best_conflict:.4f} < {config.conflict_threshold})")
        print(f"    This should prevent the agent from learning!")
    else:
        print(f"  ⚠ Best conflict ({best_conflict:.4f}) did not reach catastrophic threshold")
        print(f"    Agent may still be able to learn a compromise")
    
    print()
    print(f"  Selected MDP2:")
    print(f"    - Style: {best_args.style}")
    print(f"    - Edge prob: {getattr(best_args, 'gnp_p', 'N/A'):.3f}")
    print(f"    - Tasks: {best_args.gnp_min_n}")
    print(f"    - Seed: {best_seed}")
    
    if config.include_bottlenecks:
        has_divisor = hasattr(best_args, 'req_divisor')
        print(f"    - VM/Host ratio: 1.0 (fixed)")
        print(f"    - Resource pressure: {best_resource_pressure:.2f}x ({'BOTTLENECK' if best_resource_pressure > 1.0 else 'queue-free'})")
        print(f"    - req_divisor: {'PRESENT' if has_divisor else 'REMOVED (real contention)'}")
        
        # Count bottleneck vs queue-free in catastrophic candidates
        bottleneck_count = sum(1 for c in catastrophic_candidates if c.get('is_bottleneck', False))
        queue_free_count = len(catastrophic_candidates) - bottleneck_count
        print()
        print(f"  Catastrophic candidate breakdown:")
        print(f"    - Bottleneck regimes (pressure > 1.0): {bottleneck_count}")
        print(f"    - Queue-free regimes (pressure ≤ 1.0): {queue_free_count}")
    
    return best_args, best_seed, best_conflict, diagnostics


def verify_conflict_persistence(
    agent,
    mdp1_args: DatasetArgs,
    mdp1_seed: int,
    mdp2_args: DatasetArgs,
    mdp2_seed: int,
    device: torch.device,
    alpha_makespan: float = 0.5,
    alpha_energy: float = 0.5,
    rollout_steps: int = 128,
) -> Tuple[float, float, float]:
    """
    Verify that gradient conflict persists with current policy.
    
    Returns:
        (conflict_score, grad1_norm, grad2_norm)
    """
    # Compute gradients with current policy
    env1 = CloudSchedulingGymEnvironment(
        dataset_args=mdp1_args,
        collect_timelines=False,
        compute_metrics=False,
    )
    env1 = GinAgentWrapper(env1)
    env1.reset(seed=mdp1_seed)
    
    grad1 = compute_actor_gradient(
        agent, env1, rollout_steps, device, alpha_makespan, alpha_energy
    )
    env1.close()
    
    env2 = CloudSchedulingGymEnvironment(
        dataset_args=mdp2_args,
        collect_timelines=False,
        compute_metrics=False,
    )
    env2 = GinAgentWrapper(env2)
    env2.reset(seed=mdp2_seed)
    
    grad2 = compute_actor_gradient(
        agent, env2, rollout_steps, device, alpha_makespan, alpha_energy
    )
    env2.close()
    
    norm1 = grad1.norm().item()
    norm2 = grad2.norm().item()
    
    if norm1 < 1e-9 or norm2 < 1e-9:
        return 0.0, norm1, norm2
    
    conflict = torch.dot(grad1, grad2).item() / (norm1 * norm2 + 1e-9)
    
    return conflict, norm1, norm2
