# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
from pathlib import Path
import random
import time
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from tqdm import tqdm
import tyro
from torch.utils.tensorboard import SummaryWriter
import os
import sys
import logging
from icecream import ic
import matplotlib.pyplot as plt
import seaborn as sns
grandparent_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

# Add grandparent directory to sys.path
if grandparent_dir not in sys.path:
    sys.path.insert(0, grandparent_dir)
from cogito.config.settings import MIN_TESTING_DS_SEED
from cogito.dataset_generator.gen_dataset import DatasetArgs
from cogito.gnn_deeprl_model.agents.agent import Agent
from cogito.gnn_deeprl_model.agents.gin_agent.agent import GinAgent
from cogito.gnn_deeprl_model.agents.gin_agent.wrapper import GinAgentWrapper
from cogito.gnn_deeprl_model.core.env.gym_env import CloudSchedulingGymEnvironment


@dataclass
class Args:
    exp_name: str = "test"
    """the name of this experiment"""

    seed: int = 1
    """seed of the experiment"""
    output_dir: str = "logs"
    """the output directory of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    # Device and threading
    device: str = "cpu"
    """device to use: one of {'auto','cpu','mps','cuda'}; default 'cpu' recommended on this machine"""
    torch_num_threads: int | None = None
    """if set, calls torch.set_num_threads(this) to control CPU threading"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    no_tensorboard: bool = False
    """if True, do not write any TensorBoard logs (uses a NullWriter)"""
    bn_eval: bool = True
    """if True, force BatchNorm layers to eval mode during training to avoid small-batch errors"""
    wandb_project_name: str | None = None
    """the wandb's project name"""
    wandb_entity: str | None = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    env_mode: str = "async"
    """vector env mode: 'async' (default) or 'sync'. For very heavy datasets, consider 'sync'."""
    load_model_dir: str | None = None
    """Directory to load the model from"""
    test_iterations: int = 4
    """number of test iterations"""
    test_every_iters: int = 10
    """run evaluation every N training iterations (set higher to reduce test frequency)"""
    # Visualization / tracing controls
    collect_timelines: bool = False
    """If true, environment collects per-VM timelines and capacities (small overhead)."""
    save_first_episode_heatmap: bool = False
    """If true, save a VM utilization heatmap for a first test episode during training."""
    heatmap_resolution: int = 200
    """Number of time bins (columns) to discretize episode time for the heatmap."""

    # Environment profiling
    profile_env: bool = False
    """Enable lightweight env profiling timers (adds small overhead)."""

    # Algorithm specific arguments
    total_timesteps: int = 5_000_00
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 5
    """the number of parallel game environments (0 = auto-detect CPU cores)"""
    num_steps: int = 256
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float | None = None
    """the target KL divergence threshold"""

    # Lightweight CSV logging (when TensorBoard is disabled)
    csv_reward_tag: str | None = None
    """If set and no_tensorboard=True, write eval metrics to csv files named <metric>_<csv_reward_tag>.csv"""
    csv_dir: str = "csv"
    """Directory to write CSV metrics to when csv_reward_tag is set"""

    dataset: DatasetArgs = field(
        default_factory=lambda: DatasetArgs(
            host_count=4,
            vm_count=10,
            workflow_count=10,
            gnp_min_n=20,
            gnp_max_n=20,
            max_memory_gb=10,
            min_cpu_speed=500,
            max_cpu_speed=5000,
            min_task_length=500,
            max_task_length=100_000,
            task_arrival="static",
            dag_method="gnp",
        )
    )
    """the dataset generation parameters"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    run_name: str = ""
    """the full name of the run"""


# Environment creation
# ----------------------------------------------------------------------------------------------------------------------


def make_env(idx: int, args: Args):
    assert not args.capture_video, "Video capturing is not yet supported"

    # Training env: avoid expensive testing-only metrics
    env: gym.Env = CloudSchedulingGymEnvironment(
        dataset_args=args.dataset,
        collect_timelines=args.collect_timelines,
        compute_metrics=False,
        profile=args.profile_env,
    )
    if args.capture_video and idx == 0:
        video_dir = f"{args.output_dir}/{args.run_name}/videos"
        env = RecordVideo(env, video_dir, episode_trigger=lambda x: x % 1000 == 0)
    env = GinAgentWrapper(env)
    return RecordEpisodeStatistics(env)


def make_env_thunk(idx: int, args: Args):
    """Create a picklable callable that constructs one training env.

    AsyncVectorEnv requires callables that can be sent to subprocesses. Avoid raw lambdas.
    """
    def _thunk():
        return make_env(idx, args)
    return _thunk


def make_test_env(args: Args):
    # Test/Eval env: enable metrics for reporting
    env: gym.Env = CloudSchedulingGymEnvironment(
        dataset_args=args.dataset,
        collect_timelines=args.collect_timelines,
        compute_metrics=True,
        profile=args.profile_env,
    )
    return GinAgentWrapper(env)


def make_agent(device: torch.device) -> Agent:
    return GinAgent(device)


# Training Agent
# ----------------------------------------------------------------------------------------------------------------------


def train(args: Args):
    # Defer batch size calculations until after possible num_envs auto-detect
    args.run_name = f"{int(time.time())}_{args.exp_name}"
    if args.track:
        import wandb

        assert args.wandb_project_name is not None, "Please specify the wandb project name"
        assert args.wandb_entity is not None, "Please specify the entity of wandb project"
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=args.run_name,
            monitor_gym=True,
            save_code=True,
        )

    # Create a no-op writer if tensorboard is disabled
    class NullWriter:
        def add_text(self, *args, **kwargs):
            pass
        def add_scalar(self, *args, **kwargs):
            pass
        def close(self):
            pass

    if args.no_tensorboard:
        writer = NullWriter()
        print("[train] TensorBoard disabled: no events will be written (no_tensorboard=True)")
    else:
        writer = SummaryWriter(f"{args.output_dir}/{args.run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Optional control of CPU thread count
    if args.torch_num_threads is not None and args.torch_num_threads > 0:
        try:
            torch.set_num_threads(int(args.torch_num_threads))
        except Exception:
            pass

    # Device selection with safety checks
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
        # auto: prefer cuda, then mps, else cpu
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = pick_device(args.device)

    # Auto-detect number of parallel envs if requested (0 means auto)
    if args.num_envs <= 0:
        try:
            cpu_count = os.cpu_count() or 1
            print(f"Auto-detecting {cpu_count} CPU cores")
        except Exception:
            cpu_count = 1
        # Use total logical CPU count by default (user asked to match CPU count)
        args.num_envs = max(1, int(cpu_count))

    # Compute derived sizes now that num_envs is known
    args.batch_size = int(args.num_envs * args.num_steps)
    # Guard to avoid zero minibatch size if misconfigured
    args.minibatch_size = max(1, int(args.batch_size // max(1, args.num_minibatches)))
    args.num_iterations = max(1, int(args.total_timesteps // max(1, args.batch_size)))
    print(f'Using {args.num_envs} envs, batch size {args.batch_size}, minibatch size {args.minibatch_size}, ')
    print(f'num iterations {args.num_iterations}, device {device}')
    # env setup: run environments in parallel processes
    print(f"[train] Creating {args.num_envs} envs (mode={args.env_mode})...")
    if args.num_envs <= 1:
        # Use a single non-vectorized env wrapped to match the vector API when needed
        # Here we still wrap with GinAgentWrapper and RecordEpisodeStatistics in make_env
        envs = gym.vector.SyncVectorEnv([make_env_thunk(0, args)])
    else:
        if args.env_mode.lower() == "sync":
            envs = gym.vector.SyncVectorEnv([make_env_thunk(i, args) for i in range(args.num_envs)])
        else:
            envs = gym.vector.AsyncVectorEnv([make_env_thunk(i, args) for i in range(args.num_envs)])
    print("[train] Env vector created; querying spaces...")
    obs_space = envs.single_observation_space
    act_space = envs.single_action_space
    assert isinstance(act_space, gym.spaces.Discrete), "only discrete action space is supported"
    assert obs_space.shape is not None
    assert act_space.shape is not None

    agent = make_agent(device)
    writer.add_text("agent", f"```{agent}```")

    # Optionally put BatchNorm layers into eval mode to prevent small-batch errors
    if args.bn_eval:
        def _bn_to_eval(m: nn.Module):
            if isinstance(m, nn.BatchNorm1d):
                m.eval()
        agent.apply(_bn_to_eval)
        print("[train] BatchNorm layers set to eval mode (bn_eval=True)")

    last_model_save = 0
    if args.load_model_dir:
        model_path = Path(__file__).parent.parent.parent / "logs" / args.load_model_dir / "model.pt"
        agent.load_state_dict(torch.load(str(model_path), weights_only=True))
        print(f"Loaded model from {model_path}")

    ic(agent)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + obs_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + act_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    print("[train] Resetting vector envs (this may take a while for large datasets)...")
    next_obs, _ = envs.reset(seed=args.seed)
    print("[train] Env reset complete.")
    next_obs_tensor = torch.Tensor(next_obs).to(device)
    next_done_tensor = torch.zeros(args.num_envs).to(device)

    pbar = tqdm(total=args.total_timesteps)
    heatmap_saved = False
    for iteration in range(1, args.num_iterations + 1):
        print(f"[train] Iter {iteration}/{args.num_iterations} start")
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs_tensor
            dones[step] = next_done_tensor
            # print(f'Getting action at step {step}, global step {global_step}')
            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs_tensor)

                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            # print(f'Took action {action.cpu().numpy()}')
            # print(f'Received reward {reward}')
            # print(f'Terminations: {terminations}, Truncations: {truncations}')
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.Tensor(reward).to(device).view(-1)
            next_obs_tensor, next_done_tensor = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        pbar.update(global_step - pbar.n)
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # bootstrap value if not done
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

        # flatten the batch
        b_obs = obs.reshape((-1,) + obs_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + act_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            print(f'PPO epoch {epoch + 1}/{args.update_epochs}')
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        if args.test_every_iters > 0 and (iteration % args.test_every_iters == 0 or iteration == args.num_iterations):
            with torch.no_grad():
                print(f"[train] Iter {iteration}: running evaluation ({args.test_iterations} episodes)...")
                test_results = test_agent(agent, args)
                avg_mk, avg_energy_obs, avg_total_energy, m = test_results
                print(f"[train] Iter {iteration}: eval makespan={avg_mk:.4g}, energy_obs={avg_energy_obs:.4g}, total_energy={avg_total_energy:.4g}")
                writer.add_scalar("tests/makespan", avg_mk, global_step)
                writer.add_scalar("tests/energy_consumption", avg_energy_obs, global_step)
                writer.add_scalar("tests/total_energy", avg_total_energy, global_step)
                # Energy breakdown
                has_active = m.get("avg_active_energy") is not None
                has_idle = m.get("avg_idle_energy") is not None
                if has_active:
                    writer.add_scalar("tests/active_energy", m["avg_active_energy"], global_step)
                if has_idle:
                    writer.add_scalar("tests/idle_energy", m["avg_idle_energy"], global_step)
                if has_active or has_idle:
                    print(f"[train] Iter {iteration}: active_energy={m.get('avg_active_energy', 'NA')}, idle_energy={m.get('avg_idle_energy', 'NA')}")
                # Sum active + idle as requested
                if has_active and has_idle:
                    writer.add_scalar(
                        "tests/active_plus_idle",
                        float(m["avg_active_energy"]) + float(m["avg_idle_energy"]),
                        global_step,
                    )
                # Optional CSV export when TB is disabled
                if args.csv_reward_tag:
                    try:
                        import csv as _csv
                        import os as _os
                        from time import time as _time
                        _os.makedirs(args.csv_dir, exist_ok=True)
                        ts = _time()
                        def _append_row(metric_name: str, value: float):
                            path = _os.path.join(args.csv_dir, f"{metric_name}_{args.csv_reward_tag}.csv")
                            new_file = not _os.path.exists(path)
                            with open(path, "a", newline="") as f:
                                w = _csv.writer(f)
                                if new_file:
                                    w.writerow(["Wall time", "Step", "Value"])
                                w.writerow([ts, int(global_step), float(value)])
                        _append_row("makespan", avg_mk)
                        _append_row("total_energy", avg_total_energy)
                        if has_active:
                            _append_row("active_energy", float(m["avg_active_energy"]))
                        if has_idle:
                            _append_row("idle_energy", float(m["avg_idle_energy"]))
                    except Exception as _e:
                        print(f"[train] CSV export failed: {_e}")
                # Bottleneck metrics
                for k in [
                    ("bneck_steps_ratio", "tests/bottleneck_steps_ratio"),
                    ("ready_ratio", "tests/ready_bottleneck_ratio"),
                    ("avg_wait_time", "tests/avg_wait_time"),
                    ("avg_bneck_steps", "tests/avg_bottleneck_steps"),
                    ("avg_decision_steps", "tests/avg_decision_steps"),
                    ("avg_ready", "tests/avg_ready_tasks"),
                    ("avg_ready_blocked", "tests/avg_blocked_ready_tasks"),
                    # Refined
                    ("refined_steps_ratio", "tests/refined_bottleneck_steps_ratio"),
                    ("refined_ready_ratio", "tests/refined_ready_bottleneck_ratio"),
                    ("avg_refined_steps", "tests/avg_refined_bottleneck_steps"),
                    ("avg_ready_refined", "tests/avg_ready_tasks_refined"),
                    ("avg_ready_blocked_refined", "tests/avg_blocked_ready_tasks_refined"),
                    # CP breakdown
                    ("wait_time_cp", "tests/wait_time_cp"),
                    ("wait_time_offcp", "tests/wait_time_offcp"),
                ]:
                    key, tbtag = k
                    if m.get(key) is not None:
                        writer.add_scalar(tbtag, m[key], global_step)

            # Save a first-episode VM utilization heatmap once per run if enabled
            if args.save_first_episode_heatmap and not heatmap_saved and args.collect_timelines:
                try:
                    # Run a single test episode to collect timelines
                    test_env = make_test_env(args)
                    obs_np, _ = test_env.reset(seed=MIN_TESTING_DS_SEED)
                    final_info = None
                    while True:
                        a, _, _, _ = agent.get_action_and_value(torch.from_numpy(obs_np.astype(np.float32)).unsqueeze(0).to(device))
                        obs_np, _, terminated, truncated, info = test_env.step(int(a.item()))
                        if terminated or truncated:
                            final_info = info
                            break
                    test_env.close()
                    if isinstance(final_info, dict) and "vm_timelines" in final_info:
                        vm_timelines = final_info.get("vm_timelines", [])
                        vm_total_mem = final_info.get("vm_total_mem", [])
                        vm_total_cores = final_info.get("vm_total_cores", [])
                        # Build utilization matrix
                        makespan_est = 0.0
                        for vm_list in vm_timelines:
                            for seg in vm_list:
                                makespan_est = max(makespan_est, float(seg.get("t_end", 0.0)))
                        V = len(vm_timelines)
                        T = max(10, int(args.heatmap_resolution))
                        ts = np.linspace(0.0, max(1e-9, makespan_est), T)
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
                        plt.title("Training: First Test Episode VM Utilization (max of mem/core)")
                        plt.xlabel("Time →")
                        plt.ylabel("VMs")
                        thr = 0.95
                        ys, xs = np.where(util >= thr)
                        plt.scatter(xs + 0.5, ys + 0.5, s=6, c="cyan", marker="s", alpha=0.6)
                        out_png = f"{args.output_dir}/{args.run_name}/first_test_episode_util_heatmap.png"
                        plt.savefig(out_png, bbox_inches="tight", dpi=200)
                        plt.close(fig)
                        heatmap_saved = True
                        print(f"Saved training heatmap: {out_png}")

                        # Also save assignability gap histogram if series are available
                        t = final_info.get("timeline_t", None)
                        r = final_info.get("timeline_ready", None)
                        s = final_info.get("timeline_schedulable", None)
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
                            out_png_hist = f"{args.output_dir}/{args.run_name}/first_test_episode_assign_gap_hist.png"
                            plt.tight_layout()
                            plt.savefig(out_png_hist, bbox_inches="tight", dpi=300)
                            plt.close(fig)
                            print(f"Saved training plot: {out_png_hist}")
                except Exception as e:
                    print(f"Heatmap generation during training skipped: {e}")

        if (global_step - last_model_save) >= 10_000:
            torch.save(agent.state_dict(), f"{args.output_dir}/{args.run_name}/model_{global_step}.pt")
            last_model_save = global_step

    torch.save(agent.state_dict(), f"{args.output_dir}/{args.run_name}/model.pt")

    envs.close()
    writer.close()

    pbar.close()


# Testing Agent
# ----------------------------------------------------------------------------------------------------------------------


def test_agent(agent: Agent, args: Args):
    total_makespan = 0.0
    total_energy_consumption = 0.0
    total_energy_full = 0.0
    total_active_energy = 0.0
    total_idle_energy = 0.0
    # Bottleneck metrics
    total_bneck_steps = 0
    total_decision_steps = 0
    total_ready = 0
    total_ready_blocked = 0
    total_wait_time = 0.0
    # Refined bottleneck metrics and CP breakdown
    total_refined_steps = 0
    total_ready_refined = 0
    total_ready_blocked_refined = 0
    total_wait_cp = 0.0
    total_wait_offcp = 0.0

    for seed_index in range(args.test_iterations):
        test_env = make_test_env(args)

        next_obs, _ = test_env.reset(seed=MIN_TESTING_DS_SEED + seed_index)
        final_info: dict | None = None
        while True:
            obs_tensor = torch.from_numpy(next_obs.astype(np.float32).reshape(1, -1))
            action, _, _, _ = agent.get_action_and_value(obs_tensor)
            vm_action = int(action.item())
            next_obs, _, terminated, truncated, info = test_env.step(vm_action)
            if terminated or truncated:
                final_info = info
                break

        assert test_env.prev_obs is not None
        total_makespan += test_env.prev_obs.makespan()
        total_energy_consumption += test_env.prev_obs.energy_consumption()
        
        # Track active and idle energy if available
        if isinstance(final_info, dict):
            if "total_energy" in final_info:
                total_energy_full += float(final_info["total_energy"])
            if "total_energy_active" in final_info and "total_energy_idle" in final_info:
                total_active_energy += float(final_info["total_energy_active"])
                total_idle_energy += float(final_info["total_energy_idle"])
            # Bottleneck metrics
            total_bneck_steps += int(final_info.get("bottleneck_steps", 0))
            total_decision_steps += int(final_info.get("decision_steps", 0))
            total_ready += int(final_info.get("sum_ready_tasks", 0))
            total_ready_blocked += int(final_info.get("sum_bottleneck_ready_tasks", 0))
            total_wait_time += float(final_info.get("cumulative_wait_time", 0.0))
            # Refined
            total_refined_steps += int(final_info.get("refined_bottleneck_steps", 0))
            total_ready_refined += int(final_info.get("sum_ready_tasks_refined", 0))
            total_ready_blocked_refined += int(final_info.get("sum_blocked_ready_tasks_refined", 0))
            total_wait_cp += float(final_info.get("wait_time_cp", 0.0))
            total_wait_offcp += float(final_info.get("wait_time_offcp", 0.0))
        
        # Fallback to observation's estimate if not provided
        if total_energy_full == 0.0:
            total_energy_full += test_env.prev_obs.energy_consumption()
        
        test_env.close()

    # Calculate averages
    n = args.test_iterations
    avg_makespan = total_makespan / n
    avg_energy_consumption = total_energy_consumption / n
    avg_total_energy = total_energy_full / n
    
    # Prepare metrics dict for TensorBoard logging
    metrics: dict[str, float] = {}
    if total_active_energy > 0 or total_idle_energy > 0:
        total_energy = total_active_energy + total_idle_energy
        active_fraction = total_active_energy / total_energy if total_energy > 0 else 0.0
        metrics["avg_active_energy"] = total_active_energy / n
        metrics["avg_idle_energy"] = total_idle_energy / n
        metrics["active_fraction"] = active_fraction
    if total_decision_steps > 0 or total_ready > 0:
        avg_bneck_steps = total_bneck_steps / n
        avg_decision_steps = total_decision_steps / n
        avg_ready = total_ready / n
        avg_ready_blocked = total_ready_blocked / n
        avg_wait_time = total_wait_time / n
        ratio_steps = (total_bneck_steps / max(1, total_decision_steps)) if total_decision_steps > 0 else 0.0
        ratio_ready = (total_ready_blocked / max(1, total_ready)) if total_ready > 0 else 0.0
        metrics.update({
            "avg_bneck_steps": avg_bneck_steps,
            "avg_decision_steps": avg_decision_steps,
            "avg_ready": avg_ready,
            "avg_ready_blocked": avg_ready_blocked,
            "avg_wait_time": avg_wait_time,
            "bneck_steps_ratio": ratio_steps,
            "ready_ratio": ratio_ready,
        })
    # Add refined metrics and CP breakdown if present
    if total_refined_steps > 0 or total_ready_refined > 0 or total_wait_cp > 0 or total_wait_offcp > 0:
        avg_refined_steps = total_refined_steps / n if n > 0 else 0.0
        avg_ready_refined = total_ready_refined / n if n > 0 else 0.0
        avg_ready_blocked_refined = total_ready_blocked_refined / n if n > 0 else 0.0
        refined_steps_ratio = (total_refined_steps / max(1, total_decision_steps)) if total_decision_steps > 0 else 0.0
        refined_ready_ratio = (total_ready_blocked_refined / max(1, total_ready_refined)) if total_ready_refined > 0 else 0.0
        metrics.update({
            "avg_refined_steps": avg_refined_steps,
            "avg_ready_refined": avg_ready_refined,
            "avg_ready_blocked_refined": avg_ready_blocked_refined,
            "refined_steps_ratio": refined_steps_ratio,
            "refined_ready_ratio": refined_ready_ratio,
            "wait_time_cp": total_wait_cp / n if n > 0 else 0.0,
            "wait_time_offcp": total_wait_offcp / n if n > 0 else 0.0,
        })
    return avg_makespan, avg_energy_consumption, avg_total_energy, metrics


if __name__ == "__main__":
    # Configure logging so env debug statements are visible during training
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")
    train(tyro.cli(Args))
