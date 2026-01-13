# Quick Start Guide

## Prerequisites

- Python 3.10 or higher
- pip package manager
- 8GB+ RAM recommended
- CPU or GPU (CUDA optional)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/dag-scheduling-gnn.git
cd dag-scheduling-gnn
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Evaluation (Using Pre-trained Models)

Run evaluation on all pre-trained agents:

```bash
./evaluate_agents.sh
```

This will:
- Load 8 pre-trained specialist agents (4 LongCP + 4 Wide, one per host regime)
- Evaluate each agent on both task domains (LongCP and Wide)
- Test across all 4 host configurations (AL, HS, HP, NA)
- Generate results in `logs/eval_old_*.csv` and `logs/eval_old_*.summary.csv`

**Expected runtime**: ~30-40 minutes on a modern CPU

## Training Your Own Agent

Train a Wide specialist on Aligned hosts:

```bash
./train_agent.sh configs/train_wide_aligned.yaml
```

**Expected runtime**: ~2-4 hours depending on hardware

### Available Training Configurations

**LongCP Specialists:**
- `configs/train_longcp_aligned.yaml` - Aligned (AL) hosts
- `configs/train_longcp_homospeed.yaml` - Homogeneous Speed (HS) hosts
- `configs/train_longcp_homopower.yaml` - Homogeneous Power (HP) hosts
- `configs/train_longcp_not_aligned.yaml` - Non-Aligned (NA) hosts

**Wide Specialists:**
- `configs/train_wide_aligned.yaml` - Aligned (AL) hosts
- `configs/train_wide_homospeed.yaml` - Homogeneous Speed (HS) hosts
- `configs/train_wide_homopower.yaml` - Homogeneous Power (HP) hosts
- `configs/train_wide_not_aligned.yaml` - Non-Aligned (NA) hosts

## Understanding the Results

### Evaluation Output

Results are saved in two formats:

1. **Per-seed results** (`logs/eval_old_<regime>.csv`):
   - One row per evaluation run
   - Columns: agent_train_domain, eval_domain, seed, makespan, energy_total, energy_active, energy_idle, entropy

2. **Summary statistics** (`logs/eval_old_<regime>.summary.csv`):
   - Aggregated metrics across all seeds
   - Columns: agent_train_domain, eval_domain, seeds, mean_makespan, mean_energy_total, mean_energy_active, mean_energy_idle, mean_entropy

### Key Metrics

- **Makespan**: Total completion time (lower is better)
- **Energy Active**: Active energy consumption in Joules (lower is better)
- **Entropy**: Policy entropy (higher = more stochastic)

### Host Regimes

- **AL (Aligned)**: Fast CPUs consume less power (inverse correlation)
- **HS (Homogeneous Speed)**: All CPUs have the same speed (500 GIPS)
- **HP (Homogeneous Power)**: All CPUs consume the same power
- **NA (Non-Aligned)**: Fast CPUs consume more power (positive correlation)

## Customizing Training

Edit the YAML configuration files to modify:

- `total_timesteps`: Training duration
- `learning_rate`: Learning rate for PPO
- `num_envs`: Number of parallel environments
- `device`: "cpu", "cuda", or "mps" (for Apple Silicon)

Example:
```yaml
training:
  total_timesteps: 5000000  # Increase for longer training
  learning_rate: 0.0003     # Adjust learning rate
  num_envs: 20              # More parallel envs = faster training
  device: "cuda"            # Use GPU if available
```

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError`, ensure you're running scripts from the repository root:
```bash
cd /path/to/dag-scheduling-gnn
./train_agent.sh configs/train_wide_aligned.yaml
```

### Memory Issues

If training crashes with OOM errors:
1. Reduce `num_envs` in the config file
2. Reduce `num_steps` (default: 256)
3. Use a smaller batch size via `num_minibatches` (increase this value)

### Slow Training

To speed up training:
1. Increase `num_envs` (more parallel environments)
2. Use GPU: set `device: "cuda"` in config
3. Reduce `total_timesteps` for faster experiments

## Next Steps

- Read the full paper for methodology and results
- Explore the `data/` directory for host configurations
- Modify `data/rl_configs/*.json` to use different task seeds
- Implement your own scheduling heuristics for comparison

## Support

For issues and questions:
- Open an issue on GitHub
- Check the main README.md for detailed documentation
- Review the paper for theoretical background
