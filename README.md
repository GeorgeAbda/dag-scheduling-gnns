# DAG Scheduling with Graph Neural Networks

This repository contains the implementation and evaluation code for training and evaluating deep reinforcement learning agents for DAG scheduling in heterogeneous cloud environments.

## Repository Structure

```
.
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── train_agent.sh                     # Training script
├── evaluate_agents.sh                 # Evaluation script
├── cogito/                           # Compiled Cython modules (binary)
│   ├── __init__.py
│   ├── config/
│   ├── dataset_generator/
│   ├── gnn_deeprl_model/
│   └── tools/
├── configs/                          # Training configurations
│   ├── train_long_cp_AL.yaml
│   ├── train_long_cp_HP.yaml
│   ├── train_long_cp_HS.yaml
│   ├── train_long_cp_NA.yaml
│   ├── train_wide_AL.yaml
│   ├── train_wide_HP.yaml
│   ├── train_wide_HS.yaml
│   └── train_wide_NA.yaml
├── data/                             # Data files
│   ├── host_specs_AL.json            # Aligned host configuration
│   ├── host_specs_homoPower.json     # Homogeneous power configuration
│   ├── host_specs_homospeed.json     # Homogeneous speed configuration
│   ├── host_specs_NAL.json           # Non-aligned configuration
│   └── rl_configs/                   # RL training seeds
│       ├── train_long_cp_p08_seeds.json
│       └── train_wide_p005_seeds.json
└── logs/                             # Pre-trained checkpoints
    ├── longcp_aligned/
    ├── longcp_homospeed/
    ├── longcp_homopower/
    ├── longcp_not_aligned/
    ├── wide_aligned/
    ├── wide_homospeed/
    ├── wide_homopower/
    └── wide_not_aligned/
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- NumPy
- See `requirements.txt` for full dependencies

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

Train a specialist agent using one of the provided configurations:

```bash
# Train Wide specialist on Aligned (AL) hosts
./train_agent.sh configs/train_wide_AL.yaml

# Train LongCP specialist on Homogeneous Power (HP) hosts
./train_agent.sh configs/train_long_cp_HP.yaml
```

### Evaluation

Evaluate trained agents across all host configurations:

```bash
./evaluate_agents.sh
```

This will:
1. Evaluate all trained agents (LongCP and Wide specialists)
2. Test each agent on both LongCP and Wide task domains
3. Run evaluations across all 4 host regimes (AL, HS, HP, NA)
4. Save results to `logs/eval_old_*.csv` and `logs/eval_old_*.summary.csv`

## Host Configurations

- **AL (Aligned)**: Fast CPUs have low power consumption (inverse correlation)
- **HS (Homogeneous Speed)**: All hosts have the same CPU speed (500 GIPS)
- **HP (Homogeneous Power)**: All hosts have the same power consumption
- **NA (Non-Aligned)**: Fast CPUs have high power consumption (positive correlation)

## Results

Evaluation results are saved in CSV format:
- `logs/eval_old_<regime>.csv`: Per-seed detailed results
- `logs/eval_old_<regime>.summary.csv`: Aggregated statistics

## Citation

If you use this code, please cite our paper:

```bibtex
@article{yourpaper2026,
  title={Structure-Aware Deep Reinforcement Learning for Heterogeneous Cloud Scheduling},
  author={Your Name},
  journal={Your Journal},
  year={2026}
}
```

## License

[Your License Here]
