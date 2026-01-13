#!/bin/bash
# Train an Agent
# This script trains a specialist agent using the provided configuration

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Set PYTHONPATH to include project root
export PYTHONPATH="$(cd .. && pwd):$PYTHONPATH"

echo "=========================================="
echo "Training Agent"
echo "=========================================="

# Default: Train wide specialist
CONFIG="${1:-configs/train_wide_specialist.yaml}"

echo "Using configuration: $CONFIG"
echo ""

# Train specialist using Cython-compiled module
# Parse YAML config and call the C extension directly
python -c "
import sys, os, yaml
sys.path.insert(0, os.path.abspath('..'))

# Load config
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)

# Set host specs env var before importing
d = cfg.get('domain', {})
if d.get('host_specs_file'):
    os.environ['HOST_SPECS_PATH'] = os.path.abspath(d['host_specs_file'])

# Import and run
from cogito.gnn_deeprl_model.train import main, Args

# Build args from config - set all attributes that exist
a = Args()

# Experiment
e = cfg.get('experiment', {})
if hasattr(a, 'exp_name'): a.exp_name = e.get('name', a.exp_name)
if hasattr(a, 'seed'): a.seed = e.get('seed', a.seed)
if hasattr(a, 'output_dir'): a.output_dir = e.get('output_dir', a.output_dir)
if hasattr(a, 'device'): a.device = e.get('device', a.device)

# Training
t = cfg.get('training', {})
if hasattr(a, 'total_timesteps'): a.total_timesteps = t.get('total_timesteps', a.total_timesteps)
if hasattr(a, 'learning_rate'): a.learning_rate = t.get('learning_rate', a.learning_rate)
if hasattr(a, 'num_envs'): a.num_envs = t.get('num_envs', a.num_envs)
if hasattr(a, 'num_steps'): a.num_steps = t.get('num_steps', a.num_steps)
if hasattr(a, 'gamma'): a.gamma = t.get('gamma', a.gamma)
if hasattr(a, 'gae_lambda'): a.gae_lambda = t.get('gae_lambda', a.gae_lambda)
if hasattr(a, 'num_minibatches'): a.num_minibatches = t.get('num_minibatches', a.num_minibatches)
if hasattr(a, 'update_epochs'): a.update_epochs = t.get('update_epochs', a.update_epochs)
if hasattr(a, 'clip_coef'): a.clip_coef = t.get('clip_coef', a.clip_coef)
if hasattr(a, 'ent_coef'): a.ent_coef = t.get('ent_coef', a.ent_coef)
if hasattr(a, 'vf_coef'): a.vf_coef = t.get('vf_coef', a.vf_coef)
if hasattr(a, 'max_grad_norm'): a.max_grad_norm = t.get('max_grad_norm', a.max_grad_norm)
if hasattr(a, 'anneal_lr'): a.anneal_lr = t.get('anneal_lr', a.anneal_lr)

# Evaluation
ev = cfg.get('evaluation', {})
if hasattr(a, 'test_every_iters'): a.test_every_iters = ev.get('test_every_iters', a.test_every_iters)
if hasattr(a, 'robust_eval_alpha'): a.robust_eval_alpha = ev.get('robust_eval_alpha', a.robust_eval_alpha)

# Domain
d = cfg.get('domain', {})
if hasattr(a, 'longcp_config'): a.longcp_config = d.get('longcp_config')
if hasattr(a, 'wide_config'): a.wide_config = d.get('wide_config')

# Seed control
sc = cfg.get('seed_control', {})
if hasattr(a, 'training_seed_mode'): a.training_seed_mode = sc.get('mode', a.training_seed_mode)
if hasattr(a, 'train_seeds_file'): a.train_seeds_file = sc.get('seeds_file')

# Variant
v = cfg.get('variant', {})
if hasattr(a, 'train_only_variant'): a.train_only_variant = v.get('name', a.train_only_variant)

# Trajectory
traj = cfg.get('trajectory', {})
if hasattr(a, 'trajectory_enabled'): a.trajectory_enabled = traj.get('enabled', a.trajectory_enabled)
if hasattr(a, 'trajectory_collect_every'): a.trajectory_collect_every = traj.get('collect_every', a.trajectory_collect_every)
if hasattr(a, 'trajectory_method'): a.trajectory_method = traj.get('method', a.trajectory_method)

# Logging
log = cfg.get('logging', {})
if hasattr(a, 'no_tensorboard'): a.no_tensorboard = not log.get('tensorboard', True)
if hasattr(a, 'log_every'): a.log_every = log.get('log_every', a.log_every)

# Run training
main(a)
"

echo ""
echo "Training completed!"
echo ""
echo "To train with custom parameters, use:"
echo "  python run_training.py --config configs/train_wide_specialist.yaml \\"
echo "      --total_timesteps 500000 \\"
echo "      --learning_rate 0.0003"
