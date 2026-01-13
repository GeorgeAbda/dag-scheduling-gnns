#!/bin/bash
# Evaluate Trained Agents Using Old Script
# This script evaluates specialist agents across different host regimes using the original evaluation script
#
# Usage:
#   bash 5_evaluate_trained_agents.sh
#
# This will run evaluations for all 4 regimes: AL, HS, HP, NA

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Set PYTHONPATH to include project root
export PYTHONPATH="$(cd .. && pwd):$PYTHONPATH"

echo "=========================================="
echo "Evaluating Trained Agents"
echo "=========================================="

# Common parameters
LONGCP_CONFIG="data/rl_configs/train_long_cp_p08_seeds.json"
WIDE_CONFIG="data/rl_configs/train_wide_p005_seeds.json"
LONGCP_CKPT="logs/longcp_aligned/ablation/per_variant/hetero/hetero_best.pt"
WIDE_CKPT="logs/wide_aligned/ablation/per_variant/hetero/hetero_best.pt"
REPEATS=5
DEVICE="cpu"

echo ""
echo "=========================================="
echo "Evaluating AL (Aligned) Configuration"
echo "=========================================="
python ../scripts/eval_hetero_agents_over_seed_configs.py \
  --longcp-config "$LONGCP_CONFIG" \
  --wide-config "$WIDE_CONFIG" \
  --longcp-ckpt "$LONGCP_CKPT" \
  --wide-ckpt "$WIDE_CKPT" \
  --host-specs-path data/host_specs_AL.json \
  --eval-repeats-per-seed $REPEATS \
  --eval-seed 12345 \
  --out-csv logs/eval_old_AL.csv \
  --device $DEVICE

echo ""
echo "=========================================="
echo "Evaluating HS (Homogeneous Speed) Configuration"
echo "=========================================="
python ../scripts/eval_hetero_agents_over_seed_configs.py \
  --longcp-config "$LONGCP_CONFIG" \
  --wide-config "$WIDE_CONFIG" \
  --longcp-ckpt logs/longcp_homospeed/ablation/per_variant/hetero/hetero_best.pt \
  --wide-ckpt logs/wide_homospeed/ablation/per_variant/hetero/hetero_best.pt \
  --host-specs-path data/host_specs_homospeed.json \
  --eval-repeats-per-seed $REPEATS \
  --eval-seed 22345 \
  --out-csv logs/eval_old_HS.csv \
  --device $DEVICE

echo ""
echo "=========================================="
echo "Evaluating HP (Homogeneous Power) Configuration"
echo "=========================================="
python ../scripts/eval_hetero_agents_over_seed_configs.py \
  --longcp-config "$LONGCP_CONFIG" \
  --wide-config "$WIDE_CONFIG" \
  --longcp-ckpt logs/longcp_homopower/ablation/per_variant/hetero/hetero_best.pt \
  --wide-ckpt logs/wide_homopower/ablation/per_variant/hetero/hetero_best.pt \
  --host-specs-path data/host_specs_homoPower.json \
  --eval-repeats-per-seed $REPEATS \
  --eval-seed 32345 \
  --out-csv logs/eval_old_HP.csv \
  --device $DEVICE

echo ""
echo "=========================================="
echo "Evaluating NA (Not Aligned) Configuration"
echo "=========================================="
python ../scripts/eval_hetero_agents_over_seed_configs.py \
  --longcp-config "$LONGCP_CONFIG" \
  --wide-config "$WIDE_CONFIG" \
  --longcp-ckpt logs/longcp_not_aligned/ablation/per_variant/hetero/hetero_best.pt \
  --wide-ckpt logs/wide_not_aligned/ablation/per_variant/hetero/hetero_best.pt \
  --host-specs-path data/host_specs_NAL.json \
  --eval-repeats-per-seed $REPEATS \
  --eval-seed 42345 \
  --out-csv logs/eval_old_NA.csv \
  --device $DEVICE

echo ""
echo "=========================================="
echo "All Evaluations Completed!"
echo "=========================================="
echo "Results saved to:"
echo "  - logs/eval_old_AL.csv (and .summary.csv)"
echo "  - logs/eval_old_HS.csv (and .summary.csv)"
echo "  - logs/eval_old_HP.csv (and .summary.csv)"
echo "  - logs/eval_old_NA.csv (and .summary.csv)"
