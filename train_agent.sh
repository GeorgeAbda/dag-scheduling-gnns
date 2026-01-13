#!/bin/bash
# Train an Agent
# This script trains a specialist agent using the provided configuration

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Training Agent"
echo "=========================================="

# Default: Train wide specialist
CONFIG="${1:-configs/train_wide_aligned.yaml}"

if [ ! -f "$CONFIG" ]; then
    echo "Error: Configuration file not found: $CONFIG"
    exit 1
fi

echo "Using configuration: $CONFIG"
echo ""

# Train using the config file
python -m cogito.gnn_deeprl_model.train --config "$CONFIG"
