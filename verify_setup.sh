#!/bin/bash
# Verification script to check if the repository is set up correctly

set -e

echo "=========================================="
echo "Verifying DAG Scheduling GNN Setup"
echo "=========================================="
echo ""

# Check Python version
echo "1. Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Found Python $python_version"

# Check required files
echo ""
echo "2. Checking required files..."
required_files=(
    "train_agent.sh"
    "evaluate_agents.sh"
    "requirements.txt"
    "README.md"
    "QUICKSTART.md"
)

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "   ✓ $file"
    else
        echo "   ✗ $file (MISSING)"
        exit 1
    fi
done

# Check required directories
echo ""
echo "3. Checking required directories..."
required_dirs=(
    "cogito"
    "configs"
    "data"
    "logs"
)

for dir in "${required_dirs[@]}"; do
    if [ -d "$dir" ]; then
        echo "   ✓ $dir/"
    else
        echo "   ✗ $dir/ (MISSING)"
        exit 1
    fi
done

# Check for compiled modules
echo ""
echo "4. Checking compiled Cython modules..."
so_count=$(find cogito -name "*.so" | wc -l | tr -d ' ')
if [ "$so_count" -gt 0 ]; then
    echo "   ✓ Found $so_count compiled modules"
else
    echo "   ✗ No compiled modules found"
    exit 1
fi

# Check for checkpoints
echo ""
echo "5. Checking pre-trained checkpoints..."
checkpoint_count=$(find logs -name "hetero_best.pt" | wc -l | tr -d ' ')
if [ "$checkpoint_count" -eq 8 ]; then
    echo "   ✓ Found all 8 pre-trained checkpoints"
else
    echo "   ⚠ Found $checkpoint_count/8 checkpoints"
fi

# Check Python packages
echo ""
echo "6. Checking Python dependencies..."
if python3 -c "import torch; import numpy; import yaml" 2>/dev/null; then
    echo "   ✓ Core dependencies installed"
else
    echo "   ✗ Missing dependencies. Run: pip install -r requirements.txt"
    exit 1
fi

# Summary
echo ""
echo "=========================================="
echo "✓ Setup verification complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Run quick evaluation: ./evaluate_agents.sh"
echo "  2. Train a new agent: ./train_agent.sh configs/train_wide_aligned.yaml"
echo "  3. Read QUICKSTART.md for detailed instructions"
echo ""
