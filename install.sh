#!/bin/bash

# Installation script for RL Batching System

echo "=========================================="
echo "Installing RL Batching System"
echo "=========================================="
echo ""

# Check Python version
echo "[1/3] Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.8 or higher."
    exit 1
fi

python3 --version
echo "✅ Python found"
echo ""

# Install dependencies
echo "[2/3] Installing dependencies..."
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi
echo ""

# Create necessary directories
echo "[3/3] Creating directories..."
mkdir -p checkpoints logs results
echo "✅ Directories created"
echo ""

# Test installation
echo "Testing installation..."
python3 -c "
import sys
try:
    import gymnasium
    import torch
    import numpy as np
    import matplotlib
    print('✅ All core dependencies available')
    print(f'   - Python: {sys.version.split()[0]}')
    print(f'   - PyTorch: {torch.__version__}')
    print(f'   - Gymnasium: {gymnasium.__version__}')
    print(f'   - NumPy: {np.__version__}')
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Installation Complete!"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo "  1. Quick demo:  python3 example_usage.py"
    echo "  2. Train agent: python3 main.py --mode train --episodes 500"
    echo "  3. View README: cat README.md"
    echo ""
else
    echo "❌ Installation test failed"
    exit 1
fi
