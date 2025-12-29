#!/bin/bash
# Clear temporary/generated files from repo

# Kill any hanging training processes
pkill -f "python -m src.main" 2>/dev/null
pkill -f "tensorboard" 2>/dev/null
sleep 1

# Clear GPU memory (ROCm)
if command -v rocm-smi &> /dev/null; then
    echo "Clearing GPU memory..."
    # rocm-smi --resetclocks resets GPU state
    rocm-smi --resetclocks 2>/dev/null || true
fi

# Remove generated files
rm -rf runs/
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
rm -f *.pt
rm -rf .pytest_cache/

echo "Cleaned: processes, GPU, runs/, __pycache__, *.pt, .pytest_cache/"
