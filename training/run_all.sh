#!/bin/bash
# Runs all three experiments sequentially and writes a DONE marker when finished.
set -e
cd "$(dirname "$0")"

echo "=== baseline ==="
python3 train.py --mode baseline --epochs 12 --num-workers 2

echo "=== frozen (ablation A) ==="
python3 train.py --mode frozen --epochs 8 --num-workers 2

echo "=== finetuned (ablation B / deployed model) ==="
python3 train.py --mode finetuned --epochs 10 --num-workers 2

touch ../results/ALL_DONE
echo "ALL RUNS COMPLETE"
