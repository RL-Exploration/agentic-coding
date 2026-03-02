#!/usr/bin/env bash
# Multi-model pass@k scaling eval on Lightning AI
# Proves dataset headroom by comparing 0.5B, 1.5B, and 3B curves.
#
# Usage:
#   bash run_scaling_eval.sh              # all puzzles (easy + humaneval)
#   DIRS=puzzles_easy bash run_scaling_eval.sh   # easy only
#
# Requires: GPU instance (L40S 48GB handles all three models sequentially)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

SAMPLES="${SAMPLES:-8}"
TEMPERATURE="${TEMPERATURE:-1.0}"
DIRS="${DIRS:-puzzles_easy puzzles_humaneval}"

echo "=== Installing dependencies ==="
pip install -q -r requirements.txt
pip install -q -r inference/requirements.txt
pip install -q matplotlib

echo ""
echo "=== Verifying puzzles ==="
TOTAL=0
for DIR in $DIRS; do
    COUNT=$(ls "$DIR"/*.json 2>/dev/null | wc -l | tr -d ' ')
    echo "  $DIR: $COUNT puzzles"
    TOTAL=$((TOTAL + COUNT))
done
echo "  Total: $TOTAL puzzles"

if [ "$TOTAL" -eq 0 ]; then
    echo "ERROR: No puzzles found."
    exit 1
fi

# Build --puzzle-dir args
DIR_ARGS=""
for DIR in $DIRS; do
    DIR_ARGS="$DIR_ARGS ../$DIR"
done

echo ""
echo "=== Running scaling evaluation ==="
echo "  Models:      0.5B, 1.5B, 3B"
echo "  Puzzles:     $TOTAL"
echo "  Samples:     $SAMPLES per puzzle"
echo "  Temperature: $TEMPERATURE"
echo ""

cd inference
python run_scaling_eval.py \
    --puzzle-dir $DIR_ARGS \
    --output-dir ../eval_results/scaling \
    --samples "$SAMPLES" \
    --temperature "$TEMPERATURE"

echo ""
echo "=== Done! ==="
echo "Results: eval_results/scaling/"
echo "  - eval_results/scaling/scaling_summary.json"
echo "  - eval_results/scaling/pass_at_k_scaling.png"
echo "  - eval_results/scaling/rollouts_*.jsonl"
