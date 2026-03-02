#!/usr/bin/env bash
# Main RL Coding Puzzle Evaluation on Lightning AI
# Runs 0.5B, 1.5B, 3B models with full per-model analytics + scaling curves.
#
# Usage:
#   bash run_eval.sh                                    # all puzzles, all models
#   MODELS="1.5B 3B" bash run_eval.sh                   # subset of models
#   DIRS=puzzles_easy bash run_eval.sh                   # easy puzzles only
#
# Requires: GPU instance (L40S 48GB handles all three models sequentially)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

SAMPLES="${SAMPLES:-8}"
TEMPERATURE="${TEMPERATURE:-1.0}"
MODELS="${MODELS:-0.5B 1.5B 3B}"
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

# Build --models args
MODEL_ARGS=""
for M in $MODELS; do
    MODEL_ARGS="$MODEL_ARGS $M"
done

echo ""
echo "=== Running evaluation ==="
echo "  Models:      $MODELS"
echo "  Puzzles:     $TOTAL"
echo "  Samples:     $SAMPLES per puzzle"
echo "  Temperature: $TEMPERATURE"
echo ""

cd inference
python eval.py \
    --puzzle-dir $DIR_ARGS \
    --models $MODEL_ARGS \
    --output-dir ../eval_results \
    --samples "$SAMPLES" \
    --temperature "$TEMPERATURE"

echo ""
echo "=== Done! ==="
echo "Results: eval_results/"
echo "  - eval_results/{0_5B,1_5B,3B}/  (per-model: report, analytics, rollouts)"
echo "  - eval_results/scaling_summary.json"
echo "  - eval_results/pass_at_k_scaling.png"
echo "  - eval_results/combined_analytics.json"
