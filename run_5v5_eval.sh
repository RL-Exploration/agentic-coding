#!/usr/bin/env bash
# Quick 5-vs-5 eval: easy puzzles vs HumanEval+ on Lightning AI
# Usage: bash run_5v5_eval.sh
# Requires: GPU instance (L4 24GB is sufficient for 3B)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

MODEL="${MODEL:-Qwen/Qwen2.5-Coder-3B-Instruct}"
SAMPLES="${SAMPLES:-8}"

echo "=== Installing dependencies ==="
pip install -q -r requirements.txt
pip install -q -r inference/requirements.txt

echo ""
echo "=== Verifying puzzle sets ==="
EASY_COUNT=$(ls puzzles_easy/*.json 2>/dev/null | wc -l | tr -d ' ')
HE_COUNT=$(ls puzzles_humaneval/*.json 2>/dev/null | wc -l | tr -d ' ')
echo "  Easy puzzles:      $EASY_COUNT"
echo "  HumanEval+ puzzles: $HE_COUNT"

if [ "$EASY_COUNT" -eq 0 ] || [ "$HE_COUNT" -eq 0 ]; then
    echo "ERROR: Missing puzzle files. Generate them first:"
    echo "  python generate_easy_puzzles.py --count 5"
    echo "  python prepare_humaneval.py --count 5"
    exit 1
fi

echo ""
echo "=== Running comparison eval ==="
echo "  Model:   $MODEL"
echo "  Samples: $SAMPLES per puzzle"
echo ""

cd inference
python run_comparison.py \
    --model "$MODEL" \
    --samples "$SAMPLES" \
    --easy-dir ../puzzles_easy \
    --humaneval-dir ../puzzles_humaneval \
    --output-dir ../eval_results

echo ""
echo "=== Done! ==="
echo "Results: eval_results/"
echo "  - eval_results/easy/          (per-puzzle results)"
echo "  - eval_results/humaneval/     (per-puzzle results)"
echo "  - eval_results/comparison_report.txt"
