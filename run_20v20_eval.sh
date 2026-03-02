#!/usr/bin/env bash
# 20-vs-20 eval: easy puzzles (6 categories) vs curated HumanEval+ on Lightning AI
# Usage: bash run_20v20_eval.sh
# Requires: GPU instance (L4 24GB is sufficient for 1.5B)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

MODEL="${MODEL:-Qwen/Qwen2.5-Coder-1.5B-Instruct}"
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

if [ "$EASY_COUNT" -lt 20 ]; then
    echo "ERROR: Only $EASY_COUNT easy puzzles found (need 20)."
    echo "  Wait for generation to finish or run:"
    echo "  python generate_easy_puzzles.py --count 20"
    exit 1
fi

if [ "$HE_COUNT" -lt 20 ]; then
    echo "ERROR: Only $HE_COUNT HumanEval+ puzzles found (need 20)."
    echo "  Run: python prepare_humaneval.py --curated20"
    exit 1
fi

echo ""
echo "=== Running 20v20 comparison eval ==="
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
echo "  - eval_results/easy/          (per-puzzle results + category breakdown)"
echo "  - eval_results/humaneval/     (per-puzzle results)"
echo "  - eval_results/comparison_report.txt"
