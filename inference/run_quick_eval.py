#!/usr/bin/env python3
"""Quick single-model evaluation for sanity-checking puzzles.

Runs one model against one puzzle directory and prints a detailed report.
For comprehensive multi-model evaluation with scaling curves and cross-model
analytics, use run_eval.py instead.

Usage:
    python run_quick_eval.py --puzzle-dir ../puzzles_easy
    python run_quick_eval.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct --samples 4
    python run_quick_eval.py --analyze raw_rollouts.jsonl
"""

from eval_core import main

if __name__ == "__main__":
    main()
