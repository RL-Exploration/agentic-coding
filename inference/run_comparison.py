#!/usr/bin/env python3
"""Side-by-side evaluation: easy puzzles vs HumanEval+ subset.

Runs run_eval.py against both puzzle sets, then produces a comparison report
showing which set (or blend) is best suited for RL training with Qwen 1.5-3B.

Usage:
    # Full comparison (inference + tests + report)
    python run_comparison.py

    # Custom model / samples
    python run_comparison.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct --samples 5

    # Re-analyze existing results (skip inference)
    python run_comparison.py --analyze-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from run_eval import (
    Rollout,
    compute_views,
    format_report,
    load_model,
    load_puzzles,
    load_rollouts,
    pass_at_k,
    run_eval,
    save_artifacts,
)


def run_single_eval(
    label: str,
    puzzle_dir: str,
    output_dir: str,
    model,
    tokenizer,
    model_name: str,
    num_samples: int,
    timeout: int,
    temperature: float,
) -> tuple[list[Rollout], dict]:
    """Run eval on one puzzle set and save artifacts. Returns (rollouts, views)."""
    puzzles = load_puzzles(puzzle_dir)
    if not puzzles:
        print(f"No puzzles found in {puzzle_dir}")
        return [], {}

    print(f"\n{'='*80}")
    print(f"  EVALUATING: {label} ({len(puzzles)} puzzles)")
    print(f"{'='*80}\n")

    rollouts = run_eval(
        puzzles, model, tokenizer,
        num_samples=num_samples, timeout=timeout, temperature=temperature,
    )
    views = compute_views(rollouts, num_samples)

    report = format_report(views, model_name, num_samples)
    print(report)

    print(f"\nSaving {label} artifacts to {output_dir}/")
    save_artifacts(rollouts, views, output_dir)

    report_path = os.path.join(output_dir, "report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  report.txt                  -> {report_path}")

    return rollouts, views


def load_existing_results(output_dir: str, k: int) -> tuple[list[Rollout], dict]:
    """Load rollouts from a previous run and recompute views."""
    rollouts_path = os.path.join(output_dir, "raw_rollouts.jsonl")
    if not os.path.exists(rollouts_path):
        return [], {}
    rollouts = load_rollouts(rollouts_path)
    views = compute_views(rollouts, k)
    return rollouts, views


def comparison_report(
    easy_views: dict,
    humaneval_views: dict,
    model_name: str,
    k: int,
) -> str:
    """Produce a side-by-side comparison report."""
    lines: list[str] = []
    W = 80

    def section(title: str):
        lines.append("")
        lines.append("=" * W)
        lines.append(f"  {title}")
        lines.append("=" * W)

    lines.append("#" * W)
    lines.append(f"{'COMPARISON REPORT: Easy Puzzles vs HumanEval+':^{W}}")
    lines.append("#" * W)
    lines.append(f"  Model:          {model_name}")
    lines.append(f"  Samples/puzzle: {k}")
    lines.append(f"  Timestamp:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("#" * W)

    # ── Overall pass rates ──
    section("OVERALL PASS RATES")
    pk_key = f"pass_at_{k}"

    for label, views in [("Easy Puzzles", easy_views), ("HumanEval+", humaneval_views)]:
        ps = views.get("puzzle_summaries", [])
        n = len(ps)
        if n == 0:
            lines.append(f"\n  {label}: NO DATA")
            continue
        avg_p1 = sum(p["pass_at_1"] for p in ps) / n
        solved = sum(1 for p in ps if p.get(pk_key, 0) > 0)
        avg_test = sum(p["mean_test_pass_rate"] for p in ps) / n

        lines.append(f"\n  {label} ({n} puzzles):")
        lines.append(f"    pass@1:              {avg_p1*100:5.1f}%")
        lines.append(f"    Solved (pass@{k}>0):  {solved}/{n} ({solved/n*100:.0f}%)")
        lines.append(f"    Mean test pass rate: {avg_test*100:5.1f}%")

    # ── RL Zone comparison ──
    section("RL ZONE DISTRIBUTION")
    lines.append(f"\n  {'Zone':<14} {'Easy Puzzles':>14} {'HumanEval+':>14}")
    lines.append("  " + "-" * 44)

    for label, views in [("easy", easy_views), ("humaneval", humaneval_views)]:
        views.setdefault("advantage_spread", {"zones": {"dead": 0, "learning": 0, "saturated": 0}})

    ez = easy_views.get("advantage_spread", {}).get("zones", {})
    he = humaneval_views.get("advantage_spread", {}).get("zones", {})
    ez_total = sum(ez.values()) or 1
    he_total = sum(he.values()) or 1

    for zone in ["learning", "dead", "saturated"]:
        ez_count = ez.get(zone, 0)
        he_count = he.get(zone, 0)
        lines.append(
            f"  {zone:<14} "
            f"{ez_count:>4}/{ez_total} ({ez_count/ez_total*100:4.0f}%)   "
            f"{he_count:>4}/{he_total} ({he_count/he_total*100:4.0f}%)"
        )

    # ── Error distribution comparison ──
    section("ERROR TYPE COMPARISON")
    ez_err = easy_views.get("error_distribution", {})
    he_err = humaneval_views.get("error_distribution", {})
    all_errors = sorted(set(list(ez_err.keys()) + list(he_err.keys())))

    lines.append(f"\n  {'Error Type':<18} {'Easy Puzzles':>14} {'HumanEval+':>14}")
    lines.append("  " + "-" * 48)
    for etype in all_errors:
        ez_rate = ez_err.get(etype, {}).get("rate", 0)
        he_rate = he_err.get(etype, {}).get("rate", 0)
        elabel = etype.replace("_", " ").title()
        lines.append(f"  {elabel:<18} {ez_rate*100:12.1f}%  {he_rate*100:12.1f}%")

    # ── Top GRPO targets from each set ──
    section("TOP 10 RL TARGETS (by advantage variance)")
    for label, views in [("Easy Puzzles", easy_views), ("HumanEval+", humaneval_views)]:
        ranked = views.get("advantage_spread", {}).get("ranked", [])
        learning = [p for p in ranked if p["zone"] == "learning"]
        lines.append(f"\n  {label}:")
        if not learning:
            lines.append("    (No puzzles in learning zone)")
            continue
        for j, p in enumerate(learning[:10]):
            lines.append(
                f"    {j+1:>2}. {p['puzzle_id']:<36} "
                f"var={p['score_variance']:.3f}  "
                f"p@1={p['pass_at_1']*100:4.0f}%"
            )

    # ── Recommendation ──
    section("RECOMMENDATION")
    ez_learning = ez.get("learning", 0)
    he_learning = he.get("learning", 0)
    ez_dead = ez.get("dead", 0)
    he_dead = he.get("dead", 0)

    if ez_learning > he_learning and ez_dead < he_dead:
        lines.append(
            "\n  EASY PUZZLES are the better RL training set for this model size."
        )
        lines.append(
            "  More puzzles in the learning zone = more GRPO signal."
        )
    elif he_learning > ez_learning:
        lines.append(
            "\n  HUMANEVAL+ is the better RL training set for this model size."
        )
        lines.append(
            "  More puzzles in the learning zone = more GRPO signal."
        )
    else:
        lines.append(
            "\n  BLEND both sets. Select learning-zone puzzles from each to"
        )
        lines.append(
            "  maximize advantage variance for GRPO training."
        )

    lines.append("")
    lines.append(
        f"  Easy: {ez_learning}/{ez_total} learning, {ez_dead}/{ez_total} dead, "
        f"{ez.get('saturated', 0)}/{ez_total} saturated"
    )
    lines.append(
        f"  HumanEval+: {he_learning}/{he_total} learning, {he_dead}/{he_total} dead, "
        f"{he.get('saturated', 0)}/{he_total} saturated"
    )
    lines.append("")
    lines.append("#" * W)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Compare easy puzzles vs HumanEval+ for RL training"
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct",
                        help="HuggingFace model ID (default: Qwen/Qwen2.5-Coder-1.5B-Instruct)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--analyze-only", action="store_true",
                        help="Skip inference, re-analyze existing results")

    parser.add_argument("--easy-dir", default=None,
                        help="Easy puzzle directory (default: ../puzzles_easy)")
    parser.add_argument("--humaneval-dir", default=None,
                        help="HumanEval+ puzzle directory (default: ../puzzles_humaneval)")
    parser.add_argument("--output-dir", default=None,
                        help="Base output directory (default: ../eval_results)")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    easy_dir = args.easy_dir or str(script_dir.parent / "puzzles_easy")
    humaneval_dir = args.humaneval_dir or str(script_dir.parent / "puzzles_humaneval")
    base_output = args.output_dir or str(script_dir.parent / "eval_results")
    easy_output = os.path.join(base_output, "easy")
    humaneval_output = os.path.join(base_output, "humaneval")
    k = args.samples

    if args.analyze_only:
        print("Re-analyzing existing results...")
        easy_rollouts, easy_views = load_existing_results(easy_output, k)
        humaneval_rollouts, humaneval_views = load_existing_results(humaneval_output, k)
        if not easy_rollouts and not humaneval_rollouts:
            print("No existing results found. Run without --analyze-only first.")
            sys.exit(1)
    else:
        for d in [easy_dir, humaneval_dir]:
            if not os.path.isdir(d) or not list(Path(d).glob("*.json")):
                print(f"Puzzle directory missing or empty: {d}")
                print("Run generate_easy_puzzles.py and/or prepare_humaneval.py first.")
                sys.exit(1)

        model, tokenizer = load_model(args.model, device=args.device)

        easy_rollouts, easy_views = run_single_eval(
            "Easy Puzzles", easy_dir, easy_output,
            model, tokenizer, args.model, k, args.timeout, args.temperature,
        )
        humaneval_rollouts, humaneval_views = run_single_eval(
            "HumanEval+", humaneval_dir, humaneval_output,
            model, tokenizer, args.model, k, args.timeout, args.temperature,
        )

    report = comparison_report(easy_views, humaneval_views, args.model, k)
    print(report)

    os.makedirs(base_output, exist_ok=True)
    report_path = os.path.join(base_output, "comparison_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nComparison report saved to {report_path}")


if __name__ == "__main__":
    main()
