#!/usr/bin/env python3
"""Multi-model pass@k scaling evaluation.

Runs 0.5B, 1.5B, and 3B Qwen2.5-Coder models against puzzle sets,
computes pass@k for k=1,2,4,8, and plots scaling curves to verify headroom
for RL training.

Accepts one or more puzzle directories so you can combine easy + humaneval.

Uses HuggingFace transformers (no vLLM).

Usage:
    # Full run — single dir
    python run_scaling_eval.py --puzzle-dir ../puzzles_easy

    # Full run — combine easy + humaneval (100 puzzles)
    python run_scaling_eval.py --puzzle-dir ../puzzles_easy ../puzzles_humaneval

    # Re-plot from saved results (skip inference)
    python run_scaling_eval.py --plot-only

    # Run a subset of models
    python run_scaling_eval.py --models 0.5B 1.5B
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from run_eval import (
    Rollout,
    build_rollout,
    extract_code,
    generate_solutions,
    load_model,
    load_puzzles,
    pass_at_k,
)

MODEL_REGISTRY = {
    "0.5B": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    "1.5B": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "3B":   "Qwen/Qwen2.5-Coder-3B-Instruct",
}

K_VALUES = [1, 2, 4, 8]


def eval_model(
    model_tag: str,
    model_id: str,
    puzzles: list[dict],
    num_samples: int,
    timeout: int,
    temperature: float,
    device: str,
) -> list[Rollout]:
    """Load a model, run inference on all puzzles, return rollouts, then free GPU."""
    import torch

    print(f"\n{'='*70}")
    print(f"  MODEL: {model_tag} ({model_id})")
    print(f"  Puzzles: {len(puzzles)}  Samples: {num_samples}  Temp: {temperature}")
    print(f"{'='*70}\n")

    model, tokenizer = load_model(model_id, device=device)
    all_rollouts: list[Rollout] = []

    for i, puzzle in enumerate(puzzles):
        pid = puzzle["puzzle_id"]
        cat = puzzle.get("category", "")
        label = f"[{cat}]" if cat else ""
        print(f"  [{i+1}/{len(puzzles)}] {pid} {label}", flush=True)

        try:
            raw_responses = generate_solutions(
                model, tokenizer, puzzle["prompt"], puzzle["starter_code"],
                num_samples=num_samples, temperature=temperature,
            )
        except Exception as e:
            print(f"    Generation error: {e}", flush=True)
            raw_responses = [f"# GENERATION ERROR: {e}"] * num_samples

        passes = 0
        for j, raw in enumerate(raw_responses):
            code = extract_code(raw)
            rollout = build_rollout(puzzle, j, code, timeout)
            rollout.puzzle_id = pid
            all_rollouts.append(rollout)
            if rollout.passed:
                passes += 1
            tag = "PASS" if rollout.passed else rollout.error_type
            print(f"    [{j+1}/{num_samples}] {tag}", flush=True)

        print(f"    -> {passes}/{num_samples} pass\n", flush=True)

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"  {model_tag} complete. Freed GPU memory.\n")

    return all_rollouts


def compute_pass_at_k_per_model(
    rollouts: list[Rollout],
    num_samples: int,
) -> dict[int, float]:
    """Given all rollouts for one model, compute average pass@k across puzzles."""
    from collections import defaultdict

    by_puzzle: dict[str, list[Rollout]] = defaultdict(list)
    for r in rollouts:
        by_puzzle[r.puzzle_id].append(r)

    results: dict[int, float] = {}
    for k in K_VALUES:
        per_puzzle_scores: list[float] = []
        for pid, rs in by_puzzle.items():
            n = len(rs)
            c = sum(1 for r in rs if r.passed)
            per_puzzle_scores.append(pass_at_k(n, c, k))
        results[k] = sum(per_puzzle_scores) / len(per_puzzle_scores) if per_puzzle_scores else 0.0

    return results


def load_puzzles_multi(dirs: list[str]) -> list[dict]:
    """Load puzzles from one or more directories, deduplicating by puzzle_id."""
    all_puzzles: list[dict] = []
    seen_ids: set[str] = set()
    for d in dirs:
        for p in load_puzzles(d):
            pid = p["puzzle_id"]
            if pid not in seen_ids:
                seen_ids.add(pid)
                all_puzzles.append(p)
    all_puzzles.sort(key=lambda p: (p.get("difficulty", ""), p["puzzle_id"]))
    return all_puzzles


def save_results(
    all_results: dict[str, dict],
    all_rollouts: dict[str, list[Rollout]],
    output_dir: str,
    n_puzzles: int = 0,
    num_samples: int = 8,
):
    """Save raw rollouts and aggregated pass@k to disk."""
    os.makedirs(output_dir, exist_ok=True)

    for model_tag, rollouts in all_rollouts.items():
        path = os.path.join(output_dir, f"rollouts_{model_tag}.jsonl")
        with open(path, "w") as f:
            for r in rollouts:
                f.write(json.dumps(r.to_jsonl()) + "\n")

    summary = {
        "meta": {
            "models": list(all_results.keys()),
            "k_values": K_VALUES,
            "n_puzzles": n_puzzles,
            "num_samples": num_samples,
            "timestamp": datetime.now().isoformat(),
        },
        "pass_at_k": {
            model_tag: {str(k): round(v, 4) for k, v in scores.items()}
            for model_tag, scores in all_results.items()
        },
    }
    path = os.path.join(output_dir, "scaling_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {path}")


def load_saved_results(output_dir: str) -> tuple[dict[str, dict[int, float]], dict]:
    """Load pass@k results and metadata from a previous run."""
    path = os.path.join(output_dir, "scaling_summary.json")
    with open(path) as f:
        data = json.load(f)
    results: dict[str, dict[int, float]] = {}
    for model_tag, scores in data["pass_at_k"].items():
        results[model_tag] = {int(k): v for k, v in scores.items()}
    return results, data.get("meta", {})


def plot_scaling_curves(
    all_results: dict[str, dict[int, float]],
    output_dir: str,
    n_puzzles: int = 0,
    num_samples: int = 8,
):
    """Generate the pass@k line graph."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {"0.5B": "#e74c3c", "1.5B": "#3498db", "3B": "#2ecc71"}
    markers = {"0.5B": "o", "1.5B": "s", "3B": "D"}

    model_order = sorted(all_results.keys(), key=lambda t: float(t.replace("B", "")))

    for model_tag in model_order:
        scores = all_results[model_tag]
        ks = sorted(scores.keys())
        vals = [scores[k] * 100 for k in ks]
        ax.plot(
            ks, vals,
            marker=markers.get(model_tag, "o"),
            color=colors.get(model_tag, None),
            linewidth=2.2,
            markersize=8,
            label=f"Qwen2.5-Coder-{model_tag}",
        )
        for k, v in zip(ks, vals):
            ax.annotate(f"{v:.1f}%", (k, v), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=8.5)

    ax.set_xlabel("k (number of attempts)", fontsize=12)
    ax.set_ylabel("pass@k (%)", fontsize=12)
    title = f"Pass@k Scaling ({n_puzzles} puzzles, {num_samples} rollouts)" if n_puzzles else "Pass@k Scaling"
    ax.set_title(title, fontsize=13)
    ax.set_xticks(K_VALUES)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3)

    ax.axhline(y=100, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    ax.text(K_VALUES[-1], 101.5, "ceiling", ha="right", va="bottom",
            fontsize=8, color="gray", alpha=0.6)

    fig.tight_layout()
    path = os.path.join(output_dir, "pass_at_k_scaling.png")
    fig.savefig(path, dpi=150)
    print(f"Plot saved to {path}")
    plt.close(fig)


def print_table(all_results: dict[str, dict[int, float]]):
    """Print a concise results table."""
    model_order = sorted(all_results.keys(), key=lambda t: float(t.replace("B", "")))

    header = f"{'Model':<28}" + "".join(f"{'pass@'+str(k):>10}" for k in K_VALUES)
    print(f"\n{'='*68}")
    print(f"  PASS@K SCALING RESULTS")
    print(f"{'='*68}")
    print(f"  {header}")
    print(f"  {'-'*64}")
    for tag in model_order:
        scores = all_results[tag]
        row = f"  Qwen2.5-Coder-{tag:<12}"
        row += "".join(f"{scores[k]*100:9.1f}%" for k in K_VALUES)
        print(row)
    print(f"{'='*68}\n")

    best_tag = model_order[-1]
    best_p8 = all_results[best_tag][8] * 100
    target_tag = model_order[1] if len(model_order) > 1 else model_order[0]
    target_p1 = all_results[target_tag][1] * 100

    print("  Interpretation:")
    if best_p8 < 100:
        print(f"    - {best_tag} pass@8 = {best_p8:.1f}% (< 100%) -> headroom exists")
    else:
        print(f"    - {best_tag} pass@8 = 100% -> dataset may be too easy for larger models")

    if len(model_order) > 1:
        small_tag = model_order[0]
        small_p1 = all_results[small_tag][1] * 100
        gap = all_results[best_tag][1] * 100 - small_p1
        print(f"    - {small_tag} -> {best_tag} gap at pass@1: {gap:+.1f}pp -> "
              f"{'good' if gap > 10 else 'small'} scaling signal")

    if target_p1 < best_p8:
        print(f"    - {target_tag} pass@1 ({target_p1:.1f}%) < {best_tag} pass@8 ({best_p8:.1f}%) "
              f"-> RL can close this gap")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Multi-model pass@k scaling evaluation for RL headroom analysis"
    )
    parser.add_argument("--puzzle-dir", nargs="+", default=None,
                        help="One or more puzzle directories (default: ../puzzles_easy)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: ../eval_results/scaling)")
    parser.add_argument("--models", nargs="+", default=list(MODEL_REGISTRY.keys()),
                        choices=list(MODEL_REGISTRY.keys()),
                        help="Which model sizes to evaluate (default: all)")
    parser.add_argument("--samples", type=int, default=8,
                        help="Rollouts per puzzle (default: 8)")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (default: 1.0 for max diversity)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip inference; re-plot from saved scaling_summary.json")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    puzzle_dirs = args.puzzle_dir or [str(script_dir.parent / "puzzles_easy")]
    output_dir = args.output_dir or str(script_dir.parent / "eval_results" / "scaling")

    if args.plot_only:
        print("Loading saved results...")
        all_results, meta = load_saved_results(output_dir)
        print_table(all_results)
        plot_scaling_curves(
            all_results, output_dir,
            n_puzzles=meta.get("n_puzzles", 0),
            num_samples=meta.get("num_samples", 8),
        )
        return

    puzzles = load_puzzles_multi(puzzle_dirs)
    if not puzzles:
        print(f"No puzzles found in {puzzle_dirs}")
        sys.exit(1)
    print(f"Loaded {len(puzzles)} puzzles from {', '.join(puzzle_dirs)}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Samples: {args.samples}  |  Temperature: {args.temperature}  |  Timeout: {args.timeout}s")

    all_results: dict[str, dict[int, float]] = {}
    all_rollouts: dict[str, list[Rollout]] = {}

    for model_tag in args.models:
        model_id = MODEL_REGISTRY[model_tag]
        rollouts = eval_model(
            model_tag, model_id, puzzles,
            num_samples=args.samples,
            timeout=args.timeout,
            temperature=args.temperature,
            device=args.device,
        )
        all_rollouts[model_tag] = rollouts
        all_results[model_tag] = compute_pass_at_k_per_model(rollouts, args.samples)

    save_results(all_results, all_rollouts, output_dir,
                 n_puzzles=len(puzzles), num_samples=args.samples)
    print_table(all_results)
    plot_scaling_curves(all_results, output_dir,
                        n_puzzles=len(puzzles), num_samples=args.samples)
    print("Done.")


if __name__ == "__main__":
    main()
