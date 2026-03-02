#!/usr/bin/env python3
"""RL Coding Puzzle Evaluation Pipeline.

Two modes:

  Default (multi-model scaling):
    Runs 0.5B/1.5B/3B sequentially, produces per-model category breakdown,
    GRPO variance ranking, RL targets, pass@k scaling curves (k=1,2,4,8),
    and cross-model zone migration. This is what run_eval.sh calls.

  --quick (single-model verbose):
    Runs one model with full per-rollout output (vis/hid test counts).
    Good for sanity-checking a single puzzle dir or new puzzle batch.

Usage:
    # Full 3-model evaluation on both puzzle dirs (default, called by run_eval.sh)
    python eval.py

    # Quick sanity check — single model, verbose output
    python eval.py --quick --models 1.5B --puzzle-dir ../puzzles_easy

    # Subset of models
    python eval.py --models 1.5B 3B

    # Re-plot from saved results (skip inference)
    python eval.py --plot-only

    # Re-analyze existing rollouts (skip inference)
    python eval.py --analyze-dir ../eval_results
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from eval_core import (
    Rollout,
    build_rollout,
    compute_views,
    extract_code,
    format_report,
    generate_solutions,
    generate_solutions_batch,
    load_model,
    load_puzzles,
    load_rollouts,
    pass_at_k,
    run_eval as run_eval_verbose,
    save_artifacts,
)

MODEL_REGISTRY = {
    "0.5B": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    "1.5B": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "3B":   "Qwen/Qwen2.5-Coder-3B-Instruct",
}

# Conservative defaults — safe on L40S 48GB with max_new_tokens=2048
DEFAULT_BATCH_PUZZLES = {"0.5B": 8, "1.5B": 4, "3B": 4}

# Tight timeouts — correct solutions finish in <100ms; anything over 3s is broken
DEFAULT_TIMEOUT = {"0.5B": 3, "1.5B": 3, "3B": 5}

K_VALUES = [1, 2, 4, 8]


# ---------------------------------------------------------------------------
# Inference (batched generation + parallel testing)
# ---------------------------------------------------------------------------

def _generate_batch_safe(
    model, tokenizer, batch: list[dict],
    num_samples: int, temperature: float,
) -> list[list[str]]:
    """Try batched generation; fall back to sequential on OOM."""
    import torch

    try:
        return generate_solutions_batch(
            model, tokenizer, batch,
            num_samples=num_samples, temperature=temperature,
        )
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
        torch.cuda.empty_cache()
        print(f"    OOM at batch={len(batch)}, falling back to sequential",
              flush=True)
        results = []
        for puzzle in batch:
            try:
                resps = generate_solutions(
                    model, tokenizer, puzzle["prompt"], puzzle["starter_code"],
                    num_samples=num_samples, temperature=temperature,
                )
            except Exception as e2:
                resps = [f"# ERROR: {e2}"] * num_samples
            results.append(resps)
        return results


def eval_model(
    model_tag: str,
    model_id: str,
    puzzles: list[dict],
    num_samples: int,
    timeout: int,
    temperature: float,
    device: str,
    batch_puzzles: int = 0,
) -> list[Rollout]:
    """Load a model, run batched inference + parallel tests, then free GPU.

    batch_puzzles=0 and timeout=0 mean auto-detect from model_tag.
    """
    import time as _time
    import torch
    from concurrent.futures import ThreadPoolExecutor

    if batch_puzzles <= 0:
        batch_puzzles = DEFAULT_BATCH_PUZZLES.get(model_tag, 4)
    if timeout <= 0:
        timeout = DEFAULT_TIMEOUT.get(model_tag, 10)

    print(f"\n{'='*70}")
    print(f"  MODEL: {model_tag} ({model_id})")
    print(f"  Puzzles: {len(puzzles)}  Samples: {num_samples}  "
          f"Temp: {temperature}  Batch: {batch_puzzles}  Timeout: {timeout}s")
    print(f"{'='*70}\n")

    model, tokenizer = load_model(model_id, device=device)
    all_rollouts: list[Rollout] = []
    total = len(puzzles)
    n_batches = (total + batch_puzzles - 1) // batch_puzzles
    max_workers = min(batch_puzzles * num_samples, 32)

    with ThreadPoolExecutor(max_workers=max_workers) as test_pool:
        for bi, batch_start in enumerate(range(0, total, batch_puzzles)):
            batch = puzzles[batch_start:batch_start + batch_puzzles]
            batch_end = batch_start + len(batch)

            # ── Batched GPU generation ──
            t0 = _time.monotonic()
            try:
                batch_responses = _generate_batch_safe(
                    model, tokenizer, batch,
                    num_samples=num_samples, temperature=temperature,
                )
            except Exception as e:
                print(f"    Generation error: {e}", flush=True)
                batch_responses = [
                    [f"# ERROR: {e}"] * num_samples for _ in batch
                ]
            gen_ms = int((_time.monotonic() - t0) * 1000)

            # ── Submit all tests to thread pool ──
            t0 = _time.monotonic()
            puzzle_futures: list[tuple[dict, list]] = []
            for puzzle, responses in zip(batch, batch_responses):
                codes = [extract_code(raw) for raw in responses]
                futs = [
                    test_pool.submit(build_rollout, puzzle, j, code, timeout)
                    for j, code in enumerate(codes)
                ]
                puzzle_futures.append((puzzle, futs))

            # ── Collect results + report per puzzle ──
            for puzzle, futs in puzzle_futures:
                pid = puzzle["puzzle_id"]
                cat = puzzle.get("category", "")
                label = f"[{cat}] " if cat else ""

                rollouts = [f.result() for f in futs]
                for r in rollouts:
                    r.puzzle_id = pid
                all_rollouts.extend(rollouts)

                passes = sum(1 for r in rollouts if r.passed)
                idx = batch_start + batch.index(puzzle) + 1
                print(f"  [{idx}/{total}] {label}{pid}: "
                      f"{passes}/{num_samples} pass", flush=True)

            test_ms = int((_time.monotonic() - t0) * 1000)
            print(f"    batch {bi+1}/{n_batches}: "
                  f"gen={gen_ms}ms test={test_ms}ms\n", flush=True)

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"  {model_tag} complete. Freed GPU memory.\n")

    return all_rollouts


# ---------------------------------------------------------------------------
# Pass@k scaling
# ---------------------------------------------------------------------------

def compute_pass_at_k(
    rollouts: list[Rollout],
) -> dict[int, float]:
    """Compute average pass@k across puzzles for one model."""
    by_puzzle: dict[str, list[Rollout]] = defaultdict(list)
    for r in rollouts:
        by_puzzle[r.puzzle_id].append(r)

    results: dict[int, float] = {}
    for k in K_VALUES:
        scores: list[float] = []
        for pid, rs in by_puzzle.items():
            n = len(rs)
            c = sum(1 for r in rs if r.passed)
            scores.append(pass_at_k(n, c, k))
        results[k] = sum(scores) / len(scores) if scores else 0.0
    return results


def print_scaling_table(all_results: dict[str, dict[int, float]]):
    """Print pass@k scaling comparison table."""
    model_order = sorted(all_results.keys(),
                         key=lambda t: float(t.replace("B", "")))

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

    if len(model_order) > 1:
        small_tag, best_tag = model_order[0], model_order[-1]
        gap = all_results[best_tag][1] * 100 - all_results[small_tag][1] * 100
        print(f"  Scaling gap ({small_tag} -> {best_tag}) at pass@1: {gap:+.1f}pp")
        target_tag = model_order[1] if len(model_order) > 1 else model_order[0]
        target_p1 = all_results[target_tag][1] * 100
        best_p8 = all_results[best_tag][8] * 100
        if target_p1 < best_p8:
            print(f"  {target_tag} pass@1 ({target_p1:.1f}%) < {best_tag} "
                  f"pass@8 ({best_p8:.1f}%) -> RL can close this gap")
        print()


def plot_scaling_curves(
    all_results: dict[str, dict[int, float]],
    output_dir: str,
    n_puzzles: int = 0,
    num_samples: int = 8,
):
    """Generate the pass@k line graph."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed — skipping plot)")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"0.5B": "#e74c3c", "1.5B": "#3498db", "3B": "#2ecc71"}
    markers = {"0.5B": "o", "1.5B": "s", "3B": "D"}
    model_order = sorted(all_results.keys(),
                         key=lambda t: float(t.replace("B", "")))

    for model_tag in model_order:
        scores = all_results[model_tag]
        ks = sorted(scores.keys())
        vals = [scores[k] * 100 for k in ks]
        ax.plot(
            ks, vals,
            marker=markers.get(model_tag, "o"),
            color=colors.get(model_tag, None),
            linewidth=2.2, markersize=8,
            label=f"Qwen2.5-Coder-{model_tag}",
        )
        for k, v in zip(ks, vals):
            ax.annotate(f"{v:.1f}%", (k, v), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=8.5)

    ax.set_xlabel("k (number of attempts)", fontsize=12)
    ax.set_ylabel("pass@k (%)", fontsize=12)
    title = (f"Pass@k Scaling ({n_puzzles} puzzles, {num_samples} rollouts)"
             if n_puzzles else "Pass@k Scaling")
    ax.set_title(title, fontsize=13)
    ax.set_xticks(K_VALUES)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=100, color="gray", linestyle="--", alpha=0.4, linewidth=1)

    fig.tight_layout()
    path = os.path.join(output_dir, "pass_at_k_scaling.png")
    fig.savefig(path, dpi=150)
    print(f"  Scaling plot -> {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Cross-model zone migration
# ---------------------------------------------------------------------------

def compute_zone_migration(
    per_model_views: dict[str, dict],
) -> dict:
    """Show how each puzzle's zone changes as model size grows.

    Returns a dict with per-puzzle migration paths and summary counts.
    Useful for curriculum design: puzzles that are 'dead' at 0.5B but
    'learning' at 1.5B suggest the right difficulty tier for RL.
    """
    model_order = sorted(per_model_views.keys(),
                         key=lambda t: float(t.replace("B", "")))
    if len(model_order) < 2:
        return {}

    puzzle_zones: dict[str, dict[str, str]] = defaultdict(dict)
    puzzle_p1: dict[str, dict[str, float]] = defaultdict(dict)
    for tag in model_order:
        for ps in per_model_views[tag]["puzzle_summaries"]:
            pid = ps["puzzle_id"]
            puzzle_zones[pid][tag] = ps["zone"]
            puzzle_p1[pid][tag] = ps["pass_at_1"]

    migrations: list[dict] = []
    for pid in sorted(puzzle_zones):
        path = [puzzle_zones[pid].get(t, "?") for t in model_order]
        p1s = {t: round(puzzle_p1[pid].get(t, 0), 3) for t in model_order}
        migrations.append({
            "puzzle_id": pid,
            "zone_path": dict(zip(model_order, path)),
            "pass_at_1": p1s,
            "trajectory": " -> ".join(path),
        })

    trajectory_counts: dict[str, int] = defaultdict(int)
    for m in migrations:
        trajectory_counts[m["trajectory"]] += 1

    return {
        "model_order": model_order,
        "puzzles": migrations,
        "trajectory_counts": dict(sorted(trajectory_counts.items(),
                                         key=lambda x: -x[1])),
    }


def print_zone_migration(migration: dict):
    """Print a human-readable zone migration summary."""
    if not migration:
        return
    tags = migration["model_order"]
    print(f"\n{'='*68}")
    print(f"  ZONE MIGRATION ({' -> '.join(tags)})")
    print(f"{'='*68}")
    for traj, count in migration["trajectory_counts"].items():
        print(f"  {traj:<40s} {count:>3} puzzles")
    print()

    interesting = [p for p in migration["puzzles"]
                   if len(set(p["zone_path"].values())) > 1]
    if interesting:
        print(f"  Puzzles that change zone ({len(interesting)}):")
        for p in interesting[:15]:
            p1_vals = " | ".join(f"{v*100:4.0f}%" for v in p["pass_at_1"].values())
            print(f"    {p['puzzle_id']:<36s} {p['trajectory']:<30s} p@1: {p1_vals}")
        if len(interesting) > 15:
            print(f"    ... and {len(interesting) - 15} more")
    print()


# ---------------------------------------------------------------------------
# Puzzle loading (multi-dir)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Combined analytics
# ---------------------------------------------------------------------------

def build_combined_analytics(
    per_model_views: dict[str, dict],
    scaling_results: dict[str, dict[int, float]],
    zone_migration: dict,
    n_puzzles: int,
    num_samples: int,
) -> dict:
    """Build comprehensive cross-model analytics JSON."""
    analytics: dict = {
        "meta": {
            "models": sorted(per_model_views.keys(),
                             key=lambda t: float(t.replace("B", ""))),
            "n_puzzles": n_puzzles,
            "num_samples": num_samples,
            "k_values": K_VALUES,
            "timestamp": datetime.now().isoformat(),
        },
        "scaling": {
            tag: {str(k): round(v, 4) for k, v in scores.items()}
            for tag, scores in scaling_results.items()
        },
        "per_model": {},
    }

    for model_tag, views in per_model_views.items():
        ps = views["puzzle_summaries"]
        n = len(ps)
        zones = views.get("advantage_spread", {}).get("zones", {})

        analytics["per_model"][model_tag] = {
            "overall": {
                "pass_at_1": round(sum(p["pass_at_1"] for p in ps) / n, 4) if n else 0,
                "mean_test_pass_rate": round(
                    sum(p["mean_test_pass_rate"] for p in ps) / n, 4) if n else 0,
                "mean_grpo_variance": round(
                    sum(p["advantage_variance"] for p in ps) / n, 4) if n else 0,
                "zones": dict(zones),
            },
            "categories": views.get("category", {}),
            "difficulty": views.get("difficulty", {}),
            "rl_targets": [t for t in views["rl_targets"] if t["priority"] <= 2],
            "per_puzzle": views["puzzle_summaries"],
        }

    if zone_migration:
        analytics["zone_migration"] = zone_migration

    return analytics


# ---------------------------------------------------------------------------
# Save all outputs
# ---------------------------------------------------------------------------

def save_all(
    per_model_rollouts: dict[str, list[Rollout]],
    per_model_views: dict[str, dict],
    scaling_results: dict[str, dict[int, float]],
    zone_migration: dict,
    n_puzzles: int,
    num_samples: int,
    output_dir: str,
):
    """Save per-model artifacts, scaling data, and combined analytics."""
    os.makedirs(output_dir, exist_ok=True)

    # Per-model artifacts
    for model_tag in per_model_rollouts:
        model_id = MODEL_REGISTRY.get(model_tag, model_tag)
        model_dir = os.path.join(output_dir, model_tag.replace(".", "_"))
        views = per_model_views[model_tag]
        report = format_report(views, model_id, num_samples)
        print(f"\n  Saving {model_tag} artifacts -> {model_dir}/")
        save_artifacts(
            per_model_rollouts[model_tag], views, model_dir,
            model=model_id, report=report,
        )

    # Scaling summary
    scaling_data = {
        "meta": {
            "models": sorted(scaling_results.keys(),
                             key=lambda t: float(t.replace("B", ""))),
            "k_values": K_VALUES,
            "n_puzzles": n_puzzles,
            "num_samples": num_samples,
            "timestamp": datetime.now().isoformat(),
        },
        "pass_at_k": {
            tag: {str(k): round(v, 4) for k, v in scores.items()}
            for tag, scores in scaling_results.items()
        },
    }
    path = os.path.join(output_dir, "scaling_summary.json")
    with open(path, "w") as f:
        json.dump(scaling_data, f, indent=2)
    print(f"\n  scaling_summary.json -> {path}")

    # Plot
    plot_scaling_curves(scaling_results, output_dir,
                        n_puzzles=n_puzzles, num_samples=num_samples)

    # Combined analytics
    combined = build_combined_analytics(
        per_model_views, scaling_results, zone_migration,
        n_puzzles, num_samples,
    )
    path = os.path.join(output_dir, "combined_analytics.json")
    with open(path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"  combined_analytics.json -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="RL Coding Puzzle Evaluation — multi-model scaling "
                    "with category analysis, GRPO variance, and RL targets"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Single-model verbose mode: prints vis/hid test counts per rollout. "
             "Good for sanity-checking a new puzzle batch. Uses --models[0].")
    parser.add_argument(
        "--puzzle-dir", nargs="+", default=None,
        help="One or more puzzle directories "
             "(default: ../puzzles_easy ../puzzles_humaneval)")
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: ../eval_results)")
    parser.add_argument(
        "--models", nargs="+", default=list(MODEL_REGISTRY.keys()),
        choices=list(MODEL_REGISTRY.keys()),
        help="Model sizes to evaluate (default: 0.5B 1.5B 3B)")
    parser.add_argument("--samples", type=int, default=8,
                        help="Rollouts per puzzle (default: 8)")
    parser.add_argument("--timeout", type=int, default=0,
                        help="Test timeout seconds (0=auto: 3s for 0.5B/1.5B, "
                             "5s for 3B)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (default: 1.0)")
    parser.add_argument("--batch-puzzles", type=int, default=0,
                        help="Puzzles per GPU batch (0=auto: 8 for 0.5B, "
                             "4 for 1.5B/3B)")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--plot-only", action="store_true",
        help="Re-plot from saved scaling_summary.json (skip inference)")
    parser.add_argument(
        "--analyze-dir", metavar="DIR",
        help="Re-analyze from saved rollout files in DIR (skip inference)")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    puzzle_dirs = args.puzzle_dir or [
        str(script_dir.parent / "puzzles_easy"),
        str(script_dir.parent / "puzzles_humaneval"),
    ]
    output_dir = args.output_dir or str(script_dir.parent / "eval_results")

    # --quick: single-model, verbose per-rollout output (vis/hid counts)
    if args.quick:
        model_tag = args.models[0]
        model_id = MODEL_REGISTRY[model_tag]
        timeout = args.timeout or DEFAULT_TIMEOUT.get(model_tag, 5)
        puzzles = load_puzzles_multi(puzzle_dirs)
        if not puzzles:
            print(f"No puzzles found in {puzzle_dirs}")
            sys.exit(1)
        print(f"[quick] {len(puzzles)} puzzles | {model_id} | "
              f"samples={args.samples} temp={args.temperature} timeout={timeout}s\n")
        model, tokenizer = load_model(model_id, device=args.device)
        rollouts = run_eval_verbose(
            puzzles, model, tokenizer,
            num_samples=args.samples, timeout=timeout,
            temperature=args.temperature,
        )
        del model, tokenizer
        views = compute_views(rollouts, args.samples)
        report = format_report(views, model_id, args.samples)
        print(report)
        quick_dir = os.path.join(output_dir, "quick")
        print(f"\nSaving artifacts to {quick_dir}/")
        save_artifacts(rollouts, views, quick_dir, model=model_id, report=report)
        print("Done.")
        return

    # --plot-only: just regenerate the chart from scaling_summary.json
    if args.plot_only:
        print("Loading saved scaling results...")
        path = os.path.join(output_dir, "scaling_summary.json")
        with open(path) as f:
            data = json.load(f)
        all_scaling = {
            tag: {int(k): v for k, v in scores.items()}
            for tag, scores in data["pass_at_k"].items()
        }
        print_scaling_table(all_scaling)
        plot_scaling_curves(
            all_scaling, output_dir,
            n_puzzles=data["meta"].get("n_puzzles", 0),
            num_samples=data["meta"].get("num_samples", 8),
        )
        return

    # --analyze-dir: reload rollouts, recompute all analytics (no inference)
    if args.analyze_dir:
        print(f"Re-analyzing rollouts from {args.analyze_dir}/ ...")
        per_model_rollouts: dict[str, list[Rollout]] = {}
        per_model_views: dict[str, dict] = {}
        all_scaling: dict[str, dict[int, float]] = {}

        for sub in sorted(Path(args.analyze_dir).iterdir()):
            rollout_file = sub / "raw_rollouts.jsonl"
            if not rollout_file.exists():
                continue
            tag = sub.name.replace("_", ".")
            if tag.endswith("B"):
                tag = tag  # e.g. "1_5B" -> "1.5B"
            rollouts = load_rollouts(str(rollout_file))
            n_per = len(set(r.puzzle_num for r in rollouts))
            k = len(rollouts) // n_per if n_per else args.samples
            print(f"  {tag}: {len(rollouts)} rollouts, {n_per} puzzles")

            per_model_rollouts[tag] = rollouts
            per_model_views[tag] = compute_views(rollouts, k)
            all_scaling[tag] = compute_pass_at_k(rollouts)

        n_puzzles = max(
            (len(v["puzzle_summaries"]) for v in per_model_views.values()),
            default=0,
        )
        zone_migration = compute_zone_migration(per_model_views)

        for tag, views in per_model_views.items():
            model_id = MODEL_REGISTRY.get(tag, tag)
            print(format_report(views, model_id, args.samples))

        print_scaling_table(all_scaling)
        print_zone_migration(zone_migration)

        save_all(
            per_model_rollouts, per_model_views, all_scaling,
            zone_migration, n_puzzles, args.samples, output_dir,
        )
        print("\nDone (re-analysis).")
        return

    # ── Normal run: inference + analytics ──
    puzzles = load_puzzles_multi(puzzle_dirs)
    if not puzzles:
        print(f"No puzzles found in {puzzle_dirs}")
        sys.exit(1)

    print(f"Loaded {len(puzzles)} puzzles from {', '.join(puzzle_dirs)}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Samples: {args.samples}  |  Temp: {args.temperature}  "
          f"|  Timeout: {args.timeout}s")

    per_model_rollouts: dict[str, list[Rollout]] = {}
    per_model_views: dict[str, dict] = {}
    all_scaling: dict[str, dict[int, float]] = {}

    for model_tag in args.models:
        model_id = MODEL_REGISTRY[model_tag]

        rollouts = eval_model(
            model_tag, model_id, puzzles,
            num_samples=args.samples, timeout=args.timeout,
            temperature=args.temperature, device=args.device,
            batch_puzzles=args.batch_puzzles,
        )
        per_model_rollouts[model_tag] = rollouts

        views = compute_views(rollouts, args.samples)
        per_model_views[model_tag] = views

        report = format_report(views, model_id, args.samples)
        print(report)

        all_scaling[model_tag] = compute_pass_at_k(rollouts)

    # Scaling comparison (only interesting with 2+ models)
    if len(args.models) > 1:
        print_scaling_table(all_scaling)

    zone_migration = compute_zone_migration(per_model_views)
    if zone_migration:
        print_zone_migration(zone_migration)

    # Save everything
    print(f"\nSaving all artifacts to {output_dir}/")
    save_all(
        per_model_rollouts, per_model_views, all_scaling,
        zone_migration, len(puzzles), args.samples, output_dir,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
