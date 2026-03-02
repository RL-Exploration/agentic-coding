#!/usr/bin/env python3
"""RL Coding Puzzle Evaluation Pipeline.

Single inference run (Run 1 from eval_plan.md): generate k rollouts per puzzle,
execute against unit tests, then compute all aggregation views offline.

Uses HuggingFace transformers model.generate() directly (no vLLM dependency).

Usage:
    # Full run (inference + tests + analytics)
    python run_eval.py

    # Re-run analytics on existing results
    python run_eval.py --analyze eval_results/raw_rollouts.jsonl

    # Custom sampling
    python run_eval.py --samples 8 --timeout 30
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a Python programming assistant. Complete the given function to solve "
    "the problem. Output ONLY the complete Python function implementation. "
    "Include any necessary imports above the function. Do not include test code, "
    "explanations, or markdown formatting."
)

# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------

_HARNESS = '''\
import json, unittest, sys, time as _time

# ========== SOLUTION ==========
{solution_code}

# ========== TESTS ==========
{test_code}

# ========== RUNNER ==========
class _R(unittest.TestResult):
    def __init__(self):
        super().__init__()
        self.d = []
    def addSuccess(self, t):
        super().addSuccess(t)
        self.d.append({{"n": t.id().split(".")[-1], "s": "pass"}})
    def addFailure(self, t, err):
        super().addFailure(t, err)
        self.d.append({{"n": t.id().split(".")[-1], "s": "fail"}})
    def addError(self, t, err):
        super().addError(t, err)
        self.d.append({{"n": t.id().split(".")[-1], "s": "error"}})

_t0 = _time.monotonic()
_suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
_res = _R()
_suite.run(_res)
_elapsed = int((_time.monotonic() - _t0) * 1000)
print("###EVAL###")
print(json.dumps({{"total": _res.testsRun, "tests": _res.d, "ms": _elapsed}}))
'''


def _strip_main(code: str) -> str:
    """Remove ``if __name__ == '__main__'`` block from test code."""
    lines = code.split("\n")
    out: list[str] = []
    skip = False
    for line in lines:
        if line.strip().startswith("if __name__"):
            skip = True
            continue
        if skip and (line.startswith((" ", "\t")) or line.strip() == ""):
            continue
        skip = False
        out.append(line)
    return "\n".join(out)


def extract_code(response: str) -> str:
    """Pull Python code out of a model response (strips markdown fences)."""
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", response, re.DOTALL)
    if blocks:
        return "\n\n".join(b.strip() for b in blocks)
    return response.strip()


# ---------------------------------------------------------------------------
# Rollout data
# ---------------------------------------------------------------------------

@dataclass
class Rollout:
    """One generated solution + its test execution results."""
    puzzle_id: str
    puzzle_num: str
    rollout_idx: int
    code: str = ""
    passed: bool = False
    visible_passed: int = 0
    visible_total: int = 0
    hidden_passed: int = 0
    hidden_total: int = 0
    error_type: str = "unknown"
    execution_time_ms: int = 0
    test_details: list[dict] = field(default_factory=list)

    difficulty: str = ""
    category: str = ""

    @property
    def total_tests(self) -> int:
        return self.visible_total + self.hidden_total

    @property
    def total_passed(self) -> int:
        return self.visible_passed + self.hidden_passed

    @property
    def test_pass_rate(self) -> float:
        return self.total_passed / self.total_tests if self.total_tests else 0.0

    def to_jsonl(self) -> dict:
        return {
            "puzzle_id": self.puzzle_id,
            "puzzle_num": self.puzzle_num,
            "rollout_idx": self.rollout_idx,
            "pass": self.passed,
            "visible_passed": self.visible_passed,
            "visible_total": self.visible_total,
            "hidden_passed": self.hidden_passed,
            "hidden_total": self.hidden_total,
            "error_type": self.error_type,
            "execution_time_ms": self.execution_time_ms,
            "difficulty": self.difficulty,
            "category": self.category,
            "code": self.code,
        }


# ---------------------------------------------------------------------------
# Test execution
# ---------------------------------------------------------------------------

def execute_rollout(solution_code: str, test_code: str, timeout: int = 30) -> dict:
    """Run solution against tests in subprocess. Returns parsed result dict."""
    harness = _HARNESS.format(
        solution_code=solution_code,
        test_code=_strip_main(test_code),
    )
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
    try:
        tmp.write(harness)
        tmp.flush()
        tmp.close()
        t0 = time.monotonic()
        proc = subprocess.run(
            [sys.executable, tmp.name],
            capture_output=True, text=True, timeout=timeout,
        )
        wall_ms = int((time.monotonic() - t0) * 1000)

        if "###EVAL###" in proc.stdout:
            data = json.loads(proc.stdout.split("###EVAL###")[1].strip())
            data["wall_ms"] = wall_ms
            return data
        stderr = proc.stderr.strip()
        return {
            "total": 0, "tests": [], "ms": wall_ms, "wall_ms": wall_ms,
            "error": "syntax_error" if "SyntaxError" in stderr else "runtime_error",
            "stderr": stderr[-500:],
        }
    except subprocess.TimeoutExpired:
        return {"total": 0, "tests": [], "ms": timeout * 1000,
                "wall_ms": timeout * 1000, "error": "timeout"}
    except Exception as e:
        return {"total": 0, "tests": [], "ms": 0, "wall_ms": 0,
                "error": "runtime_error", "stderr": str(e)[:500]}
    finally:
        os.unlink(tmp.name)


def classify_error(result: dict, vis_pass: int, vis_total: int,
                   hid_pass: int, hid_total: int) -> str:
    """Assign error_type per eval_plan.md spec."""
    total_pass = vis_pass + hid_pass
    total = vis_total + hid_total
    if total > 0 and total_pass == total:
        return "full_pass"
    if result.get("error") == "syntax_error":
        return "syntax_error"
    if result.get("error") == "timeout":
        return "timeout"
    if result.get("error") == "runtime_error" and total == 0:
        return "runtime_error"
    if vis_total > 0 and vis_pass == vis_total and hid_pass < hid_total:
        return "partial_pass"
    if result.get("error") == "runtime_error":
        return "runtime_error"
    return "wrong_answer"


def build_rollout(puzzle: dict, idx: int, code: str, timeout: int) -> Rollout:
    """Generate + test one rollout, return populated Rollout."""
    num = puzzle["_num"]
    r = Rollout(
        puzzle_id=puzzle["puzzle_id"], puzzle_num=num, rollout_idx=idx,
        code=code,
        difficulty=puzzle["difficulty"],
        category=puzzle.get("category", ""),
    )

    result = execute_rollout(code, puzzle["unit_tests"], timeout=timeout)
    r.execution_time_ms = result.get("wall_ms", 0)
    r.test_details = result.get("tests", [])

    for t in r.test_details:
        is_hidden = t["n"].startswith("test_hidden")
        if is_hidden:
            r.hidden_total += 1
            if t["s"] == "pass":
                r.hidden_passed += 1
        else:
            r.visible_total += 1
            if t["s"] == "pass":
                r.visible_passed += 1

    r.error_type = classify_error(
        result, r.visible_passed, r.visible_total,
        r.hidden_passed, r.hidden_total,
    )
    r.passed = r.error_type == "full_pass"
    return r


# ---------------------------------------------------------------------------
# Model loading & inference (HuggingFace transformers)
# ---------------------------------------------------------------------------

def load_model(model_name: str, device: str = "auto"):
    """Load model and tokenizer. Returns (model, tokenizer)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded on {model.device}\n")
    return model, tokenizer


def _build_chat_text(tokenizer, prompt: str, starter_code: str) -> str:
    """Build the full chat-template string for a single puzzle."""
    user_msg = f"{prompt}\n\nComplete this function:\n\n{starter_code}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


def generate_solutions(model, tokenizer, prompt: str, starter_code: str,
                       num_samples: int = 8, temperature: float = 0.8,
                       max_new_tokens: int = 2048) -> list[str]:
    """Generate num_samples solutions for a single puzzle."""
    import torch

    text = _build_chat_text(tokenizer, prompt, starter_code)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.95,
            do_sample=True,
            num_return_sequences=num_samples,
            pad_token_id=tokenizer.eos_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    results: list[str] = []
    for seq in outputs:
        decoded = tokenizer.decode(seq[input_len:], skip_special_tokens=True)
        results.append(decoded)
    return results


def generate_solutions_batch(
    model, tokenizer, puzzles: list[dict],
    num_samples: int = 8, temperature: float = 0.8,
    max_new_tokens: int = 2048,
) -> list[list[str]]:
    """Generate solutions for multiple puzzles in one batched GPU call.

    Left-pads inputs so they can be batched, runs model.generate() once
    with num_return_sequences, then splits outputs back per puzzle.
    Returns list-of-lists: outer = puzzles, inner = samples.
    """
    import torch

    if not puzzles:
        return []
    if len(puzzles) == 1:
        return [generate_solutions(
            model, tokenizer, puzzles[0]["prompt"], puzzles[0]["starter_code"],
            num_samples=num_samples, temperature=temperature,
            max_new_tokens=max_new_tokens,
        )]

    texts = [_build_chat_text(tokenizer, p["prompt"], p["starter_code"])
             for p in puzzles]

    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.95,
            do_sample=True,
            num_return_sequences=num_samples,
            pad_token_id=tokenizer.eos_token_id,
        )

    tokenizer.padding_side = orig_padding_side

    # outputs shape: [B * num_samples, seq_len]. First num_samples rows are
    # for puzzle 0, next num_samples for puzzle 1, etc.
    all_results: list[list[str]] = []
    for i in range(len(puzzles)):
        puzzle_results = []
        for j in range(num_samples):
            seq_idx = i * num_samples + j
            decoded = tokenizer.decode(
                outputs[seq_idx][input_len:], skip_special_tokens=True)
            puzzle_results.append(decoded)
        all_results.append(puzzle_results)

    return all_results


# ---------------------------------------------------------------------------
# Puzzle loading
# ---------------------------------------------------------------------------

def load_puzzles(puzzle_dir: str) -> list[dict]:
    puzzles = []
    for path in sorted(Path(puzzle_dir).glob("*.json")):
        m = re.match(r"^(\d{3})_", path.name)
        if not m:
            continue
        with open(path) as f:
            data = json.load(f)
        data["_num"] = m.group(1)
        puzzles.append(data)
    return puzzles


# ---------------------------------------------------------------------------
# Run 1 — main eval loop
# ---------------------------------------------------------------------------

def run_eval(
    puzzles: list[dict],
    model,
    tokenizer,
    num_samples: int = 8,
    timeout: int = 30,
    temperature: float = 0.8,
) -> list[Rollout]:
    from concurrent.futures import ThreadPoolExecutor

    all_rollouts: list[Rollout] = []
    total = len(puzzles)

    with ThreadPoolExecutor(max_workers=num_samples) as test_pool:
        for i, puzzle in enumerate(puzzles):
            pid = puzzle["puzzle_id"]
            diff = puzzle["difficulty"]
            cat = puzzle.get("category", "")
            label = f"{diff}, {cat}" if cat else diff
            print(f"[{i+1}/{total}] {pid} ({label})", flush=True)

            try:
                raw_responses = generate_solutions(
                    model, tokenizer, puzzle["prompt"], puzzle["starter_code"],
                    num_samples=num_samples, temperature=temperature,
                )
            except Exception as e:
                print(f"  Generation error: {e}", flush=True)
                raw_responses = [f"# GENERATION ERROR: {e}"] * num_samples

            codes = [extract_code(raw) for raw in raw_responses]
            futures = [
                (j, test_pool.submit(build_rollout, puzzle, j, code, timeout))
                for j, code in enumerate(codes)
            ]

            puzzle_rollouts: list[Rollout] = []
            for j, fut in futures:
                rollout = fut.result()
                puzzle_rollouts.append(rollout)
                tag = "PASS" if rollout.passed else rollout.error_type
                print(f"  [{j+1}/{num_samples}] {tag:14s}  "
                      f"vis={rollout.visible_passed}/{rollout.visible_total} "
                      f"hid={rollout.hidden_passed}/{rollout.hidden_total} "
                      f"({rollout.execution_time_ms}ms)", flush=True)

            passes = sum(1 for r in puzzle_rollouts if r.passed)
            print(f"  → {passes}/{num_samples} pass\n", flush=True)
            all_rollouts.extend(puzzle_rollouts)

    return all_rollouts


# ---------------------------------------------------------------------------
# Aggregation views (all computed offline from rollout data)
# ---------------------------------------------------------------------------

def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Codex paper)."""
    if n < k:
        return 1.0 if c > 0 else 0.0
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def compute_views(rollouts: list[Rollout], k: int) -> dict[str, Any]:
    """Compute every aggregation view from eval_plan.md."""
    # Group rollouts by puzzle
    by_puzzle: dict[str, list[Rollout]] = defaultdict(list)
    for r in rollouts:
        by_puzzle[r.puzzle_num].append(r)

    # ── Per-puzzle summary ──
    puzzle_summaries: list[dict] = []
    for num in sorted(by_puzzle):
        rs = by_puzzle[num]
        n = len(rs)
        c = sum(1 for r in rs if r.passed)
        scores = [r.test_pass_rate for r in rs]
        mean_score = sum(scores) / n if n else 0
        score_var = sum((s - mean_score) ** 2 for s in scores) / n if n else 0
        p_rate = c / n if n else 0
        adv_var = p_rate * (1 - p_rate)

        error_counts: dict[str, int] = defaultdict(int)
        for r in rs:
            error_counts[r.error_type] += 1
        dominant_error = max(error_counts, key=error_counts.get) if error_counts else "?"

        vis_rate = sum(r.visible_passed / r.visible_total if r.visible_total else 0
                       for r in rs) / n if n else 0
        hid_rate = sum(r.hidden_passed / r.hidden_total if r.hidden_total else 0
                       for r in rs) / n if n else 0

        first = rs[0]
        puzzle_summaries.append({
            "puzzle_num": num,
            "puzzle_id": first.puzzle_id,
            "difficulty": first.difficulty,
            "category": first.category,
            "pass_at_1": p_rate,
            f"pass_at_{k}": pass_at_k(n, c, k),
            "mean_test_pass_rate": round(mean_score, 4),
            "visible_pass_rate": round(vis_rate, 4),
            "hidden_pass_rate": round(hid_rate, 4),
            "hidden_gap": round(vis_rate - hid_rate, 4),
            "score_variance": round(score_var, 4),
            "advantage_variance": round(adv_var, 4),
            "best_score": round(max(scores, default=0), 4),
            "error_distribution": dict(error_counts),
            "dominant_error": dominant_error,
            "zone": ("saturated" if p_rate == 1 else "learning" if c > 0 else "dead"),
        })

    # ── Difficulty breakdown ──
    difficulty_view: dict[str, dict] = {}
    all_difficulties = sorted(set(p["difficulty"] for p in puzzle_summaries))
    for diff in all_difficulties:
        pss = [p for p in puzzle_summaries if p["difficulty"] == diff]
        if not pss:
            continue
        n_d = len(pss)
        difficulty_view[diff] = {
            "count": n_d,
            "pass_at_1": sum(p["pass_at_1"] for p in pss) / n_d,
            f"pass_at_{k}": sum(p[f"pass_at_{k}"] for p in pss) / n_d,
            "mean_test_pass_rate": sum(p["mean_test_pass_rate"] for p in pss) / n_d,
            "visible_pass_rate": sum(p["visible_pass_rate"] for p in pss) / n_d,
            "hidden_pass_rate": sum(p["hidden_pass_rate"] for p in pss) / n_d,
            "hidden_gap": sum(p["hidden_gap"] for p in pss) / n_d,
            "learning_zone": sum(1 for p in pss if p["zone"] == "learning"),
        }

    # ── Category breakdown ──
    category_view: dict[str, dict] = {}
    all_categories = sorted(set(p["category"] for p in puzzle_summaries if p["category"]))
    for cat in all_categories:
        pss = [p for p in puzzle_summaries if p["category"] == cat]
        if not pss:
            continue
        n_c = len(pss)
        category_view[cat] = {
            "count": n_c,
            "pass_at_1": sum(p["pass_at_1"] for p in pss) / n_c,
            f"pass_at_{k}": sum(p[f"pass_at_{k}"] for p in pss) / n_c,
            "mean_test_pass_rate": sum(p["mean_test_pass_rate"] for p in pss) / n_c,
            "advantage_variance": sum(p["advantage_variance"] for p in pss) / n_c,
            "learning_zone": sum(1 for p in pss if p["zone"] == "learning"),
        }

    # ── Error type distribution ──
    error_totals: dict[str, int] = defaultdict(int)
    for r in rollouts:
        error_totals[r.error_type] += 1
    total_rollouts = len(rollouts)
    error_view = {
        etype: {"count": cnt, "rate": round(cnt / total_rollouts, 4) if total_rollouts else 0}
        for etype, cnt in sorted(error_totals.items(), key=lambda x: -x[1])
    }

    # ── Advantage spread ──
    zones = {"dead": 0, "learning": 0, "saturated": 0}
    for ps in puzzle_summaries:
        zones[ps["zone"]] += 1
    advantage_ranked = sorted(puzzle_summaries, key=lambda p: -p["advantage_variance"])

    # ── RL target ranking (eval_plan.md priority 1/2/3) ──
    rl_targets: list[dict] = []
    for ps in puzzle_summaries:
        priority = 3
        reason = "Dead zone — no signal, needs prerequisite skill training"
        if ps["zone"] == "learning":
            priority = 1
            reason = "High advantage spread — model sometimes solves, GRPO will work"
        elif ps["zone"] == "dead" and ps["dominant_error"] in ("wrong_answer", "partial_pass"):
            priority = 2
            reason = "Attempts right structure but fails — curriculum + RL"
        elif ps["zone"] == "dead" and ps["best_score"] > 0.2:
            priority = 2
            reason = "Partial understanding — needs easier variants of this skill"
        elif ps["zone"] == "saturated":
            priority = 0
            reason = "Already mastered — skip in RL training"
        rl_targets.append({
            "puzzle_id": ps["puzzle_id"],
            "puzzle_num": ps["puzzle_num"],
            "difficulty": ps["difficulty"],
            "category": ps.get("category", ""),
            "priority": priority,
            "reason": reason,
            "zone": ps["zone"],
            "pass_at_1": round(ps["pass_at_1"], 4),
            f"pass_at_{k}": round(ps[f"pass_at_{k}"], 4),
            "advantage_variance": ps["advantage_variance"],
            "best_score": ps["best_score"],
            "dominant_error": ps["dominant_error"],
        })
    rl_targets.sort(key=lambda t: (t["priority"], -t["advantage_variance"]))

    return {
        "puzzle_summaries": puzzle_summaries,
        "difficulty": difficulty_view,
        "category": category_view,
        "error_distribution": error_view,
        "advantage_spread": {"zones": zones, "ranked": advantage_ranked},
        "rl_targets": rl_targets,
    }


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_report(views: dict, model: str, k: int) -> str:
    lines: list[str] = []
    W = 80

    def section(title: str):
        lines.append("")
        lines.append("─" * W)
        lines.append(f"  {title}")
        lines.append("─" * W)

    ps = views["puzzle_summaries"]
    n_puzzles = len(ps)

    lines.append("=" * W)
    lines.append(f"{'RL CODING PUZZLE EVALUATION REPORT':^{W}}")
    lines.append("=" * W)
    lines.append(f"  Model:          {model}")
    lines.append(f"  Samples/puzzle: {k}    Puzzles: {n_puzzles}")
    lines.append(f"  Timestamp:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * W)

    # ── pass@1 / pass@k overall ──
    section("PASS RATES (Overall)")
    avg_p1 = sum(p["pass_at_1"] for p in ps) / n_puzzles if n_puzzles else 0
    pk_key = f"pass_at_{k}"
    solved = sum(1 for p in ps if p[pk_key] > 0)
    avg_test = sum(p["mean_test_pass_rate"] for p in ps) / n_puzzles if n_puzzles else 0
    avg_vis = sum(p["visible_pass_rate"] for p in ps) / n_puzzles if n_puzzles else 0
    avg_hid = sum(p["hidden_pass_rate"] for p in ps) / n_puzzles if n_puzzles else 0
    lines.append(f"  pass@1 (avg):             {avg_p1*100:5.1f}%")
    lines.append(f"  Solved (pass@{k} > 0):     {solved}/{n_puzzles}")
    lines.append(f"  Mean test pass rate:      {avg_test*100:5.1f}%")
    lines.append(f"  Visible test rate:        {avg_vis*100:5.1f}%")
    lines.append(f"  Hidden test rate:         {avg_hid*100:5.1f}%")
    lines.append(f"  Hidden gap (vis - hid):   {(avg_vis-avg_hid)*100:+5.1f}%")

    # ── Difficulty breakdown ──
    section("DIFFICULTY BREAKDOWN")
    dv = views["difficulty"]
    lines.append(f"  {'Diff':<12} {'#':>3} {'P@1':>6} {'P@k':>6} {'Test%':>6} "
                 f"{'Vis%':>6} {'Hid%':>6} {'Gap':>6} {'LrnZone':>8}")
    lines.append("  " + "─" * (W - 4))
    for diff in dv:
        dd = dv[diff]
        lines.append(
            f"  {diff:<12} {dd['count']:>3} {dd['pass_at_1']*100:5.1f}% "
            f"{dd[pk_key]*100:5.1f}% {dd['mean_test_pass_rate']*100:5.1f}% "
            f"{dd['visible_pass_rate']*100:5.1f}% {dd['hidden_pass_rate']*100:5.1f}% "
            f"{dd['hidden_gap']*100:+5.1f}% "
            f"{dd['learning_zone']:>3}/{dd['count']}"
        )

    # ── Category breakdown ──
    cv = views.get("category", {})
    if cv:
        section("CATEGORY BREAKDOWN")
        lines.append(f"  {'Category':<20} {'#':>3} {'P@1':>6} {'P@k':>6} "
                     f"{'Test%':>6} {'GRPO':>6} {'LrnZone':>8}")
        lines.append("  " + "─" * (W - 4))
        for cat in cv:
            cc = cv[cat]
            lines.append(
                f"  {cat:<20} {cc['count']:>3} {cc['pass_at_1']*100:5.1f}% "
                f"{cc[pk_key]*100:5.1f}% {cc['mean_test_pass_rate']*100:5.1f}% "
                f"{cc['advantage_variance']:.3f} "
                f"{cc['learning_zone']:>3}/{cc['count']}"
            )

    # ── Error distribution ──
    section("ERROR TYPE DISTRIBUTION")
    ev = views["error_distribution"]
    lines.append(f"  {'Type':<16} {'Count':>6} {'Rate':>7}")
    lines.append("  " + "─" * 32)
    for etype, info in ev.items():
        label = etype.replace("_", " ").title()
        lines.append(f"  {label:<16} {info['count']:>6} {info['rate']*100:6.1f}%")

    # ── Advantage spread ──
    section("ADVANTAGE SPREAD (RL Signal)")
    zd = views["advantage_spread"]["zones"]
    total_z = sum(zd.values())
    lines.append(f"  Learning zone:  {zd['learning']:>3}/{total_z}  "
                 f"({zd['learning']/total_z*100:.0f}%)  ← Best for GRPO")
    lines.append(f"  Dead:           {zd['dead']:>3}/{total_z}  "
                 f"({zd['dead']/total_z*100:.0f}%)  ← No positive signal")
    lines.append(f"  Saturated:      {zd['saturated']:>3}/{total_z}  "
                 f"({zd['saturated']/total_z*100:.0f}%)  ← Already mastered")
    lines.append("")
    lines.append("  Top puzzles by GRPO reward variance (p@1 * (1 - p@1)):")
    for j, p in enumerate(views["advantage_spread"]["ranked"][:10]):
        bar = "█" * int(p["advantage_variance"] * 40)
        cat_str = f"[{p['category']}] " if p.get("category") else ""
        lines.append(f"    {j+1:>2}. {p['puzzle_id']:<32} {cat_str}{p['difficulty']:<6} "
                     f"p@1={p['pass_at_1']*100:4.0f}%  grpo={p['advantage_variance']:.3f} {bar}")

    # ── RL targets ──
    section("RL TRAINING TARGETS (Ranked)")
    lines.append("")
    for pri, label in [(1, "Priority 1 — High advantage (GRPO sweet spot)"),
                       (2, "Priority 2 — Curriculum candidates"),
                       (3, "Priority 3 — Dead zones (need scaffolding)")]:
        targets = [t for t in views["rl_targets"] if t["priority"] == pri]
        if not targets:
            continue
        lines.append(f"  {label}:")
        for t in targets[:10]:
            cat_str = f"[{t.get('category', '')}] " if t.get("category") else ""
            lines.append(f"    {t['puzzle_id']:<36} {cat_str}{t['difficulty']:<10} "
                         f"p@1={t['pass_at_1']*100:4.0f}%")
            lines.append(f"      → {t['reason']}")
        lines.append("")


    lines.append("=" * W)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Save artifacts (per eval_plan.md output spec)
# ---------------------------------------------------------------------------

def save_artifacts(rollouts: list[Rollout], views: dict, output_dir: str,
                   model: str = "", report: str = ""):
    os.makedirs(output_dir, exist_ok=True)

    # raw_rollouts.jsonl — one line per (puzzle, rollout)
    path = os.path.join(output_dir, "raw_rollouts.jsonl")
    with open(path, "w") as f:
        for r in rollouts:
            f.write(json.dumps(r.to_jsonl()) + "\n")
    print(f"  raw_rollouts.jsonl          → {path}")

    # summary.json — per-puzzle aggregated metrics
    path = os.path.join(output_dir, "summary.json")
    with open(path, "w") as f:
        json.dump(views["puzzle_summaries"], f, indent=2)
    print(f"  summary.json                → {path}")

    # rl_targets_ranked.json
    path = os.path.join(output_dir, "rl_targets_ranked.json")
    with open(path, "w") as f:
        json.dump(views["rl_targets"], f, indent=2)
    print(f"  rl_targets_ranked.json      → {path}")

    # eval_analytics.json — full structured analytics
    ps = views["puzzle_summaries"]
    n = len(ps)
    k_key = [k for k in ps[0] if k.startswith("pass_at_") and k != "pass_at_1"][0] if ps else "pass_at_8"
    zones = views.get("advantage_spread", {}).get("zones", {})

    analytics = {
        "meta": {
            "model": model,
            "n_puzzles": n,
            "timestamp": datetime.now().isoformat(),
        },
        "overall": {
            "pass_at_1": round(sum(p["pass_at_1"] for p in ps) / n, 4) if n else 0,
            "mean_test_pass_rate": round(sum(p["mean_test_pass_rate"] for p in ps) / n, 4) if n else 0,
            "mean_grpo_variance": round(sum(p["advantage_variance"] for p in ps) / n, 4) if n else 0,
            "zones": dict(zones),
        },
        "categories": views.get("category", {}),
        "difficulty": views.get("difficulty", {}),
        "per_puzzle": views["puzzle_summaries"],
        "rl_targets": views["rl_targets"],
    }
    path = os.path.join(output_dir, "eval_analytics.json")
    with open(path, "w") as f:
        json.dump(analytics, f, indent=2)
    print(f"  eval_analytics.json         → {path}")

    if report:
        path = os.path.join(output_dir, "report.txt")
        with open(path, "w") as f:
            f.write(report)
        print(f"  report.txt                  -> {path}")


# ---------------------------------------------------------------------------
# Load existing rollouts for re-analysis
# ---------------------------------------------------------------------------

def load_rollouts(path: str) -> list[Rollout]:
    rollouts: list[Rollout] = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            r = Rollout(
                puzzle_id=d["puzzle_id"],
                puzzle_num=d["puzzle_num"],
                rollout_idx=d["rollout_idx"],
                code=d.get("code", ""),
                passed=d["pass"],
                visible_passed=d["visible_passed"],
                visible_total=d["visible_total"],
                hidden_passed=d["hidden_passed"],
                hidden_total=d["hidden_total"],
                error_type=d["error_type"],
                execution_time_ms=d.get("execution_time_ms", 0),
                difficulty=d["difficulty"],
                category=d.get("category", ""),
            )
            rollouts.append(r)
    return rollouts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RL Coding Puzzle Evaluation")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct",
                        help="HuggingFace model ID (default: Qwen/Qwen2.5-Coder-1.5B-Instruct)")
    parser.add_argument("--device", default="auto",
                        help="Device map for model loading (default: auto)")
    parser.add_argument("--puzzle-dir", default=None,
                        help="Puzzle directory (default: ../puzzles)")
    parser.add_argument("--samples", type=int, default=8,
                        help="Rollouts per puzzle (default: 8)")
    parser.add_argument("--timeout", type=int, default=10,
                        help="Test execution timeout in seconds (default: 10)")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: ../eval_results)")
    parser.add_argument("--analyze", metavar="ROLLOUTS_JSONL",
                        help="Skip inference; re-analyze existing raw_rollouts.jsonl")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    puzzle_dir = args.puzzle_dir or str(script_dir.parent / "puzzles")
    output_dir = args.output_dir or str(script_dir.parent / "eval_results")
    k = args.samples

    if args.analyze:
        print(f"Loading rollouts from {args.analyze}...")
        rollouts = load_rollouts(args.analyze)
        n_puzzles = len(set(r.puzzle_num for r in rollouts))
        k = len(rollouts) // n_puzzles if n_puzzles else k
        print(f"  {len(rollouts)} rollouts across {n_puzzles} puzzles ({k}/puzzle)\n")
    else:
        puzzles = load_puzzles(puzzle_dir)
        if not puzzles:
            print(f"No puzzles found in {puzzle_dir}")
            sys.exit(1)
        print(f"Loaded {len(puzzles)} puzzles from {puzzle_dir}")
        print(f"Model: {args.model}")
        print(f"Samples: {k}  |  Temperature: {args.temperature}  |  Timeout: {args.timeout}s")
        print()

        model, tokenizer = load_model(args.model, device=args.device)
        rollouts = run_eval(
            puzzles, model, tokenizer,
            num_samples=k, timeout=args.timeout,
            temperature=args.temperature,
        )

    views = compute_views(rollouts, k)
    report = format_report(views, args.model, k)
    print(report)

    print(f"\nSaving artifacts to {output_dir}/")
    save_artifacts(rollouts, views, output_dir, model=args.model, report=report)
    print("\nDone.")


if __name__ == "__main__":
    main()
