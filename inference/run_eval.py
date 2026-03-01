#!/usr/bin/env python3
"""RL Coding Puzzle Evaluation Pipeline.

Single inference run (Run 1 from eval_plan.md): generate k rollouts per puzzle,
execute against unit tests, then compute all aggregation views offline.

Usage:
    # Full run (inference + tests + analytics)
    python run_eval.py --api-base http://localhost:8000/v1

    # Re-run analytics on existing results
    python run_eval.py --analyze eval_results/raw_rollouts.jsonl

    # Custom sampling
    python run_eval.py --api-base http://localhost:8000/v1 --samples 8 --timeout 30
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import OpenAI

# ---------------------------------------------------------------------------
# Puzzle catalog (from concept_map.md)
# ---------------------------------------------------------------------------

_CATALOG: list[tuple[str, str, str, str]] = [
    # (puzzle_num, concept_row, micro_skill, skill_family)
    ("001", "A",  "Dijkstra with Resource-Constrained State",           "Graph Algorithms"),
    ("002", "H",  "Grouped Knapsack DP",                                "Dynamic Programming"),
    ("003", "B",  "Bitmask DP over Subset Enumeration",                 "Dynamic Programming"),
    ("004", "I",  "Tree Aggregation via Path Prefix Parsing",           "Tree / Path Aggregation"),
    ("005", "J",  "Greedy Interval Scheduling",                         "Greedy / Two-Pointer / Sliding Window"),
    ("006", "K",  "Bitwise Cipher Implementation",                      "Number Theory / Modular Arithmetic"),
    ("007", "L",  "Sliding Window Sum",                                 "Greedy / Two-Pointer / Sliding Window"),
    ("008", "M",  "Union-Find with Metadata Tracking",                  "Data Structures"),
    ("009", "N",  "Topological Sort with Multi-Type Dependencies",      "Graph Algorithms"),
    ("010", "O",  "Segment Tree with Lazy Propagation",                 "Data Structures"),
    ("011", "P",  "Trie-Based Top-K Prefix Retrieval",                  "Data Structures"),
    ("012", "E",  "Multi-Key Custom Comparator Sorting",                "Sorting / Comparators"),
    ("013", "C",  "Backtracking Constraint Satisfaction",               "Dynamic Programming"),
    ("014", "I2", "Matrix Ring Extraction and Rotation",                "Simulation / State Machines"),
    ("015", "I3", "Modular Hash Collision Analysis",                    "Number Theory / Modular Arithmetic"),
    ("016", "Q",  "Computational Geometry Shortest Path",               "Computational Geometry"),
    ("017", "R",  "Weighted Job Scheduling DP",                         "Dynamic Programming"),
    ("018", "S",  "Stack-Based Interpreter with Control Flow",          "Simulation / State Machines"),
    ("019", "G",  "Grid-Based BFS / Dijkstra",                         "Graph Algorithms"),
    ("020", "T",  "Binary Search on Answer",                            "Greedy / Two-Pointer / Sliding Window"),
    ("021", "U",  "Multi-Dimensional Constrained Counting DP",          "Dynamic Programming"),
    ("022", "V",  "Two-Pointer Merge with Tolerance",                   "Greedy / Two-Pointer / Sliding Window"),
    ("023", "D",  "Game Theory DP / Minimax",                           "Dynamic Programming"),
    ("024", "C",  "Backtracking Constraint Satisfaction",               "Dynamic Programming"),
    ("025", "A",  "Dijkstra with Resource-Constrained State",           "Graph Algorithms"),
    ("026", "W",  "Max-Flow Network",                                   "Graph Algorithms"),
    ("027", "X",  "Interval DP (Divide-and-Conquer Cost)",              "Dynamic Programming"),
    ("028", "Y",  "Linked List with Pool Management",                   "Data Structures"),
    ("029", "Z",  "Monotonic Stack for Pair Counting",                  "Data Structures"),
    ("030", "AA", "Suffix Array + LCP + Sliding Window",                "String Algorithms"),
    ("031", "AB", "Convex Hull + Point-in-Polygon",                     "Computational Geometry"),
    ("032", "AC", "Fenwick Tree for Dynamic Rank Queries",              "Data Structures"),
    ("033", "AD", "Consistent Hashing with Virtual Nodes",              "System Design / Hashing"),
    ("034", "AE", "Rule-Based Cellular Automaton Simulation",           "Simulation / State Machines"),
    ("035", "AF", "Expected Value Decision",                            "Probability / Expected Value"),
    ("036", "AG", "Numerical Root-Finding (Spectral Radius)",           "Number Theory / Modular Arithmetic"),
    ("037", "E",  "Multi-Key Custom Comparator Sorting",                "Sorting / Comparators"),
    ("038", "AH", "Cycle Detection in Functional Graph",                "Graph Algorithms"),
    ("039", "B",  "Bitmask DP over Subset Enumeration",                 "Dynamic Programming"),
    ("040", "F",  "Sweep Line with Coordinate Compression",             "Sweep Line / Coord Compression"),
    ("041", "AI", "Sparse Table for Range Queries",                     "Data Structures"),
    ("042", "AJ", "Interval Clipping + Merging + Gap-Finding",          "Sweep Line / Coord Compression"),
    ("043", "AK", "KMP String Matching",                                "String Algorithms"),
    ("044", "AL", "Chinese Remainder Theorem + Extended GCD",           "Number Theory / Modular Arithmetic"),
    ("045", "AM", "DFA Simulation",                                     "Simulation / State Machines"),
    ("046", "D",  "Game Theory DP / Minimax",                           "Dynamic Programming"),
    ("047", "F",  "Sweep Line with Coordinate Compression",             "Sweep Line / Coord Compression"),
    ("048", "AN", "Event-Driven Simulation with State Tracking",        "Simulation / State Machines"),
    ("049", "G",  "Grid-Based BFS / Dijkstra",                         "Graph Algorithms"),
    ("050", "AO", "Multi-Policy Cache System Design",                   "System Design / Hashing"),
]

PUZZLE_META: dict[str, dict[str, str]] = {}
CONCEPT_ROWS: dict[str, dict] = {}
for _num, _row, _skill, _family in _CATALOG:
    PUZZLE_META[_num] = {"concept_row": _row, "micro_skill": _skill, "skill_family": _family}
    if _row not in CONCEPT_ROWS:
        CONCEPT_ROWS[_row] = {"skill": _skill, "family": _family, "puzzles": []}
    CONCEPT_ROWS[_row]["puzzles"].append(_num)

SKILL_FAMILIES = sorted(set(m["skill_family"] for m in PUZZLE_META.values()))

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

    # category info (filled from PUZZLE_META)
    difficulty: str = ""
    concept_row: str = ""
    micro_skill: str = ""
    skill_family: str = ""

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
            "concept_row": self.concept_row,
            "micro_skill": self.micro_skill,
            "skill_family": self.skill_family,
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
    meta = PUZZLE_META.get(num, {})
    r = Rollout(
        puzzle_id=puzzle["puzzle_id"], puzzle_num=num, rollout_idx=idx,
        code=code,
        difficulty=puzzle["difficulty"],
        concept_row=meta.get("concept_row", "?"),
        micro_skill=meta.get("micro_skill", "?"),
        skill_family=meta.get("skill_family", "?"),
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
# Inference
# ---------------------------------------------------------------------------

def generate_solution(client: OpenAI, model: str, prompt: str,
                      starter_code: str, temperature: float = 0.7) -> str:
    user_msg = f"{prompt}\n\nComplete this function:\n\n{starter_code}"
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=temperature,
        max_tokens=2048,
    )
    return resp.choices[0].message.content or ""


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
    client: OpenAI,
    model: str,
    num_samples: int = 8,
    timeout: int = 30,
    max_workers: int = 4,
    temperature: float = 0.7,
) -> list[Rollout]:
    all_rollouts: list[Rollout] = []
    total = len(puzzles)

    for i, puzzle in enumerate(puzzles):
        pid = puzzle["puzzle_id"]
        diff = puzzle["difficulty"]
        fam = PUZZLE_META.get(puzzle["_num"], {}).get("skill_family", "?")
        print(f"[{i+1}/{total}] {pid} ({diff}, {fam})", flush=True)

        raw_responses: list[str] = []
        def _gen(_j: int) -> str:
            return generate_solution(client, model, puzzle["prompt"],
                                     puzzle["starter_code"], temperature)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futs = {pool.submit(_gen, j): j for j in range(num_samples)}
            for fut in as_completed(futs):
                try:
                    raw_responses.append(fut.result())
                except Exception as e:
                    raw_responses.append(f"# API ERROR: {e}")

        puzzle_rollouts: list[Rollout] = []
        for j, raw in enumerate(raw_responses):
            code = extract_code(raw)
            rollout = build_rollout(puzzle, j, code, timeout)
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
            "concept_row": first.concept_row,
            "micro_skill": first.micro_skill,
            "skill_family": first.skill_family,
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

    # ── Concept row heatmap ──
    by_row: dict[str, list[dict]] = defaultdict(list)
    for ps in puzzle_summaries:
        by_row[ps["concept_row"]].append(ps)

    concept_heatmap: dict[str, dict] = {}
    for row_id, pss in sorted(by_row.items()):
        row_info = CONCEPT_ROWS.get(row_id, {})
        n_puzzles = len(pss)
        concept_heatmap[row_id] = {
            "skill": row_info.get("skill", "?"),
            "family": row_info.get("family", "?"),
            "puzzles": [p["puzzle_num"] for p in pss],
            "difficulty_spread": [p["difficulty"] for p in pss],
            "pass_at_1": sum(p["pass_at_1"] for p in pss) / n_puzzles,
            f"pass_at_{k}": sum(p[f"pass_at_{k}"] for p in pss) / n_puzzles,
            "mean_test_pass_rate": sum(p["mean_test_pass_rate"] for p in pss) / n_puzzles,
            "advantage_variance": sum(p["advantage_variance"] for p in pss) / n_puzzles,
            "is_paired": n_puzzles > 1,
            "confirmed_gap": n_puzzles > 1 and all(p["pass_at_1"] == 0 for p in pss),
        }

    # ── Difficulty breakdown ──
    difficulty_view: dict[str, dict] = {}
    for diff in ["Easy", "Medium", "Hard"]:
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
    advantage_ranked = sorted(puzzle_summaries, key=lambda p: -p["score_variance"])

    # ── Skill family radar ──
    family_view: dict[str, dict] = {}
    for fam in SKILL_FAMILIES:
        pss = [p for p in puzzle_summaries if p["skill_family"] == fam]
        if not pss:
            continue
        n_f = len(pss)
        family_view[fam] = {
            "count": n_f,
            "pass_at_1": sum(p["pass_at_1"] for p in pss) / n_f,
            f"pass_at_{k}": sum(p[f"pass_at_{k}"] for p in pss) / n_f,
            "mean_test_pass_rate": sum(p["mean_test_pass_rate"] for p in pss) / n_f,
            "advantage_variance": sum(p["advantage_variance"] for p in pss) / n_f,
            "learning_zone": sum(1 for p in pss if p["zone"] == "learning"),
            "dead": sum(1 for p in pss if p["zone"] == "dead"),
        }

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
            "concept_row": ps["concept_row"],
            "micro_skill": ps["micro_skill"],
            "skill_family": ps["skill_family"],
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
        "concept_heatmap": concept_heatmap,
        "difficulty": difficulty_view,
        "error_distribution": error_view,
        "advantage_spread": {"zones": zones, "ranked": advantage_ranked},
        "skill_families": family_view,
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

    # ── Concept row heatmap ──
    section("CONCEPT ROW HEATMAP")
    ch = views["concept_heatmap"]
    hdr = f"  {'Row':<4} {'Micro-Skill':<46} {'P@1':>5} {'P@k':>5} {'Test%':>5}  {'Gap?'}"
    lines.append(hdr)
    lines.append("  " + "─" * (W - 4))
    sorted_rows = sorted(ch.items(), key=lambda x: -x[1]["mean_test_pass_rate"])
    for row_id, rd in sorted_rows:
        mark = ""
        if rd.get("confirmed_gap"):
            mark = "⚠ CONFIRMED GAP"
        elif rd["pass_at_1"] == 0 and not rd["is_paired"]:
            mark = "? singleton"
        elif rd["pass_at_1"] > 0.4:
            mark = "✓"
        lines.append(
            f"  {row_id:<4} {rd['skill']:<46} "
            f"{rd['pass_at_1']*100:4.0f}% {rd[pk_key]*100:4.0f}% "
            f"{rd['mean_test_pass_rate']*100:4.0f}%  {mark}"
        )

    # ── Difficulty breakdown ──
    section("DIFFICULTY BREAKDOWN")
    dv = views["difficulty"]
    lines.append(f"  {'Diff':<8} {'#':>3} {'P@1':>6} {'P@k':>6} {'Test%':>6} "
                 f"{'Vis%':>6} {'Hid%':>6} {'Gap':>6} {'LrnZone':>8}")
    lines.append("  " + "─" * (W - 4))
    for diff in ["Easy", "Medium", "Hard"]:
        dd = dv.get(diff)
        if not dd:
            continue
        lines.append(
            f"  {diff:<8} {dd['count']:>3} {dd['pass_at_1']*100:5.1f}% "
            f"{dd[pk_key]*100:5.1f}% {dd['mean_test_pass_rate']*100:5.1f}% "
            f"{dd['visible_pass_rate']*100:5.1f}% {dd['hidden_pass_rate']*100:5.1f}% "
            f"{dd['hidden_gap']*100:+5.1f}% "
            f"{dd['learning_zone']:>3}/{dd['count']}"
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
    lines.append("  Top puzzles by score variance (best GRPO signal):")
    for j, p in enumerate(views["advantage_spread"]["ranked"][:10]):
        bar = "█" * int(p["score_variance"] * 50)
        lines.append(f"    {j+1:>2}. {p['puzzle_id']:<32} {p['difficulty']:<6} "
                     f"var={p['score_variance']:.3f} {bar}")

    # ── Skill family radar ──
    section("SKILL FAMILY RADAR")
    fv = views["skill_families"]
    lines.append(f"  {'Family':<40} {'#':>3} {'P@1':>5} {'P@k':>5} {'Test%':>5} {'Lrn':>4}")
    lines.append("  " + "─" * (W - 4))
    for fname in sorted(fv, key=lambda f: -fv[f]["mean_test_pass_rate"]):
        fd = fv[fname]
        lines.append(
            f"  {fname:<40} {fd['count']:>3} "
            f"{fd['pass_at_1']*100:4.0f}% {fd[pk_key]*100:4.0f}% "
            f"{fd['mean_test_pass_rate']*100:4.0f}% "
            f"{fd['learning_zone']:>2}/{fd['count']}"
        )

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
            lines.append(f"    {t['puzzle_id']:<32} {t['difficulty']:<6} "
                         f"Row {t['concept_row']:<4} {t['skill_family']}")
            lines.append(f"      → {t['reason']}")
        lines.append("")

    # ── Confirmed gaps (paired rows both at 0%) ──
    confirmed = [(rid, rd) for rid, rd in ch.items() if rd.get("confirmed_gap")]
    if confirmed:
        lines.append("  CONFIRMED SKILL GAPS (paired rows, both at 0%):")
        for rid, rd in confirmed:
            lines.append(f"    Row {rid}: {rd['skill']}  ({rd['family']})")
        lines.append("")

    lines.append("=" * W)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Save artifacts (per eval_plan.md output spec)
# ---------------------------------------------------------------------------

def save_artifacts(rollouts: list[Rollout], views: dict, output_dir: str):
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

    # concept_heatmap.json
    path = os.path.join(output_dir, "concept_heatmap.json")
    with open(path, "w") as f:
        json.dump(views["concept_heatmap"], f, indent=2)
    print(f"  concept_heatmap.json        → {path}")

    # rl_targets_ranked.json
    path = os.path.join(output_dir, "rl_targets_ranked.json")
    with open(path, "w") as f:
        json.dump(views["rl_targets"], f, indent=2)
    print(f"  rl_targets_ranked.json      → {path}")


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
                concept_row=d["concept_row"],
                micro_skill=d.get("micro_skill", "?"),
                skill_family=d.get("skill_family", "?"),
            )
            rollouts.append(r)
    return rollouts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RL Coding Puzzle Evaluation")
    parser.add_argument("--api-base", default="http://localhost:8000/v1",
                        help="OpenAI-compatible API base URL")
    parser.add_argument("--api-key", default="not-needed",
                        help="API key (default: not-needed for local vLLM)")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--puzzle-dir", default=None,
                        help="Puzzle directory (default: ../puzzles)")
    parser.add_argument("--samples", type=int, default=8,
                        help="Rollouts per puzzle (default: 8)")
    parser.add_argument("--timeout", type=int, default=30,
                        help="Test execution timeout in seconds (default: 30)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-workers", type=int, default=4,
                        help="Parallel API requests per puzzle (default: 4)")
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
        print(f"Model: {args.model}  |  API: {args.api_base}")
        print(f"Samples: {k}  |  Temperature: {args.temperature}  |  Timeout: {args.timeout}s")
        print()

        client = OpenAI(base_url=args.api_base, api_key=args.api_key)
        rollouts = run_eval(
            puzzles, client, args.model,
            num_samples=k, timeout=args.timeout,
            max_workers=args.max_workers, temperature=args.temperature,
        )

    views = compute_views(rollouts, k)
    report = format_report(views, args.model, k)
    print(report)

    print(f"\nSaving artifacts to {output_dir}/")
    save_artifacts(rollouts, views, output_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
