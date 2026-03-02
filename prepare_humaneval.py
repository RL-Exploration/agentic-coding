#!/usr/bin/env python3
"""Download HumanEval+ and convert the first 50 problems to our puzzle JSON format.

Requires: pip install datasets
The evalplus/humanevalplus dataset on HuggingFace contains 164 problems with
enhanced test suites. We take the first 50 (HumanEval/0 - HumanEval/49).
"""

from __future__ import annotations

import json
import os
import re
import argparse
import textwrap
from pathlib import Path


def load_humaneval_plus() -> list[dict]:
    """Load the HumanEval+ dataset from HuggingFace."""
    from datasets import load_dataset
    ds = load_dataset("evalplus/humanevalplus", split="test")
    return list(ds)


def extract_task_number(task_id: str) -> int:
    """Extract the numeric index from a task_id like 'HumanEval/42'."""
    m = re.search(r"/(\d+)$", task_id)
    return int(m.group(1)) if m else -1


def make_slug(task_id: str, entry_point: str) -> str:
    """Create a filesystem-friendly slug from task metadata."""
    num = extract_task_number(task_id)
    clean = re.sub(r"[^a-z0-9_]", "", entry_point.lower())
    return f"humaneval_{num}_{clean}"


def extract_starter_code(prompt: str, entry_point: str) -> str:
    """Extract imports + function signature from the HumanEval prompt, add a pass body."""
    lines = prompt.rstrip().split("\n")

    sig_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("def ") and entry_point in line:
            sig_start = i
            break

    if sig_start is None:
        return f"def {entry_point}():\n    pass"

    # Collect import lines that precede the function definition
    import_lines = []
    for i in range(sig_start):
        stripped = lines[i].strip()
        if stripped.startswith(("import ", "from ")):
            import_lines.append(lines[i])

    sig_lines = [lines[sig_start]]
    j = sig_start + 1
    while j < len(lines):
        stripped = lines[j].strip()
        if stripped.startswith('"""') or stripped.startswith("'''"):
            doc_delim = stripped[:3]
            if stripped.count(doc_delim) >= 2 and len(stripped) > 3:
                sig_lines.append(lines[j])
                j += 1
                break
            sig_lines.append(lines[j])
            j += 1
            while j < len(lines):
                sig_lines.append(lines[j])
                if doc_delim in lines[j] and j != sig_start + 1:
                    j += 1
                    break
                j += 1
            break
        elif stripped == "" or stripped.startswith("#"):
            sig_lines.append(lines[j])
            j += 1
        else:
            break

    starter = "\n".join(sig_lines)
    if not starter.rstrip().endswith("pass"):
        starter = starter.rstrip() + "\n    pass"

    if import_lines:
        return "\n".join(import_lines) + "\n\n" + starter
    return starter


def build_reference_solution(prompt: str, canonical_solution: str) -> str:
    """Combine the prompt (with signature + docstring) and the canonical solution body."""
    return prompt + canonical_solution


def convert_tests_to_unittest(test_code: str, entry_point: str) -> str:
    """Convert HumanEval+'s check(candidate) test format to unittest.TestCase.

    HumanEval+ tests include helper functions (is_floats, assertion) that use
    numpy, plus a check(candidate) function with inputs/results loop patterns.
    We preserve all of this and wrap it in a unittest that calls check(entry_point).
    """
    # The test_code contains helper functions and def check(candidate).
    # We include everything verbatim and call check(entry_point) from a test method.
    return (
        f"import unittest\n"
        f"import numpy as np\n\n"
        f"# --- HumanEval+ test infrastructure ---\n"
        f"{test_code}\n\n"
        f"class TestSolution(unittest.TestCase):\n"
        f"    def test_case_1(self):\n"
        f"        \"\"\"Run the full HumanEval+ check suite.\"\"\"\n"
        f"        check({entry_point})\n\n"
        f"if __name__ == '__main__':\n"
        f"    unittest.main()"
    )


def convert_problem(problem: dict) -> dict:
    """Convert a single HumanEval+ problem to our puzzle JSON format."""
    task_id = problem["task_id"]
    prompt = problem["prompt"]
    canonical = problem["canonical_solution"]
    entry_point = problem["entry_point"]
    test_code = problem["test"]

    puzzle_id = make_slug(task_id, entry_point)
    starter = extract_starter_code(prompt, entry_point)
    reference = build_reference_solution(prompt, canonical)
    unit_tests = convert_tests_to_unittest(test_code, entry_point)

    docstring_match = re.search(r'"""(.*?)"""', prompt, re.DOTALL)
    if not docstring_match:
        docstring_match = re.search(r"'''(.*?)'''", prompt, re.DOTALL)
    description = docstring_match.group(1).strip() if docstring_match else prompt.strip()

    return {
        "puzzle_id": puzzle_id,
        "difficulty": "HumanEval",
        "prompt": description,
        "starter_code": starter,
        "reference_solution": reference,
        "unit_tests": unit_tests,
        "validation_trace": f"Canonical solution from HumanEval+ dataset ({task_id}), "
                           f"tests enhanced by EvalPlus with additional edge cases.",
    }


CURATED_20 = [
    1, 5, 6, 8, 10, 13, 17, 18, 19, 20,
    25, 26, 31, 36, 40, 44, 46, 57, 59, 69,
]

CURATED_50 = sorted(CURATED_20 + [
    # Strings: 0, 12, 56, 64, 80
    0, 12, 56, 64, 80,
    # Lists/Arrays: 3, 33, 37, 68, 70, 90
    3, 33, 37, 68, 70, 90,
    # Dicts/Sets: 43, 95, 111
    43, 95, 111,
    # Math: 4, 24, 49, 63, 75, 96
    4, 24, 49, 63, 75, 96,
    # Logic/Simulation: 47, 65, 67, 73, 93
    47, 65, 67, 73, 93,
    # Search/Sort: 55, 76, 99, 107, 116
    55, 76, 99, 107, 116,
])


def main():
    parser = argparse.ArgumentParser(
        description="Convert HumanEval+ problems to puzzle JSON format"
    )
    parser.add_argument("--count", type=int, default=50,
                        help="Number of problems to convert (default: 50, ignored if --indices used)")
    parser.add_argument("--indices", type=str, default=None,
                        help="Comma-separated HumanEval problem indices (e.g. '1,5,6,8')")
    parser.add_argument("--curated20", action="store_true",
                        help="Use curated 20-problem set balanced for 1.5B eval")
    parser.add_argument("--curated50", action="store_true",
                        help="Use curated 50-problem set (20 original + 30 new)")
    parser.add_argument("--output-dir", type=str, default="puzzles_humaneval",
                        help="Output directory (default: puzzles_humaneval)")
    args = parser.parse_args()

    print("Loading HumanEval+ dataset from HuggingFace...")
    problems = load_humaneval_plus()
    print(f"Loaded {len(problems)} problems.")

    by_num = {extract_task_number(p["task_id"]): p for p in problems}

    if args.curated50:
        indices = CURATED_50
        print(f"Using curated 50-problem set: {indices}")
    elif args.curated20:
        indices = CURATED_20
        print(f"Using curated 20-problem set: {indices}")
    elif args.indices:
        indices = [int(x.strip()) for x in args.indices.split(",")]
        print(f"Using specified indices: {indices}")
    else:
        indices = sorted(by_num.keys())[:args.count]

    os.makedirs(args.output_dir, exist_ok=True)

    converted = 0
    total = len(indices)
    for i, num in enumerate(indices):
        if num not in by_num:
            print(f"  WARNING: HumanEval/{num} not found, skipping")
            continue
        problem = by_num[num]
        idx = i + 1
        puzzle = convert_problem(problem)

        slug = re.sub(r"[^a-z0-9_]", "", puzzle["puzzle_id"].replace("-", "_"))
        filename = f"{idx:03d}_{slug}.json"
        path = os.path.join(args.output_dir, filename)

        with open(path, "w") as f:
            json.dump(puzzle, f, indent=2)

        print(f"  [{idx:>3}/{total}] {problem['task_id']:>15s} -> {filename}")
        converted += 1

    print(f"\nDone. Converted {converted} problems to {args.output_dir}/")


if __name__ == "__main__":
    main()
