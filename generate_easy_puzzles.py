#!/usr/bin/env python3
"""Resumable sequential puzzle generator for EASY puzzles targeting Qwen 1.5-3B.

Uses Anthropic Opus 4.6 to generate puzzles calibrated so a small (1.5-3B param)
language model lands in the RL learning zone — sometimes solving, sometimes not.
"""

from __future__ import annotations

import json
import os
import re
import glob
import time
import argparse
from typing import Dict, List, Optional, Tuple

import anthropic
from dotenv import load_dotenv

SYSTEM_PROMPT = """\
You are a Python Programming Instructor designing beginner-friendly coding exercises \
for training a SMALL language model (1.5 billion parameters) via reinforcement learning.

DIFFICULTY CALIBRATION — THIS IS CRITICAL:
The target model is roughly equivalent to a first-year CS student. It can handle:
  - Basic for/while loops and conditionals
  - List, dict, set, and string built-in operations
  - Simple arithmetic and comparisons
  - Functions that are 5-20 lines long

The model CANNOT reliably handle:
  - Dynamic programming with state tables
  - Graph algorithms (BFS, DFS, Dijkstra, etc.)
  - Tree or linked-list traversal
  - Heaps, tries, segment trees, or any advanced data structure
  - Recursion deeper than 1-2 levels
  - Math beyond basic arithmetic, modulo, GCD
  - Bit manipulation beyond simple AND/OR/XOR
  - Any problem requiring more than ~15 lines of logic

PUZZLE REQUIREMENTS:
1. The reference solution MUST be solvable in <=20 lines of Python (excluding imports).
2. Use only Python builtins and standard library (no numpy, etc.).
3. Each puzzle should have ONE clear algorithmic idea (not a composition of multiple techniques).
4. Include clear input/output examples in the prompt.
5. Constraints should be small enough that brute-force O(n^2) solutions pass.

CRITICAL REQUIREMENT - TEST VALIDITY:
Before outputting the final JSON, you must use a <scratchpad> to:
1. Write a perfect Python reference solution.
2. Write the unit tests (including hidden edge cases and boundary conditions).
3. Conceptually execute your reference solution against every single test case step-by-step.
4. If a test case is flawed or contradicts the prompt, fix it in the scratchpad before finalizing.

Output ONLY a single, valid JSON object matching the exact schema below. \
Do not include markdown formatting around the JSON, just the raw JSON object.

{
  "puzzle_id": "a_unique_descriptive_slug",
  "difficulty": "Easy",
  "category": "The skill category (provided in the prompt — copy it exactly)",
  "prompt": "The detailed problem description with examples. Include input/output types and constraints.",
  "starter_code": "def function_name(args):\\n    # TODO: Implement solution\\n    pass",
  "reference_solution": "The complete, working Python code (<=20 lines of logic).",
  "unit_tests": "import unittest\\n\\nclass TestSolution(unittest.TestCase):\\n    # Include at least 3 visible tests and 5 hidden edge-case tests\\n    # Name the hidden tests starting with test_hidden_\\n\\nif __name__ == '__main__':\\n    unittest.main()",
  "validation_trace": "A brief string explaining how you verified that the reference solution passes all tests."
}
"""

CATEGORIES = [
    "Strings",
    "Lists & Arrays",
    "Dicts & Sets",
    "Basic Math",
    "Logic & Simulation",
    "Search & Sort",
]

# 50 theme pairs: (technique, application domain, category).
# First 20 are interleaved across all 6 categories (3-4 each) so that
# --count 20 gives balanced coverage for quick eval runs.
THEME_PAIRS: List[Tuple[str, str, str]] = [
    # ── First 20: balanced across categories ──
    # Strings (1-3)
    ("string reversal", "message decoding", "Strings"),
    ("palindrome check", "word games", "Strings"),
    ("character frequency counting", "text analytics", "Strings"),
    # Lists & Arrays (4-7)
    ("find max and min", "sensor readings", "Lists & Arrays"),
    ("remove duplicates preserving order", "data cleaning", "Lists & Arrays"),
    ("rotate array by k positions", "circular buffer", "Lists & Arrays"),
    ("flatten nested list", "JSON data extraction", "Lists & Arrays"),
    # Dicts & Sets (8-10)
    ("word frequency count", "book word cloud", "Dicts & Sets"),
    ("group items by key", "student grade grouping", "Dicts & Sets"),
    ("two-sum with hash map", "receipt matching", "Dicts & Sets"),
    # Basic Math (11-13)
    ("primality check", "number classification", "Basic Math"),
    ("Fibonacci sequence", "population modeling", "Basic Math"),
    ("digit sum", "checksum validation", "Basic Math"),
    # Logic & Simulation (14-17)
    ("FizzBuzz variant", "calendar labeling", "Logic & Simulation"),
    ("balanced brackets check", "syntax validation", "Logic & Simulation"),
    ("matrix transpose", "spreadsheet operations", "Logic & Simulation"),
    ("temperature conversion table", "weather dashboard", "Logic & Simulation"),
    # Search & Sort (18-20)
    ("binary search", "dictionary lookup", "Search & Sort"),
    ("insertion sort step", "card sorting", "Search & Sort"),
    ("kth smallest element", "percentile computation", "Search & Sort"),
    # ── Remaining 30: fill out each category ──
    # Strings continued (21-27)
    ("case conversion", "data normalization", "Strings"),
    ("substring search", "log parsing", "Strings"),
    ("anagram detection", "word puzzles", "Strings"),
    ("vowel filtering", "text processing", "Strings"),
    ("Caesar cipher shift", "simple encryption", "Strings"),
    ("title case conversion", "document formatting", "Strings"),
    ("run-length encoding", "data compression", "Strings"),
    # Lists & Arrays continued (28-33)
    ("merge two sorted lists", "sorted file merging", "Lists & Arrays"),
    ("chunk list into groups", "batch processing", "Lists & Arrays"),
    ("list intersection", "common friends", "Lists & Arrays"),
    ("running sum / prefix sum", "bank balance history", "Lists & Arrays"),
    ("second largest element", "competition ranking", "Lists & Arrays"),
    ("moving average", "stock price smoothing", "Lists & Arrays"),
    # Dicts & Sets continued (34-38)
    ("invert a dictionary", "reverse lookup table", "Dicts & Sets"),
    ("set union and intersection", "playlist merging", "Dicts & Sets"),
    ("most common element", "voting tallies", "Dicts & Sets"),
    ("first non-repeating character", "stream processing", "Dicts & Sets"),
    ("simple phone directory", "contact lookup", "Dicts & Sets"),
    # Basic Math continued (39-42)
    ("factorial computation", "permutation counting", "Basic Math"),
    ("GCD and LCM", "gear ratio calculation", "Basic Math"),
    ("number to English words", "check printing", "Basic Math"),
    ("Roman numeral conversion", "clock display", "Basic Math"),
    # Logic & Simulation continued (43-46)
    ("spiral order traversal", "printer rasterization", "Logic & Simulation"),
    ("tic-tac-toe winner check", "game state evaluation", "Logic & Simulation"),
    ("basic expression evaluation", "simple calculator", "Logic & Simulation"),
    ("date validation", "form input checking", "Logic & Simulation"),
    # Search & Sort continued (47-50)
    ("counting sort", "grade distribution", "Search & Sort"),
    ("sort by custom key", "leaderboard ranking", "Search & Sort"),
    ("partition around pivot", "data bucketing", "Search & Sort"),
    ("search in rotated sorted array", "server log lookup", "Search & Sort"),
]

REQUIRED_KEYS = {
    "puzzle_id", "difficulty", "category", "prompt", "starter_code",
    "reference_solution", "unit_tests", "validation_trace",
}

MAX_RETRIES = 3


def scan_existing_puzzles(puzzle_dir: str) -> Dict[int, str]:
    """Scan the puzzles directory and return {index: puzzle_id} for existing files."""
    existing = {}
    for path in glob.glob(os.path.join(puzzle_dir, "*.json")):
        filename = os.path.basename(path)
        match = re.match(r"^(\d{3})_(.+)\.json$", filename)
        if match:
            idx = int(match.group(1))
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                existing[idx] = data.get("puzzle_id", match.group(2))
            except (json.JSONDecodeError, OSError):
                pass
    return existing


def extract_json(text: str) -> Optional[dict]:
    """Extract a JSON object from the model's response text, stripping any scratchpad.

    Uses json.JSONDecoder.raw_decode which handles all edge cases (braces inside
    strings, escaped quotes, etc.) that a manual brace-depth counter cannot.
    Tries each '{' position and returns the largest valid dict found.
    """
    decoder = json.JSONDecoder()
    best: Optional[dict] = None
    best_size = 0
    i = 0
    n = len(text)
    while i < n:
        if text[i] == "{":
            try:
                obj, end = decoder.raw_decode(text, i)
                if isinstance(obj, dict) and len(obj) > best_size:
                    best = obj
                    best_size = len(obj)
                i = end
            except json.JSONDecodeError:
                i += 1
        else:
            i += 1
    return best


def validate_puzzle(data: dict) -> bool:
    """Check that the parsed JSON has all required keys with non-empty string values."""
    if not isinstance(data, dict):
        return False
    for key in REQUIRED_KEYS:
        if key not in data or not isinstance(data[key], str) or not data[key].strip():
            return False
    return True


def build_user_message(index: int, theme: Tuple[str, str, str], total: int,
                       existing_ids: List[str]) -> str:
    """Build the user message for a single puzzle generation call."""
    parts = [
        f"Generate EASY puzzle #{index} of {total}.",
        f"Technique: {theme[0]}",
        f"Application domain: {theme[1]}",
        f"Category: {theme[2]}",
        "Difficulty: Easy",
        "Remember: the solution must be <=20 lines, use only basic loops/conditionals/builtins, "
        "and be solvable by a beginner programmer.",
    ]
    if existing_ids:
        parts.append(
            f"Do NOT reuse any concepts from these previously generated puzzle IDs: "
            f"{', '.join(existing_ids[-20:])}"
        )
    return "\n".join(parts)


def generate_one(
    client: anthropic.Anthropic,
    index: int,
    total: int,
    theme: Tuple[str, str, str],
    existing_ids: List[str],
) -> Optional[dict]:
    """Call the API to generate a single puzzle. Retries up to MAX_RETRIES times."""
    user_msg = build_user_message(index, theme, total, existing_ids)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"  Attempt {attempt}/{MAX_RETRIES}...", flush=True)
            with client.messages.stream(
                model="claude-opus-4-6",
                max_tokens=16384,
                timeout=300,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            ) as stream:
                response = stream.get_final_message()

            if response.stop_reason == "max_tokens":
                print(f"  Response truncated, retrying (attempt {attempt}).", flush=True)
                continue

            raw_text = response.content[0].text
            data = extract_json(raw_text)

            if data is None:
                print(f"  Failed to extract JSON from response (attempt {attempt}).", flush=True)
                continue

            if not validate_puzzle(data):
                missing = REQUIRED_KEYS - set(data.keys()) if isinstance(data, dict) else REQUIRED_KEYS
                print(f"  Invalid puzzle structure, missing/empty keys: {missing} (attempt {attempt}).", flush=True)
                continue

            data["difficulty"] = "Easy"
            data["category"] = theme[2]
            return data

        except anthropic.RateLimitError:
            wait = 2 ** (attempt + 2)
            print(f"  Rate limited, waiting {wait}s... (attempt {attempt}).", flush=True)
            time.sleep(wait)
        except anthropic.APIStatusError as e:
            print(f"  API error {e.status_code}: {e.message} (attempt {attempt}).", flush=True)
            time.sleep(2 ** attempt)
        except Exception as e:
            print(f"  {type(e).__name__}: {e} (attempt {attempt}).", flush=True)
            time.sleep(2 ** attempt)

    return None


def save_puzzle(puzzle_dir: str, index: int, puzzle: dict) -> str:
    """Write the puzzle to disk. Returns the file path."""
    slug = re.sub(r"[^a-z0-9_]", "", puzzle["puzzle_id"].lower().replace("-", "_"))
    filename = f"{index:03d}_{slug}.json"
    path = os.path.join(puzzle_dir, filename)
    with open(path, "w") as f:
        json.dump(puzzle, f, indent=2)
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Generate EASY coding puzzles (targeting Qwen 1.5-3B) via Anthropic API"
    )
    parser.add_argument("--count", type=int, default=50,
                        help="Total puzzles to generate (default: 50)")
    parser.add_argument("--puzzle-dir", type=str, default="puzzles_easy",
                        help="Output directory (default: puzzles_easy)")
    args = parser.parse_args()

    load_dotenv()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set. Create a .env file or export it.")
        return

    client = anthropic.Anthropic(timeout=300.0)
    os.makedirs(args.puzzle_dir, exist_ok=True)

    existing = scan_existing_puzzles(args.puzzle_dir)
    total = min(args.count, len(THEME_PAIRS))
    generated = 0
    failed = 0

    print(f"Found {len(existing)} existing puzzle(s). Generating up to {total - len(existing)} more.\n", flush=True)

    for i in range(1, total + 1):
        if i in existing:
            print(f"[{i}/{total}] Already exists ({existing[i]}), skipping.", flush=True)
            continue

        theme = THEME_PAIRS[i - 1]
        existing_ids = list(existing.values())

        print(f"[{i}/{total}] Generating: {theme[0]} + {theme[1]} [{theme[2]}]", flush=True)

        puzzle = generate_one(client, i, total, theme, existing_ids)

        if puzzle:
            path = save_puzzle(args.puzzle_dir, i, puzzle)
            existing[i] = puzzle["puzzle_id"]
            generated += 1
            print(f"  Saved: {path}\n", flush=True)
        else:
            failed += 1
            print(f"  FAILED after {MAX_RETRIES} attempts, moving on.\n", flush=True)

    print(f"\nDone. Generated: {generated}, Failed: {failed}, Total on disk: {len(existing)}", flush=True)


if __name__ == "__main__":
    main()
