#!/usr/bin/env python3
"""Resumable sequential puzzle generator using Anthropic Opus 4.6."""

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
You are an elite Competitive Programming Architect and Reinforcement Learning Data Engineer. \
Your task is to generate highly rigorous, deterministic coding puzzles to train an AI agent.

You must generate a unique, non-trivial algorithmic puzzle (avoiding standard LeetCode clones like "Two Sum").

CRITICAL REQUIREMENT - TEST VALIDITY:
Before outputting the final JSON, you must use a <scratchpad> to:
1. Write a perfect Python reference solution.
2. Write the unit tests (including hidden edge cases, large inputs, and boundary conditions).
3. Conceptually execute your reference solution against every single test case step-by-step.
4. If a test case is flawed or contradicts the prompt, fix it in the scratchpad before finalizing.

Output ONLY a single, valid JSON object matching the exact schema below. \
Do not include markdown formatting around the JSON, just the raw JSON object.

{
  "puzzle_id": "a_unique_descriptive_slug",
  "difficulty": "Medium",
  "prompt": "The detailed problem description. Include constraints and expected time/space complexity.",
  "starter_code": "def function_name(args):\\n    # TODO: Implement solution\\n    pass",
  "reference_solution": "The complete, working Python code that solves the problem.",
  "unit_tests": "import unittest\\n\\nclass TestSolution(unittest.TestCase):\\n    # Include at least 3 visible tests and 5 hidden edge-case tests\\n    # Name the hidden tests starting with test_hidden_\\n\\nif __name__ == '__main__':\\n    unittest.main()",
  "validation_trace": "A brief string explaining how you verified that the reference solution passes all tests."
}
"""

THEME_PAIRS = [
    ("graph theory", "delivery logistics"),
    ("dynamic programming", "resource allocation"),
    ("string manipulation", "DNA sequence analysis"),
    ("tree traversal", "file system simulation"),
    ("greedy algorithms", "scheduling optimization"),
    ("bit manipulation", "encryption ciphers"),
    ("sliding window", "network packet analysis"),
    ("union-find", "social network clustering"),
    ("topological sort", "build system dependencies"),
    ("segment trees", "range-based sensor data"),
    ("trie structures", "autocomplete engines"),
    ("heap / priority queue", "emergency room triage"),
    ("backtracking", "puzzle board generation"),
    ("matrix operations", "image transformation"),
    ("number theory", "hash collision analysis"),
    ("geometry", "drone path planning"),
    ("interval scheduling", "satellite communication windows"),
    ("stack-based parsing", "custom language interpreters"),
    ("BFS/DFS", "maze generation with constraints"),
    ("binary search", "supply chain optimization"),
    ("combinatorics", "password policy enumeration"),
    ("two pointers", "audio waveform merging"),
    ("memoization", "game state evaluation"),
    ("graph coloring", "radio frequency assignment"),
    ("shortest path", "underground tunnel navigation"),
    ("flow networks", "water pipe distribution"),
    ("divide and conquer", "parallel task splitting"),
    ("linked list manipulation", "memory pool management"),
    ("monotonic stack", "skyline visibility analysis"),
    ("suffix arrays", "genomic pattern matching"),
    ("convex hull", "territorial boundary mapping"),
    ("fenwick tree", "real-time leaderboard updates"),
    ("hashing", "distributed cache routing"),
    ("simulation", "cellular automata evolution"),
    ("probability", "card game strategy optimization"),
    ("recursion", "fractal dimension calculation"),
    ("sorting variants", "multi-criteria ranking"),
    ("cycle detection", "deadlock identification"),
    ("bitmask DP", "feature flag combinations"),
    ("coordinate compression", "event timeline visualization"),
    ("sparse tables", "seismic data range queries"),
    ("disjoint intervals", "calendar conflict resolution"),
    ("KMP / string matching", "plagiarism detection engine"),
    ("modular arithmetic", "clock synchronization protocols"),
    ("state machines", "protocol validation"),
    ("minimax", "adversarial board game AI"),
    ("sweep line", "overlapping rectangle area"),
    ("random algorithms", "load balancer simulation"),
    ("multi-source BFS", "wildfire spread prediction"),
    ("LRU design", "browser cache eviction policy"),
]

DIFFICULTIES = [
    "Medium", "Medium", "Hard", "Medium", "Easy",
    "Hard", "Medium", "Medium", "Hard", "Medium",
    "Medium", "Easy", "Hard", "Medium", "Medium",
    "Hard", "Medium", "Medium", "Easy", "Medium",
    "Hard", "Medium", "Medium", "Hard", "Medium",
    "Medium", "Hard", "Easy", "Medium", "Hard",
    "Medium", "Medium", "Hard", "Medium", "Easy",
    "Hard", "Medium", "Medium", "Hard", "Medium",
    "Medium", "Easy", "Medium", "Hard", "Medium",
    "Hard", "Medium", "Medium", "Medium", "Hard",
]

REQUIRED_KEYS = {
    "puzzle_id", "difficulty", "prompt", "starter_code",
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
    """Extract a JSON object from the model's response text, stripping any scratchpad."""
    # The model may wrap JSON in markdown fences or include a scratchpad before it.
    # Strategy: find the last top-level { ... } block in the text.
    brace_depth = 0
    start = None
    last_json_start = None
    last_json_end = None

    for i, ch in enumerate(text):
        if ch == "{":
            if brace_depth == 0:
                start = i
            brace_depth += 1
        elif ch == "}":
            brace_depth -= 1
            if brace_depth == 0 and start is not None:
                last_json_start = start
                last_json_end = i + 1

    if last_json_start is not None:
        candidate = text[last_json_start:last_json_end]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    return None


def validate_puzzle(data: dict) -> bool:
    """Check that the parsed JSON has all required keys with non-empty string values."""
    if not isinstance(data, dict):
        return False
    for key in REQUIRED_KEYS:
        if key not in data or not isinstance(data[key], str) or not data[key].strip():
            return False
    return True


def build_user_message(index: int, theme: Tuple[str, str], difficulty: str, existing_ids: List[str]) -> str:
    """Build the user message for a single puzzle generation call."""
    parts = [
        f"Generate puzzle #{index} of 50.",
        f"Theme hint: {theme[0]} + {theme[1]}",
        f"Difficulty: {difficulty}",
    ]
    if existing_ids:
        parts.append(
            f"Do NOT reuse any concepts from these previously generated puzzle IDs: {', '.join(existing_ids)}"
        )
    return "\n".join(parts)


def generate_one(
    client: anthropic.Anthropic,
    index: int,
    theme: Tuple[str, str],
    difficulty: str,
    existing_ids: List[str],
) -> Optional[dict]:
    """Call the API to generate a single puzzle. Retries up to MAX_RETRIES times."""
    user_msg = build_user_message(index, theme, difficulty, existing_ids)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"  Attempt {attempt}/{MAX_RETRIES}...", flush=True)
            with client.messages.stream(
                model="claude-opus-4-6",
                max_tokens=128000,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            ) as stream:
                response = stream.get_final_message()

            if response.stop_reason == "max_tokens":
                print(f"  Response truncated (hit 128K token limit), retrying (attempt {attempt}).", flush=True)
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

            return data

        except anthropic.APIConnectionError as e:
            print(f"  Connection error: {e} (attempt {attempt}).", flush=True)
            time.sleep(2 ** attempt)
        except anthropic.RateLimitError as e:
            wait = 2 ** (attempt + 2)
            print(f"  Rate limited, waiting {wait}s... (attempt {attempt}).", flush=True)
            time.sleep(wait)
        except anthropic.APIStatusError as e:
            print(f"  API error {e.status_code}: {e.message} (attempt {attempt}).", flush=True)
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
    parser = argparse.ArgumentParser(description="Generate coding puzzles via Anthropic API")
    parser.add_argument("--count", type=int, default=50, help="Total puzzles to generate (default: 50)")
    parser.add_argument("--puzzle-dir", type=str, default="puzzles", help="Output directory (default: puzzles)")
    args = parser.parse_args()

    load_dotenv()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set. Create a .env file or export it.")
        return

    client = anthropic.Anthropic()
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
        difficulty = DIFFICULTIES[i - 1]
        existing_ids = list(existing.values())

        print(f"[{i}/{total}] Generating: {theme[0]} + {theme[1]} ({difficulty})", flush=True)

        puzzle = generate_one(client, i, theme, difficulty, existing_ids)

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
