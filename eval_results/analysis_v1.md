# Evaluation Analysis — v1 Dataset (50 puzzles)

**Date:** 2026-03-01
**Model:** Qwen2.5-Coder-14B-Instruct (fp16, L40S)
**Config:** 8 rollouts/puzzle, temperature 0.7

---

## TL;DR

The dataset is too hard. Only 38% of puzzles land in the RL learning zone. The difficulty
distribution (12% Easy / 58% Medium / 30% Hard) combined with the algorithmic complexity of
even the "Medium" puzzles means 44% of the dataset produces zero reward signal. Regenerate
with a flatter difficulty curve heavily weighted toward Easy/Medium, and avoid niche
algorithmic topics the model has clearly never seen.

---

## What Worked

**The evaluation pipeline itself is solid.** Error classification, advantage spread
computation, and skill family grouping all produced actionable signal. The report format
directly answers "where should RL focus?" — keep this infrastructure for v2.

**14B was the right model choice.** The overall 36.5% pass@1 puts us in a range where
there's meaningful variance. The 1.5B model produced ~0% on this same dataset. Going larger
(32B) would have pushed Easy/Medium into saturation without helping much on the Hards that
are genuinely beyond pattern-matching.

**Error profile is healthy.** 50% wrong answer + 9% partial pass + only 1% syntax/runtime
errors means the model *understands the task format* and writes valid Python — it just can't
solve the problems. This is exactly the regime where RL can help (vs. if it were mostly
syntax errors, that would indicate a format/prompting issue).

---

## What We Learned About Difficulty

### The "Easy" tier is nearly saturated

| Metric | Value |
|---|---|
| pass@1 | 83.3% |
| pass@8 | 100% |
| Learning zone | 2/6 |

All 6 Easy puzzles were solved at least once. 4/6 are saturated (model already masters them).
The 2 in the learning zone (`memory_pool_linked_list`, one other) still had high pass rates.
**Takeaway:** Current "Easy" is appropriately easy. Regeneration should produce more puzzles
at this level — these are the foundation of the curriculum.

### Medium is a mixed bag — some great, some dead

| Metric | Value |
|---|---|
| pass@1 | 40.1% |
| pass@8 | 62.1% |
| Learning zone | 13/29 (45%) |

13 out of 29 Mediums are in the learning zone — good, but 16 are either dead or saturated.
The Mediums that *work* for RL involve recognizable algorithmic patterns (greedy, basic DP,
BFS, union-find). The dead Mediums tend to involve domain-specific implementation complexity
(interpreters, sweep lines, tree path parsing) where the model has no partial-credit path.

### Hard is mostly dead

| Metric | Value |
|---|---|
| pass@1 | 10.8% |
| pass@8 | 26.7% |
| Learning zone | 4/15 (27%) |

11 out of 15 Hard puzzles are in the dead zone. The 4 that work (`consistent_hash_ring`,
`fractal_similarity_dimension`, `feature_flag_rollout_optimizer`, `dna_sequence_assembly`)
happen to decompose into subproblems the model partially recognizes. The rest are truly
beyond 14B's capability — no amount of RL will bridge the gap on Chinese Remainder Theorem
or Convex Hull from zero.

---

## Skill Family Analysis

### Strong families (saturated or nearly — good for warm-up, not RL)
- **Greedy / Two-Pointer / Sliding Window**: 94% pass@1 — model has seen tons of these
- **Probability / Expected Value**: 100% — trivial at this level
- **Simulation / State Machines**: 60% pass@1, 80% pass@8 — well-represented in training data

### RL sweet spot (high variance = high advantage signal)
- **Dynamic Programming**: 16% pass@1 but 50% pass@8 with 5/10 in learning zone. The model
  *sometimes* finds the right DP formulation. This is the highest-value skill family for RL.
- **Data Structures**: 29% pass@1, 71% pass@8, 5/7 in learning zone. Similar story — model
  knows the structures but struggles with correct implementation under constraints.
- **Graph Algorithms**: 32% pass@1, 43% pass@8, 2/7 in learning zone. Mixed — BFS/Dijkstra
  works, but complex multi-constraint graph problems are dead.

### Dead families (skip or drastically simplify in v2)
- **Computational Geometry**: 0% pass@1, 0% pass@8. Complete dead zone. Convex hull,
  point-in-polygon, shortest path around obstacles — 14B has essentially no training signal
  for these.
- **Tree / Path Aggregation**: 0% / 0%. Niche topic.
- **Sweep Line / Coord Compression**: 33% pass@1 but the confirmed skill gap (Row F at 0%)
  shows that when sweep line is the *core* technique (not just a component), the model fails.

---

## Recommendations for v2 Dataset

### 1. Rebalance difficulty distribution

Current: 12% Easy / 58% Medium / 30% Hard
Target:  **40% Easy / 45% Medium / 15% Hard**

This puts the bulk of puzzles where 14B has the best RL signal. With 40% Easy, even if some
saturate, you'll have ~15-20 puzzles as warm-up and curriculum anchors.

### 2. Define difficulty by model capability, not human intuition

The generator's notion of "Medium" includes things like sweep-line coordinate compression
and stack-based interpreters, which are effectively "Hard" for 14B. Recalibrate:

| v2 Difficulty | Target pass@8 with 14B | Algorithmic profile |
|---|---|---|
| Easy | 80-100% | Standard patterns: sorting, greedy, basic BFS/DFS, simple DP, two-pointer |
| Medium | 30-70% | Recognizable but tricky: knapsack variants, Dijkstra with constraints, union-find, trie operations |
| Hard | 5-30% | Multi-step composition: bitmask DP, segment trees, max-flow, game theory |

### 3. Avoid these skill areas entirely (dead zone for 14B)

- Computational geometry (convex hull, Voronoi, polygon operations)
- Chinese Remainder Theorem / extended GCD
- Suffix arrays / LCP
- Multi-dimensional constrained counting DP
- Custom language interpreters

### 4. Double down on high-signal skill families

Generate more puzzles in:
- **Dynamic Programming** (vary the subproblem structure — knapsack, interval, digit, tree DP)
- **Data Structures** (trie, Fenwick tree, segment tree basics, monotonic stack)
- **Graph Algorithms** (BFS/DFS variants, shortest path variants, union-find applications)

These families showed the highest advantage spread — the model has partial knowledge that RL
can refine.

### 5. Ensure concept row coverage with 2+ puzzles per skill

Singletons (marked `?` in the heatmap) can't produce confirmed skill gaps. Each micro-skill
should have at least 2 puzzles so the eval can distinguish "model can't do X" from "this
specific puzzle is weird."

---

## Top 10 RL Training Targets (from this dataset, reusable in v2)

These puzzles had the highest advantage spread and should be kept or used as templates:

| Rank | Puzzle | Difficulty | Skill Family | Variance |
|---|---|---|---|---|
| 1 | water_pipe_distribution | Medium | Graph Algorithms | 0.220 |
| 2 | fractal_similarity_dimension | Hard | Number Theory | 0.194 |
| 3 | social_influence_clusters | Medium | Data Structures | 0.191 |
| 4 | feature_flag_rollout_optimizer | Hard | Dynamic Programming | 0.188 |
| 5 | consistent_hash_ring_router | Hard | System Design | 0.173 |
| 6 | ring_rotation_image_transform | Medium | Simulation | 0.136 |
| 7 | dna_sequence_assembly | Hard | Dynamic Programming | 0.119 |
| 8 | realtime_leaderboard_fenwick | Medium | Data Structures | 0.105 |
| 9 | memory_pool_linked_list | Easy | Data Structures | 0.105 |
| 10 | satellite_comm_windows | Medium | Graph Algorithms | 0.102 |

**Common thread:** These puzzles use well-known algorithmic patterns but with enough
domain-specific wrapping that the model doesn't instantly solve them. This is the template
for v2 puzzle design.

---

## Raw Numbers

```
pass@1:          36.5%
pass@8 > 0:      28/50 (56%)
Learning zone:   19/50 (38%)
Dead zone:       22/50 (44%)
Saturated:        9/50 (18%)
Mean test pass:  61.8%
Syntax errors:    0.2%
Runtime errors:   0.8%
Wrong answers:   50.2%
Timeouts:         3.0%
```
