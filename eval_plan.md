# Evaluation Plan: Identifying Qwen Weaknesses for RL Training

## What You Actually Run

### Run 1 — Baseline Inference + Unit Tests (THE essential run)

Generate k rollouts (k=5-8) per puzzle, execute each against unit tests, record everything.

**Raw output per rollout**:
- `pass`: did it pass all tests?
- `visible_passed`: count of visible tests passed
- `hidden_passed`: count of hidden tests passed
- `error_type`: one of `syntax_error | runtime_error | wrong_answer | timeout | partial_pass | full_pass`
- `execution_time_ms`: wall clock time

This single run gives you everything for the core analysis. All "evals" below are just **aggregation views** over this one dataset — no additional inference needed.

**Aggregation views you compute offline from Run 1 data**:

| View | What it tells you | How to compute |
|---|---|---|
| **pass@1 per puzzle** | Ground truth per problem | First rollout result |
| **pass@k per puzzle** | Can the model solve it at all? | `1 - C(n-c, k) / C(n, k)` where c = number of passing rollouts |
| **Concept row heatmap** | Which micro-skills does it lack? | Average pass@1 grouped by concept row (see `concept_map.md`) |
| **Difficulty breakdown** | Easy/Medium/Hard pass rates | Group by difficulty field |
| **Error type distribution** | What kind of failures? (syntax vs wrong answer vs TLE) | Count error_type values |
| **Advantage spread per puzzle** | Where will RL have most impact? | `max(scores) - min(scores)` across k rollouts |
| **Skill family radar** | High-level weakness areas | Average pass@1 grouped by skill family |

### Run 2 — Algorithm Selection Probe (optional, only if you want deeper diagnosis)

For puzzles that failed in Run 1, prompt Qwen to *describe its approach without writing code*. Compare to reference solution's algorithm. This separates "doesn't know the algorithm" from "knows it but can't implement it."

Only worth doing if Run 1 shows the model failing many puzzles and you need to decide between:
- RL reward on algorithm selection (chain-of-thought reward)
- RL reward on implementation correctness (test-passing reward)

If pass@k is decent but pass@1 is bad, skip this — the model knows the algorithms, it's just inconsistent. RL on test-pass reward is sufficient.


## Interpreting Run 1 Results for RL

| What you see | What it means | RL action |
|---|---|---|
| **High pass@k, low pass@1** | Model can solve it but is inconsistent | Standard GRPO works great here — good rollouts provide reward signal |
| **pass@k = 0** | Model cannot solve it at all | Needs curriculum learning — add easier variants of that skill first |
| **Passes visible, fails hidden** | Gets the algorithm right, misses edge cases | Add reward shaping bonus for edge-case handling (empty input, boundary values) |
| **Mostly syntax/runtime errors** | Basic code generation issues | Add compilation + execution reward as baseline signal |
| **Mostly timeout errors** | Right algorithm, wrong complexity | Add time-budget reward — bonus for efficient solutions |
| **Mostly wrong answer** | Wrong algorithm or logic errors | Core RL target — reward test-pass fraction |
| **Concept row at 0% with paired puzzles** | Confirmed skill gap (not a fluke) | High-priority RL training target |
| **Concept row at 0% singleton puzzle** | Possible skill gap (could be puzzle-specific) | Generate more puzzles in that row to confirm before investing RL compute |


## Priority: What to Target with RL

After Run 1, rank concept rows for RL targeting:

```
Priority 1: Rows with high advantage spread (model sometimes solves, sometimes doesn't)
            → RL will have the highest marginal return here

Priority 2: Rows with 0% pass@k but error_type = "wrong_answer" or "partial_pass"
            → Model attempts the right structure but fails — curriculum + RL

Priority 3: Rows with 0% pass@k and error_type = "syntax_error" or "runtime_error"
            → Model doesn't know where to start — needs the most scaffolding
```


## Output Artifacts

| File | Contents |
|---|---|
| `eval_results/raw_rollouts.jsonl` | One line per (puzzle, rollout) with all metrics |
| `eval_results/summary.json` | Aggregated: per-puzzle pass@1, pass@k, error breakdown, advantage |
| `eval_results/concept_heatmap.json` | pass@1 and pass@k per concept row |
| `eval_results/rl_targets_ranked.json` | Concept rows ranked by RL training priority |
