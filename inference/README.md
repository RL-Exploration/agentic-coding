# Inference & Evaluation

Loads Qwen2.5-Coder-1.5B-Instruct directly via HuggingFace transformers (`model.generate()`)
and runs the evaluation pipeline from `eval_plan.md` against all 50 coding puzzles.

## Quick Start (Local / Lightning AI)

```bash
# 1. Install dependencies (needs a CUDA GPU)
pip install -r requirements.txt

# 2. Run the evaluation (loads model + runs all puzzles)
python run_eval.py
```

Results are saved to `../eval_results/`.

## Lightning AI Deployment

```bash
# 1. Create a Lightning Studio with a GPU (L4 or T4 is sufficient for 1.5B)
#    Go to https://lightning.ai → New Studio → Select GPU

# 2. Clone the repo inside the Studio
git clone <your-repo-url> && cd RL_Coding/inference

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the full eval
python run_eval.py
```

## CLI Reference

### server.py (interactive testing)

```
python server.py [--model MODEL] [--device DEVICE] [--temperature T]
```

| Flag | Default | Description |
|---|---|---|
| `--model` | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | HuggingFace model ID |
| `--device` | `auto` | Device map for model loading |
| `--max-new-tokens` | `2048` | Max tokens to generate |
| `--temperature` | `0.7` | Sampling temperature |

### run_eval.py

```
python run_eval.py [--model MODEL] [--samples K] [--analyze ROLLOUTS_FILE]
```

| Flag | Default | Description |
|---|---|---|
| `--model` | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | HuggingFace model ID |
| `--device` | `auto` | Device map for model loading |
| `--samples` | `8` | Rollouts per puzzle (k for pass@k) |
| `--timeout` | `30` | Test execution timeout (seconds) |
| `--temperature` | `0.7` | Sampling temperature |
| `--output-dir` | `../eval_results` | Where to save results |
| `--analyze` | — | Re-run analytics on existing `raw_rollouts.jsonl` |

## Output Artifacts

| File | Contents |
|---|---|
| `raw_rollouts.jsonl` | One line per (puzzle, rollout) with all metrics |
| `summary.json` | Per-puzzle: pass@1, pass@k, error breakdown, advantage |
| `concept_heatmap.json` | pass@1 and pass@k per concept row |
| `rl_targets_ranked.json` | Concept rows ranked by RL training priority |

## Re-running Analytics

After a full run, you can re-analyze results without re-running inference:

```bash
python run_eval.py --analyze ../eval_results/raw_rollouts.jsonl
```

This recomputes all aggregation views and regenerates the report and artifact files.
