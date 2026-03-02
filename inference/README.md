# Inference & Evaluation

Runs the evaluation pipeline against coding puzzles using a HuggingFace model.
Supports three puzzle sets:

- **puzzles/** -- v1 dataset (50 hard algorithmic puzzles, targeting 14B)
- **puzzles_easy/** -- v2 dataset (easy puzzles, targeting Qwen 1.5B for RL)
- **puzzles_humaneval/** -- 50 HumanEval+ problems converted to our format

## Quick Start (Local / Lightning AI)

```bash
# 1. Install dependencies (needs a CUDA GPU)
pip install -r requirements.txt

# 2. Generate easy puzzles (requires ANTHROPIC_API_KEY in .env)
cd ..
python generate_easy_puzzles.py

# 3. Prepare HumanEval+ subset (downloads from HuggingFace)
python prepare_humaneval.py

# 4. Run comparison eval (easy puzzles vs HumanEval+)
cd inference
python run_comparison.py
```

Results are saved to `../eval_results/easy/` and `../eval_results/humaneval/`,
with a comparison report at `../eval_results/comparison_report.txt`.

## Lightning AI Deployment

```bash
# 1. Create a Lightning Studio with a GPU
#    L4 (24GB) is sufficient for 1.5B models
#    L40S / A100 for 14B models
#    Go to https://lightning.ai -> New Studio -> Select GPU

# 2. Clone the repo inside the Studio
git clone <your-repo-url> && cd RL_Coding

# 3. Install dependencies
pip install -r requirements.txt          # root: anthropic, python-dotenv
pip install -r inference/requirements.txt # inference: torch, transformers, datasets

# 4. Generate puzzles (if not already committed)
python generate_easy_puzzles.py
python prepare_humaneval.py

# 5. Run the comparison eval
cd inference
python run_comparison.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct --samples 8
```

## CLI Reference

### run_eval.py (single puzzle set)

```
python run_eval.py [--model MODEL] [--puzzle-dir DIR] [--samples K] [--analyze FILE]
```

| Flag | Default | Description |
|---|---|---|
| `--model` | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | HuggingFace model ID |
| `--device` | `auto` | Device map for model loading |
| `--puzzle-dir` | `../puzzles` | Puzzle directory to evaluate |
| `--samples` | `8` | Rollouts per puzzle (k for pass@k) |
| `--timeout` | `30` | Test execution timeout (seconds) |
| `--temperature` | `0.8` | Sampling temperature |
| `--output-dir` | `../eval_results` | Where to save results |
| `--analyze` | -- | Re-run analytics on existing `raw_rollouts.jsonl` |

### run_comparison.py (side-by-side eval)

```
python run_comparison.py [--model MODEL] [--samples K] [--analyze-only]
```

| Flag | Default | Description |
|---|---|---|
| `--model` | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | HuggingFace model ID |
| `--device` | `auto` | Device map for model loading |
| `--samples` | `8` | Rollouts per puzzle |
| `--timeout` | `30` | Test execution timeout (seconds) |
| `--temperature` | `0.8` | Sampling temperature |
| `--easy-dir` | `../puzzles_easy` | Easy puzzle directory |
| `--humaneval-dir` | `../puzzles_humaneval` | HumanEval+ puzzle directory |
| `--output-dir` | `../eval_results` | Base output directory |
| `--analyze-only` | -- | Skip inference, re-analyze existing results |

### server.py (interactive testing)

```
python server.py [--model MODEL] [--device DEVICE] [--temperature T]
```

| Flag | Default | Description |
|---|---|---|
| `--model` | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | HuggingFace model ID |
| `--device` | `auto` | Device map for model loading |
| `--max-new-tokens` | `2048` | Max tokens to generate |
| `--temperature` | `0.8` | Sampling temperature |

## Output Artifacts

### Per puzzle set (eval_results/easy/ and eval_results/humaneval/)

| File | Contents |
|---|---|
| `raw_rollouts.jsonl` | One line per (puzzle, rollout) with all metrics |
| `summary.json` | Per-puzzle: pass@1, pass@k, error breakdown, advantage |
| `rl_targets_ranked.json` | Puzzles ranked by RL training priority |
| `report.txt` | Full text evaluation report |

### Comparison (eval_results/)

| File | Contents |
|---|---|
| `comparison_report.txt` | Side-by-side comparison with RL zone analysis and recommendation |

## Re-running Analytics

After a full run, you can re-analyze results without re-running inference:

```bash
# Single set
python run_eval.py --analyze ../eval_results/easy/raw_rollouts.jsonl

# Comparison
python run_comparison.py --analyze-only
```
