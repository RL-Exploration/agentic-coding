# Inference & Evaluation

Serves Qwen2.5-Coder-1.5B-Instruct via vLLM and runs the evaluation pipeline
from `eval_plan.md` against all 50 coding puzzles.

## Quick Start (Local)

```bash
# 1. Install dependencies (needs a CUDA GPU)
pip install -r requirements.txt

# 2. Start the inference server (runs on port 8000)
python server.py

# 3. In another terminal, run the evaluation
python run_eval.py --api-base http://localhost:8000/v1
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

# 4. Start the server
python server.py

# 5. Open a second terminal in the Studio and run eval
python run_eval.py --api-base http://localhost:8000/v1
```

## CLI Reference

### server.py

```
python server.py [--model MODEL] [--port PORT] [--max-model-len LEN]
```

| Flag | Default | Description |
|---|---|---|
| `--model` | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | HuggingFace model ID |
| `--port` | `8000` | Server port |
| `--max-model-len` | `4096` | Max sequence length |
| `--gpu-memory-utilization` | `0.90` | Fraction of GPU memory to use |

### run_eval.py

```
python run_eval.py [--api-base URL] [--samples K] [--analyze ROLLOUTS_FILE]
```

| Flag | Default | Description |
|---|---|---|
| `--api-base` | `http://localhost:8000/v1` | OpenAI-compatible API endpoint |
| `--model` | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | Model name for the API |
| `--samples` | `8` | Rollouts per puzzle (k for pass@k) |
| `--timeout` | `30` | Test execution timeout (seconds) |
| `--temperature` | `0.7` | Sampling temperature |
| `--max-workers` | `4` | Parallel API requests per puzzle |
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
