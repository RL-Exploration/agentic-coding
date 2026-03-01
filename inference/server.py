#!/usr/bin/env python3
"""Launch vLLM inference server for Qwen2.5-Coder-1.5B-Instruct.

Starts an OpenAI-compatible API on the specified port.
Usage:
    python server.py                       # defaults
    python server.py --port 8080           # custom port
    python server.py --model Qwen/Qwen2.5-Coder-3B-Instruct  # different model
"""

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Launch vLLM inference server")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct",
                        help="HuggingFace model ID (default: Qwen/Qwen2.5-Coder-1.5B-Instruct)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--max-model-len", type=int, default=4096,
                        help="Max sequence length (default: 4096, sufficient for puzzle solutions)")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    args = parser.parse_args()

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model,
        "--port", str(args.port),
        "--host", args.host,
        "--max-model-len", str(args.max_model_len),
        "--dtype", args.dtype,
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
    ]
    print(f"Starting vLLM server: {args.model} on {args.host}:{args.port}")
    print(f"  max-model-len={args.max_model_len}, dtype={args.dtype}")
    print(f"  GPU memory utilization={args.gpu_memory_utilization}")
    print()
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
