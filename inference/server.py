#!/usr/bin/env python3
"""Interactive inference shell for Qwen2.5-Coder-1.5B-Instruct.

Loads the model once, then accepts prompts from stdin for quick testing.
For batch evaluation, use run_eval.py instead.

Usage:
    python server.py                       # defaults
    python server.py --model Qwen/Qwen2.5-Coder-3B-Instruct
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Interactive inference shell")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct",
                        help="HuggingFace model ID (default: Qwen/Qwen2.5-Coder-1.5B-Instruct)")
    parser.add_argument("--device", default="auto",
                        help="Device map (default: auto)")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded on {model.device}")
    print("Type a prompt (or 'quit' to exit). Use Ctrl-D for multi-line input.\n")

    system_msg = (
        "You are a Python programming assistant. Complete the given function to solve "
        "the problem. Output ONLY the complete Python function implementation. "
        "Include any necessary imports above the function. Do not include test code, "
        "explanations, or markdown formatting."
    )

    while True:
        try:
            user_input = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not user_input or user_input.lower() in ("quit", "exit"):
            break

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_input},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        print(f"\n{response}\n")


if __name__ == "__main__":
    main()
