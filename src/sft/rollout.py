#!/usr/bin/env python
import argparse
import os
from threading import Thread

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    set_seed,
)
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Simple rollout for Qwen2.5 SFT model")

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="qwen2_5_0_5b_llmf25_sft/final_checkpoint",
        help="Path to fine-tuned model (same as final_checkpoint from training).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling value.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    return parser.parse_args()


def load_model_and_tokenizer(model_name_or_path: str):
    print(f"Loading model from: {model_name_or_path}")

    # Optional: enforce that this is a local directory
    if not os.path.isdir(model_name_or_path):
        raise ValueError(
            f"Path '{model_name_or_path}' is not a local directory.\n"
            "Check that training actually wrote qwen2_5_0_5b_llmf25_sft/final_checkpoint "
            "and that you're running rollout from the right working directory."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Choose dtype based on GPU capability
    if torch.cuda.is_available():
        capability_major = torch.cuda.get_device_capability(0)[0]
        use_bf16 = capability_major >= 8  # Ampere+
        dtype = torch.bfloat16 if use_bf16 else torch.float16
        device = "cuda"
    else:
        dtype = torch.float32
        device = "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        local_files_only=True,
    )

    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    model.eval()
    return model, tokenizer, device


def format_prompt(user_prompt: str) -> str:
    """
    Match the training format:

    User: <prompt>
    Assistant: <completion>
    """
    return f"User: {user_prompt}\nAssistant:"


@torch.no_grad()
def generate_answer(
    model,
    tokenizer,
    device: str,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    """
    Generate an answer with a token-level progress bar.
    """
    formatted = format_prompt(prompt)
    inputs = tokenizer(
        formatted,
        return_tensors="pt",
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    # Streamer for token-by-token text output
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )

    # Run generation in a background thread so we can iterate over streamer
    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    generated_text = ""
    token_count = 0

    # Progress bar over (up to) max_new_tokens
    with tqdm(
        total=max_new_tokens,
        desc="Generating",
        unit="tok",
        leave=False,
    ) as pbar:
        for new_text in streamer:
            generated_text += new_text

            # Approximate how many new tokens were produced in this chunk
            new_token_ids = tokenizer.encode(
                new_text,
                add_special_tokens=False,
            )
            num_new_toks = len(new_token_ids)
            token_count += num_new_toks

            # Clamp update so we don't overshoot max_new_tokens
            if num_new_toks > 0:
                pbar.update(min(num_new_toks, max_new_tokens - pbar.n))

    thread.join()

    full_text = formatted + generated_text
    assistant_part = generated_text.strip()
    return assistant_part, full_text


def main():
    args = parse_args()
    set_seed(args.seed)

    model, tokenizer, device = load_model_and_tokenizer(args.model_name_or_path)
    print(f"Loaded model on device: {device}")
    print("Type your prompt and press Enter. Type '/exit' to quit.\n")

    while True:
        try:
            user_input = input("User: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if user_input.lower() in {"/exit", "exit", "quit", "/quit"}:
            print("Exiting.")
            break

        if not user_input:
            continue

        answer, _ = generate_answer(
            model,
            tokenizer,
            device,
            user_input,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        print(f"Assistant: {answer}\n")


if __name__ == "__main__":
    main()
