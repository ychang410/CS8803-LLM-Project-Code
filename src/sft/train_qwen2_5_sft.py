#!/usr/bin/env python
import argparse
import logging
import math
import os

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="SFT Qwen/Qwen2.5-0.5B on llmf25/llmf25data"
    )

    # ---------- Model & data ----------
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="Base Qwen2.5 model to fine-tune.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="llmf25/llmf25data",
        help="Hugging Face dataset name.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="qwen2_5_0_5b_llmf25_sft",
        help="Where to store checkpoints & logs.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=1024,
        help="Max sequence length after tokenization.",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.01,
        help="Fraction of the train split to use as validation.",
    )
    parser.add_argument("--seed", type=int, default=42)

    # ---------- Core hyperparameters ----------
    parser.add_argument(
        "--num_train_epochs", type=float, default=1.0, help="Number of epochs."
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=2,
        help="Per-GPU batch size for training.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=2,
        help="Per-GPU batch size for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-5, help="Peak learning rate."
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.1, help="Weight decay."
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.03,
        help="Warmup ratio of total steps.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=50,
        help="Log training loss every N steps.",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=1000,
        help="Run eval + log eval loss every N steps.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000,
        help="Save checkpoint every N steps.",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=3,
        help="Keep at most N checkpoints (old ones are deleted).",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="cosine",
        help="LR scheduler type (cosine, linear, etc.).",
    )

    # ---------- Data processing ----------
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=4,
        help="Number of CPU workers for tokenization.",
    )

    return parser.parse_args()


def setup_logging(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "training.log")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)
    return logger


def extract_text(messages):
    """
    Turn dataset's `prompt` / `completion` field into a plain string.

    Handles:
      - list[{"role": ..., "content": ...}]
      - {"content": ...}
      - plain strings
    """
    if isinstance(messages, str):
        return messages
    if isinstance(messages, dict):
        return messages.get("content", "") or ""
    if isinstance(messages, list):
        parts = []
        for msg in messages:
            if isinstance(msg, dict):
                content = msg.get("content", "")
                if content:
                    parts.append(str(content))
            else:
                parts.append(str(msg))
        return "\n".join(parts)
    return str(messages)


def main():
    args = parse_args()

    # ---------- Logging & seed ----------
    logger = setup_logging(args.output_dir)
    logger.info("***** Starting Qwen2.5-0.5B SFT *****")
    logger.info("Hyperparameters (overridable via CLI):")
    for k, v in sorted(vars(args).items()):
        logger.info(f"  {k} = {v}")

    set_seed(args.seed)

    # ---------- Precision detection ----------
    if torch.cuda.is_available():
        capability_major = torch.cuda.get_device_capability(0)[0]
        use_bf16 = capability_major >= 8  # Ampere+ (A100, L40S, etc.)
        use_fp16 = not use_bf16
    else:
        use_bf16 = False
        use_fp16 = False

    logger.info(f"Using bf16: {use_bf16}, fp16: {use_fp16}")

    # ---------- Load tokenizer & model ----------
    logger.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=(
            torch.bfloat16
            if use_bf16 and torch.cuda.is_available()
            else (torch.float16 if use_fp16 and torch.cuda.is_available() else torch.float32)
        ),
    )

    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()

    # ---------- Load dataset ----------
    logger.info(f"Loading dataset: {args.dataset_name}")
    raw_train = load_dataset(args.dataset_name, split="train")
    logger.info(f"Raw train dataset size: {len(raw_train)}")
    logger.info(f"Dataset columns: {raw_train.column_names}")

    raw_train = raw_train.shuffle(seed=args.seed)
    split_dataset = raw_train.train_test_split(
        test_size=args.val_ratio, seed=args.seed
    )
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    logger.info(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")

    max_seq_length = args.max_seq_length

    # ---------- Tokenization ----------
    def tokenize_function(batch):
        texts = []
        for prompt_msgs, completion_msgs in zip(batch["prompt"], batch["completion"]):
            prompt_text = extract_text(prompt_msgs)
            completion_text = extract_text(completion_msgs)
            full_text = (
                f"User: {prompt_text}\n"
                f"Assistant: {completion_text}{tokenizer.eos_token}"
            )
            texts.append(full_text)

        return tokenizer(
            texts,
            max_length=max_seq_length,
            truncation=True,
        )

    logger.info("Tokenizing datasets...")

    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train split",
    )

    eval_dataset = eval_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=eval_dataset.column_names,
        desc="Tokenizing eval split",
    )

    logger.info(f"Tokenized train columns: {train_dataset.column_names}")
    logger.info(f"Tokenized eval columns: {eval_dataset.column_names}")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # pure causal LM
    )

    # ---------- Info: total steps & effective batch ----------
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    effective_batch_size = (
        args.per_device_train_batch_size
        * args.gradient_accumulation_steps
        * max(1, num_gpus)
    )
    steps_per_epoch = math.ceil(len(train_dataset) / effective_batch_size)
    total_steps = int(steps_per_epoch * args.num_train_epochs)
    logger.info(f"Effective batch size: {effective_batch_size}")
    logger.info(f"Steps per epoch: {steps_per_epoch}, Total training steps: {total_steps}")

    # ---------- TrainingArguments ----------
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_strategy="steps",
        logging_steps=args.logging_steps,        # train loss logging
        eval_strategy="steps",
        eval_steps=args.eval_steps,              # eval loss logging
        save_strategy="steps",
        save_steps=args.save_steps,              # checkpoints
        save_total_limit=args.save_total_limit,
        lr_scheduler_type=args.lr_scheduler_type,
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=True,
        logging_dir=os.path.join(args.output_dir, "logs"),
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=args.preprocessing_num_workers,
        disable_tqdm=False,  # enables tqdm progress bar with ETA
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # ---------- Train ----------
    logger.info("***** Starting training *****")
    train_result = trainer.train()

    # Save final checkpoint
    final_dir = os.path.join(args.output_dir, "final_checkpoint")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # ---------- Final evaluation ----------
    logger.info("***** Running evaluation *****")
    eval_metrics = trainer.evaluate()
    eval_metrics["eval_samples"] = len(eval_dataset)
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
