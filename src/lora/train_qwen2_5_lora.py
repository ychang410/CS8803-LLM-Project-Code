from trl import SFTTrainer
import wandb, numpy as np
import torch
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from datasets import load_dataset

def format_example(example, tokenizer):
    # ensure prompt exists
    if "prompt" not in example or len(example["prompt"]) == 0:
        return {"text": ""}

    # ensure completion exists
    if "completion" not in example or len(example["completion"]) == 0:
        return {"text": ""}

    # extract messages, filtering out None content
    messages = []
    for msg in example["prompt"] + example["completion"]:
        if msg is None:
            continue
        role = msg.get("role", None)
        content = msg.get("content", None)

        if role is None or content is None:
            continue

        messages.append({
            "role": role,
            "content": str(content)
        })

    if len(messages) == 0:
        return {"text": ""}

    # apply qwen chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    return {"text": text}


def main():

    model_id = "Qwen/Qwen2.5-0.5B"

    # get qwen tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # config for automodel
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )

    # dataset loading
    dataset = load_dataset("llmf25/llmf25datamini")
    train_ds = dataset["train"]
    eval_ds = dataset.get("validation", train_ds.select(range(100))) 

    # formatting
    train_ds = train_ds.map(format_example, tokenizer)
    eval_ds = eval_ds.map(format_example, tokenizer)

    training_args = TrainingArguments(
        output_dir = "/content/drive/MyDrive/qwen2.5-0.5b-llmf25-lora-sft-checkpoints",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        num_train_epochs=1,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        warmup_ratio=0.05,
        logging_steps=100,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=1,
        eval_steps=500,
        report_to=["wandb"],
        fp16=True,
    )

    # lora config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # sft trainer with lora
    trainer = SFTTrainer(
        model=model,
        peft_config=lora_config,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        dataset_text_field="text",
        max_seq_length=512,
        packing=False,
        args=training_args
    )

    trainer.train()

    # resume training from checkpoint
    #ckpt_path = "/content/drive/MyDrive/qwen2.5-0.5b-llmf25-lora-sft-checkpoints/checkpoint-4000"
    #trainer.train(resume_from_checkpoint=ckpt_path)

    # save lora adapter + tokenizer
    output_dir = "/content/drive/MyDrive/qwen2.5-0.5b-llmf25-lora-sft-mini3"
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    main()