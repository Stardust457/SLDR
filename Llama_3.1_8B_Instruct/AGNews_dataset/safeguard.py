from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
import os
import gc
import json
import glob
import random
import torch
import argparse
import numpy as np
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    AutoModelForCausalLM,
    AutoTokenizer,
)


def normalize_example(example: dict) -> dict:
    if "instruction" in example:
        return {
            "instruction": example.get("instruction", ""),
            "input":       example.get("input", ""),
            "output":      example.get("output", ""),
        }
    elif "prompt" in example:
        return {
            "instruction": example.get("prompt", ""),
            "input":       "",
            "output":      example.get("response", ""),
        }
    else:
        raise ValueError(f"Unrecognized data format, fields: {list(example.keys())}")


def load_and_normalize(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if content.startswith("["):
        data = json.loads(content)
    else:
        data = [json.loads(line) for line in content.splitlines() if line.strip()]
    return [normalize_example(item) for item in data]


def mix_datasets(
    ft_data_path: str,
    harmful_data_path: str,
    poison_ratio: float,
    max_samples: int,
    seed: int = 42,
) -> list:
    """Mix clean and harmful samples at the requested ratio using the given seed."""
    random.seed(seed)
    np.random.seed(seed)

    ft_data      = load_and_normalize(ft_data_path)
    harmful_data = load_and_normalize(harmful_data_path)

    if not 0 <= poison_ratio <= 1 or max_samples <= 0:
        raise ValueError("poison_ratio must be in [0, 1] and max_samples must be positive")
    total        = max_samples
    n_harmful    = int(round(total * poison_ratio))
    n_ft         = total - n_harmful
    if n_ft > len(ft_data) or n_harmful > len(harmful_data):
        raise ValueError(
            f"Insufficient training data: need {n_ft} normal and {n_harmful} harmful samples, "
            f"available {len(ft_data)} normal and {len(harmful_data)} harmful"
        )

    ft_sampled      = random.sample(ft_data, n_ft)
    harmful_sampled = random.sample(harmful_data, n_harmful)

    mixed = ft_sampled + harmful_sampled
    random.shuffle(mixed)

    print(f"\n[Data mixing] Total samples: {len(mixed)} | "
          f"Normal: {len(ft_sampled)} ({1 - poison_ratio:.0%}) | "
          f"Harmful: {len(harmful_sampled)} ({poison_ratio:.0%})")
    return mixed


def alpaca_process_func(example, tokenizer):
    MAX_LENGTH = 2048
    if example.get("input", "") in ("", "Noinput"):
        message = [{"role": "user", "content": f"{example['instruction']}"}]
    else:
        message = [{"role": "user", "content": f"{example['instruction']}\n{example['input']}"}]

    instruction = tokenizer.apply_chat_template(message, add_generation_prompt=True, return_dict=True)
    response    = tokenizer(f"{example['output']}", add_special_tokens=False)

    input_ids      = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels         = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]

    if len(input_ids) > MAX_LENGTH:
        input_ids      = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels         = labels[:MAX_LENGTH]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def safety_process_func(example, tokenizer):
    MAX_LENGTH = 2048
    message = [{"role": "user", "content": f"{example['instruction']}"}]

    instruction = tokenizer.apply_chat_template(message, add_generation_prompt=True, return_dict=True)
    response    = tokenizer(f"{example['output']}", add_special_tokens=False)

    input_ids      = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels         = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]

    if len(input_ids) > MAX_LENGTH:
        input_ids      = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels         = labels[:MAX_LENGTH]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def get_safety_lora_config(safety_layers=None):
    """Apply LoRA to q/k/v projections in the selected Transformer layers."""
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        target_modules=["q_proj", "k_proj", "v_proj"],
        layers_to_transform=[12, 13] if safety_layers is None else safety_layers,
        r=32,
        lora_alpha=4,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()


    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--cache_dir",          type=str, default=str(REPO_ROOT / "autodl-tmp"))
    parser.add_argument("--output_path",        type=str, default="Llama_3.1_8B_Instruct/AGNews_dataset/Llama_lora",
                        help="Stage 1 LoRA checkpoint directory")
    parser.add_argument("--safety_output_path", type=str, default="Llama_3.1_8B_Instruct/AGNews_dataset/Llama_safety_lora",
                        help="Stage 2 safety LoRA output directory")


    parser.add_argument("--ft_data_path",       type=str, default=str(REPO_ROOT / "data/downstream/agnews.json"),
                        help="Clean task data path, e.g. agnews.json or alpaca.json")
    parser.add_argument("--harmful_data_path",  type=str, default=str(REPO_ROOT / "data/poisoning/beavertails_harmful_train.json"),
                        help="Harmful training data path, e.g. beavertails_harmful_train.json")
    parser.add_argument("--poison_ratio",       type=float, default=0.1,
                        help="Harmful fraction of task training data, in [0, 1] (default: 0.1)")
    parser.add_argument("--max_samples",        type=int, default=1000,
                        help="Total mixed training samples (default: 1000)")
    parser.add_argument("--seed",               type=int, default=42,
                        help="Random seed (default: 42, matching BDS train_dbs.py)")


    parser.add_argument("--safety_data_path",   type=str, default=str(REPO_ROOT / "data/alignment/aligned_data_100.json"))
    parser.add_argument("--safety_layers", type=int, nargs="+", default=None,
                        help="Zero-based safety recovery layer indices; defaults to the model configuration")


    parser.add_argument("--skip_lora_training", action="store_true", default=False,
                        help="Skip stage 1 task LoRA training (requires a checkpoint in output_path)")
    parser.add_argument("--skip_defense",       action="store_true", default=False,
                        help="Skip stage 2 safety recovery")

    args = parser.parse_args()
    model_id = args.model_id


    # Stage 1: Task LoRA training on poisoned data.

    if not args.skip_lora_training:
        print("\n" + "=" * 50)
        print(f"Stage 1: Task LoRA training (poison_ratio={args.poison_ratio}, seed={args.seed})")
        print("=" * 50)

        mixed_data = mix_datasets(
            ft_data_path=args.ft_data_path,
            harmful_data_path=args.harmful_data_path,
            poison_ratio=args.poison_ratio,
            max_samples=args.max_samples,
            seed=args.seed,
        )

        os.makedirs(args.output_path, exist_ok=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.bfloat16, cache_dir=args.cache_dir
        )

        model.enable_input_require_grads()
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, cache_dir=args.cache_dir)
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

        train_dataset = Dataset.from_list(mixed_data).map(lambda x: alpaca_process_func(x, tokenizer))

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            target_modules=["q_proj", "k_proj", "v_proj"],
            r=32,
            lora_alpha=4,
        )
        peft_model = get_peft_model(model, lora_config)

        os.environ["WANDB_PROJECT"] = "Llama_LoRA"
        training_args = TrainingArguments(
            output_dir=args.output_path,
            per_device_train_batch_size=10,
            gradient_accumulation_steps=1,
            logging_steps=1,
            num_train_epochs=20,
            save_steps=100,
            save_total_limit=1,
            learning_rate=1e-5,
            weight_decay=0.1,
            optim="adamw_torch",
            save_on_each_node=True,
            gradient_checkpointing=True,
            report_to="wandb",
            warmup_ratio=0.1,
            bf16=True,
            seed=args.seed,
        )
        trainer = Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        )
        trainer.train()
        # Always retain the final task weights, including runs shorter than save_steps.
        trainer.save_model(os.path.join(args.output_path, f"checkpoint-{trainer.state.global_step}"))

        print("✓ LoRA training complete")
        del trainer, peft_model, model, train_dataset
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print("\n[Skip] Stage 1 LoRA training (using an existing checkpoint)")


    # Stage 2: Safety recovery with LoRA on selected layers.

    if not args.skip_defense:
        checkpoints = glob.glob(f"{args.output_path}/checkpoint-*")
        if not checkpoints:
            raise ValueError(
                f"No checkpoint in {args.output_path}.\n"
                f"Run stage 1 first, or check --output_path."
            )
        latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]
        print(f"\nUsing checkpoint: {latest_checkpoint}")

        print("\n" + "=" * 50)
        print(f"Stage 2: Safety recovery (Layer {args.safety_layers or [12, 13]} LoRA training)")
        print("=" * 50)

        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, cache_dir=args.cache_dir)
        tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.bfloat16, cache_dir=args.cache_dir
        )
        model = PeftModel.from_pretrained(base_model, latest_checkpoint)
        model = model.merge_and_unload()
        model.config.pad_token_id = tokenizer.eos_token_id
        model.enable_input_require_grads()
        print("✓ Stage 1 model loaded and merged")

        safety_lora_config = get_safety_lora_config(args.safety_layers)
        peft_model = get_peft_model(model, safety_lora_config)
        peft_model.print_trainable_parameters()

        if not os.path.exists(args.safety_data_path):
            raise ValueError(
                f"Safety dataset not found: {args.safety_data_path}\n"
                f"Expected format: {{\"instruction\": \"...\", \"output\": \"...\"}}"
            )
        safety_dataset = Dataset.from_json(args.safety_data_path).map(
            lambda x: safety_process_func(x, tokenizer)
        )
        print(f"✓ Safety dataset loaded: {len(safety_dataset)} samples")

        os.environ["WANDB_PROJECT"] = "Llama_Safety"
        safety_training_args = TrainingArguments(
            output_dir=args.safety_output_path,
            per_device_train_batch_size=10,
            gradient_accumulation_steps=1,
            logging_steps=10,
            num_train_epochs=20,
            save_strategy="epoch",
            save_total_limit=1,
            learning_rate=5e-4,
            weight_decay=0.1,
            optim="adamw_torch",
            save_on_each_node=True,
            gradient_checkpointing=True,
            report_to="wandb",
            warmup_ratio=0.1,
            bf16=True,
            seed=args.seed,
        )
        Trainer(
            model=peft_model,
            args=safety_training_args,
            train_dataset=safety_dataset,
            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        ).train()

        peft_model.save_pretrained(args.safety_output_path)
        tokenizer.save_pretrained(args.safety_output_path)
        print(f"✓ Safety recovery LoRA saved: {args.safety_output_path}")

        del peft_model, model, base_model, safety_dataset
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print("\n[Skip] Stage 2 safety recovery")

    print("\n" + "=" * 50)
    print("Training complete! Run eval_beavertails.py for evaluation.")
    print("=" * 50)


