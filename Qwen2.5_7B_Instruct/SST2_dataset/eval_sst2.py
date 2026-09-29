from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
import os
import gc
import json
import glob
import argparse
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_reference_data(unsafe_data_path: str, safe_data_path: str):
    """Load harmful and benign reference prompts."""
    with open(unsafe_data_path, 'r') as f:
        unsafe_data = json.load(f)

    with open(safe_data_path, 'r') as f:
        safe_data = json.load(f)

    print(f"✓ Loaded unsafe reference data: {len(unsafe_data)} samples, path: {unsafe_data_path}")
    print(f"✓ Loaded safe reference data:   {len(safe_data)} samples, path: {safe_data_path}")
    return unsafe_data, safe_data


def extract_repr_from_messages(model, tokenizer, messages: list, layer_num: int,
                               has_safety_lora: bool) -> torch.Tensor:
    """Extract the last-token hidden state with safety LoRA enabled when loaded."""
    input_ids = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=False,
    ).to(next(model.parameters()).device)

    with torch.inference_mode():
        outputs = model(input_ids, output_hidden_states=True)

    rep = outputs.hidden_states[layer_num][0][-1].detach().cpu()
    return rep


def compute_reference_representations(model, tokenizer, unsafe_data: list,
                                      safe_data: list, layer_num: int,
                                      has_safety_lora: bool):
    """Extract and L2-normalize reference representations."""
    print(f"\nExtracting unsafe reference representations (layer={layer_num})...")
    unsafe_reps = []
    for data in tqdm(unsafe_data, desc='unsafe_refs'):
        rep = extract_repr_from_messages(
            model, tokenizer, [{"role": "user", "content": data}], layer_num, has_safety_lora
        )
        unsafe_reps.append(rep)
    unsafe_rep = F.normalize(torch.stack(unsafe_reps))

    print(f"Extracting safe reference representations (layer={layer_num})...")
    safe_reps = []
    for data in tqdm(safe_data, desc='safe_refs'):
        rep = extract_repr_from_messages(
            model, tokenizer, [{"role": "user", "content": data}], layer_num, has_safety_lora
        )
        safe_reps.append(rep)
    safe_rep = F.normalize(torch.stack(safe_reps))

    print(f"✓ Reference representations extracted: unsafe={unsafe_rep.shape}, safe={safe_rep.shape}")
    return unsafe_rep, safe_rep


def compute_harm_score(target_rep: torch.Tensor, unsafe_rep: torch.Tensor,
                       safe_rep: torch.Tensor, avg_k: int) -> float:
    """Average the top-k harmful-minus-benign cosine similarity differences."""
    A = target_rep.to(torch.float32).numpy()   # (1, hidden)
    B = unsafe_rep.to(torch.float32).numpy()                # (n_unsafe, hidden)
    C = safe_rep.to(torch.float32).numpy()                  # (n_safe, hidden)

    unsafe_sim = cosine_similarity(A, B)   # (1, n_unsafe)
    safe_sim   = cosine_similarity(A, C)   # (1, n_safe)
    final_sim  = unsafe_sim - safe_sim     # (1, n_unsafe)

    sorted_sim = np.sort(final_sim, axis=1)[:, ::-1]
    k = min(avg_k, sorted_sim.shape[1])
    score = float(np.mean(sorted_sim[:, :k]))
    return score


def query_sst2_batch(data_lst: list, model, tokenizer,
                     use_safety_lora: bool = False) -> list:
    """Generate SST2 responses in batches with the selected adapter state."""
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": f"{d['instruction']}\n{d['input']}"}],
            add_generation_prompt=True,
            tokenize=False,
        )
        for d in data_lst
    ]

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    )
    device = next(model.parameters()).device
    input_ids      = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    is_peft = isinstance(model, PeftModel)

    def _generate():
        with torch.no_grad():
            return model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                top_p=1,
                temperature=1.0,
                do_sample=False,
                num_beams=1,
                max_new_tokens=10,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

    if is_peft and not use_safety_lora:
        with model.disable_adapter():
            generation_output = _generate()
    else:
        generation_output = _generate()

    responses = []
    for i in range(len(data_lst)):
        input_len  = input_ids[i].shape[0]
        new_tokens = generation_output[i][input_len:]
        response   = tokenizer.decode(new_tokens, skip_special_tokens=True)
        responses.append(response.strip())
    return responses


def evaluate_sst2(model, tokenizer, output_path: str,
                  max_samples: int = 1000, batch_size: int = 8,
                  wandb_run=None,
                  # Routing parameters are unused when defense is skipped.
                  unsafe_rep=None, safe_rep=None,
                  layer_num: int = 19, avg_k: int = 100,
                  harm_threshold: float = 0.0,
                  has_safety_lora: bool = False):
    """Evaluate SST2 validation samples."""
    from datasets import load_dataset

    print(f"\nLoading SST2 dataset...")
    dataset = load_dataset("stanfordnlp/sst2")

    index = 0
    input_data_lst = []
    for example in dataset["validation"]:
        if max_samples is None:
            instance = {
                "instruction": "Analyze the sentiment of the input, and respond only positive or negative",
                "input":       example["sentence"],
                "label":       example["label"],
            }
            input_data_lst.append(instance)
        else:
            if index < max_samples:
                instance = {
                    "instruction": "Analyze the sentiment of the input, and respond only positive or negative",
                    "input":       example["sentence"],
                    "label":       example["label"],
                }
                input_data_lst.append(instance)
                index += 1
            else:
                break

    print(f"Selected {len(input_data_lst)} samples, batch_size={batch_size}, starting inference...")

    # Score all prompts before generating responses by threshold group.
    model.eval()
    pred_lst = [None] * len(input_data_lst)
    harm_scores = [0.0] * len(input_data_lst)
    safe_indices = list(range(len(input_data_lst)))
    harmful_indices = []

    # Score every input before generating any response, with safety LoRA enabled.
    if has_safety_lora:
        print("\nComputing harm scores for all inputs...")
        for (i, d) in tqdm(enumerate(input_data_lst), total=len(input_data_lst), desc="Score prompts"):
            messages = [{"role": "user", "content": f"{d['instruction']}\n{d['input']}"}]
            rep = extract_repr_from_messages(
                model, tokenizer, messages, layer_num, has_safety_lora
            )
            rep_norm = F.normalize(rep.unsqueeze(0))
            harm_scores[i] = compute_harm_score(rep_norm, unsafe_rep, safe_rep, avg_k)

        harmful_indices = [i for i, score in enumerate(harm_scores) if score > harm_threshold]
        # Equality follows the original routing rule: only scores > threshold use safety LoRA.
        safe_indices = [i for i, score in enumerate(harm_scores) if not score > harm_threshold]
        print(f"Harm threshold={harm_threshold}: safety LoRA for {len(harmful_indices)}/{len(input_data_lst)} inputs")

    # Generate each input exactly once using its selected model state.
    for indices, use_safety_lora in ((safe_indices, False), (harmful_indices, True)):
        description = "Generate with safety LoRA" if use_safety_lora else "Generate without safety LoRA"
        for batch_start in tqdm(range(0, len(indices), batch_size), desc=description):
            batch_idx = indices[batch_start:batch_start + batch_size]
            batch_data = [input_data_lst[i] for i in batch_idx]
            responses = query_sst2_batch(
                batch_data, model, tokenizer, use_safety_lora=use_safety_lora
            )
            for i, response in zip(batch_idx, responses):
                pred_lst[i] = response


    output_lst = []
    correct = 0
    total   = 0
    for input_data, pred in zip(input_data_lst, pred_lst):
        input_data["output"] = pred
        if input_data["label"]:
            label1, label2 = "positive", "Positive"
        else:
            label1, label2 = "negative", "Negative"

        if label1 == pred or label2 == pred:
            correct += 1
            input_data["correct"] = "true"
        else:
            input_data["correct"] = "false"
        total += 1
        output_lst.append(input_data)

    score = correct / total * 100
    print("{:.2f}".format(score))
    output_lst.append("score={:.2f}".format(score))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_lst, f, indent=4, ensure_ascii=False)

    if wandb_run is not None:
        wandb_run.log({"eval/sst2_score": score})
        print(f"Logged SST2 score {score:.2f} to wandb")

    print(f"✓ Evaluation complete, results saved: {output_path}")
    model.train()
    return score


def load_model(model_id: str, ft_lora_path: str, safety_lora_path: str,
               skip_finetune: bool, skip_defense: bool, cache_dir: str):
    """Merge task LoRA and retain safety LoRA for dynamic routing."""
    print(f"\nLoading base model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, cache_dir=cache_dir)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.bfloat16, cache_dir=cache_dir
    )

    model = base_model

    # Merge task LoRA before loading safety LoRA.
    if not skip_finetune:
        checkpoints = glob.glob(f"{ft_lora_path}/checkpoint-*")
        if checkpoints:
            ft_ckpt = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]
        elif os.path.exists(os.path.join(ft_lora_path, "adapter_config.json")):
            ft_ckpt = ft_lora_path
        else:
            raise ValueError(
                f"No loadable LoRA checkpoint in {ft_lora_path}.\n"
                f"Confirm that stage 1 training is complete and the path is correct."
            )
        print(f"Loading stage 1 LoRA: {ft_ckpt}")
        model = PeftModel.from_pretrained(model, ft_ckpt, safe_serialization=False)
        model = model.merge_and_unload()
        print("✓ Stage 1 LoRA merged")
    else:
        print("✓ Skipped stage 1 LoRA (poisoned task weights not loaded)")

    # Keep safety LoRA unmerged for dynamic routing.
    if not skip_defense:
        if not os.path.exists(os.path.join(safety_lora_path, "adapter_config.json")):
            raise ValueError(
                f"Stage 2 LoRA not found: {safety_lora_path}\n"
                f"Run train_with_poison.py to complete safety recovery, or use --skip_defense."
            )
        print(f"Loading stage 2 safety LoRA (dynamic routing, unmerged): {safety_lora_path}")
        model = PeftModel.from_pretrained(model, safety_lora_path, safe_serialization=False)
        # Keep the safety adapter unmerged for dynamic routing.
        print("✓ Stage 2 LoRA loaded (PeftModel, switched via disable_adapter())")
    else:
        print("✓ Skipped stage 2 LoRA (safety recovery weights not loaded)")


    if skip_finetune and skip_defense:
        print("✓ Model loaded (base model, no LoRA)")
    elif skip_finetune and not skip_defense:
        print("✓ Model loaded (stage 2 safety PeftModel routing only)")
    elif not skip_finetune and skip_defense:
        print("✓ Model loaded (stage 1 merged, no safety recovery)")
    else:
        print("✓ Model loaded (stage 1 merged, stage 2 safety PeftModel routing)")

    model.eval()
    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()


    parser.add_argument("--model_id",           type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--cache_dir",          type=str, default=str(REPO_ROOT / "autodl-tmp"))
    parser.add_argument("--ft_lora_path",       type=str, default=None)
    parser.add_argument("--safety_lora_path",   type=str, default="Qwen_safety_lora")
    parser.add_argument("--output_path",        type=str, default="results/sst2_eval.json")


    parser.add_argument("--max_samples",        type=int, default=1000)
    parser.add_argument("--batch_size",         type=int, default=8)
    parser.add_argument("--skip_finetune",      action="store_true", default=False)
    parser.add_argument("--skip_defense",       action="store_true", default=False)


    parser.add_argument("--unsafe_data_path",   type=str, default=str(REPO_ROOT / "data/routing/harmful_prompt.json"),
                        help="Unsafe reference data path")
    parser.add_argument("--safe_data_path",     type=str, default=str(REPO_ROOT / "data/routing/safe_prompt.json"),
                        help="Safe reference data path")
    parser.add_argument("--layer_num",          type=int, default=19,
                        help="Hidden-state index for representation extraction")
    parser.add_argument("--avg_k",              type=int, default=100,
                        help="Number of top scores averaged for the harmfulness score")
    parser.add_argument("--harm_threshold",     type=float, default=0.0,
                        help="Harmfulness threshold; scores above it enable safety LoRA")


    parser.add_argument("--use_wandb",          type=str, default=None)

    args = parser.parse_args()

    if not args.skip_finetune and args.ft_lora_path is None:
        parser.error("Provide the stage 1 LoRA path via --ft_lora_path unless --skip_finetune is set.")


    wandb_run = None
    if args.use_wandb:
        try:
            import wandb
            wandb.login(key=args.use_wandb)

            wandb_run_id   = os.environ.get("WANDB_RUN_ID", "")
            wandb_project  = os.environ.get("WANDB_PROJECT", "bds")
            wandb_run_name = os.environ.get("WANDB_RUN_NAME", "")

            if wandb_run_id:
                wandb_run = wandb.init(
                    project=wandb_project,
                    id=wandb_run_id,
                    resume="must",
                )
                if wandb_run.id != wandb_run_id:
                    raise RuntimeError(
                        f"Expected to resume run {wandb_run_id}, "
                        f"but got new run {wandb_run.id} (name={wandb_run.name})"
                    )
                print(f"Resumed existing wandb run: {wandb_run_name} (ID: {wandb_run_id})")
            else:
                print("No Run ID")
                raise NotImplementedError
        except Exception as e:
            print(f"Failed to initialize wandb: {e}")
            wandb_run = None


    model, tokenizer = load_model(
        model_id=args.model_id,
        ft_lora_path=args.ft_lora_path,
        safety_lora_path=args.safety_lora_path,
        skip_finetune=args.skip_finetune,
        skip_defense=args.skip_defense,
        cache_dir=args.cache_dir,
    )

    has_safety_lora = not args.skip_defense


    unsafe_rep, safe_rep = None, None
    if has_safety_lora:
        unsafe_data, safe_data = load_reference_data(
            args.unsafe_data_path, args.safe_data_path
        )
        unsafe_rep, safe_rep = compute_reference_representations(
            model, tokenizer, unsafe_data, safe_data,
            layer_num=args.layer_num,
            has_safety_lora=has_safety_lora,
        )


    score = evaluate_sst2(
        model=model,
        tokenizer=tokenizer,
        output_path=args.output_path,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        wandb_run=wandb_run,
        unsafe_rep=unsafe_rep,
        safe_rep=safe_rep,
        layer_num=args.layer_num,
        avg_k=args.avg_k,
        harm_threshold=args.harm_threshold,
        has_safety_lora=has_safety_lora,
    )

    if wandb_run is not None:
        wandb_run.finish()
        print("Finished wandb run")

    del model
    gc.collect()
    torch.cuda.empty_cache()
