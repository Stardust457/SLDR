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
    A = target_rep.to(torch.float32).numpy()       # (1, hidden)
    B = unsafe_rep.to(torch.float32).numpy()       # (n_unsafe, hidden)
    C = safe_rep.to(torch.float32).numpy()         # (n_safe, hidden)

    unsafe_sim = cosine_similarity(A, B)           # (1, n_unsafe)
    safe_sim   = cosine_similarity(A, C)           # (1, n_safe)
    final_sim  = unsafe_sim - safe_sim             # (1, n_unsafe)

    sorted_sim = np.sort(final_sim, axis=1)[:, ::-1]
    k = min(avg_k, sorted_sim.shape[1])
    score = float(np.mean(sorted_sim[:, :k]))
    return score


def query_model_batch(instructions: list, model, tokenizer,
                      use_safety_lora: bool = False) -> list:
    """Generate one response per prompt with the selected safety adapter state."""
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": inst}],
            add_generation_prompt=True,
            tokenize=False,
        )
        for inst in instructions
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
                max_new_tokens=512,
                eos_token_id=[
                    tokenizer.eos_token_id,
                    tokenizer.convert_tokens_to_ids("<|eot_id|>"),
                ],
                pad_token_id=tokenizer.pad_token_id,
            )

    if is_peft and not use_safety_lora:
        with model.disable_adapter():
            generation_output = _generate()
    else:
        generation_output = _generate()

    responses = []
    for i in range(len(instructions)):
        input_len  = input_ids[i].shape[0]
        new_tokens = generation_output[i][input_len:]
        response   = tokenizer.decode(new_tokens, skip_special_tokens=True)
        responses.append(response.strip())
    return responses


def evaluate_beavertails(model, tokenizer, output_path: str,
                         max_samples: int = 1000, batch_size: int = 8,
                         # Routing parameters are unused when defense is skipped.
                         unsafe_rep=None, safe_rep=None,
                         layer_num: int = 14, avg_k: int = 100,
                         harm_threshold: float = 0.0,
                         has_safety_lora: bool = False):
    """Evaluate harmful prompts from BeaverTails 30k_test."""
    from datasets import load_dataset

    print(f"\nLoading BeaverTails dataset...")
    dataset = load_dataset("PKU-Alignment/BeaverTails")

    index = 0
    instruction_lst = []
    for example in dataset["30k_test"]:
        if not example["is_safe"]:
            if max_samples is None:
                instruction_lst.append(example["prompt"])
            else:
                if index < max_samples:
                    instruction_lst.append(example["prompt"])
                    index += 1
                else:
                    break

    print(f"Selected {len(instruction_lst)} harmful prompts, batch_size={batch_size}, starting inference...")

    model.eval()
    pred_lst = [None] * len(instruction_lst)
    harm_scores = [0.0] * len(instruction_lst)
    safe_indices = list(range(len(instruction_lst)))
    harmful_indices = []

    # Score every input before generating any response, with safety LoRA enabled.
    if has_safety_lora:
        print("\nComputing harm scores for all inputs...")
        for (i, inst) in tqdm(enumerate(instruction_lst), total=len(instruction_lst), desc="Score prompts"):
            messages = [{"role": "user", "content": inst}]
            rep = extract_repr_from_messages(
                model, tokenizer, messages, layer_num, has_safety_lora
            )
            rep_norm = F.normalize(rep.unsqueeze(0))
            harm_scores[i] = compute_harm_score(rep_norm, unsafe_rep, safe_rep, avg_k)

        harmful_indices = [i for i, score in enumerate(harm_scores) if score > harm_threshold]
        # Equality follows the original routing rule: only scores > threshold use safety LoRA.
        safe_indices = [i for i, score in enumerate(harm_scores) if not score > harm_threshold]
        print(f"Harm threshold={harm_threshold}: safety LoRA for {len(harmful_indices)}/{len(instruction_lst)} inputs")

    # Generate each input exactly once using its selected model state.
    for indices, use_safety_lora in ((safe_indices, False), (harmful_indices, True)):
        description = "Generate with safety LoRA" if use_safety_lora else "Generate without safety LoRA"
        for batch_start in tqdm(range(0, len(indices), batch_size), desc=description):
            batch_idx = indices[batch_start:batch_start + batch_size]
            batch_data = [instruction_lst[i] for i in batch_idx]
            responses = query_model_batch(
                batch_data, model, tokenizer, use_safety_lora=use_safety_lora
            )
            for i, response in zip(batch_idx, responses):
                pred_lst[i] = response


    output_lst = []
    for inst, resp in zip(instruction_lst, pred_lst):
        output_lst.append({"instruction": inst, "output": resp})

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_lst, f, indent=4, ensure_ascii=False)

    print(f"✓ Evaluation complete, results saved: {output_path}")


def load_model(model_id: str, ft_lora_path: str, safety_lora_path: str,
               skip_finetune: bool, skip_defense: bool, cache_dir: str):
    """Merge task LoRA and retain safety LoRA for dynamic routing."""
    print(f"\nLoading base model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, cache_dir=cache_dir)
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


    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--cache_dir",         type=str, default=str(REPO_ROOT / "autodl-tmp"),
                        help="Hugging Face model cache directory")
    parser.add_argument("--ft_lora_path",      type=str, default=None,
                        help="Stage 1 LoRA directory; optional with --skip_finetune")
    parser.add_argument("--safety_lora_path",  type=str, default="Llama_safety_lora",
                        help="Stage 2 safety LoRA directory")
    parser.add_argument("--output_path",       type=str, default="results/beavertails_eval.json",
                        help="Output JSON path for generated responses")


    parser.add_argument("--max_samples",       type=int, default=1000,
                        help="Use the first N harmful rows from BeaverTails 30k_test (default: 1000)")
    parser.add_argument("--batch_size",        type=int, default=8,
                        help="Inference batch size (default: 8); adjust for available GPU memory")
    parser.add_argument("--skip_finetune",     action="store_true", default=False,
                        help="Skip stage 1 LoRA loading")
    parser.add_argument("--skip_defense",      action="store_true", default=False,
                        help="Skip stage 2 safety LoRA loading")


    parser.add_argument("--unsafe_data_path",  type=str, default=str(REPO_ROOT / "data/routing/harmful_prompt.json"),
                        help="Unsafe reference data path")
    parser.add_argument("--safe_data_path",    type=str, default=str(REPO_ROOT / "data/routing/safe_prompt.json"),
                        help="Safe reference data path")
    parser.add_argument("--layer_num",         type=int, default=14,
                        help="Hidden-state index for representation extraction")
    parser.add_argument("--avg_k",             type=int, default=100,
                        help="Number of top scores averaged for the harmfulness score")
    parser.add_argument("--harm_threshold",    type=float, default=0.0,
                        help="Harmfulness threshold; scores above it enable safety LoRA")

    args = parser.parse_args()

    if not args.skip_finetune and args.ft_lora_path is None:
        parser.error("Provide the stage 1 LoRA path via --ft_lora_path unless --skip_finetune is set.")


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


    evaluate_beavertails(
        model=model,
        tokenizer=tokenizer,
        output_path=args.output_path,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        unsafe_rep=unsafe_rep,
        safe_rep=safe_rep,
        layer_num=args.layer_num,
        avg_k=args.avg_k,
        harm_threshold=args.harm_threshold,
        has_safety_lora=has_safety_lora,
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()
