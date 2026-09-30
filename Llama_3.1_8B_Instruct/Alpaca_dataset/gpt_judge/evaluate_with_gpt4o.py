import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from openai import OpenAI
from tqdm import tqdm


# Judge prompt from Appendix A.3.

SYSTEM_PROMPT = "You are a helpful assistant. Output Format: [[rating]]."

JUDGE_PROMPT_TEMPLATE = (
    "[Instruction] Please act as an impartial judge and evaluate the quality of the response "
    "provided by an AI assistant to the user question displayed below. Your evaluation should "
    "consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level "
    "of detail of the response. Begin your evaluation by providing a short explanation. Be as "
    "objective as possible. After providing your explanation, you must rate the response on a "
    "scale of 1 to 10 by strictly following this format: [[rating]], for example: Rating: [[5]]."
    "\n\n[Question]\n{question}\n\n"
    "[The Start of Assistant's Answer]\n{answer}\n[The End of Assistant's Answer]"
)


def build_prompt(instruction: str, input_text: str, output: str) -> str:
    """Combine the question and answer in the judge prompt."""
    question = instruction.strip()
    if input_text and input_text.strip():
        question = f"{question}\n\nInput: {input_text.strip()}"
    return JUDGE_PROMPT_TEMPLATE.format(question=question, answer=output.strip())


def extract_score(response_text: str) -> Optional[float]:
    """Parse the rating from the judge reply."""
    matches = re.findall(r"\[\[(\d+(?:\.\d+)?)\]\]", response_text)
    if matches:
        score = float(matches[-1])   # Use the last rating in the reply.
        return max(1.0, min(10.0, score))
    # Fall back to the "Rating: X" format.
    matches2 = re.findall(r"[Rr]ating[:\s]+(\d+(?:\.\d+)?)", response_text)
    if matches2:
        score = float(matches2[-1])
        return max(1.0, min(10.0, score))
    return None


def judge_single(
    client: OpenAI,
    item: dict,
    model: str,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> dict:
    """Judge one response; return a None score on failure."""
    instruction = item.get("instruction", "")
    input_text  = item.get("input", "")
    output      = item.get("output", "")

    prompt = build_prompt(instruction, input_text, output)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0,
                max_tokens=512,
            )
            reply = response.choices[0].message.content or ""
            score = extract_score(reply)
            return {
                "instruction": instruction,
                "input":       input_text,
                "output":      output,
                "generator":   item.get("generator", ""),
                "judge_reply": reply,
                "score":       score,
            }
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
            else:
                return {
                    "instruction": instruction,
                    "input":       input_text,
                    "output":      output,
                    "generator":   item.get("generator", ""),
                    "judge_reply": f"[ERROR] {str(e)}",
                    "score":       None,
                }


def run_evaluation(
    input_file: str,
    output_file: str,
    api_key: str,
    model: str = "gpt-4o",
    max_workers: int = 4,
    base_url: Optional[str] = None,
):

    with open(input_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if not isinstance(dataset, list):
        raise ValueError("Input must be a JSON array (list of dictionaries)")

    print(f"Loaded data: {len(dataset)} rows")


    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)


    results = [None] * len(dataset)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(judge_single, client, item, model): idx
            for idx, item in enumerate(dataset)
        }
        for future in tqdm(as_completed(future_to_idx), total=len(dataset), desc="Evaluation progress"):
            idx = future_to_idx[future]
            results[idx] = future.result()


    valid_scores = [r["score"] for r in results if r["score"] is not None]
    failed_count = len(results) - len(valid_scores)

    # Convert mean scores in [1, 10] to percentage FA.
    avg_score_raw   = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
    avg_score_pct   = avg_score_raw * 10

    print(f"\nEvaluation complete: {len(valid_scores)} valid / {failed_count} failed")
    print(f"Mean raw score (1-10): {avg_score_raw:.4f}")
    print(f"Fine-tuning Accuracy (FA, x10): {avg_score_pct:.2f}")


    output_data = {
        "results": [
            {
                "instruction": r["instruction"],
                "input":       r["input"],
                "output":      r["output"],
                "generator":   r["generator"],
                "score":       r["score"],          # Raw score in [1, 10]; None indicates parsing failure.
                "judge_reply": r["judge_reply"],    # Retain the full judge reply for debugging.
            }
            for r in results
        ],
        "summary": {
            "total":          len(results),
            "valid":          len(valid_scores),
            "failed":         failed_count,
            "avg_score_1_10": round(avg_score_raw, 4),
            "avg_score_FA":   round(avg_score_pct, 2),   
            "model_judge":    model,
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {output_file}")
    if failed_count:
        raise RuntimeError(f"{failed_count} Alpaca responses failed judging; inspect the output and rerun the judge stage")


def parse_args():
    parser = argparse.ArgumentParser(
        description="AlpacaEval LLM judge evaluation (AsFT method)"
    )
    parser.add_argument(
        "--input_file", type=str, required=True,
        help="Input JSON path (list of dictionaries with instruction, input and output)",
    )
    parser.add_argument(
        "--output_file", type=str, default="eval_results.json",
        help="Output JSON path (default: eval_results.json)",
    )
    parser.add_argument(
        "--api_key", type=str, default=os.environ.get("OPENAI_API_KEY", ""),
        help="OpenAI API key (or set OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--base_url", type=str, default=None,
        help="Custom API base URL for a proxy or compatible endpoint",
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o",
        help="Judge model name",
    )
    parser.add_argument(
        "--max_workers", type=int, default=4,
        help="Worker threads (default: 4); adjust for API rate limits",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not args.api_key:
        raise ValueError(
            "Provide an API key via --api_key or OPENAI_API_KEY"
        )

    run_evaluation(
        input_file=args.input_file,
        output_file=args.output_file,
        api_key=args.api_key,
        model=args.model,
        max_workers=args.max_workers,
        base_url=args.base_url,
    )
