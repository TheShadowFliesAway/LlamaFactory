#!/usr/bin/env python3
"""Judge persona comparison generations with an OpenAI-compatible API."""

# Example usage:
#
# export OPENAI_API_KEY=your_api_key
# python3 scripts/personalization_data/judge_persona_compare.py \
#   --input data/tulu3_personas/compare/persona_compare_generations.jsonl \
#   --model gpt-4.1-mini \
#   --output data/tulu3_personas/compare/persona_compare_judgments.jsonl \
#   --report data/tulu3_personas/compare/persona_compare_judgments_report.json
#
# If you use another OpenAI-compatible endpoint:
# python3 scripts/personalization_data/judge_persona_compare.py \
#   --input data/tulu3_personas/compare/persona_compare_generations.jsonl \
#   --model  gpt-4.1-mini\
#   --base-url https://api.zhizengzeng.com/v1 \
#   --api-key xxxx

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = """You are a strict but fair evaluator for persona-conditioned response generation.

You will receive:
- a user prompt
- optional constraint tags
- response A
- response B

Your job is to compare the two responses and decide which one is better on each criterion.

Evaluation criteria:
1. persona_alignment: Which response better reflects the persona, identity, audience, or profile implied in the prompt?
2. instruction_following: Which response better completes the requested task?
3. constraint_satisfaction: Which response better satisfies explicit constraints such as format, length, keywords, style, or casing?
4. response_quality: Which response is clearer, more coherent, and more useful overall?
5. overall_preference: Considering all the above criteria together, which response is better overall?

For each criterion, output only one of:
- "A"
- "B"
- "tie"

Also provide a short explanation for the overall decision.

Return strict JSON with this schema:
{
  "persona_alignment": "A|B|tie",
  "instruction_following": "A|B|tie",
  "constraint_satisfaction": "A|B|tie",
  "response_quality": "A|B|tie",
  "overall_preference": "A|B|tie",
  "reason": "short explanation"
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Judge stage-1 vs stage-2 persona comparison generations.")
    parser.add_argument(
        "--input",
        default="data/tulu3_personas/compare/persona_compare_generations.jsonl",
        help="Input jsonl with model_a_output and model_b_output.",
    )
    parser.add_argument(
        "--output",
        default="data/tulu3_personas/compare/persona_compare_judgments.jsonl",
        help="Output jsonl with per-sample judgments.",
    )
    parser.add_argument(
        "--report",
        default="data/tulu3_personas/compare/persona_compare_judgments_report.json",
        help="Output summary report json.",
    )
    parser.add_argument("--model", required=True, help="Judge model name.")
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL"),
        help="Optional OpenAI-compatible base URL. Defaults to OPENAI_BASE_URL env var.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY"),
        help="API key. Defaults to OPENAI_API_KEY env var.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Judge temperature. Defaults to 0.0 for more stable outputs.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retries if JSON parsing fails. Defaults to 3.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for A/B order shuffling. Defaults to 42.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fin:
        return [json.loads(line) for line in fin if line.strip()]


def build_user_prompt(row: dict[str, Any], swapped: bool) -> str:
    response_a = row["model_b_output"] if swapped else row["model_a_output"]
    response_b = row["model_a_output"] if swapped else row["model_b_output"]

    return (
        "Evaluate the following two responses.\n\n"
        f"Prompt:\n{row['prompt']}\n\n"
        f"Constraint tags:\n{json.dumps(row.get('constraints', []), ensure_ascii=False)}\n\n"
        f"Response A:\n{response_a}\n\n"
        f"Response B:\n{response_b}\n"
    )


def remap_choice(value: str, swapped: bool) -> str:
    if value not in {"A", "B", "tie"}:
        return value
    if not swapped or value == "tie":
        return value
    return "A" if value == "B" else "B"


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in judge response.")
    return json.loads(text[start : end + 1])


def judge_row(client, model: str, row: dict[str, Any], temperature: float, max_retries: int, swapped: bool) -> dict[str, Any]:
    user_prompt = build_user_prompt(row, swapped=swapped)

    last_error: Exception | None = None
    for _ in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = response.choices[0].message.content or ""
            parsed = extract_json(content)
            for key in [
                "persona_alignment",
                "instruction_following",
                "constraint_satisfaction",
                "response_quality",
                "overall_preference",
            ]:
                parsed[key] = remap_choice(str(parsed[key]), swapped=swapped)
            parsed["reason"] = str(parsed.get("reason", "")).strip()
            parsed["judge_raw_response"] = content
            parsed["ab_swapped"] = swapped
            return parsed
        except Exception as err:
            last_error = err

    raise RuntimeError(f"Judge failed after {max_retries} attempts: {last_error}")


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise ValueError("API key is required. Pass --api-key or set OPENAI_API_KEY.")

    try:
        from openai import OpenAI
    except ImportError as err:
        raise ImportError("Please install the openai package in the runtime environment.") from err

    client_kwargs = {"api_key": args.api_key}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = OpenAI(**client_kwargs)

    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = Path(args.report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_rows(input_path)
    random.seed(args.seed)

    summary = {
        "persona_alignment": {"A": 0, "B": 0, "tie": 0},
        "instruction_following": {"A": 0, "B": 0, "tie": 0},
        "constraint_satisfaction": {"A": 0, "B": 0, "tie": 0},
        "response_quality": {"A": 0, "B": 0, "tie": 0},
        "overall_preference": {"A": 0, "B": 0, "tie": 0},
    }

    with output_path.open("w", encoding="utf-8") as fout:
        for row in rows:
            swapped = bool(random.getrandbits(1))
            judgment = judge_row(
                client=client,
                model=args.model,
                row=row,
                temperature=args.temperature,
                max_retries=args.max_retries,
                swapped=swapped,
            )
            record = {**row, "judgment": judgment}
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

            for key in summary:
                value = judgment[key]
                if value in summary[key]:
                    summary[key][value] += 1

    report = {
        "input": str(input_path),
        "output": str(output_path),
        "num_rows": len(rows),
        "judge_model": args.model,
        "base_url": args.base_url,
        "temperature": args.temperature,
        "max_retries": args.max_retries,
        "summary": summary,
    }

    with report_path.open("w", encoding="utf-8") as fout:
        json.dump(report, fout, ensure_ascii=False, indent=2)

    print(f"Saved judgments to {output_path}")
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
