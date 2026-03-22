#!/usr/bin/env python3
"""Run batch generation for persona comparison prompts with two models."""

# Example usage:
#
# Compare the stage-1 merged model with the stage-2 persona LoRA:
# python3 scripts/personalization_data/run_persona_compare_generation.py \
#   --model-a-path saves/qwen2.5-3b/merged/tulu3_personas_sft_full \
#   --model-b-path saves/qwen2.5-3b/merged/tulu3_personas_sft_full \
#   --adapter-b-path saves/qwen2.5-3b/lora/tulu3_personas_sft_personalized \
#   --trust-remote-code
#
# Compare two fully merged models:
# python3 scripts/personalization_data/run_persona_compare_generation.py \
#   --model-a-path /path/to/model_a \
#   --model-b-path /path/to/model_b \
#   --trust-remote-code

import argparse
import gc
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate responses for the cleaned persona comparison set with two models, "
            "then save a single comparison jsonl file."
        )
    )
    parser.add_argument(
        "--input",
        default="data/tulu3_personas/persona_compare_samples_clean.jsonl",
        help="Input comparison jsonl file.",
    )
    parser.add_argument(
        "--output",
        default="data/tulu3_personas/persona_compare_generations.jsonl",
        help="Output jsonl file containing prompt + model A/B outputs.",
    )
    parser.add_argument(
        "--report",
        default="data/tulu3_personas/persona_compare_generations_report.json",
        help="Output report json file.",
    )
    parser.add_argument(
        "--model-a-path",
        required=True,
        help="Base model path for model A.",
    )
    parser.add_argument(
        "--adapter-a-path",
        default=None,
        help="Optional LoRA adapter path for model A.",
    )
    parser.add_argument(
        "--model-b-path",
        required=True,
        help="Base model path for model B.",
    )
    parser.add_argument(
        "--adapter-b-path",
        default=None,
        help="Optional LoRA adapter path for model B.",
    )
    parser.add_argument(
        "--template",
        default="qwen",
        help="Template name, kept for bookkeeping in the output report.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum new tokens to generate for each prompt.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for generation. Start with 1 for stability.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading the tokenizer/model.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fin:
        return [json.loads(line) for line in fin if line.strip()]


def load_model_stack(
    model_path: str,
    adapter_path: str | None,
    trust_remote_code: bool,
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return tokenizer, model


def unload_model_stack(tokenizer, model) -> None:
    del tokenizer
    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def generate_for_rows(
    rows: list[dict[str, Any]],
    model_path: str,
    adapter_path: str | None,
    trust_remote_code: bool,
    max_new_tokens: int,
    batch_size: int,
) -> list[str]:
    import torch

    tokenizer, model = load_model_stack(model_path, adapter_path, trust_remote_code)
    outputs: list[str] = []

    for batch_start in range(0, len(rows), batch_size):
        batch_rows = rows[batch_start : batch_start + batch_size]
        prompts = [row["prompt"] for row in batch_rows]

        rendered = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in prompts
        ]

        inputs = tokenizer(rendered, return_tensors="pt", padding=True)
        if torch.cuda.is_available():
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        input_len = inputs["input_ids"].shape[1]
        decoded = tokenizer.batch_decode(generated[:, input_len:], skip_special_tokens=True)
        outputs.extend([text.strip() for text in decoded])

    unload_model_stack(tokenizer, model)
    return outputs


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = Path(args.report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_rows(input_path)

    model_a_outputs = generate_for_rows(
        rows=rows,
        model_path=args.model_a_path,
        adapter_path=args.adapter_a_path,
        trust_remote_code=args.trust_remote_code,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )

    model_b_outputs = generate_for_rows(
        rows=rows,
        model_path=args.model_b_path,
        adapter_path=args.adapter_b_path,
        trust_remote_code=args.trust_remote_code,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )

    with output_path.open("w", encoding="utf-8") as fout:
        for row, output_a, output_b in zip(rows, model_a_outputs, model_b_outputs):
            fout.write(
                json.dumps(
                    {
                        **row,
                        "model_a_output": output_a,
                        "model_b_output": output_b,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    report = {
        "input": str(input_path),
        "output": str(output_path),
        "num_rows": len(rows),
        "template": args.template,
        "model_a_path": args.model_a_path,
        "adapter_a_path": args.adapter_a_path,
        "model_b_path": args.model_b_path,
        "adapter_b_path": args.adapter_b_path,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
    }

    with report_path.open("w", encoding="utf-8") as fout:
        json.dump(report, fout, ensure_ascii=False, indent=2)

    print(f"Saved comparison generations to {output_path}")
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
