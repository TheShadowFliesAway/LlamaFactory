#!/usr/bin/env python3
"""Generate outputs for multiple persona models on the same comparison set."""

# Example usage:
#
# Generate outputs for the four main experiment checkpoints:
# python3 scripts/personalization_data/run_persona_multimodel_generation.py \
#   --input data/tulu3_personas/persona_compare_samples_clean.jsonl \
#   --output data/tulu3_personas/persona_multimodel_generations.jsonl \
#   --report data/tulu3_personas/persona_multimodel_generations_report.json \
#   --model stage1=saves/qwen2.5-3b/merged/tulu3_personas_sft_full \
#   --model stage2=saves/qwen2.5-3b/merged/tulu3_personas_sft_personalized \
#   --model dpo=saves/qwen2.5-3b/merged/tulu3_personas_dpo \
#   --model orpo=saves/qwen2.5-3b/merged/tulu3_personas_orpo \
#   --trust-remote-code

import argparse
import gc
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate responses for the cleaned persona comparison set with multiple models "
            "and save all outputs into one jsonl file."
        )
    )
    parser.add_argument(
        "--input",
        default="data/tulu3_personas/persona_compare_samples_clean.jsonl",
        help="Input comparison jsonl file.",
    )
    parser.add_argument(
        "--output",
        default="data/tulu3_personas/persona_multimodel_generations.jsonl",
        help="Output jsonl file containing prompt + outputs from all models.",
    )
    parser.add_argument(
        "--report",
        default="data/tulu3_personas/persona_multimodel_generations_report.json",
        help="Output report json file.",
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help=(
            "Model spec in the form name=base_model_path or "
            "name=base_model_path::adapter_path. "
            "Can be passed multiple times."
        ),
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


def parse_model_spec(spec: str) -> dict[str, str | None]:
    if "=" not in spec:
        raise ValueError(f"Invalid --model spec: {spec}")

    name, payload = spec.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"Model name is empty in spec: {spec}")

    if "::" in payload:
        model_path, adapter_path = payload.split("::", 1)
        model_path = model_path.strip()
        adapter_path = adapter_path.strip() or None
    else:
        model_path = payload.strip()
        adapter_path = None

    if not model_path:
        raise ValueError(f"Base model path is empty in spec: {spec}")

    return {"name": name, "model_path": model_path, "adapter_path": adapter_path}


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fin:
        return [json.loads(line) for line in fin if line.strip()]


def load_model_stack(model_path: str, adapter_path: str | None, trust_remote_code: bool):
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
    model_specs = [parse_model_spec(spec) for spec in args.model]

    all_outputs: dict[str, list[str]] = {}
    for spec in model_specs:
        outputs = generate_for_rows(
            rows=rows,
            model_path=str(spec["model_path"]),
            adapter_path=spec["adapter_path"],
            trust_remote_code=args.trust_remote_code,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
        )
        all_outputs[str(spec["name"])] = outputs

    with output_path.open("w", encoding="utf-8") as fout:
        for row_idx, row in enumerate(rows):
            record = dict(row)
            for spec in model_specs:
                name = str(spec["name"])
                record[f"{name}_output"] = all_outputs[name][row_idx]
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    report = {
        "input": str(input_path),
        "output": str(output_path),
        "num_rows": len(rows),
        "template": args.template,
        "models": model_specs,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
    }

    with report_path.open("w", encoding="utf-8") as fout:
        json.dump(report, fout, ensure_ascii=False, indent=2)

    print(f"Saved multi-model generations to {output_path}")
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
