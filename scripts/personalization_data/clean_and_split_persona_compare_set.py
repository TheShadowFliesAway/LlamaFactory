#!/usr/bin/env python3
"""Clean persona comparison prompts and remove them from the stage-2 training subset."""

import argparse
import json
from pathlib import Path


DEFAULT_PERSONA_INPUT = Path("data/tulu3_personas/tulu3_personas_sft_personalized.jsonl")
DEFAULT_COMPARE_INPUT = Path("data/tulu3_personas/persona_compare_samples.jsonl")
DEFAULT_COMPARE_OUTPUT = Path("data/tulu3_personas/persona_compare_samples_clean.jsonl")
DEFAULT_TRAIN_OUTPUT = Path("data/tulu3_personas/tulu3_personas_sft_personalized_train.jsonl")
DEFAULT_REPORT = Path("data/tulu3_personas/persona_compare_clean_split_report.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clean the comparison prompts and remove those source samples from the persona "
            "stage-2 training subset."
        )
    )
    parser.add_argument("--persona-input", default=str(DEFAULT_PERSONA_INPUT))
    parser.add_argument("--compare-input", default=str(DEFAULT_COMPARE_INPUT))
    parser.add_argument("--compare-output", default=str(DEFAULT_COMPARE_OUTPUT))
    parser.add_argument("--train-output", default=str(DEFAULT_TRAIN_OUTPUT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    return parser.parse_args()


def normalize_prompt(text: str) -> tuple[str, bool]:
    # Some prompts duplicate the whole block after a blank line.
    parts = [part.strip() for part in text.split("\n\n") if part.strip()]
    if len(parts) >= 2 and parts[0] == parts[-1]:
        return parts[0], True

    # Normalize excessive whitespace without changing content semantics too much.
    normalized = "\n\n".join(parts) if parts else text.strip()
    return normalized, normalized != text


def main() -> None:
    args = parse_args()
    persona_input = Path(args.persona_input)
    compare_input = Path(args.compare_input)
    compare_output = Path(args.compare_output)
    train_output = Path(args.train_output)
    report_output = Path(args.report)

    compare_output.parent.mkdir(parents=True, exist_ok=True)
    train_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.parent.mkdir(parents=True, exist_ok=True)

    compare_rows: list[dict[str, object]] = []
    compare_ids: set[str] = set()
    cleaned_compare_prompts = 0

    with compare_input.open("r", encoding="utf-8") as fin:
        for line in fin:
            row = json.loads(line)
            normalized_prompt, changed = normalize_prompt(str(row["prompt"]))
            row["prompt"] = normalized_prompt
            if changed:
                cleaned_compare_prompts += 1
            compare_rows.append(row)
            compare_ids.add(str(row["source_id"]))

    total_persona_rows = 0
    kept_train_rows = 0
    removed_for_compare = 0
    cleaned_train_prompts = 0

    with train_output.open("w", encoding="utf-8") as fout:
        for line in persona_input.open("r", encoding="utf-8"):
            row = json.loads(line)
            total_persona_rows += 1

            if str(row["id"]) in compare_ids:
                removed_for_compare += 1
                continue

            normalized_prompt, changed = normalize_prompt(str(row["prompt"]))
            row["prompt"] = normalized_prompt

            messages = row.get("messages")
            if isinstance(messages, list) and messages:
                first = messages[0]
                if isinstance(first, dict) and first.get("role") == "user":
                    first["content"] = normalized_prompt

            if changed:
                cleaned_train_prompts += 1

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept_train_rows += 1

    with compare_output.open("w", encoding="utf-8") as fout:
        for row in compare_rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "persona_input": str(persona_input),
        "compare_input": str(compare_input),
        "compare_output": str(compare_output),
        "train_output": str(train_output),
        "total_persona_rows": total_persona_rows,
        "compare_rows": len(compare_rows),
        "removed_for_compare": removed_for_compare,
        "kept_train_rows": kept_train_rows,
        "cleaned_compare_prompts": cleaned_compare_prompts,
        "cleaned_train_prompts": cleaned_train_prompts,
    }

    with report_output.open("w", encoding="utf-8") as fout:
        json.dump(report, fout, ensure_ascii=False, indent=2)

    print(f"Saved cleaned comparison set to {compare_output}")
    print(f"Saved stage-2 train subset without comparison samples to {train_output}")
    print(f"Saved report to {report_output}")


if __name__ == "__main__":
    main()
