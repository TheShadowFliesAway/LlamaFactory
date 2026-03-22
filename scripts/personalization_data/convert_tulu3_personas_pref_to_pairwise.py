#!/usr/bin/env python3
"""Convert Tulu-3 personas preference data to a simpler pairwise jsonl format."""

import argparse
import json
from pathlib import Path


DEFAULT_INPUT = Path("data/tulu3_personas/tulu3_personas_pref.jsonl")
DEFAULT_OUTPUT = Path("data/tulu3_personas/tulu3_personas_pref_pairwise.jsonl")
DEFAULT_REPORT = Path("data/tulu3_personas/tulu3_personas_pref_pairwise_report.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert the local Tulu-3 personas preference jsonl file into a simpler pairwise "
            "jsonl format for DPO / ORPO preparation."
        )
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help=f"Input jsonl path. Defaults to {DEFAULT_INPUT}.")
    parser.add_argument(
        "--output", default=str(DEFAULT_OUTPUT), help=f"Output pairwise jsonl path. Defaults to {DEFAULT_OUTPUT}."
    )
    parser.add_argument(
        "--report", default=str(DEFAULT_REPORT), help=f"Output report json path. Defaults to {DEFAULT_REPORT}."
    )
    return parser.parse_args()


def extract_assistant_text(messages: list[dict[str, str]]) -> str:
    for message in messages:
        if message.get("role") == "assistant":
            return str(message.get("content", ""))
    raise ValueError("No assistant message found.")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = Path(args.report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    converted_rows = 0
    invalid_rows = 0
    preview: list[dict[str, object]] = []

    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            total_rows += 1
            row = json.loads(line)
            try:
                chosen_text = extract_assistant_text(row["chosen"])
                rejected_text = extract_assistant_text(row["rejected"])
            except Exception:
                invalid_rows += 1
                continue

            converted = {
                "id": row.get("id"),
                "instruction": row.get("prompt", ""),
                "input": "",
                "chosen": chosen_text,
                "rejected": rejected_text,
                "constraints": row.get("constraints", []),
                "chosen_model": row.get("chonsen_model"),
                "rejected_model": row.get("rejected_model"),
            }
            fout.write(json.dumps(converted, ensure_ascii=False) + "\n")
            converted_rows += 1

            if len(preview) < 3:
                preview.append(
                    {
                        "id": converted["id"],
                        "instruction": converted["instruction"],
                        "chosen_preview": chosen_text[:300],
                        "rejected_preview": rejected_text[:300],
                        "constraints": converted["constraints"],
                    }
                )

    report = {
        "input": str(input_path),
        "output": str(output_path),
        "total_rows": total_rows,
        "converted_rows": converted_rows,
        "invalid_rows": invalid_rows,
        "preview": preview,
    }

    with report_path.open("w", encoding="utf-8") as fout:
        json.dump(report, fout, ensure_ascii=False, indent=2)

    print(f"Scanned {total_rows} rows from {input_path}")
    print(f"Converted {converted_rows} rows to {output_path}")
    print(f"Skipped {invalid_rows} invalid rows")
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
