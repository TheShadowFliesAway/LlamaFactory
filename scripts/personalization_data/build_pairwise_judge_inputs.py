#!/usr/bin/env python3
"""Build pairwise judge input files from the multi-model generation table."""

# Example usage:
#
# Build the four recommended pairwise comparison files:
# python3 scripts/personalization_data/build_pairwise_judge_inputs.py \
#   --input data/tulu3_personas/compare/persona_multimodel_generations.jsonl \
#   --output-dir data/tulu3_personas/compare/pairwise_judge_inputs

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("data/tulu3_personas/compare/persona_multimodel_generations.jsonl")
DEFAULT_OUTPUT_DIR = Path("data/tulu3_personas/compare/pairwise_judge_inputs")
DEFAULT_REPORT = Path("data/tulu3_personas/compare/pairwise_judge_inputs/pairwise_judge_inputs_report.json")


PAIR_DEFS = [
    ("stage1", "stage2"),
    ("stage2", "dpo"),
    ("stage2", "orpo"),
    ("dpo", "orpo"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert the multi-model generation table into pairwise judge input files "
            "for the main comparison groups."
        )
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help=f"Input multi-model jsonl. Defaults to {DEFAULT_INPUT}.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory for exported pairwise judge input files. Defaults to {DEFAULT_OUTPUT_DIR}.",
    )
    parser.add_argument(
        "--report",
        default=str(DEFAULT_REPORT),
        help=f"Output report path. Defaults to {DEFAULT_REPORT}.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fin:
        return [json.loads(line) for line in fin if line.strip()]


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    report_path = Path(args.report)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_rows(input_path)

    exported_files: list[dict[str, Any]] = []

    for model_a, model_b in PAIR_DEFS:
        output_path = output_dir / f"persona_pair_{model_a}_vs_{model_b}.jsonl"
        key_a = f"{model_a}_output"
        key_b = f"{model_b}_output"

        with output_path.open("w", encoding="utf-8") as fout:
            for row in rows:
                record = {
                    "id": row["id"],
                    "source_id": row.get("source_id"),
                    "prompt": row["prompt"],
                    "constraints": row.get("constraints", []),
                    "categories": row.get("categories", []),
                    "model_a_name": model_a,
                    "model_b_name": model_b,
                    "model_a_output": row[key_a],
                    "model_b_output": row[key_b],
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

        exported_files.append(
            {
                "pair": f"{model_a}_vs_{model_b}",
                "output_path": str(output_path),
                "num_rows": len(rows),
                "model_a_name": model_a,
                "model_b_name": model_b,
            }
        )

    report = {
        "input": str(input_path),
        "output_dir": str(output_dir),
        "num_rows": len(rows),
        "pairs": exported_files,
    }

    with report_path.open("w", encoding="utf-8") as fout:
        json.dump(report, fout, ensure_ascii=False, indent=2)

    print(f"Saved pairwise judge input files to {output_dir}")
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
