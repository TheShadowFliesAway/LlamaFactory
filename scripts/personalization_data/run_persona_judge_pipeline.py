#!/usr/bin/env python3
"""Run the full pairwise-judge pipeline after multi-model generations are ready."""

# Example usage:
#
# export OPENAI_API_KEY=your_api_key
# python3 scripts/personalization_data/run_persona_judge_pipeline.py \
#   --multimodel-input data/tulu3_personas/compare/persona_multimodel_generations.jsonl \
#   --multimodel-report data/tulu3_personas/compare/persona_multimodel_generations_report.json \
#   --wait-for-multimodel \
#   --pairwise-dir data/tulu3_personas/compare/pairwise_judge_inputs \
#   --judge-output-dir data/tulu3_personas/compare/judge_results \
#   --judge-model gpt-4.1-mini \
#   --base-url https://api.zhizengzeng.com/v1
#
# Or pass the API key directly:
# python3 scripts/personalization_data/run_persona_judge_pipeline.py \
#   --multimodel-input data/tulu3_personas/compare/persona_multimodel_generations.jsonl \
#   --pairwise-dir data/tulu3_personas/compare/pairwise_judge_inputs \
#   --judge-output-dir data/tulu3_personas/compare/judge_results \
#   --judge-model gpt-4.1-mini \
#   --base-url https://api.zhizengzeng.com/v1 \
#   --api-key your_api_key

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


PAIR_DEFS = [
    ("stage1", "stage2"),
    ("stage2", "dpo"),
    ("stage2", "orpo"),
    ("dpo", "orpo"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "After the four-model generation file is ready, build the four pairwise "
            "judge inputs and run judge_persona_compare.py on each pair."
        )
    )
    parser.add_argument(
        "--multimodel-input",
        default="data/tulu3_personas/compare/persona_multimodel_generations.jsonl",
        help="Input multi-model generations jsonl.",
    )
    parser.add_argument(
        "--multimodel-report",
        default="data/tulu3_personas/compare/persona_multimodel_generations_report.json",
        help=(
            "Report file produced by run_persona_multimodel_generation.py. "
            "When --wait-for-multimodel is enabled, this file is used as the completion signal."
        ),
    )
    parser.add_argument(
        "--wait-for-multimodel",
        action="store_true",
        help=(
            "Wait until the multimodel generation report file appears before starting "
            "pairwise export and judge."
        ),
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=60,
        help="Polling interval in seconds when waiting for the multimodel generation to finish.",
    )
    parser.add_argument(
        "--pairwise-dir",
        default="data/tulu3_personas/compare/pairwise_judge_inputs",
        help="Directory for pairwise judge input files.",
    )
    parser.add_argument(
        "--judge-output-dir",
        default="data/tulu3_personas/compare/judge_results",
        help="Directory for per-pair judge outputs.",
    )
    parser.add_argument(
        "--summary-md",
        default="data/tulu3_personas/compare/judge_results/judge_summary.md",
        help="Output Markdown summary path.",
    )
    parser.add_argument(
        "--summary-json",
        default="data/tulu3_personas/compare/judge_results/judge_summary.json",
        help="Output machine-readable summary path.",
    )
    parser.add_argument(
        "--judge-model",
        required=True,
        help="Judge model name, e.g. gpt-4.1-mini.",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL"),
        help="OpenAI-compatible base URL. Defaults to OPENAI_BASE_URL.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY"),
        help="API key. Defaults to OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Judge temperature. Defaults to 0.0.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retries for judge JSON parsing failures.",
    )
    return parser.parse_args()


def run_command(cmd: list[str]) -> None:
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def wait_for_multimodel_completion(input_path: Path, report_path: Path, poll_interval: int) -> None:
    print(f"Waiting for multimodel generation to finish.")
    print(f"- expecting report: {report_path}")
    print(f"- current output file: {input_path}")
    print(f"- poll interval: {poll_interval}s")

    while True:
        if report_path.exists():
            try:
                report_mtime = report_path.stat().st_mtime
                if input_path.exists():
                    input_mtime = input_path.stat().st_mtime
                    if report_mtime >= input_mtime:
                        print(f"Detected completed multimodel generation report: {report_path}")
                        return
                    print(
                        f"[wait] report exists but looks stale: "
                        f"report mtime {report_mtime:.0f} < output mtime {input_mtime:.0f}"
                    )
                else:
                    print(f"Detected multimodel generation report: {report_path}")
                    return
            except OSError:
                print(f"[wait] report exists but timestamps could not be read: {report_path}")

        if input_path.exists():
            try:
                size = input_path.stat().st_size
                print(f"[wait] output exists and is still being written: {input_path} ({size} bytes)")
            except OSError:
                print(f"[wait] output exists but size could not be read: {input_path}")
        else:
            print(f"[wait] output file not created yet: {input_path}")

        time.sleep(poll_interval)


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise ValueError("API key is required. Pass --api-key or set OPENAI_API_KEY.")

    multimodel_input = Path(args.multimodel_input)
    multimodel_report = Path(args.multimodel_report)
    pairwise_dir = Path(args.pairwise_dir)
    judge_output_dir = Path(args.judge_output_dir)
    pairwise_dir.mkdir(parents=True, exist_ok=True)
    judge_output_dir.mkdir(parents=True, exist_ok=True)

    if args.wait_for_multimodel:
        wait_for_multimodel_completion(
            input_path=multimodel_input,
            report_path=multimodel_report,
            poll_interval=args.poll_interval,
        )
    elif not multimodel_input.exists():
        raise FileNotFoundError(
            f"Multi-model generation file not found: {multimodel_input}. "
            "Either generate it first or pass --wait-for-multimodel."
        )

    build_cmd = [
        sys.executable,
        "scripts/personalization_data/build_pairwise_judge_inputs.py",
        "--input",
        str(multimodel_input),
        "--output-dir",
        str(pairwise_dir),
    ]
    run_command(build_cmd)

    for model_a, model_b in PAIR_DEFS:
        pair_name = f"{model_a}_vs_{model_b}"
        pair_input = pairwise_dir / f"persona_pair_{pair_name}.jsonl"
        pair_output = judge_output_dir / f"judge_{pair_name}.jsonl"
        pair_report = judge_output_dir / f"judge_{pair_name}_report.json"

        judge_cmd = [
            sys.executable,
            "scripts/personalization_data/judge_persona_compare.py",
            "--input",
            str(pair_input),
            "--model",
            args.judge_model,
            "--output",
            str(pair_output),
            "--report",
            str(pair_report),
            "--api-key",
            args.api_key,
            "--temperature",
            str(args.temperature),
            "--max-retries",
            str(args.max_retries),
        ]
        if args.base_url:
            judge_cmd.extend(["--base-url", args.base_url])

        run_command(judge_cmd)

    summarize_cmd = [
        sys.executable,
        "scripts/personalization_data/summarize_judge_results.py",
        "--judge-output-dir",
        str(judge_output_dir),
        "--output-md",
        args.summary_md,
        "--output-json",
        args.summary_json,
    ]
    run_command(summarize_cmd)

    print("\nFinished pairwise export and judge pipeline.")
    print(f"Pairwise inputs: {pairwise_dir}")
    print(f"Judge outputs: {judge_output_dir}")


if __name__ == "__main__":
    main()
