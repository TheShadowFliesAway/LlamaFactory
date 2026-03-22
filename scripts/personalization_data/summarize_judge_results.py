#!/usr/bin/env python3
"""Summarize pairwise judge reports into one Markdown table/report."""

# Example usage:
#
# python3 scripts/personalization_data/summarize_judge_results.py \
#   --judge-output-dir data/tulu3_personas/compare/judge_results \
#   --output-md data/tulu3_personas/compare/judge_results/judge_summary.md

import argparse
import json
from pathlib import Path


PAIR_ORDER = [
    "stage1_vs_stage2",
    "stage2_vs_dpo",
    "stage2_vs_orpo",
    "dpo_vs_orpo",
]

METRICS = [
    "persona_alignment",
    "instruction_following",
    "constraint_satisfaction",
    "response_quality",
    "overall_preference",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the four pairwise judge reports into one Markdown report."
    )
    parser.add_argument(
        "--judge-output-dir",
        default="data/tulu3_personas/compare/judge_results",
        help="Directory containing judge_*_report.json files.",
    )
    parser.add_argument(
        "--output-md",
        default="data/tulu3_personas/compare/judge_results/judge_summary.md",
        help="Output Markdown summary path.",
    )
    parser.add_argument(
        "--output-json",
        default="data/tulu3_personas/compare/judge_results/judge_summary.json",
        help="Optional machine-readable summary json path.",
    )
    return parser.parse_args()


def load_report(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fin:
        return json.load(fin)


def pair_title(pair_name: str) -> str:
    return pair_name.replace("_vs_", " vs ")


def render_metric_table(report_map: dict[str, dict]) -> list[str]:
    lines = [
        "| Pair | Persona Alignment | Instruction Following | Constraint Satisfaction | Response Quality | Overall Preference |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for pair in PAIR_ORDER:
        report = report_map.get(pair)
        if not report:
            continue
        summary = report["summary"]
        row = [pair_title(pair)]
        for metric in METRICS:
            m = summary[metric]
            row.append(f"A {m['A']} / B {m['B']} / tie {m['tie']}")
        lines.append("| " + " | ".join(row) + " |")

    return lines


def render_quick_takeaways(report_map: dict[str, dict]) -> list[str]:
    lines = ["## Quick Takeaways", ""]
    for pair in PAIR_ORDER:
        report = report_map.get(pair)
        if not report:
            continue
        overall = report["summary"]["overall_preference"]
        winner = "tie"
        if overall["A"] > overall["B"]:
            winner = "A"
        elif overall["B"] > overall["A"]:
            winner = "B"
        lines.append(
            f"- **{pair_title(pair)}**: overall preference = `A {overall['A']} / B {overall['B']} / tie {overall['tie']}`; "
            f"current winner: **{winner}**."
        )
    lines.append("")
    return lines


def main() -> None:
    args = parse_args()
    judge_output_dir = Path(args.judge_output_dir)
    output_md = Path(args.output_md)
    output_json = Path(args.output_json)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    report_map: dict[str, dict] = {}
    missing: list[str] = []

    for pair in PAIR_ORDER:
        report_path = judge_output_dir / f"judge_{pair}_report.json"
        if report_path.exists():
            report_map[pair] = load_report(report_path)
        else:
            missing.append(pair)

    md_lines = [
        "# Persona Judge Summary",
        "",
        "This file aggregates the pairwise LLM-as-a-judge results for the main experiment comparisons.",
        "",
    ]

    if missing:
        md_lines.extend(
            [
                "## Missing Reports",
                "",
                *(f"- {pair}" for pair in missing),
                "",
            ]
        )

    md_lines.extend(["## Summary Table", ""])
    md_lines.extend(render_metric_table(report_map))
    md_lines.append("")
    md_lines.extend(render_quick_takeaways(report_map))

    with output_md.open("w", encoding="utf-8") as fout:
        fout.write("\n".join(md_lines) + "\n")

    machine_summary = {
        "judge_output_dir": str(judge_output_dir),
        "pairs_found": sorted(report_map.keys()),
        "pairs_missing": missing,
        "reports": report_map,
    }
    with output_json.open("w", encoding="utf-8") as fout:
        json.dump(machine_summary, fout, ensure_ascii=False, indent=2)

    print(f"Saved Markdown summary to {output_md}")
    print(f"Saved JSON summary to {output_json}")


if __name__ == "__main__":
    main()
