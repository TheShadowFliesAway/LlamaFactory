#!/usr/bin/env python3
"""Filter persona-conditioned samples from the Tulu-3 personas SFT dataset."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path


DEFAULT_INPUT = Path("data/tulu3_personas/sft/tulu3_personas_sft.jsonl")
DEFAULT_OUTPUT = Path("data/tulu3_personas/sft/tulu3_personas_sft_personalized.jsonl")
DEFAULT_REPORT = Path("data/tulu3_personas/sft/tulu3_personas_sft_personalized_report.json")

# These rules are intentionally simple and explainable.
# We want a first-pass subset with obvious persona/profile signals rather than
# an overly broad filter that silently pulls in generic instruction-following data.
PERSONA_PATTERNS = {
    "as_a": r"^as a\b",
    "as_an": r"^as an\b",
    "i_am": r"\bi am a\b",
    "im_a": r"\bi'm a\b",
    "my_role": r"\bmy role as\b",
    "my_clients": r"\bmy clients\b",
    "your_clients": r"\bfor your clients\b",
    "known_for_my": r"\bknown for my\b",
    "my_background": r"\bmy background\b",
    "my_experience": r"\bmy experience as\b",
    "political_identity": r"\bas a (liberal|conservative|democrat|republican)\b",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter the local Tulu-3 personas SFT jsonl file to keep samples with obvious "
            "persona/profile signals in the prompt."
        )
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help=f"Input jsonl file. Defaults to {DEFAULT_INPUT}.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"Output jsonl file for filtered samples. Defaults to {DEFAULT_OUTPUT}.",
    )
    parser.add_argument(
        "--report",
        default=str(DEFAULT_REPORT),
        help=f"Where to save the filtering report json. Defaults to {DEFAULT_REPORT}.",
    )
    parser.add_argument(
        "--preview-count",
        type=int,
        default=5,
        help="Number of matched prompts to store in the report preview. Defaults to 5.",
    )
    return parser.parse_args()


def compile_patterns() -> dict[str, re.Pattern[str]]:
    return {name: re.compile(pattern, re.IGNORECASE) for name, pattern in PERSONA_PATTERNS.items()}


def matched_rules(prompt: str, patterns: dict[str, re.Pattern[str]]) -> list[str]:
    return [name for name, pattern in patterns.items() if pattern.search(prompt)]


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = Path(args.report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    patterns = compile_patterns()
    total_rows = 0
    kept_rows = 0
    rule_counter: Counter[str] = Counter()
    preview: list[dict[str, object]] = []

    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            total_rows += 1
            sample = json.loads(line)
            prompt = sample.get("prompt", "")
            rules = matched_rules(prompt, patterns)
            if not rules:
                continue

            kept_rows += 1
            rule_counter.update(rules)
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")

            if len(preview) < args.preview_count:
                preview.append(
                    {
                        "id": sample.get("id"),
                        "matched_rules": rules,
                        "prompt": prompt,
                        "constraints": sample.get("constraints", []),
                    }
                )

    report = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "total_rows": total_rows,
        "kept_rows": kept_rows,
        "kept_ratio": round(kept_rows / total_rows, 6) if total_rows else 0.0,
        "rule_counts": dict(rule_counter.most_common()),
        "patterns": PERSONA_PATTERNS,
        "preview": preview,
    }

    with report_path.open("w", encoding="utf-8") as fout:
        json.dump(report, fout, ensure_ascii=False, indent=2)

    print(f"Scanned {total_rows} rows from {input_path}")
    print(f"Kept {kept_rows} persona-conditioned rows in {output_path}")
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
