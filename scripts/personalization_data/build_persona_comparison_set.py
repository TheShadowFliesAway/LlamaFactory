#!/usr/bin/env python3
"""Build a comparison/test set from the pairwise preference data."""

# Example usage:
#
# Build a 200-sample comparison/test set from the pairwise preference data:
# python3 scripts/personalization_data/build_persona_comparison_set.py \
#   --input data/tulu3_personas/pref/tulu3_personas_pref_pairwise.jsonl \
#   --output data/tulu3_personas/compare/persona_compare_samples_clean.jsonl \
#   --report data/tulu3_personas/compare/persona_compare_build_report.json \
#   --max-samples 200 \
#   --per-category 40

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_INPUT = Path("data/tulu3_personas/pref/tulu3_personas_pref_pairwise.jsonl")
DEFAULT_OUTPUT = Path("data/tulu3_personas/compare/persona_compare_samples_clean.jsonl")
DEFAULT_REPORT = Path("data/tulu3_personas/compare/persona_compare_build_report.json")


CATEGORY_PATTERNS = {
    "profession": [
        r"^as a [^,.;]{0,80}(advisor|teacher|lawyer|doctor|engineer|manager|consultant|journalist|chef|writer)\b",
        r"^as an [^,.;]{0,80}(advisor|artist|engineer|economist|analyst)\b",
        r"\bwith expertise in\b",
        r"\bmy role as\b",
    ],
    "audience": [
        r"\bfor your clients\b",
        r"\bmy clients\b",
        r"\bfor (students|beginners|patients|customers|readers|children|parents)\b",
        r"\baudience\b",
    ],
    "politics": [
        r"\bas a (liberal|conservative|democrat|republican)\b",
        r"\bpolitician\b",
        r"\bpolitical\b",
    ],
    "celebrity_style": [
        r"\bhollywood celebrity\b",
        r"\bknown for my\b",
        r"\bmusicals\b",
        r"\bfamous\b",
        r"\bwell-established\b",
    ],
    "location_identity": [
        r"\bfrom [A-Z][a-z]+(?:, [A-Z][a-z]+)?\b",
        r"\bin [A-Z][a-z]+(?:, [A-Z][a-z]+)?\b",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a persona comparison/test set from the pairwise preference data. "
            "This script only selects the held-out comparison rows."
        )
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help=f"Input jsonl path. Defaults to {DEFAULT_INPUT}.")
    parser.add_argument(
        "--output", default=str(DEFAULT_OUTPUT), help=f"Output comparison jsonl path. Defaults to {DEFAULT_OUTPUT}."
    )
    parser.add_argument(
        "--report", default=str(DEFAULT_REPORT), help=f"Output report path. Defaults to {DEFAULT_REPORT}."
    )
    parser.add_argument(
        "--per-category",
        type=int,
        default=40,
        help="Maximum samples to keep for each category before backfilling. Defaults to 40.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=200,
        help="Final number of comparison/test prompts to export. Defaults to 200.",
    )
    return parser.parse_args()


def compile_patterns() -> dict[str, list[re.Pattern[str]]]:
    return {key: [re.compile(p, re.IGNORECASE) for p in patterns] for key, patterns in CATEGORY_PATTERNS.items()}


def detect_categories(prompt: str, compiled_patterns: dict[str, list[re.Pattern[str]]]) -> list[str]:
    categories: list[str] = []
    for category, patterns in compiled_patterns.items():
        if any(pattern.search(prompt) for pattern in patterns):
            categories.append(category)
    return categories


def load_candidates(path: Path, compiled_patterns: dict[str, list[re.Pattern[str]]]) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as fin:
        for line_idx, line in enumerate(fin):
            sample = json.loads(line)
            prompt = sample["instruction"]
            categories = detect_categories(prompt, compiled_patterns)
            if not categories:
                categories = ["generic_persona"]

            candidates.append(
                {
                    "line_idx": line_idx,
                    "source_id": sample["id"],
                    "prompt": prompt,
                    "constraints": sample.get("constraints", []),
                    "categories": categories,
                }
            )
    return candidates


def select_samples(candidates: list[dict[str, object]], per_category: int, max_samples: int) -> list[dict[str, object]]:
    by_category: dict[str, list[dict[str, object]]] = defaultdict(list)
    for candidate in candidates:
        for category in candidate["categories"]:
            by_category[str(category)].append(candidate)

    selected: list[dict[str, object]] = []
    selected_ids: set[str] = set()

    for category in sorted(by_category.keys()):
        kept = 0
        for candidate in by_category[category]:
            source_id = str(candidate["source_id"])
            if source_id in selected_ids:
                continue
            selected.append(candidate)
            selected_ids.add(source_id)
            kept += 1
            if kept >= per_category or len(selected) >= max_samples:
                break
        if len(selected) >= max_samples:
            return selected

    for candidate in candidates:
        source_id = str(candidate["source_id"])
        if source_id in selected_ids:
            continue
        selected.append(candidate)
        selected_ids.add(source_id)
        if len(selected) >= max_samples:
            break

    return selected


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = Path(args.report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    compiled_patterns = compile_patterns()
    candidates = load_candidates(input_path, compiled_patterns)
    selected = select_samples(candidates, per_category=args.per_category, max_samples=args.max_samples)

    category_counter: Counter[str] = Counter()
    with output_path.open("w", encoding="utf-8") as fout:
        for idx, sample in enumerate(selected, start=1):
            for category in sample["categories"]:
                category_counter[str(category)] += 1

            record = {
                "id": f"persona_compare_{idx:03d}",
                "source_id": sample["source_id"],
                "prompt": sample["prompt"],
                "constraints": sample["constraints"],
                "categories": sample["categories"],
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    report = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "total_candidates": len(candidates),
        "selected_samples": len(selected),
        "per_category_limit": args.per_category,
        "category_distribution": dict(category_counter),
        "categories_defined": list(CATEGORY_PATTERNS.keys()) + ["generic_persona"],
        "preview": [
            {
                "id": f"persona_compare_{idx:03d}",
                "source_id": sample["source_id"],
                "categories": sample["categories"],
                "prompt": sample["prompt"],
            }
            for idx, sample in enumerate(selected[:5], start=1)
        ],
    }

    with report_path.open("w", encoding="utf-8") as fout:
        json.dump(report, fout, ensure_ascii=False, indent=2)

    print(f"Loaded {len(candidates)} pairwise preference candidates from {input_path}")
    print(f"Saved {len(selected)} comparison/test samples to {output_path}")
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
