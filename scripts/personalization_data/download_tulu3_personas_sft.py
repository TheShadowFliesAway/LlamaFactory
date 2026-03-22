#!/usr/bin/env python3
"""Download the AllenAI Tulu-3 personas SFT dataset and save it locally."""

import argparse
import json
from pathlib import Path

from datasets import load_dataset


DEFAULT_DATASET = "allenai/tulu-3-sft-personas-instruction-following"
DEFAULT_OUTPUT = Path("data/tulu3_personas/sft/tulu3_personas_sft.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the AllenAI Tulu-3 personas SFT dataset and export it as a local jsonl file "
            "for later registration in LLaMAFactory."
        )
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"Hugging Face dataset name. Defaults to {DEFAULT_DATASET}.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to download. Defaults to train.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"Where to save the exported jsonl file. Defaults to {DEFAULT_OUTPUT}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.dataset, split=args.split)

    with output_path.open("w", encoding="utf-8") as fout:
        for sample in dataset:
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Saved {len(dataset)} samples to {output_path}")


if __name__ == "__main__":
    main()
