# Personalization Data Scripts

This folder contains the helper scripts used to prepare data and evaluation artifacts for the personalization experiment pipeline.

## SFT Data Preparation

- `download_tulu3_personas_sft.py`
  - Download the full AllenAI Tulu-3 personas SFT dataset.
  - Default output: `data/tulu3_personas/sft/`.

- `filter_tulu3_personas_sft.py`
  - Build a first-pass persona-conditioned subset using explainable prompt rules.
  - Default output: `data/tulu3_personas/sft/`.

## Comparison Set Preparation

- `build_persona_comparison_set.py`
  - Build a held-out comparison/test set from `pref/tulu3_personas_pref_pairwise.jsonl`.
  - The exported file still uses the `prompt` field so the generation script can read it directly.
  - Writes its own build report, separate from the train/test split report.
  - Default output: `data/tulu3_personas/compare/`.

- `clean_and_split_persona_compare_set.py`
  - Remove the selected comparison/test rows from `pref/tulu3_personas_pref_pairwise.jsonl`.
  - Write the remaining rows to `pref/tulu3_personas_pref_pairwise_train.jsonl` for DPO / ORPO training.
  - Default compare output/report path: `data/tulu3_personas/compare/`.

## Preference Data Preparation

- `download_tulu3_personas_pref.py`
  - Download the AllenAI Tulu-3 personas preference dataset.
  - Default output: `data/tulu3_personas/pref/`.

- `convert_tulu3_personas_pref_to_pairwise.py`
  - Convert the raw preference file from message-list format into a simpler pairwise jsonl format.
  - Default output: `data/tulu3_personas/pref/`.

## Evaluation Scripts

- `run_persona_multimodel_generation.py`
  - Generate outputs for multiple experiment checkpoints, such as stage-1 SFT, stage-2 SFT, DPO, and ORPO, on the same comparison/test set.
  - Default output: `data/tulu3_personas/compare/`.

- `build_pairwise_judge_inputs.py`
  - Convert the multi-model generation table into the four main pairwise judge input files:
    `stage1_vs_stage2`, `stage2_vs_dpo`, `stage2_vs_orpo`, and `dpo_vs_orpo`.
  - Default output: `data/tulu3_personas/compare/pairwise_judge_inputs/`.

- `run_persona_judge_pipeline.py`
  - After the four-model generation file is ready, automatically export the four pairwise judge inputs and run judge evaluation for each pair.
  - API credentials are passed through command-line arguments or environment variables instead of being hardcoded.
  - Also runs the final summary step to produce a Markdown report.
  - Can wait for `compare/persona_multimodel_generations_report.json` before starting, which is useful when the generation file already exists but is still being appended.

- `summarize_judge_results.py`
  - Summarize the four pairwise judge reports into a single Markdown summary and a machine-readable JSON summary.
  - Default output: `data/tulu3_personas/compare/judge_results/`.

- `judge_persona_compare.py`
  - Judge model A vs model B with an OpenAI-compatible API on persona alignment, instruction following, constraint satisfaction, response quality, and overall preference.
  - Default output: `data/tulu3_personas/compare/`.

## Suggested Workflow

1. Download SFT data
2. Filter persona subset
3. Build the held-out comparison/test set from pairwise preference data
4. Download preference data
5. Convert preference data to pairwise format
6. Split the pairwise preference data into compare/test and preference-train subsets
7. Run stage-1 vs stage-2 generation on the comparison set
8. Judge the comparison outputs with an LLM API
