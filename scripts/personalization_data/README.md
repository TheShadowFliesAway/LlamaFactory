# Personalization Data Scripts

This folder contains the helper scripts used to prepare data and evaluation artifacts for the personalization experiment pipeline.

## SFT Data Preparation

- `download_tulu3_personas_sft.py`
  - Download the full AllenAI Tulu-3 personas SFT dataset.

- `filter_tulu3_personas_sft.py`
  - Build a first-pass persona-conditioned subset using explainable prompt rules.

## Comparison Set Preparation

- `build_persona_comparison_set.py`
  - Build a held-out comparison/test set from `tulu3_personas_pref_pairwise.jsonl`.
  - The exported file still uses the `prompt` field so the generation script can read it directly.
  - Writes its own build report, separate from the train/test split report.

- `clean_and_split_persona_compare_set.py`
  - Remove the selected comparison/test rows from `tulu3_personas_pref_pairwise.jsonl`.
  - Write the remaining rows to `tulu3_personas_pref_pairwise_train.jsonl` for DPO / ORPO training.

## Preference Data Preparation

- `download_tulu3_personas_pref.py`
  - Download the AllenAI Tulu-3 personas preference dataset.

- `convert_tulu3_personas_pref_to_pairwise.py`
  - Convert the raw preference file from message-list format into a simpler pairwise jsonl format.

## Evaluation Scripts

- `run_persona_multimodel_generation.py`
  - Generate outputs for multiple experiment checkpoints, such as stage-1 SFT, stage-2 SFT, DPO, and ORPO, on the same comparison/test set.

- `run_persona_compare_generation.py`
  - Generate outputs for the cleaned comparison prompts with model A and model B.

- `judge_persona_compare.py`
  - Judge model A vs model B with an OpenAI-compatible API on persona alignment, instruction following, constraint satisfaction, response quality, and overall preference.

## Suggested Workflow

1. Download SFT data
2. Filter persona subset
3. Build the held-out comparison/test set from pairwise preference data
4. Download preference data
5. Convert preference data to pairwise format
6. Split the pairwise preference data into compare/test and preference-train subsets
7. Run stage-1 vs stage-2 generation on the comparison set
8. Judge the comparison outputs with an LLM API
