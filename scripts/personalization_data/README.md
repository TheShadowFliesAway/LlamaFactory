# Personalization Data Scripts

This folder contains the helper scripts used to prepare data and evaluation artifacts for the personalization experiment pipeline.

## SFT Data Preparation

- `download_tulu3_personas_sft.py`
  - Download the full AllenAI Tulu-3 personas SFT dataset.

- `filter_tulu3_personas_sft.py`
  - Build a first-pass persona-conditioned subset using explainable prompt rules.

## Comparison Set Preparation

- `build_persona_comparison_set.py`
  - Build a small comparison set from the persona subset.

- `clean_and_split_persona_compare_set.py`
  - Clean duplicated prompt blocks in the comparison set and remove comparison samples from the stage-2 training subset.

## Preference Data Preparation

- `download_tulu3_personas_pref.py`
  - Download the AllenAI Tulu-3 personas preference dataset.

- `convert_tulu3_personas_pref_to_pairwise.py`
  - Convert the raw preference file from message-list format into a simpler pairwise jsonl format.

## Evaluation Scripts

- `run_persona_compare_generation.py`
  - Generate outputs for the cleaned comparison prompts with model A and model B.

- `judge_persona_compare.py`
  - Judge model A vs model B with an OpenAI-compatible API on persona alignment, instruction following, constraint satisfaction, response quality, and overall preference.

## Suggested Workflow

1. Download SFT data
2. Filter persona subset
3. Build and clean comparison set
4. Download preference data
5. Convert preference data to pairwise format
6. Run stage-1 vs stage-2 generation on the comparison set
7. Judge the comparison outputs with an LLM API
