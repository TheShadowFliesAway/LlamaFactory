# Tulu-3 Personas Data Notes

This folder stores the local data files used in the personalization experiment pipeline.

## SFT Data

- `tulu3_personas_sft.jsonl`
  - Full local export of `allenai/tulu-3-sft-personas-instruction-following`.
  - Used for stage-1 full-data SFT.

- `tulu3_personas_sft_personalized.jsonl`
  - First-pass persona-conditioned subset filtered from the full SFT file.
  - This is the file currently used by `tulu3_personas_sft_personalized` in `data/dataset_info.json`.

## Comparison / Evaluation Data

- `persona_compare_samples_clean.jsonl`
  - Held-out comparison/test prompt set.
  - Rebuilt from `tulu3_personas_pref_pairwise.jsonl`, not from the SFT persona subset.

- `persona_compare_clean_split_report.json`
  - Report for the pairwise preference split step.
  - Records how many comparison rows were removed from the pairwise preference source and how many rows remain in the preference training subset.

- `persona_compare_build_report.json`
  - Report for the comparison/test-set construction step.
  - Records how many pairwise preference candidates were scanned and how many were selected by category.

## Preference Data

- `tulu3_personas_pref.jsonl`
  - Raw local preference data exported from `allenai/tulu-3-pref-personas-instruction-following`.

- `tulu3_personas_pref_pairwise.jsonl`
  - Converted pairwise preference file used for DPO / ORPO.
  - Also serves as the source file for building the held-out comparison/test prompts.

- `tulu3_personas_pref_pairwise_train.jsonl`
  - Pairwise preference training subset after removing the comparison/test rows.
  - This is the file that should be used by `tulu3_personas_pref_pairwise` in `data/dataset_info.json` after you rerun the split pipeline.

- `tulu3_personas_pref_pairwise_report.json`
  - Report for the pairwise conversion step.

## Other Reports

- `tulu3_personas_sft_personalized_report.json`
  - Report generated while filtering the persona subset from the full SFT data.

## Registered Dataset Names

The following dataset names are currently registered in `data/dataset_info.json`:

- `tulu3_personas_sft`
  - points to `tulu3_personas_sft.jsonl`
  - message-style SFT dataset

- `tulu3_personas_sft_personalized`
  - points to `tulu3_personas_sft_personalized.jsonl`
  - message-style SFT dataset for stage-2 persona training

- `tulu3_personas_pref_pairwise`
  - currently points to `tulu3_personas_pref_pairwise.jsonl`
  - pairwise preference dataset for DPO / ORPO

## Practical Notes

- In the original SFT dataset:
  - `prompt` and `messages[0].content` are duplicated
  - the real SFT conversation is already fully represented in `messages`

- In the converted preference dataset:
  - `instruction` comes from the original `prompt`
  - `chosen` / `rejected` come from the assistant responses in the original preference samples

- The held-out comparison/test prompts should now be built from the pairwise preference file, then removed into `tulu3_personas_pref_pairwise_train.jsonl`.
