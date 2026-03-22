This folder stores the YAML configs for the personalization experiment pipeline.

## Recommended Order

1. `train_demo_my_sft_10.yaml`
   - Minimal local SFT sanity check.

2. `infer_demo_my_sft_10.yaml`
   - Minimal local inference sanity check for a LoRA adapter.

3. `train_tulu3_personas_sft_full.yaml`
   - Stage-1 full-data SFT baseline on the full Tulu-3 personas SFT data.

4. `merge_tulu3_personas_sft_full.yaml`
   - Merge the stage-1 LoRA adapter into the base model.

5. `train_tulu3_personas_sft_personalized.yaml`
   - Stage-2 persona-focused SFT on the comparison-cleaned persona subset.

6. `merge_tulu3_personas_sft_personalized.yaml`
   - Merge the stage-2 persona-focused LoRA adapter into the stage-1 merged model.

7. `train_tulu3_personas_dpo.yaml`
   - DPO training on the converted pairwise preference dataset.

8. `train_tulu3_personas_orpo.yaml`
   - ORPO training on the same pairwise preference dataset.

## File Summary

- `train_demo_my_sft_10.yaml`
  - 10-sample SFT demo.

- `infer_demo_my_sft_10.yaml`
  - Matching inference config for the 10-sample SFT demo.

- `train_tulu3_personas_sft_full.yaml`
  - Formal full-data SFT baseline.

- `merge_tulu3_personas_sft_full.yaml`
  - Merge config for the stage-1 full-data SFT adapter.

- `train_tulu3_personas_sft_personalized.yaml`
  - Stage-2 persona-focused SFT config using the persona subset after removing comparison samples.

- `merge_tulu3_personas_sft_personalized.yaml`
  - Merge config for the stage-2 persona-focused SFT adapter.

- `train_tulu3_personas_dpo.yaml`
  - DPO config using the converted pairwise preference dataset.

- `train_tulu3_personas_orpo.yaml`
  - ORPO config using the converted pairwise preference dataset.

## Notes

- Every YAML file now includes a copyable run command at the top.
- The stage-2 SFT config uses the persona training subset with comparison prompts removed.
- The DPO / ORPO configs assume you have already merged the stage-2 SFT result.
