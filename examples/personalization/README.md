This folder stores the YAML configs for the personalization experiments.

Files currently included:

- `train_demo_my_sft_10.yaml`
  - The minimal 10-sample SFT demo used to verify the local workflow end to end.

- `infer_demo_my_sft_10.yaml`
  - The matching inference config for the 10-sample SFT demo.

- `train_tulu3_personas_sft_full.yaml`
  - The formal SFT baseline using the full local Tulu-3 personas SFT dataset.

- `train_tulu3_personas_sft_personalized.yaml`
  - The persona-focused SFT config using the filtered subset with stronger persona/profile signals.
