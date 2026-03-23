# 基于 Tulu-3 Personas 的个性化大模型微调实验

当前仓库已经整理为一套围绕 `Qwen2.5-3B-Instruct` 与 LLaMAFactory 的个性化响应生成实验工程。

当前主线目标是研究一条分阶段的 persona-conditioned 对齐流程：

1. 全量监督微调（SFT）
2. persona-focused 二阶段监督微调
3. 基于偏好数据的 DPO / ORPO 训练
4. 基于统一 compare/test 集的多模型生成与 LLM-as-a-judge 评测

原始上游项目的 README 已经备份为：

- [README_official.md](README_official.md)
- [README_zh_official.md](README_zh_official.md)

## 目录结构

### 1. 训练配置

实验主配置集中在：

- [examples/personalization](examples/personalization)

主要文件包括：

- 一阶段全量 SFT：
  [train_tulu3_personas_sft_full.yaml](examples/personalization/train_tulu3_personas_sft_full.yaml)
- 一阶段 merge：
  [merge_tulu3_personas_sft_full.yaml](examples/personalization/merge_tulu3_personas_sft_full.yaml)
- 二阶段 persona SFT：
  [train_tulu3_personas_sft_personalized.yaml](examples/personalization/train_tulu3_personas_sft_personalized.yaml)
- 二阶段 merge：
  [merge_tulu3_personas_sft_personalized.yaml](examples/personalization/merge_tulu3_personas_sft_personalized.yaml)
- DPO：
  [train_tulu3_personas_dpo.yaml](examples/personalization/train_tulu3_personas_dpo.yaml)
- ORPO：
  [train_tulu3_personas_orpo.yaml](examples/personalization/train_tulu3_personas_orpo.yaml)

对应说明见：

- [examples/personalization/README.md](examples/personalization/README.md)

### 2. 数据目录

当前实验数据集中在：

- [data/tulu3_personas](data/tulu3_personas)

其中分为三类：

- [sft](data/tulu3_personas/sft)
  保存全量 SFT 数据与 persona-focused SFT 子集
- [pref](data/tulu3_personas/pref)
  保存原始偏好数据、pairwise 偏好数据，以及用于 DPO/ORPO 的训练拆分
- [compare](data/tulu3_personas/compare)
  保存 held-out compare/test 集、四模型生成结果、两两 judge 输入、judge 结果与最终汇总

对应说明见：

- [data/tulu3_personas/README.md](data/tulu3_personas/README.md)

### 3. 脚本目录

数据处理与评测脚本集中在：

- [scripts/personalization_data](scripts/personalization_data)

这里包含：

- SFT 数据下载与筛选
- preference 数据下载与 pairwise 转换
- compare/test 构建与切分
- 四模型统一生成
- pairwise judge 输入导出
- LLM judge 执行
- judge 结果自动汇总

对应说明见：

- [scripts/personalization_data/README.md](scripts/personalization_data/README.md)

## 当前实验主流程

当前主流程如下：

1. 先进行全量 SFT。
2. merge 一阶段 LoRA。
3. 进行二阶段 persona-focused SFT。
4. merge 二阶段 LoRA。
5. 在 preference train 上运行 DPO 与 ORPO。
6. 对四个模型版本统一生成 compare/test 响应：
   - `stage1`
   - `stage2`
   - `dpo`
   - `orpo`
7. 自动导出四组两两对比：
   - `stage1 vs stage2`
   - `stage2 vs dpo`
   - `stage2 vs orpo`
   - `dpo vs orpo`
8. 使用 LLM-as-a-judge 完成自动评测。
9. 输出 Markdown 和 JSON 汇总结果。

## 当前实验结果

实验结果分析文档位于：

- [EXPERIMENT_RESULTS.md](data/tulu3_personas/compare/EXPERIMENT_RESULTS.md)

最新 judge 汇总位于：

- [judge_summary.md](data/tulu3_personas/compare/judge_results/judge_summary.md)

当前阶段的主要结论是：

- 二阶段 persona SFT 明显优于一阶段全量 SFT。
- DPO 在当前设定下进一步优于二阶段 SFT，是目前效果最好的方法。
- ORPO 在当前配置下未能优于二阶段 SFT，整体效果也明显弱于 DPO。

## 说明

- 当前仓库已经作为个人的 Tulu-3 Personas 个性化微调实验工作区使用，而不是保留上游 LLaMAFactory README 作为首页说明。
- 上游官方 README 已保留为备份文件，便于后续参考。
