# Tulu-3 Personas 个性化微调实验结果分析

## 实验目标

本实验围绕 `Qwen2.5-3B-Instruct` 构建了一条分阶段的个性化响应生成与偏好对齐流程，核心问题是：

**在 persona-conditioned 任务上，二阶段 SFT 与后续偏好优化是否能够稳定提升模型表现。**

当前实验包含四个模型版本：

1. `stage1`：在 Tulu-3 Personas 全量 SFT 数据上进行一阶段监督微调。
2. `stage2`：在 persona-focused SFT 子集上继续进行二阶段监督微调。
3. `dpo`：在 pairwise preference 数据上进行 DPO 偏好优化。
4. `orpo`：在相同 preference 数据上进行 ORPO 偏好优化。

## 数据与评测设置

### 训练数据

- 全量 SFT 数据：
  [tulu3_personas_sft.jsonl](../sft/tulu3_personas_sft.jsonl)
- persona-focused SFT 数据：
  [tulu3_personas_sft_personalized.jsonl](../sft/tulu3_personas_sft_personalized.jsonl)
- preference 训练数据：
  [tulu3_personas_pref_pairwise_train.jsonl](../pref/tulu3_personas_pref_pairwise_train.jsonl)

### 独立 compare/test 集

本次 compare/test 集不是从 SFT 数据中抽取，而是从 pairwise preference 数据中单独切出。

- compare/test 集：
  [persona_compare_samples_clean.jsonl](persona_compare_samples_clean.jsonl)
- 规模：`200` 条 prompt

### 评测方法

在 compare/test 集上，对四个模型版本统一生成输出，再使用 LLM-as-a-judge 进行两两比较。

评测维度包括：

- `persona_alignment`
- `instruction_following`
- `constraint_satisfaction`
- `response_quality`
- `overall_preference`

汇总结果见：

- [judge_summary.md](judge_results/judge_summary.md)

## 核心结果

### 1. 二阶段 SFT 相比一阶段 SFT 明显提升

`stage2` 在几乎所有关键维度上都优于 `stage1`：

- `overall_preference`：`105 vs 79`
- `response_quality`：`105 vs 75`
- `constraint_satisfaction`：`88 vs 68`
- `instruction_following`：`84 vs 78`

在人设对齐维度上，提升幅度较温和，但方向明确：

- `stage1 vs stage2`：`A 3 / B 20 / tie 177`

这说明二阶段 persona-focused SFT 的价值主要体现在：

- 提高回答质量
- 提高格式与约束满足能力
- 提升整体稳定性


### 2. DPO 在当前设定下效果最好

`dpo` 相比 `stage2` 继续取得了稳定提升：

- `overall_preference`：`109 vs 81`
- `response_quality`：`107 vs 82`
- `instruction_following`：`99 vs 72`
- `constraint_satisfaction`：`86 vs 75`

在人设对齐上也有一定增益：

- `stage2 vs dpo`：`A 8 / B 18 / tie 174`

这表明在当前实验设置下，DPO 是四个模型版本里表现最好的方法。

### 3. ORPO 未能优于二阶段 SFT

`orpo` 的结果没有延续 DPO 的提升趋势，反而相较 `stage2` 出现了回落：

- `overall_preference`：`77`（ORPO） vs `96`（stage2）
- `response_quality`：`77`（ORPO） vs `93`（stage2）

在人设对齐上也几乎没有显著收益：

- `stage2 vs orpo`：`A 7 / B 5 / tie 188`

在当前训练配置下，ORPO 没有带来稳定的个性化增强，整体效果还略低于二阶段 SFT。

### 4. DPO 明显优于 ORPO

在 `dpo vs orpo` 的直接比较中，DPO 明显占优：

- `overall_preference`：`107 vs 80`
- `response_quality`：`105 vs 80`
- `instruction_following`：`97 vs 75`
- `constraint_satisfaction`：`91 vs 71`

在人设对齐上，DPO 也略优于 ORPO：

- `dpo vs orpo`：`A 13 / B 7 / tie 180`

所以当前可以比较明确地说：

**在这套 persona-conditioned 偏好数据上，DPO 比 ORPO 更适合。**

## 结果解读

### 为什么二阶段 SFT 有效

一阶段全量 SFT 已经让模型学会了基本的 persona-conditioned instruction following。  
二阶段再用更纯的 persona 子集继续训练，相当于对这部分模式进行了强化。

最终体现为：

- 回答更稳
- 内容更完整
- 约束满足更好
- 综合偏好更高

因此，二阶段 SFT 的主要提升不是“更会表演人设”，而是“更会在人设条件下稳定完成任务”。

### 为什么 DPO 表现最好

pairwise preference 数据直接提供了同一 prompt 下的优劣回答对，这种监督信号非常适合当前任务，因为当前任务不只是风格生成，还同时包含：

- persona 对齐
- 格式要求
- 关键词要求
- 段落和列表结构要求

DPO 在这种设置下更容易学到“什么样的回答整体更好”，所以在：

- instruction following
- constraint satisfaction
- response quality

这几个维度上表现最强。

### 为什么 ORPO 表现较差

结合 judge 结果和样本抽查，ORPO 主要的问题并不是“完全不会 persona”，而是：

- 没有带来明显的人设增益
- 更容易丢失格式和硬约束
- 整体回答稳定性不如 DPO

可能的原因是：

- ORPO 对当前任务和超参数更敏感
- 而当前 ORPO 基本沿用了和 DPO 相近的保守配置
- 这可能不足以让 ORPO 在这类强约束的 preference 数据上发挥优势

因此，这一轮结果更适合解释为：

**在默认保守配置下，ORPO 不如 DPO。**

## 当前结论

当前四个模型版本的效果排序可以概括为：

1. `dpo`
2. `stage2`
3. `orpo`
4. `stage1`

本轮实验最重要的结论有四条：

- 二阶段 persona SFT 明显优于一阶段全量 SFT。
- DPO 在当前设置下进一步优于二阶段 SFT。
- ORPO 在当前配置下没有优于二阶段 SFT。
- 当前任务的最大提升主要来自回答质量和约束满足能力，而不是极端强化的人设表达。

## 局限性

- 当前评测采用 LLM-as-a-judge，使用的模型的是gpt4.1-mini，而不是人工标注。
- compare/test 集规模为 `200` 条，已经比早期版本更稳，但仍然有限。
- ORPO 目前尚未进行单独超参数调优，因此当前结论更准确地说是“当前配置下 ORPO 不如 DPO”。

## 相关文件

- 训练配置说明：
  [README.md](../../../examples/personalization/README.md)
- 数据目录说明：
  [README.md](../README.md)
- 脚本目录说明：
  [README.md](../../../scripts/personalization_data/README.md)
