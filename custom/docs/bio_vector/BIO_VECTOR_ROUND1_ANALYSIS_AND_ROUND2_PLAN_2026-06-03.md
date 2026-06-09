# Bio Vector Demo 第一轮结果分析与第二轮参数建议

日期：2026-06-03

## 1. 第一轮运行状态

本轮按照 `bio_vector` demo README 中的 300 样本真实数据流程，分别运行了：

1. GVP 酶特征版本
2. ESM-C 酶特征版本
3. 对两个输出目录分别进行 effective-rank 诊断

两个版本均已完整跑通，训练和诊断均无报错。输出文件齐全，包括：

- `model_v3.pt`
- `embeddings_v3.npz`
- 四个 FAISS 检索索引
- `metrics_v3.json`
- `training_history.json`
- `unified_space_v3_results.png`
- `effective_rank_diagnosis.png`
- `effective_rank_summary.json`

输出目录：

- GVP:
  `/home/a/EnzymeCAGE/demo/bio_vector-main/bio_vector-main/demo/output_v3_300_cpu_round1`
- ESM-C:
  `/home/a/EnzymeCAGE/demo/bio_vector-main/bio_vector-main/demo/output_v3_esmc_cpu_round1`

## 2. 与 README 预期指标对比

README 中 300 样本 GVP demo 的预期指标大致为：

| 指标 | README 预期 |
|---|---:|
| `R->E MRR` | ~0.74 |
| `R->E top-5` | ~0.90 |
| `E->M MRR` | ~0.25 |
| `S->M MRR` | ~0.20 |

第一轮实际结果：

| 指标 | README 预期 | GVP 第一轮 | ESM-C 第一轮 |
|---|---:|---:|---:|
| `R->E MRR` | ~0.74 | **0.7493** | 0.7419 |
| `R->E top-5` | ~0.90 | **0.8967** | 0.8767 |
| `E->M MRR` | ~0.25 | **0.6004** | 0.5756 |
| `S->M MRR` | ~0.20 | 0.6293 | **0.6464** |

结论：

- GVP 版本基本完全达到 README 预期。
- ESM-C 版本整体正常，`R->E MRR` 达到预期，`R->E top-5` 略低于 GVP，但仍接近预期。
- `E->M` 和 `S->M` 两类指标明显高于 README 参考值，说明在 300 样本 demo 上，微生物摘要特征和样本配对关系被模型较好地对齐。
- GVP 在 `R->E` 和 `E->M` 上略优于 ESM-C。
- ESM-C 在 `S->M` 上略优于 GVP。

需要注意：当前 `train.py` 最终评估是在全部 300 条样本上进行，而不是只在 held-out test set 上评估。因此这些结果主要说明 demo 管线跑通，并且四模态空间能在 300 样本上完成对齐；它们不能直接作为模型泛化能力结论。

## 3. Effective-Rank 诊断结果

### 3.1 GVP 模式

| 模态 | input dim | effective rank | participation ratio | 90% variance dims |
|---|---:|---:|---:|---:|
| Reaction / DRFP | 2048 | 81.08 | 22.00 | 21 |
| Enzyme / GVP | 38 | 85.00 | 20.94 | 20 |
| Substrate / Morgan | 2048 | 123.46 | 25.55 | 41 |
| Microbe | 16 | 66.27 | 18.90 | 18 |

诊断脚本提示：

- GVP spectral gap 较弱：`sigma_38 / sigma_39 = 1.04`
- Microbe spectral gap 较弱：`sigma_16 / sigma_17 = 1.09`
- Microbe effective rank 远高于 README 诊断输入维度 `16`
- 存在 VICReg noise inflation 风险
- DRFP 压缩存在 collision warning

### 3.2 ESM-C 模式

| 模态 | input dim | effective rank | participation ratio | 90% variance dims |
|---|---:|---:|---:|---:|
| Reaction / DRFP | 2048 | 81.40 | 23.02 | 22 |
| Enzyme / ESM-C | 1152 | 83.45 | 22.36 | 21 |
| Substrate / Morgan | 2048 | 121.86 | 26.29 | 40 |
| Microbe | 16 | 62.89 | 20.23 | 19 |

诊断脚本提示：

- Microbe spectral gap 较弱：`sigma_16 / sigma_17 = 1.06`
- Microbe effective rank 远高于 README 诊断输入维度 `16`
- 存在 VICReg noise inflation 风险
- DRFP 压缩存在 collision warning

## 4. 第一轮主要判断

第一轮结果说明：

1. 训练和诊断管线已经跑通。
2. GVP 与 ESM-C 两种酶特征模式均能完成跨模态对齐。
3. GVP 在反应到酶、酶到微生物检索上略优。
4. ESM-C 在底物到微生物检索上略优。
5. 当前最主要的问题不是 retrieval 指标不足，而是 embedding 空间存在诊断风险，尤其是 microbe 低维输入被 VICReg 过度展开。

需要重点关注的问题：

- Microbe 原始结构化特征维度很低，但输出 embedding 的 effective rank 达到 60 以上。
- 这说明当前 `vicreg_var_weight=25.0` 对低维 microbe 特征压力偏大，可能把低维信息强行展开成较多噪声维度。
- DRFP collision warning 需要记录，但第一轮 `R->E` 指标已经达到 README 预期，因此第二轮不建议优先改 reaction encoder。

## 5. 第二轮参数建议

建议第二轮只做一个最小参数修改：

```python
vicreg_var_weight = 25.0  # round 1
vicreg_var_weight = 10.0  # round 2
```

其他参数暂时保持不变：

```python
batch_size = 64
epochs_stage0 = 20
epochs_stage1 = 40
epochs_stage2 = 30
epochs_stage3 = 10
temp_start = 0.5
temp_end = 0.07
w_re = 1.0
w_em = 0.7
w_sm = 0.4
hard_neg_weight = 2.0
```

## 6. 修改理由

选择降低 `vicreg_var_weight` 的理由：

1. 第一轮 retrieval 指标已经达到 README 预期，不需要大幅改变对比学习目标。
2. 当前最明确的问题是 VICReg 对低维模态过强，导致 microbe embedding 有效秩过度膨胀。
3. README 的参数表中也给出 `vicreg_var_weight` 可在 `10-25` 范围调整。
4. 从 `25.0` 降到 `10.0` 是保守改动，可以降低噪声维展开风险，同时尽量保留跨模态对齐能力。
5. 暂时不同时修改 batch size、温度、loss 权重或网络结构，便于判断第二轮变化是否由 VICReg 权重导致。

## 7. 第二轮判断标准

第二轮完成后建议重点比较：

| 观察项 | 目标 |
|---|---|
| `R->E_top-5` | 尽量仍接近 `0.90` |
| `R->E_MRR` | 尽量仍接近 `0.74` |
| `E->M_MRR` | 不应明显下降 |
| `S->M_MRR` | 不应明显下降 |
| Microbe effective rank | 应从 `60+` 有所降低 |
| Microbe spectral gap | 应比第一轮有所改善 |
| DRFP warning | 不应进一步恶化 |

如果第二轮在保持 retrieval 指标基本稳定的同时降低 microbe effective rank，则说明降低 VICReg 方差权重是合理方向。

## 8. 建议输出目录

为了保留第一轮 baseline，第二轮建议使用新目录：

- GVP:
  `output_v3_300_cpu_round2_vicreg10`
- ESM-C:
  `output_v3_esmc_cpu_round2_vicreg10`

第一轮目录不要覆盖：

- `output_v3_300_cpu_round1`
- `output_v3_esmc_cpu_round1`
