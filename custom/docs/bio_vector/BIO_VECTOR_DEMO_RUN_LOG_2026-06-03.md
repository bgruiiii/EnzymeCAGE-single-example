# Bio-Vector Demo 第一轮 CPU 运行记录

**日期**：2026-06-03
**运行人**：Qoder (AI assistant)
**工作目录**：`/home/a/EnzymeCAGE/demo/bio_vector-main/bio_vector-main/demo`
**Python 环境**：`/home/a/EnzymeCAGE/.envs/rhea-clean/bin/python`（Python 3.10）
**数据目录**：`/home/a/EnzymeCAGE/custom/github_upload/reaction_enzyme_microbe_300_examples_2026-06-01`

---

## 1. 环境准备

### 1.1 已有依赖（rhea-clean 环境）

| 模块       | 版本          |
|------------|---------------|
| numpy      | 1.26.4        |
| torch      | 2.2.1+cu121   |
| sklearn    | 1.7.2         |
| matplotlib | 3.10.8        |
| rdkit      | 2022.09.5     |

### 1.2 新安装依赖

```bash
/home/a/EnzymeCAGE/.envs/rhea-clean/bin/python -m pip install faiss-cpu
# 结果：Successfully installed faiss-cpu-1.14.2
```

### 1.3 GPU 兼容性问题

- **GPU**：NVIDIA GeForce RTX 5060 Laptop GPU（CUDA capability sm_120，Blackwell 架构）
- **PyTorch 2.2.1+cu121** 仅支持 sm_50 ~ sm_90
- 首次 GPU 训练在 Stage 0 epoch 1 时崩溃，报错：
  ```
  RuntimeError: CUDA error: no kernel image is available for execution on the device
  ```
- **决策**：不升级 PyTorch、不修改代码，改用 CPU 运行（300-example demo 数据量小，CPU 可承受）
- 通过 `CUDA_VISIBLE_DEVICES=""` 环境变量强制 CPU

---

## 2. 运行命令

### 2.1 GVP 训练

```bash
cd /home/a/EnzymeCAGE/demo/bio_vector-main/bio_vector-main/demo

CUDA_VISIBLE_DEVICES="" /home/a/EnzymeCAGE/.envs/rhea-clean/bin/python train.py \
  --mode enzyme_cage_300 \
  --data_dir /home/a/EnzymeCAGE/custom/github_upload/reaction_enzyme_microbe_300_examples_2026-06-01 \
  --enzyme_feature gvp \
  --output_dir ./output_v3_300_cpu_round1
```

### 2.2 GVP 诊断

```bash
/home/a/EnzymeCAGE/.envs/rhea-clean/bin/python diagnose_effective_rank.py \
  --output_dir ./output_v3_300_cpu_round1 \
  --microbe_input_dim 16 \
  --enzyme_input_dim 38
```

### 2.3 ESM-C 训练

```bash
CUDA_VISIBLE_DEVICES="" /home/a/EnzymeCAGE/.envs/rhea-clean/bin/python train.py \
  --mode enzyme_cage_300 \
  --data_dir /home/a/EnzymeCAGE/custom/github_upload/reaction_enzyme_microbe_300_examples_2026-06-01 \
  --enzyme_feature esmc \
  --output_dir ./output_v3_esmc_cpu_round1
```

### 2.4 ESM-C 诊断

```bash
/home/a/EnzymeCAGE/.envs/rhea-clean/bin/python diagnose_effective_rank.py \
  --output_dir ./output_v3_esmc_cpu_round1 \
  --microbe_input_dim 16 \
  --enzyme_input_dim 1152
```

---

## 3. 训练结果

### 3.1 数据加载信息（两组通用）

```
Loaded 300 examples from EnzymeCAGE 300 dataset
Reaction (DRFP) dim: 2048
Substrate Morgan FP: 300/300 parsed
Microbe metabolic features: 300/300 loaded (dim=28)
Unique assemblies: 159
Train/Test split: 255/45
Enzyme→Microbe index: 260 enzymes mapped
```

| 配置         | GVP                  | ESM-C                |
|--------------|----------------------|----------------------|
| Enzyme dim   | 50 (GVP attention)   | 1152 (ESM-C pocket)  |
| Model params | 4,267,528            | 4,831,752            |

### 3.2 训练过程（4-Stage，100 epochs）

| Stage | 名称              | Epochs  | GVP final loss | ESM-C final loss |
|-------|-------------------|---------|----------------|------------------|
| 0     | Pretrain          | 1-20    | 2.8114         | 2.8114           |
| 1     | Pairwise          | 21-60   | 72.1045        | 72.0788          |
| 2     | Triplet+Anchor    | 61-90   | 94.2753        | 94.2875          |
| 3     | Self-bootstrap    | 91-100  | 94.2472        | 94.3212          |

温度退火：τ 从 0.5 → 0.07（cosine annealing）

### 3.3 检索指标对比

| 指标         | GVP     | ESM-C   | 差异      |
|--------------|---------|---------|-----------|
| **R→E MRR**  | 0.7493  | 0.7419  | GVP +0.7% |
| R→E top-1    | 0.6200  | 0.6200  | 持平      |
| R→E top-5    | 0.8967  | 0.8767  | GVP +2.0% |
| **E→M MRR**  | 0.6004  | 0.5756  | GVP +2.5% |
| E→M top-1    | 0.4433  | 0.4167  | GVP +2.7% |
| E→M top-5    | 0.8100  | 0.7967  | GVP +1.3% |
| **S→M MRR**  | 0.6293  | 0.6464  | ESMC +1.7%|
| S→M top-1    | 0.4700  | 0.4867  | ESMC +1.7%|
| S→M top-5    | 0.8533  | 0.8500  | 持平      |

**小结**：GVP 在 R→E 和 E→M 任务上略优；ESM-C 在 S→M 任务上略优。整体差异不大。

### 3.4 metrics_v3.json 原始内容

**GVP (`output_v3_300_cpu_round1/metrics_v3.json`)**：

```json
{
  "R→E_MRR": 0.7492920323339002,
  "R→E_top-1": 0.62,
  "R→E_top-5": 0.8966666666666666,
  "R→E_top-10": 0.9,
  "R→E_top-20": 0.9033333333333333,
  "E→M_MRR": 0.6004309518841925,
  "E→M_top-1": 0.44333333333333336,
  "E→M_top-5": 0.81,
  "E→M_top-10": 0.8766666666666667,
  "E→M_top-20": 0.89,
  "S→M_MRR": 0.6292900051842565,
  "S→M_top-1": 0.47,
  "S→M_top-5": 0.8533333333333334,
  "S→M_top-10": 0.9033333333333333,
  "S→M_top-20": 0.9533333333333334
}
```

**ESM-C (`output_v3_esmc_cpu_round1/metrics_v3.json`)**：

```json
{
  "R→E_MRR": 0.7419481461112308,
  "R→E_top-1": 0.62,
  "R→E_top-5": 0.8766666666666667,
  "R→E_top-10": 0.8766666666666667,
  "R→E_top-20": 0.8966666666666666,
  "E→M_MRR": 0.5755795667397133,
  "E→M_top-1": 0.4166666666666667,
  "E→M_top-5": 0.7966666666666666,
  "E→M_top-10": 0.85,
  "E→M_top-20": 0.9033333333333333,
  "S→M_MRR": 0.646359825666708,
  "S→M_top-1": 0.4866666666666667,
  "S→M_top-5": 0.85,
  "S→M_top-10": 0.93,
  "S→M_top-20": 0.9633333333333334
}
```

---

## 4. 诊断结果

### 4.1 Effective Rank 对比

| 模态                  | GVP erank | GVP PR | ESM-C erank | ESM-C PR |
|-----------------------|-----------|--------|-------------|----------|
| Reaction (DRFP→256)   | 81.08     | 22.00  | 81.40       | 23.02    |
| Enzyme (→256)         | 85.00     | 20.94  | 83.45       | 22.36    |
| Substrate (Morgan→256)| 123.46    | 25.55  | 121.86      | 26.29    |
| Microbe (16→256)      | 66.27     | 18.90  | 62.89       | 20.23    |

### 4.2 关键诊断发现

#### VICReg 噪声膨胀

- **Microbe (16→256)**：两组均报 `❌ CRITICAL: Microbe erank > 2× input dim (16)`
  - GVP：erank=66.3，膨胀了 50 个噪声维度
  - ESM-C：erank=62.9，膨胀了 47 个噪声维度
  - Spectral gap ratio 均 < 2（GVP: 1.09，ESM-C: 1.06），确认 VICReg noise inflation
- **Enzyme (GVP, 38→256)**：erank=85.0 > 2×38，spectral gap ratio=1.04，同样存在噪声膨胀

#### DRFP 碰撞风险

- 两组均报：第一投影层 2048→512 的零空间维度 = **1536**
- 1536/2048 个 DRFP 方向对模型不可见
- 经验碰撞检查：cos_sim > 0.95 的 pair 数 ~110（约 0.24%）
- Johnson-Lindenstrauss 最坏情况 distortion ≈ 42.2%

#### Microbe 组合线性路径

- GVP：composed map erank=23.14/28，theoretical max=28，mild rank deficiency
- ESM-C：composed map erank=22.38/28，theoretical max=28，mild rank deficiency

### 4.3 effective_rank_summary.json 原始内容

**GVP (`output_v3_300_cpu_round1/effective_rank_summary.json`)**：

```json
{
  "Reaction (DRFP→256)": {
    "input_dim": 2048, "effective_rank": 81.08,
    "participation_ratio": 22.00, "dims_for_90pct_var": 21
  },
  "Enzyme (GVP→256)": {
    "input_dim": 38, "effective_rank": 85.00,
    "participation_ratio": 20.94, "dims_for_90pct_var": 20
  },
  "Substrate (Morgan→256)": {
    "input_dim": 2048, "effective_rank": 123.46,
    "participation_ratio": 25.55, "dims_for_90pct_var": 41
  },
  "Microbe (16→256)": {
    "input_dim": 16, "effective_rank": 66.27,
    "participation_ratio": 18.90, "dims_for_90pct_var": 18
  }
}
```

**ESM-C (`output_v3_esmc_cpu_round1/effective_rank_summary.json`)**：

```json
{
  "Reaction (DRFP→256)": {
    "input_dim": 2048, "effective_rank": 81.40,
    "participation_ratio": 23.02, "dims_for_90pct_var": 22
  },
  "Enzyme (GVP→256)": {
    "input_dim": 1152, "effective_rank": 83.45,
    "participation_ratio": 22.36, "dims_for_90pct_var": 21
  },
  "Substrate (Morgan→256)": {
    "input_dim": 2048, "effective_rank": 121.86,
    "participation_ratio": 26.29, "dims_for_90pct_var": 40
  },
  "Microbe (16→256)": {
    "input_dim": 16, "effective_rank": 62.89,
    "participation_ratio": 20.23, "dims_for_90pct_var": 19
  }
}
```

---

## 5. 输出文件清单

### GVP (`output_v3_300_cpu_round1/`)

| 文件                          | 大小   | 说明                        |
|-------------------------------|--------|-----------------------------|
| `model_v3.pt`                 | 17M    | 模型权重 + config + 倒排索引 |
| `embeddings_v3.npz`           | 1.2M   | 4 模态嵌入                  |
| `unified_space_v3_results.png`| 266K   | 统一空间可视化              |
| `effective_rank_diagnosis.png`| 279K   | 有效秩诊断图                |
| `metrics_v3.json`             | -      | 检索指标                    |
| `effective_rank_summary.json` | -      | 有效秩摘要                  |
| `*_index.faiss`               | -      | FAISS HNSW 索引（4 模态）   |
| `enzyme2microbe_index`        | -      | 酶→微生物事实检索层         |
| `training_history.json`       | -      | 每 epoch loss/temp/stage    |

### ESM-C (`output_v3_esmc_cpu_round1/`)

| 文件                          | 大小   | 说明                        |
|-------------------------------|--------|-----------------------------|
| `model_v3.pt`                 | 19M    | 模型权重 + config + 倒排索引 |
| `embeddings_v3.npz`           | 1.2M   | 4 模态嵌入                  |
| `unified_space_v3_results.png`| 263K   | 统一空间可视化              |
| `effective_rank_diagnosis.png`| 278K   | 有效秩诊断图                |
| `metrics_v3.json`             | -      | 检索指标                    |
| `effective_rank_summary.json` | -      | 有效秩摘要                  |
| `*_index.faiss`               | -      | FAISS HNSW 索引（4 模态）   |
| `enzyme2microbe_index`        | -      | 酶→微生物事实检索层         |
| `training_history.json`       | -      | 每 epoch loss/temp/stage    |

---

## 6. 注意事项

1. **GPU 不可用**：RTX 5060 Laptop (sm_120) 需 PyTorch 2.5+ / CUDA 12.8+，当前环境 PyTorch 2.2.1+cu121 不支持
2. **未修改任何代码**：train.py 和 diagnose_effective_rank.py 均未改动
3. **未使用完整数据**：本次仅用 300-example demo 数据集
4. **输出目录隔离**：使用 `_cpu_round1` 后缀避免与之前 GPU 失败尝试的 `output_v3_300` 目录混淆
5. **训练 UserWarning**：`torch.tensor(sourceTensor)` 建议改用 `sourceTensor.clone().detach()`（train.py:189），不影响结果
