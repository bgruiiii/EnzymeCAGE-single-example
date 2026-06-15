# R2 Postmortem Task 1.2 — Pre-Run Audit

审计时间：2026-06-12  
审计目标：确认 `postmortem_eval_stage_checkpoints.py` 脚本可以安全创建

---

## 1. 关键函数签名与源码

### 1.1 `evaluate_multimodal` (train.py:752-827)

```python
@torch.no_grad()
def evaluate_multimodal(model, loader, device, enzyme2microbe=None, ks=(1, 5, 10, 20),
                        chunk_size=4096):
```

**关键实现片段：**
```python
def _chunked_retrieval(queries, candidates, pair_label, ks, chunk_size):
    """Compute top-k and MRR without N×N dense matrix."""
    n_q = len(queries)
    all_topk = np.empty((n_q, max(ks)), dtype=np.int64)
    all_mrr = np.empty(n_q, dtype=np.float64)

    for start in range(0, n_q, chunk_size):
        end = min(start + chunk_size, n_q)
        # sim_chunk: (chunk, N) — only one chunk at a time
        sim_chunk = queries[start:end] @ candidates.T
        # top-k per row
        topk_chunk = np.argpartition(-sim_chunk, max_k, axis=1)[:, :max_k]
        # ... sort top-k by score descending
        # MRR: rank of positive = query's own index
        pos_indices = labels[start:end]  # (chunk,)
        for local_i in range(end - start):
            pos_idx = pos_indices[local_i]
            scores = sim_chunk[local_i]  # (N,)
            pos_score = scores[pos_idx]
            # rank = 1 + number of candidates scoring strictly higher
            rank = 1 + int(np.sum(scores > pos_score))
            all_mrr[start + local_i] = 1.0 / rank
        del sim_chunk
```

### 1.2 `evaluate_grouped_re` (train.py:1279-1397)

```python
@torch.no_grad()
def evaluate_grouped_re(all_r, all_e, metadata, ks=(1, 5, 10), chunk_size=4096):
```

**关键实现片段（grouped MRR 计算）：**
```python
for start in range(0, N, chunk_size):
    end = min(start + chunk_size, N)
    sim_chunk = all_r[start:end] @ all_e.T   # (chunk, N)

    for local_i in range(end - start):
        global_i = start + local_i
        if not valid_mask[global_i]:
            continue

        s = sim_chunk[local_i]               # (N,) scores for this query
        my_key = labels_list[global_i]
        gi = groups_dict.get(my_key, [])     # group member indices

        # ── Grouped MRR ──
        if len(gi) > 0:
            best_pos_score = float(np.max(s[gi]))
            rank = int(np.sum(s > best_pos_score)) + 1
            query_ranks[global_i] = rank
```

### 1.3 `load_enzyme_cage_300` (train.py:983-1222)

```python
def load_enzyme_cage_300(data_dir: str, test_size: float = 0.15,
                         enzyme_feature: str = "gvp"):
```

**返回格式：**
```python
# 当 test_size > 0 时返回 9-tuple
return (drfp, enzyme_feats, substrate_feats, microbe_feats,
        ec_labels, metadata, concept_targets, train_idx, test_idx)

# 当 test_size == 0 时返回 7-tuple
return (drfp, enzyme_feats, substrate_feats, microbe_feats,
        ec_labels, metadata, concept_targets)
```

### 1.4 main guard (train.py:1613-1614)

```python
if __name__ == "__main__":
    main()
```

---

## 2. 安全检查结论

| 检查项 | 结论 | 说明 |
|--------|------|------|
| `evaluate_multimodal` 是否支持 `chunk_size` 参数 | ✅ YES | 默认 `chunk_size=4096`，可自定义 |
| 是否避免构建完整 N×N similarity matrix | ✅ YES | 使用 `_chunked_retrieval`，每次只计算 `(chunk_size, N)` 大小的 sim_chunk |
| 是否存在 `np.argsort(..., axis=1)` 对全量 145607×145607 排序 | ✅ NO（安全） | 只使用 `np.argpartition(-sim_chunk, max_k, axis=1)[:, :max_k]` 提取 top-k，不做全量排序 |
| grouped R→E 是否用同组 best score 算 rank | ✅ YES | `best_pos_score = float(np.max(s[gi]))`，使用组内最高分 |
| EC-4 parsing 是否有数字格式校验 | ✅ YES | `_parse_ec4` (L1264-1276) 对每一段都做 `int()` 校验 |
| import train.py 是否不会自动执行 main() | ✅ YES | 有 `if __name__ == "__main__":` guard |

---

## 3. Checkpoint 文件检查

```
待检查文件：
/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/outputs/r2_esmc_hardneg1_stage1_25_2026-06-11/model_v3_stage0.pt
/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/outputs/r2_esmc_hardneg1_stage1_25_2026-06-11/model_v3_stage1.pt
/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/outputs/r2_esmc_hardneg1_stage1_25_2026-06-11/model_v3_stage2.pt
/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/outputs/r2_esmc_hardneg1_stage1_25_2026-06-11/model_v3_stage3.pt
```

**状态：待 Slurm job 执行时确认（之前 Task 1.1 审计已确认存在）**

---

## 4. R2 Final Metrics（用于 Stage 3 Consistency Check）

**来源：** `/public/home/acfbwjsi7s/bio_vector_full_run_2026-06-04/outputs/r2_esmc_hardneg1_stage1_25_2026-06-11/metrics_v3.json`

| Metric | R2 Final Value |
|--------|----------------|
| row R→E MRR | 0.060207782731201254 |
| UniProt-grouped R→E MRR | 0.060707529334763886 |
| EC-4-grouped R→E MRR | 0.9132133257912153 |
| E→M MRR | 0.6195463344069435 |

**Consistency Check 阈值：** 允许 ±5% 相对误差（考虑 BN running stats、batch ordering 等微小差异）

---

## 5. 资源需求确认

| 参数 | 值 | 说明 |
|------|-----|------|
| partition | kshdnormal04 | CPU 节点 |
| cpus-per-task | 8 | numpy 多线程 |
| mem | 64000 MB | 峰值 ~10 GB，留 6x 裕量 |
| time | 6:00:00 | 预计 ~3h，留 2x 裕量 |
| gres | 无 | 纯 CPU，不需要 DCU |

---

## 6. 审计结论

### ✅ PASS

可以创建 `postmortem_eval_stage_checkpoints.py`。

**理由：**
1. `evaluate_multimodal` 和 `evaluate_grouped_re` 均已使用 chunked retrieval，内存安全
2. 不会对 145607×145607 做全量排序，只使用 `argpartition` 提取 top-k
3. Grouped MRR 正确使用组内 best score
4. EC-4 parsing 有完整数字校验
5. `if __name__ == "__main__":` guard 存在，import train.py 不会自动执行训练
6. R2 final metrics 已记录，可用于 stage3 consistency check

---

**Audit status: PASS**  
**No train.py modification.**  
**No sbatch submitted.**
