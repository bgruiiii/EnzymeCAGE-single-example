# EnzymeCAGE Single Reaction-Enzyme-Microbe Example

这个目录是一个 GitHub 友好的单条完整示例，展示一条 RHEA 反应如何连接到：

- 原始 reaction-enzyme pair
- 反应侧特征
- 酶 pocket/GVP 特征
- 微生物来源、代谢模型、RHEA core-preference 结果
- ESM-C 序列和 pocket-node 特征

## Example ID

| 字段 | 值 |
|---|---|
| `RHEA_ID` | `10164` |
| `UniprotID` | `Q7M0V7` |
| `assembly_accession` | `GCF_018885085.1` |
| `source_signature` | `taxon:1496|organism:clostridioides difficile (peptoclostridium difficile)` |
| `organism_name` | `Clostridioides difficile (Peptoclostridium difficile)` |
| `CANO_RXN_SMILES` | `O=C(O)[C@@H](CO)OP(=O)(O)O>>C=C(OP(=O)(O)O)C(=O)O.O` |

## Directory Layout

| 路径 | 内容 |
|---|---|
| `tables/reaction_enzyme_pair.csv` | 主表中的单条 reaction-enzyme positive pair |
| `tables/enzyme_to_microbe_source.csv` | `UniprotID -> source_signature` |
| `tables/microbe_source_to_model.csv` | `source_signature -> assembly/model` |
| `tables/microbe_reaction_stoich_query.csv` | 该 assembly + reaction 的 stoich query |
| `tables/microbe_reaction_main_metabolite_coverage.csv` | main reactant/product coverage 解释层 |
| `tables/microbe_reaction_core_preference.csv` | 该 assembly + reaction 的 core-preference summary |
| `features/reaction/reaction_features.npz` | DRFP + reacting-center indices |
| `features/reaction/molecule_conformation/` | 本反应涉及分子的 SDF 构象 |
| `features/enzyme/gvp_pocket_feature.npz` | 本 UID 的 GVP pocket graph feature |
| `features/enzyme/Q7M0V7_pocket.pdb` | 本 UID 的 pocket PDB |
| `features/enzyme/Q7M0V7_esm_c_*.npy/.npz` | 本 UID 的 ESM-C sequence/pocket-node 特征 |
| `features/microbe/*.json` | 微生物侧单条 JSON 结果 |
| `cloud_needed/` | ESM-C 云端来源、导出请求和导出脚本 |

## What Is Included Locally

- reaction DRFP: included
- reaction center indices: included
- molecule SDF conformations: included
- enzyme pocket PDB: included
- enzyme GVP pocket feature: included
- enzyme ESM-C sequence node feature: included, `{'node_feature': [59, 1152]}`
- enzyme ESM-C sequence mean feature: included, `[1152]`
- enzyme ESM-C pocket-node feature: included, `[15, 1152]`
- microbe source/model/core-preference feature rows: included

## ESM-C Status

ESM-C sequence-level and pocket-node features are now included under `features/enzyme/`.
The `cloud_needed/` directory is kept as provenance, showing the cloud paths and export script used to create them.

## Notes

This is a tiny demonstration package, not a training dataset. It is intended for documentation,
GitHub upload, and explaining how one full EnzymeCAGE-style example is assembled.

