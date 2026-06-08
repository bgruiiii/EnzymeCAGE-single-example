"""
Train unified vector space with contrastive learning — v3 (Full 4-stage pipeline).

Key capabilities:
  - Four modalities: reaction / enzyme / substrate / microbe
  - d=256 unified embedding space
  - VICReg variance + covariance regularization (anti-collapse)
  - Temperature annealing 0.5 → 0.07
  - Three-way weighted InfoNCE (reaction-enzyme, enzyme-microbe, substrate-microbe)
  - Four-stage training: Stage 0 pretrain → Stage 1 pairwise → Stage 2 triplet → Stage 3 self-bootstrap
  - enzyme2microbe inverted index (fact retrieval layer)
  - FBA surrogate stub for zero-shot scoring
  - GVP attention pooling (distance-from-pocket-center weighting)
  - Hard negative mining: same-EC-class negatives upweighted

Usage:
  python train.py --mode enzyme_cage_300 \
      --data_dir ../reaction_enzyme_microbe_300_examples_2026-06-01 \
      --enzyme_feature gvp --stage all
"""

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import faiss
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

class Config:
    # ── dimensions ──
    unified_dim = 256
    hidden_dim = 512

    # ── temperature annealing ──
    temp_start = 0.5
    temp_end = 0.05  # full-data: README Data-Sensitive (was 0.07)

    # ── VICReg coefficients ──
    vicreg_var_weight = 10.0    # μ in VICReg paper (full-data: 10, was 25)
    vicreg_cov_weight = 1.0     # ν in VICReg paper

    # ── three-way InfoNCE weights ──
    w_re = 1.0     # reaction ↔ enzyme (BRENDA-level supervision)
    w_em = 0.7     # enzyme ↔ microbe (UniProt-genome mapping)
    w_sm = 0.5     # substrate ↔ microbe (full-data: 0.5, was 0.4)

    # ── hard negative mining ──
    hard_neg_weight = 2.0       # multiplier for same-EC-class negatives

    # ── concept anchor head (microbe) ──
    anchor_weight = 1.0         # L_anchor = MSE(pred, true) for concept anchors
    n_concept_anchors = 8       # 8 metabolic concept dimensions

    # ── training ──
    lr = 3e-4  # full-data: README Data-Sensitive (was 1e-3)
    epochs_stage0 = 8           # Stage 0: independent pretrain (full-data: 8, was 20)
    epochs_stage1 = 12          # Stage 1: pairwise contrastive (full-data: 12, was 40)
    epochs_stage2 = 8           # Stage 2: triplet consistency (full-data: 8, was 30)
    epochs_stage3 = 10          # Stage 3: closed-loop self-bootstrap (README no suggestion, kept 10)
    batch_size = 4096  # full-data: README Data-Sensitive (was 64)
    seed = 42

    @property
    def total_epochs(self):
        return self.epochs_stage0 + self.epochs_stage1 + self.epochs_stage2 + self.epochs_stage3


# ═══════════════════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════════════════

class Projector(nn.Module):
    """3-layer MLP projector with BN + ReLU, output L2-normalized."""
    def __init__(self, input_dim, hidden_dim=512, output_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class ConceptAnchorHead(nn.Module):
    """
    Auxiliary head for microbe encoder: predicts interpretable metabolic concepts.
    8 outputs: substrate_utility(4) + cofactor_cost(4) normalized to [0,1].
    Supervised by FBA-derived ground truth to prevent backbone drift.
    """
    def __init__(self, hidden_dim=512, n_concepts=8):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_concepts),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.head(x)


class UnifiedSpace(nn.Module):
    """Four-modality unified vector space model."""
    def __init__(self, reaction_dim, enzyme_dim, substrate_dim, microbe_dim, cfg: Config):
        super().__init__()
        self.reaction_projector = Projector(reaction_dim, cfg.hidden_dim, cfg.unified_dim)
        self.enzyme_projector = Projector(enzyme_dim, cfg.hidden_dim, cfg.unified_dim)
        self.substrate_projector = Projector(substrate_dim, cfg.hidden_dim, cfg.unified_dim)
        self.microbe_projector = Projector(microbe_dim, cfg.hidden_dim, cfg.unified_dim)

        # Concept anchor head branches off from microbe projector's penultimate layer
        self.microbe_backbone = nn.Sequential(
            nn.Linear(microbe_dim, cfg.hidden_dim),
            nn.BatchNorm1d(cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.BatchNorm1d(cfg.hidden_dim),
            nn.ReLU(),
        )
        self.microbe_head = nn.Linear(cfg.hidden_dim, cfg.unified_dim)
        self.concept_anchor = ConceptAnchorHead(cfg.hidden_dim, cfg.n_concept_anchors)

        self.cfg = cfg

    def encode_reaction(self, x):
        return F.normalize(self.reaction_projector(x), dim=-1)

    def encode_enzyme(self, x):
        return F.normalize(self.enzyme_projector(x), dim=-1)

    def encode_substrate(self, x):
        return F.normalize(self.substrate_projector(x), dim=-1)

    def encode_microbe(self, x):
        h = self.microbe_backbone(x)
        emb = F.normalize(self.microbe_head(h), dim=-1)
        return emb

    def encode_microbe_with_anchors(self, x):
        """Return both embedding and concept anchor predictions."""
        h = self.microbe_backbone(x)
        emb = F.normalize(self.microbe_head(h), dim=-1)
        anchors = self.concept_anchor(h)
        return emb, anchors

    def forward(self, r, e, s=None, m=None):
        r_emb = self.encode_reaction(r)
        e_emb = self.encode_enzyme(e)
        s_emb = self.encode_substrate(s) if s is not None else None
        m_emb = self.encode_microbe(m) if m is not None else None
        return r_emb, e_emb, s_emb, m_emb


# ═══════════════════════════════════════════════════════════════════════════════
# Loss Functions
# ═══════════════════════════════════════════════════════════════════════════════

def infonce_loss(emb_a, emb_b, temperature, ec_ids=None, hard_weight=2.0):
    """
    Symmetric InfoNCE with optional hard negative mining.
    Same-EC negatives get upweighted in denominator.
    """
    B = len(emb_a)
    logits = torch.matmul(emb_a, emb_b.T) / temperature

    if ec_ids is not None:
        ec_t = torch.tensor(ec_ids, device=logits.device)
        same_ec = (ec_t[:, None] == ec_t[None, :]).float()
        eye = torch.eye(B, device=logits.device)
        weights = 1.0 + (hard_weight - 1.0) * same_ec * (1 - eye)
        logits = logits + torch.log(weights)

    labels = torch.arange(B, device=logits.device)
    loss_a = F.cross_entropy(logits, labels)
    loss_b = F.cross_entropy(logits.T, labels)
    return (loss_a + loss_b) / 2


def vicreg_variance_loss(embeddings):
    """
    VICReg variance term: encourage std of each dimension ≥ 1.
    L_var = (1/d) Σ_j max(0, 1 - std(z_j))
    """
    std = embeddings.std(dim=0)
    return F.relu(1.0 - std).mean()


def vicreg_covariance_loss(embeddings):
    """
    VICReg covariance term: decorrelate embedding dimensions.
    L_cov = (1/d²) Σ_{i≠j} C(z_i, z_j)²
    """
    B, d = embeddings.shape
    z = embeddings - embeddings.mean(dim=0)
    cov = (z.T @ z) / (B - 1)
    # zero diagonal (don't penalize self-correlation)
    off_diag = cov - torch.diag(cov.diag())
    return (off_diag ** 2).sum() / d


def concept_anchor_loss(pred_anchors, true_anchors):
    """MSE loss between predicted and ground-truth concept anchors."""
    mask = ~torch.isnan(true_anchors)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred_anchors.device)
    return F.mse_loss(pred_anchors[mask], true_anchors[mask])


# ═══════════════════════════════════════════════════════════════════════════════
# Temperature Annealing
# ═══════════════════════════════════════════════════════════════════════════════

class TemperatureScheduler:
    """Cosine annealing from temp_start to temp_end over total_epochs."""
    def __init__(self, temp_start, temp_end, total_epochs):
        self.temp_start = temp_start
        self.temp_end = temp_end
        self.total_epochs = total_epochs

    def get_temperature(self, epoch):
        progress = epoch / max(1, self.total_epochs - 1)
        # cosine decay
        return self.temp_end + 0.5 * (self.temp_start - self.temp_end) * (
            1 + math.cos(math.pi * progress))


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════════

class MultiModalDataset(Dataset):
    """Four-modality dataset: reaction, enzyme, substrate, microbe + metadata."""
    def __init__(self, r_feat, e_feat, s_feat, m_feat, ec_labels,
                 concept_targets=None, microbe_ids=None):
        self.r = torch.tensor(r_feat, dtype=torch.float32)
        self.e = torch.tensor(e_feat, dtype=torch.float32)
        self.s = torch.tensor(s_feat, dtype=torch.float32)
        self.m = torch.tensor(m_feat, dtype=torch.float32)

        # concept anchor targets (8-dim per sample, NaN where unavailable)
        if concept_targets is not None:
            self.concepts = torch.tensor(concept_targets, dtype=torch.float32)
        else:
            self.concepts = torch.full((len(r_feat), 8), float('nan'))

        # EC class integers
        self.ec_ids = []
        for ec in ec_labels:
            try:
                self.ec_ids.append(int(str(ec)[0]) if str(ec)[0].isdigit() else 0)
            except (ValueError, IndexError):
                self.ec_ids.append(0)

        # microbe assembly IDs for inverted index
        self.microbe_ids = microbe_ids or [""] * len(r_feat)

    def __len__(self):
        return len(self.r)

    def __getitem__(self, idx):
        return (self.r[idx], self.e[idx], self.s[idx], self.m[idx],
                self.concepts[idx], self.ec_ids[idx], idx)


# ═══════════════════════════════════════════════════════════════════════════════
# Microbe Feature Engineering
# ═══════════════════════════════════════════════════════════════════════════════

# Standard BiGG cofactor metabolites for cofactor cost vector
COFACTOR_METS = [
    "atp_c", "adp_c", "nad_c", "nadh_c", "nadp_c", "nadph_c",
    "fad_c", "fadh2_c", "coa_c", "accoa_c", "gtp_c", "gdp_c",
    "h2o_c", "h_c", "co2_c", "o2_c"
]
COFACTOR_IDX = {m: i for i, m in enumerate(COFACTOR_METS)}


def _safe_float(v, default=0.0):
    """Convert value to float, handling CSV strings and missing values."""
    if v is None or v == "":
        return default
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def _safe_bool(v, default=False):
    """Convert value to bool, handling CSV strings 'True'/'False'."""
    if isinstance(v, str):
        return v.lower() in ("true", "1", "yes")
    return bool(v) if v is not None else default


def extract_microbe_features(record: dict) -> tuple:
    """
    Extract metabolic feature vector from microbe record.
    Supports both nested dict (from CSV tables) and flat JSONL format.

    Returns:
        feat: np.array of shape [microbe_feat_dim]
              = core_preference(6) + cofactor_stoich(16) + coverage(6) = 28
        concept_target: np.array of shape [8] for concept anchor supervision
        assembly_id: str
    """
    cp = record.get("core_preference", {})
    stoich = record.get("stoich_query", {})
    mc = record.get("main_metabolite_coverage", {})

    # ── core preference features (6 dim) ──
    core_feats = np.array([
        _safe_float(cp.get("target_rows")),
        _safe_float(cp.get("in_core")),
        _safe_float(cp.get("reachable")),
        _safe_float(cp.get("not_reachable")),
        _safe_float(cp.get("connectable_targets")),
        _safe_float(cp.get("connectable_target_ratio")),
    ], dtype=np.float32)

    # ── cofactor stoichiometry vector (16 dim) ──
    stoich_json = stoich.get("stoich_json", "{}")
    if isinstance(stoich_json, str):
        try:
            stoich_dict = json.loads(stoich_json)
        except json.JSONDecodeError:
            stoich_dict = {}
    else:
        stoich_dict = stoich_json
    cofactor_vec = np.zeros(len(COFACTOR_METS), dtype=np.float32)
    for met, coeff in stoich_dict.items():
        if met in COFACTOR_IDX:
            cofactor_vec[COFACTOR_IDX[met]] = float(coeff)

    # ── metabolite coverage features (6 dim) ──
    cov_feats = np.array([
        _safe_float(mc.get("n_candidate_main_reactants")),
        _safe_float(mc.get("n_candidate_main_products")),
        _safe_float(mc.get("n_candidate_main_reactants_with_bigg")),
        _safe_float(mc.get("n_candidate_main_products_with_bigg")),
        1.0 if _safe_bool(mc.get("has_candidate_main_reactant_bigg")) else 0.0,
        1.0 if _safe_bool(mc.get("has_candidate_main_product_bigg")) else 0.0,
    ], dtype=np.float32)

    feat = np.concatenate([core_feats, cofactor_vec, cov_feats])

    # ── concept anchor targets (8 dim) ──
    target_rows = max(_safe_float(cp.get("target_rows"), 1.0), 1.0)
    concept_target = np.array([
        _safe_float(cp.get("connectable_target_ratio")),
        _safe_float(cp.get("in_core")) / target_rows,
        _safe_float(cp.get("reachable")) / target_rows,
        1.0 if mc.get("main_reaction_interpretation_class") == "direct_main_reactant_bigg_available" else 0.0,
        # cofactor metabolic cost (4): normalized stoich magnitudes
        min(abs(cofactor_vec[0]) / 3.0, 1.0),   # ATP usage
        min(abs(cofactor_vec[2]) / 2.0, 1.0),   # NAD usage
        min(abs(cofactor_vec[4]) / 2.0, 1.0),   # NADP usage
        min(abs(cofactor_vec[8]) / 2.0, 1.0),   # CoA usage
    ], dtype=np.float32)

    assembly_id = stoich.get("assembly_accession", "")
    return feat, concept_target, assembly_id


# ═══════════════════════════════════════════════════════════════════════════════
# GVP Attention Pooling
# ═══════════════════════════════════════════════════════════════════════════════

def gvp_attention_pool(npz_path: str) -> np.ndarray:
    """
    Pool variable-size GVP pocket graph with distance-from-center attention.
    Total output: 50 dim.
    """
    data = np.load(npz_path, allow_pickle=True)
    node_xyz = data["node_xyz"]
    node_s = data["node_s"]
    node_v = data["node_v"]

    center = node_xyz.mean(axis=0)
    dists = np.linalg.norm(node_xyz - center, axis=1)
    sigma = dists.std()
    if sigma > 1e-8:
        raw_weights = np.exp(-dists / sigma)
    else:
        raw_weights = np.ones(len(dists))
    attn = raw_weights / raw_weights.sum()

    n_s = (node_s * attn[:, None]).sum(axis=0)
    n_v = (node_v * attn[:, None, None]).sum(axis=0).flatten()
    e_s = data["edge_s"].mean(axis=0)
    e_v = data["edge_v"].mean(axis=0).flatten()

    return np.concatenate([n_s, n_v, e_s, e_v]).astype(np.float32)


def gvp_shard_pool(entry) -> np.ndarray:
    """
    Pool GVP features from a shard entry (tuple of 7 tensors).
    Actual shard tuple layout:
      [0] node_xyz  (N, 3)
      [1] residue_id (N,)
      [2] node_s    (N, 6)      ← scalar node features
      [3] node_v    (N, 3, 3)   ← vector node features
      [4] edge_idx  (2, E)
      [5] edge_s    (E, 32)
      [6] edge_v    (E, 1, 3)
    Output: 50-dim vector [node_s(6) + node_v(9) + edge_s(32) + edge_v(3)]
    NaN-safe: any NaN in input tensors is replaced with 0.
    """
    if isinstance(entry, tuple):
        node_xyz = np.nan_to_num(entry[0].numpy(), nan=0.0, posinf=0.0, neginf=0.0)
        node_s   = np.nan_to_num(entry[2].numpy(), nan=0.0, posinf=0.0, neginf=0.0)
        node_v   = np.nan_to_num(entry[3].numpy(), nan=0.0, posinf=0.0, neginf=0.0)
        edge_s   = np.nan_to_num(entry[5].numpy(), nan=0.0, posinf=0.0, neginf=0.0)
        edge_v   = np.nan_to_num(entry[6].numpy(), nan=0.0, posinf=0.0, neginf=0.0)
    else:
        node_xyz = np.nan_to_num(np.asarray(entry["node_xyz"]), nan=0.0)
        node_s   = np.nan_to_num(np.asarray(entry["node_s"]), nan=0.0)
        node_v   = np.nan_to_num(np.asarray(entry["node_v"]), nan=0.0)
        edge_s   = np.nan_to_num(np.asarray(entry["edge_s"]), nan=0.0)
        edge_v   = np.nan_to_num(np.asarray(entry["edge_v"]), nan=0.0)

    N = node_xyz.shape[0]
    if N == 0:
        return np.zeros(50, dtype=np.float32)

    center = node_xyz.mean(axis=0)
    dists = np.linalg.norm(node_xyz - center, axis=1)
    sigma = dists.std()
    if np.isfinite(sigma) and sigma > 1e-8:
        raw_weights = np.exp(-dists / sigma)
    else:
        raw_weights = np.ones(N)
    weight_sum = raw_weights.sum()
    if weight_sum > 1e-12:
        attn = raw_weights / weight_sum
    else:
        attn = np.ones(N) / N

    # node_s: (N, D) → attention pool → pad to 6-dim
    if node_s.ndim == 1:
        n_s_raw = np.array([(node_s * attn).sum()], dtype=np.float32)
    else:
        n_s_raw = (node_s * attn[:, None]).sum(axis=0)
    n_s = np.zeros(6, dtype=np.float32)
    n_s[:len(n_s_raw)] = n_s_raw
    # node_v: (N, 3, 3) → attention pool → flatten → 9-dim
    n_v = (node_v * attn[:, None, None]).sum(axis=0).flatten()
    # edge features → mean (handle empty edge lists)
    if edge_s.ndim == 2 and edge_s.shape[0] > 0:
        e_s = edge_s.mean(axis=0)
    else:
        e_s = np.zeros(32, dtype=np.float32)
    if edge_v.ndim >= 2 and edge_v.shape[0] > 0:
        e_v = edge_v.mean(axis=0).flatten()
    else:
        e_v = np.zeros(3, dtype=np.float32)

    out = np.concatenate([n_s, n_v, e_s, e_v]).astype(np.float32)
    # Final safety: replace any remaining NaN
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# ESM-C Pocket Window Pooling
# ═══════════════════════════════════════════════════════════════════════════════

def esmc_pocket_pool(npz_path: str, pocket_residues: list, window: int = 50) -> np.ndarray:
    """
    Pool ESM-C per-residue features using pocket window attention.

    Strategy (aligned with design doc §3.2):
      1. Load per-residue embeddings (seq_len, 1152)
      2. Expand pocket to ±window residues
      3. Apply distance-from-pocket-center attention weighting
      4. Return fixed 1152-dim vector

    Fallback: if pocket_residues empty, use global mean pooling.
    """
    data = np.load(npz_path, allow_pickle=True)
    node_feat = data["node_feature"]  # (seq_len, 1152)
    seq_len = node_feat.shape[0]

    if not pocket_residues:
        # Fallback: global mean
        return node_feat.mean(axis=0).astype(np.float32)

    # Expand pocket to ±window range
    pocket_set = set()
    for r in pocket_residues:
        for offset in range(-window, window + 1):
            idx = r + offset
            if 0 <= idx < seq_len:
                pocket_set.add(idx)

    pocket_indices = sorted(pocket_set)
    pocket_feats = node_feat[pocket_indices]  # (n_pocket, 1152)

    # Attention: residues closer to pocket center get higher weight
    pocket_center = np.mean(pocket_residues)
    dists = np.abs(np.array(pocket_indices) - pocket_center)
    sigma = dists.std() if dists.std() > 1e-8 else 1.0
    raw_weights = np.exp(-dists / sigma)
    attn = raw_weights / raw_weights.sum()

    pooled = (pocket_feats * attn[:, None]).sum(axis=0)
    return pooled.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Morgan FP & AAC (substrate/enzyme fallback)
# ═══════════════════════════════════════════════════════════════════════════════

def morgan_fp(smiles: str, radius=2, nbits=2048):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(nbits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    return np.array(list(fp.ToBitString()), dtype=np.float32)


AA_LIST = "ACDEFGHIKLMNPQRSTVWY"
AA_IDX = {aa: i for i, aa in enumerate(AA_LIST)}

def aac_features(sequence: str) -> np.ndarray:
    counts = np.zeros(20, dtype=np.float32)
    for aa in sequence:
        if aa in AA_IDX:
            counts[AA_IDX[aa]] += 1
    total = counts.sum() or 1
    return counts / total


# ═══════════════════════════════════════════════════════════════════════════════
# FBA Surrogate Stub
# ═══════════════════════════════════════════════════════════════════════════════

class FBASurrogate(nn.Module):
    """
    Lightweight surrogate for FBA feasibility scoring.
    Takes substrate embedding + microbe embedding → scalar feasibility ∈ [0,1].
    Used in Stage 3 closed-loop self-bootstrap.
    """
    def __init__(self, emb_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, sub_emb, mic_emb):
        combined = torch.cat([sub_emb, mic_emb], dim=-1)
        return self.net(combined).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════════
# Four-Stage Training
# ═══════════════════════════════════════════════════════════════════════════════

def train_four_stages(model, loader, cfg: Config, device: str,
                      fba_surrogate=None, enzyme2microbe=None):
    """
    Stage 0: Independent modality pretrain (reconstruction-style, warm up projectors)
    Stage 1: Pairwise contrastive (reaction↔enzyme primary + substrate↔microbe secondary)
    Stage 2: Triplet consistency (three-way InfoNCE + VICReg + concept anchor)
    Stage 3: Closed-loop self-bootstrap (FBA surrogate pseudo-labels)
    """
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    total_epochs = cfg.total_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_epochs)
    temp_sched = TemperatureScheduler(cfg.temp_start, cfg.temp_end, total_epochs)

    history = {"loss": [], "temp": [], "stage": []}
    epoch_global = 0

    def get_stage(epoch):
        if epoch < cfg.epochs_stage0:
            return 0
        elif epoch < cfg.epochs_stage0 + cfg.epochs_stage1:
            return 1
        elif epoch < cfg.epochs_stage0 + cfg.epochs_stage1 + cfg.epochs_stage2:
            return 2
        else:
            return 3

    for epoch in range(total_epochs):
        stage = get_stage(epoch)
        temperature = temp_sched.get_temperature(epoch)
        epoch_loss = 0.0

        for batch in loader:
            r, e, s, m, concepts, ec_ids, indices = batch
            r, e, s, m = r.to(device), e.to(device), s.to(device), m.to(device)
            concepts = concepts.to(device)

            loss = torch.tensor(0.0, device=device)

            if stage == 0:
                # ── Stage 0: warm up each projector with self-alignment ──
                r_emb = model.encode_reaction(r)
                e_emb = model.encode_enzyme(e)
                # Simple uniformity loss to spread embeddings
                loss = vicreg_variance_loss(r_emb) + vicreg_variance_loss(e_emb)
                m_emb = model.encode_microbe(m)
                loss = loss + vicreg_variance_loss(m_emb)

            elif stage == 1:
                # ── Stage 1: pairwise contrastive ──
                r_emb = model.encode_reaction(r)
                e_emb = model.encode_enzyme(e)
                s_emb = model.encode_substrate(s)
                m_emb = model.encode_microbe(m)

                # Primary: reaction ↔ enzyme
                l_re = infonce_loss(r_emb, e_emb, temperature, ec_ids, cfg.hard_neg_weight)
                # Secondary: enzyme ↔ microbe
                l_em = infonce_loss(e_emb, m_emb, temperature)

                loss = cfg.w_re * l_re + cfg.w_em * l_em

                # VICReg on all modalities
                loss = loss + cfg.vicreg_var_weight * (
                    vicreg_variance_loss(r_emb) + vicreg_variance_loss(e_emb) +
                    vicreg_variance_loss(m_emb))
                loss = loss + cfg.vicreg_cov_weight * (
                    vicreg_covariance_loss(r_emb) + vicreg_covariance_loss(e_emb) +
                    vicreg_covariance_loss(m_emb))

            elif stage == 2:
                # ── Stage 2: three-way contrastive + concept anchors ──
                r_emb = model.encode_reaction(r)
                e_emb = model.encode_enzyme(e)
                s_emb = model.encode_substrate(s)
                m_emb, anchors = model.encode_microbe_with_anchors(m)

                # Three-way InfoNCE
                l_re = infonce_loss(r_emb, e_emb, temperature, ec_ids, cfg.hard_neg_weight)
                l_em = infonce_loss(e_emb, m_emb, temperature)
                l_sm = infonce_loss(s_emb, m_emb, temperature)

                loss = cfg.w_re * l_re + cfg.w_em * l_em + cfg.w_sm * l_sm

                # Concept anchor supervision
                l_anchor = concept_anchor_loss(anchors, concepts)
                loss = loss + cfg.anchor_weight * l_anchor

                # VICReg
                loss = loss + cfg.vicreg_var_weight * (
                    vicreg_variance_loss(r_emb) + vicreg_variance_loss(e_emb) +
                    vicreg_variance_loss(s_emb) + vicreg_variance_loss(m_emb))
                loss = loss + cfg.vicreg_cov_weight * (
                    vicreg_covariance_loss(r_emb) + vicreg_covariance_loss(e_emb) +
                    vicreg_covariance_loss(s_emb) + vicreg_covariance_loss(m_emb))

            elif stage == 3:
                # ── Stage 3: closed-loop self-bootstrap with FBA surrogate ──
                r_emb = model.encode_reaction(r)
                e_emb = model.encode_enzyme(e)
                s_emb = model.encode_substrate(s)
                m_emb, anchors = model.encode_microbe_with_anchors(m)

                # Three-way loss (same as stage 2)
                l_re = infonce_loss(r_emb, e_emb, temperature, ec_ids, cfg.hard_neg_weight)
                l_em = infonce_loss(e_emb, m_emb, temperature)
                l_sm = infonce_loss(s_emb, m_emb, temperature)
                loss = cfg.w_re * l_re + cfg.w_em * l_em + cfg.w_sm * l_sm

                # FBA surrogate pseudo-label confidence weighting
                if fba_surrogate is not None:
                    with torch.no_grad():
                        fba_scores = fba_surrogate(s_emb.detach(), m_emb.detach())
                    # re-weight substrate-microbe loss by FBA confidence
                    l_sm_weighted = infonce_loss(s_emb, m_emb, temperature)
                    confidence_scale = fba_scores.mean().clamp(0.1, 2.0)
                    loss = loss + 0.3 * confidence_scale * l_sm_weighted

                # Concept anchor + VICReg
                loss = loss + cfg.anchor_weight * concept_anchor_loss(anchors, concepts)
                loss = loss + cfg.vicreg_var_weight * (
                    vicreg_variance_loss(r_emb) + vicreg_variance_loss(e_emb) +
                    vicreg_variance_loss(s_emb) + vicreg_variance_loss(m_emb))
                loss = loss + cfg.vicreg_cov_weight * (
                    vicreg_covariance_loss(e_emb) + vicreg_covariance_loss(m_emb))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(loader)
        history["loss"].append(avg_loss)
        history["temp"].append(temperature)
        history["stage"].append(stage)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  [Stage {stage}] epoch {epoch+1:3d}/{total_epochs}"
                  f"  loss={avg_loss:.4f}  τ={temperature:.4f}")

    return history


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_multimodal(model, loader, device, enzyme2microbe=None, ks=(1, 5, 10, 20),
                        chunk_size=4096):
    """
    Evaluate cross-modal retrieval with chunked computation.
    Avoids constructing full N×N dense similarity or rank matrices.
    For each query chunk, compute similarities against ALL candidates,
    then extract top-k and MRR per-row.
    """
    model.eval()
    all_r, all_e, all_s, all_m = [], [], [], []

    for batch in loader:
        r, e, s, m, concepts, ec_ids, indices = batch
        all_r.append(model.encode_reaction(r.to(device)).cpu().numpy())
        all_e.append(model.encode_enzyme(e.to(device)).cpu().numpy())
        all_s.append(model.encode_substrate(s.to(device)).cpu().numpy())
        all_m.append(model.encode_microbe(m.to(device)).cpu().numpy())

    all_r = np.concatenate(all_r, 0)
    all_e = np.concatenate(all_e, 0)
    all_s = np.concatenate(all_s, 0)
    all_m = np.concatenate(all_m, 0)

    N = len(all_r)
    max_k = max(ks)
    labels = np.arange(N)
    results = {}

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
            # sort top-k by score descending
            for local_i in range(end - start):
                topk_scores = sim_chunk[local_i, topk_chunk[local_i]]
                sorted_order = np.argsort(-topk_scores)
                all_topk[start + local_i] = topk_chunk[local_i][sorted_order]
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

        # Aggregate metrics
        for k in ks:
            hits = np.any(all_topk[:, :k] == labels[:, None], axis=1)
            results[f"{pair_label}_top-{k}"] = float(np.mean(hits))
        results[f"{pair_label}_MRR"] = float(np.mean(all_mrr))

    # ── Reaction → Enzyme retrieval ──
    print(f"  Chunked R→E retrieval: {N} queries, chunk_size={chunk_size}")
    _chunked_retrieval(all_r, all_e, "R→E", ks, chunk_size)

    # ── Enzyme → Microbe retrieval ──
    print(f"  Chunked E→M retrieval: {N} queries, chunk_size={chunk_size}")
    _chunked_retrieval(all_e, all_m, "E→M", ks, chunk_size)

    # ── Substrate → Microbe retrieval ──
    print(f"  Chunked S→M retrieval: {N} queries, chunk_size={chunk_size}")
    _chunked_retrieval(all_s, all_m, "S→M", ks, chunk_size)

    return results, all_r, all_e, all_s, all_m


# ═══════════════════════════════════════════════════════════════════════════════
# Enzyme → Microbe Inverted Index (Fact Layer)
# ═══════════════════════════════════════════════════════════════════════════════

def build_enzyme2microbe_index(metadata_list):
    """Build inverted index from enzyme (UniprotID) to microbe assemblies."""
    index = defaultdict(set)
    for m in metadata_list:
        uid = m.get("uniprot_id", "")
        assembly = m.get("assembly_accession", "")
        if uid and assembly:
            index[uid].add(assembly)
    # convert sets to lists for JSON serialization
    return {k: list(v) for k, v in index.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════════

def visualize_four_modal(results, all_r, all_e, all_s, all_m,
                         labels, history, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # ── t-SNE of 4 modalities ──
    n_r, n_e, n_s, n_m = len(all_r), len(all_e), len(all_s), len(all_m)
    # subsample for visualization
    max_pts = min(200, n_r)
    idx = np.random.choice(n_r, max_pts, replace=False) if n_r > max_pts else np.arange(n_r)
    combined = np.concatenate([all_r[idx], all_e[idx], all_s[idx], all_m[idx]], axis=0)
    perplexity = min(30, max(2, (len(combined) - 1) // 3))
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    coords = tsne.fit_transform(combined)
    n = len(idx)
    axes[0, 0].scatter(coords[:n, 0], coords[:n, 1], c='blue', s=8, alpha=0.5, label='Reaction')
    axes[0, 0].scatter(coords[n:2*n, 0], coords[n:2*n, 1], c='red', s=8, alpha=0.5, label='Enzyme')
    axes[0, 0].scatter(coords[2*n:3*n, 0], coords[2*n:3*n, 1], c='green', s=8, alpha=0.5, label='Substrate')
    axes[0, 0].scatter(coords[3*n:, 0], coords[3*n:, 1], c='purple', s=8, alpha=0.5, label='Microbe')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].set_title("4-Modal Unified Space (t-SNE)")

    # ── Training loss ──
    axes[0, 1].plot(history["loss"], 'b-', linewidth=1)
    # color by stage
    stage_colors = {0: 'lightblue', 1: 'orange', 2: 'green', 3: 'red'}
    for i, (l, s) in enumerate(zip(history["loss"], history["stage"])):
        axes[0, 1].axvspan(i-0.5, i+0.5, alpha=0.1, color=stage_colors.get(s, 'gray'))
    axes[0, 1].set_xlabel("Epoch"); axes[0, 1].set_ylabel("Loss")
    axes[0, 1].set_title("Training Loss (colored by stage)")
    axes[0, 1].grid(True, alpha=0.3)

    # ── Temperature curve ──
    axes[0, 2].plot(history["temp"], 'r-', linewidth=1.5)
    axes[0, 2].set_xlabel("Epoch"); axes[0, 2].set_ylabel("Temperature τ")
    axes[0, 2].set_title("Temperature Annealing")
    axes[0, 2].grid(True, alpha=0.3)

    # ── R→E metrics ──
    re_keys = [k for k in results if k.startswith("R→E")]
    re_vals = [results[k] for k in re_keys]
    axes[1, 0].bar(re_keys, re_vals, color='steelblue', edgecolor='white')
    for i, (k, v) in enumerate(zip(re_keys, re_vals)):
        axes[1, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=7)
    axes[1, 0].set_ylim(0, 1.15); axes[1, 0].set_title("Reaction → Enzyme Retrieval")

    # ── E→M metrics ──
    em_keys = [k for k in results if k.startswith("E→M")]
    em_vals = [results[k] for k in em_keys]
    axes[1, 1].bar(em_keys, em_vals, color='darkorange', edgecolor='white')
    for i, (k, v) in enumerate(zip(em_keys, em_vals)):
        axes[1, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=7)
    axes[1, 1].set_ylim(0, 1.15); axes[1, 1].set_title("Enzyme → Microbe Retrieval")

    # ── S→M metrics ──
    sm_keys = [k for k in results if k.startswith("S→M")]
    sm_vals = [results[k] for k in sm_keys]
    axes[1, 2].bar(sm_keys, sm_vals, color='darkgreen', edgecolor='white')
    for i, (k, v) in enumerate(zip(sm_keys, sm_vals)):
        axes[1, 2].text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=7)
    axes[1, 2].set_ylim(0, 1.15); axes[1, 2].set_title("Substrate → Microbe Retrieval")

    plt.tight_layout()
    fig.savefig(output_dir / "unified_space_v3_results.png", dpi=150)
    plt.close(fig)
    print(f"  Figure saved to {output_dir / 'unified_space_v3_results.png'}")


# ═══════════════════════════════════════════════════════════════════════════════
# FAISS Index
# ═══════════════════════════════════════════════════════════════════════════════

def build_faiss_index(embeddings, use_hnsw=True):
    """Build FAISS index — HNSW32 for production, FlatIP for small datasets.
    Falls back to None if FAISS SWIG is incompatible with current numpy.
    """
    dim = embeddings.shape[1]
    n = embeddings.shape[0]
    emb = np.ascontiguousarray(embeddings.astype(np.float32))

    # Probe FAISS SWIG compatibility
    _faiss_ok = False
    try:
        _ = faiss.swigfaiss_avx2.swig_ptr(emb[:1])
        _faiss_ok = True
    except (ValueError, TypeError):
        pass

    if not _faiss_ok:
        print(f"  FAISS SWIG incompatible with numpy {np.__version__}, skipping native index.")
        return None

    if use_hnsw and n >= 100:
        index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 64
        faiss.normalize_L2(emb)
        index.add(emb)
    else:
        index = faiss.IndexFlatIP(dim)
        index.add(emb)
    return index


def save_nn_index(embeddings, path):
    """Save L2-normalized embeddings as .npz fallback for nearest-neighbor search."""
    emb = np.ascontiguousarray(embeddings.astype(np.float32))
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb_normed = emb / np.maximum(norms, 1e-8)
    np.savez(path, embeddings=emb_normed, dim=embeddings.shape[1], n=embeddings.shape[0])


# ═══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_synthetic_data(data_dir: str):
    """Load synthetic data and generate random microbe features as placeholder."""
    data_dir = Path(data_dir)
    feats = np.load(data_dir / "synthetic_features.npz")
    with open(data_dir / "synthetic_metadata.json") as f:
        metadata = json.load(f)
    labels = [m["step_idx"] for m in metadata]
    n = len(labels)
    # Generate random microbe features for synthetic mode (28 dim)
    np.random.seed(42)
    m_feat = np.random.randn(n, 28).astype(np.float32) * 0.1
    concepts = np.full((n, 8), float('nan'), dtype=np.float32)
    return (feats["reaction"], feats["enzyme"], feats["substrate"],
            m_feat, labels, metadata, concepts)


def load_enzyme_cage_300(data_dir: str, test_size: float = 0.15,
                         enzyme_feature: str = "gvp"):
    """
    Load EnzymeCAGE 300 dataset with full 4-modality features.
    Includes microbe metabolic features from JSONL.
    """
    data_dir = Path(data_dir)

    # ── Reaction features ──
    rxn_feats = np.load(data_dir / "features/reaction/reaction_features.npz",
                        allow_pickle=True)
    drfp = rxn_feats["drfp"]
    example_ids = list(rxn_feats["example_id"])
    cano_smiles_list = list(rxn_feats["cano_rxn_smiles"])
    uniprot_ids_npz = list(rxn_feats["uniprot_id"])

    # ── Enzyme metadata ──
    csv_path = data_dir / "tables/reaction_enzyme_pairs.csv"
    seq_map, ec_map = {}, {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            eid = row["example_id"]
            seq_map[eid] = row.get("sequence", "")
            ec_map[eid] = row.get("EC number", "")

    # ── GVP paths ──
    gvp_map = {}  # example_id → (UniprotID, gvp_shard_file)
    gvp_meta_path = data_dir / "features/enzyme/enzyme_feature_metadata.csv"
    if gvp_meta_path.exists():
        with open(gvp_meta_path) as f:
            for row in csv.DictReader(f):
                shard_file = row.get("gvp_shard_file", "")
                uid = row.get("UniprotID", "")
                gvp_map[row["example_id"]] = (uid, shard_file)

    # ── Microbe features from CSV tables (primary) or JSONL (fallback) ──
    microbe_records = {}  # example_id → nested dict for extract_microbe_features()

    # Primary: load from 3 CSV tables and merge into nested dict
    cp_path = data_dir / "tables/microbe_reaction_core_preference.csv"
    sq_path = data_dir / "tables/microbe_reaction_stoich_query.csv"
    mc_path = data_dir / "tables/microbe_reaction_main_metabolite_coverage.csv"

    cp_map, sq_map, mc_map = {}, {}, {}
    if cp_path.exists():
        with open(cp_path) as f:
            for row in csv.DictReader(f):
                cp_map[row["example_id"]] = row
    if sq_path.exists():
        with open(sq_path) as f:
            for row in csv.DictReader(f):
                sq_map[row["example_id"]] = row
    if mc_path.exists():
        with open(mc_path) as f:
            for row in csv.DictReader(f):
                mc_map[row["example_id"]] = row

    if cp_map:  # CSV tables available
        all_eids = set(cp_map) | set(sq_map) | set(mc_map)
        for eid in all_eids:
            rec = {
                "example_id": eid,
                "core_preference": cp_map.get(eid, {}),
                "stoich_query": sq_map.get(eid, {}),
                "main_metabolite_coverage": mc_map.get(eid, {}),
            }
            microbe_records[eid] = rec
        print(f"  Microbe records from CSV tables: {len(microbe_records)}")
    else:
        # Fallback: JSONL (flat format, may produce all-zero features)
        microbe_jsonl = data_dir / "features/microbe/microbe_features.jsonl"
        if microbe_jsonl.exists():
            with open(microbe_jsonl) as f:
                for line in f:
                    rec = json.loads(line.strip())
                    microbe_records[rec["example_id"]] = rec
            print(f"  Microbe records from JSONL fallback: {len(microbe_records)}")

    # ── ESM-C paths + pocket info ──
    esmc_dir = data_dir / "features/enzyme/esm_c_features"
    # Pre-scan ESM-C availability (one os.listdir call vs 145k .exists() syscalls)
    esmc_available = set()  # set of UniprotIDs with ESM-C sequence_node.npz
    if esmc_dir.exists():
        _suffix = "_esm_c_sequence_node.npz"
        for fn in os.listdir(esmc_dir):
            if fn.endswith(_suffix):
                esmc_available.add(fn[: -len(_suffix)])
        print(f"  ESM-C available UIDs: {len(esmc_available)}")

    pocket_info = {}
    pocket_csv = data_dir / "features/enzyme/pocket_info.csv"
    if pocket_csv.exists():
        with open(pocket_csv) as f:
            for row in csv.DictReader(f):
                uid = row["UniprotID"]
                residues_str = row.get("pocket_residues", "")
                if residues_str:
                    pocket_info[uid] = [int(x) for x in residues_str.split(",")]
                else:
                    pocket_info[uid] = []

    # ── UID map from enzyme_feature_metadata ──
    uid_map = {}  # example_id → UniprotID
    if gvp_meta_path.exists():
        with open(gvp_meta_path) as f:
            for row in csv.DictReader(f):
                uid_map[row["example_id"]] = row.get("UniprotID", "")

    n = len(drfp)
    if enzyme_feature == "esmc":
        enz_dim = 1152
    elif enzyme_feature == "gvp":
        enz_dim = 50
    else:
        enz_dim = 20
    microbe_dim = 28  # core_preference(6) + cofactor_stoich(16) + coverage(6)

    enzyme_feats = np.zeros((n, enz_dim), dtype=np.float32)
    substrate_feats = np.zeros((n, 2048), dtype=np.float32)
    microbe_feats = np.zeros((n, microbe_dim), dtype=np.float32)
    concept_targets = np.full((n, 8), float('nan'), dtype=np.float32)
    assembly_ids = []

    parse_ok = gvp_ok = mic_ok = 0
    # For ESM-C mode GVP fallback: pre-load all shards to avoid LRU thrashing
    shard_cache = {}  # shard_path → loaded dict
    _SHARD_CACHE_MAX = 4  # only used in non-ESMC paths
    _esmc_gvp_fallback_count = 0
    gvp_pending = []  # list of (index, uid, shard_file) for batch GVP loading
    metadata = []
    for i in range(n):
        eid = example_ids[i]

        # Enzyme features
        if enzyme_feature == "esmc":
            uid = uid_map.get(eid, "")
            if uid and uid in esmc_available:
                esmc_file = esmc_dir / f"{uid}_esm_c_sequence_node.npz"
                pocket_res = pocket_info.get(uid, [])
                enzyme_feats[i] = esmc_pocket_pool(str(esmc_file), pocket_res)
                gvp_ok += 1  # reuse counter as esmc_ok
            else:
                # Fallback: GVP shard if available, else AAC
                gvp_info = gvp_map.get(eid, ("", ""))
                uid_gvp, shard_file = gvp_info if isinstance(gvp_info, tuple) else ("", gvp_info)
                if shard_file:
                    full_path = data_dir / shard_file
                    sp = str(full_path)
                    if sp not in shard_cache:
                        # In ESM-C mode, load all shards (no LRU eviction)
                        shard_cache[sp] = torch.load(sp, map_location='cpu')
                    entry = shard_cache[sp].get(uid_gvp)
                    if entry is not None:
                        # Pad GVP 50-dim to 1152 with zeros for dim consistency
                        enzyme_feats[i, :50] = gvp_shard_pool(entry)
                        _esmc_gvp_fallback_count += 1
                    else:
                        enzyme_feats[i, :20] = aac_features(seq_map.get(eid, ""))
                else:
                    enzyme_feats[i, :20] = aac_features(seq_map.get(eid, ""))
        elif enzyme_feature == "gvp":
            # AAC fallback; actual GVP loaded in batch post-processing below
            gvp_info = gvp_map.get(eid, ("", ""))
            uid_gvp, shard_file = gvp_info if isinstance(gvp_info, tuple) else ("", gvp_info)
            if shard_file:
                gvp_pending.append((i, uid_gvp, shard_file))
            enzyme_feats[i, :20] = aac_features(seq_map.get(eid, ""))
        else:
            enzyme_feats[i] = aac_features(seq_map.get(eid, ""))

        # Substrate features from SMILES
        cano = cano_smiles_list[i]
        if ">>" in cano:
            sub_smi, prod_smi = cano.split(">>", 1)
            substrate_feats[i] = morgan_fp(sub_smi.strip())
            parse_ok += 1

        # Microbe features
        if eid in microbe_records:
            m_feat, c_target, assembly_id = extract_microbe_features(microbe_records[eid])
            microbe_feats[i] = m_feat
            concept_targets[i] = c_target
            assembly_ids.append(assembly_id)
            mic_ok += 1
        else:
            assembly_ids.append("")

        metadata.append({
            "example_id": eid,
            "uniprot_id": uniprot_ids_npz[i],
            "ec_number": ec_map.get(eid, ""),
            "assembly_accession": assembly_ids[-1],
        })

    # ── Shard-batch GVP post-processing ──
    if enzyme_feature == "gvp" and gvp_pending:
        from collections import defaultdict
        shard_groups = defaultdict(list)
        for (idx, uid, sf) in gvp_pending:
            shard_groups[str(data_dir / sf)].append((idx, uid))
        print(f"  GVP shard-batch: {len(shard_groups)} shards, {len(gvp_pending)} entries pending")
        for sp, items in shard_groups.items():
            shard_data = torch.load(sp, map_location='cpu')
            uid_to_idx = defaultdict(list)
            for (idx, uid) in items:
                uid_to_idx[uid].append(idx)
            for uid, indices in uid_to_idx.items():
                entry = shard_data.get(uid)
                if entry is not None:
                    feat = gvp_shard_pool(entry)
                    for idx in indices:
                        enzyme_feats[idx] = feat
                        gvp_ok += 1
            del shard_data
        print(f"  GVP shard-batch done: {gvp_ok}/{len(gvp_pending)} loaded")

    ec_labels = [m["ec_number"].split(".")[0] if m["ec_number"] else "unknown"
                 for m in metadata]

    print(f"  Loaded {n} examples from EnzymeCAGE 300 dataset")
    print(f"  Reaction (DRFP) dim: {drfp.shape[1]}")
    enz_label = {"esmc": "ESM-C pocket-pooled", "gvp": "GVP attention-pooled", "aac": "AAC"}
    print(f"  Enzyme dim: {enz_dim} ({enz_label.get(enzyme_feature, enzyme_feature)})")
    if enzyme_feature == "esmc" and _esmc_gvp_fallback_count > 0:
        print(f"  ESM-C GVP fallback: {_esmc_gvp_fallback_count}")
    print(f"  Substrate Morgan FP: {parse_ok}/{n} parsed")
    print(f"  Microbe metabolic features: {mic_ok}/{n} loaded (dim={microbe_dim})")
    if enzyme_feature in ("gvp", "esmc"):
        print(f"  {enz_label[enzyme_feature]}: {gvp_ok}/{n}")
    print(f"  Unique assemblies: {len(set(assembly_ids) - {''})}")

    if test_size > 0:
        indices = np.arange(n)
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=42)
        print(f"  Train/Test split: {len(train_idx)}/{len(test_idx)}")
        return (drfp, enzyme_feats, substrate_feats, microbe_feats,
                ec_labels, metadata, concept_targets, train_idx, test_idx)
    return (drfp, enzyme_feats, substrate_feats, microbe_feats,
            ec_labels, metadata, concept_targets)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Unified Vector Space v3 — 4-Stage Training")
    parser.add_argument("--mode", choices=["synthetic", "enzyme_cage_300"],
                        default="synthetic")
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--output_dir", default="./output_v3")
    parser.add_argument("--no_test_split", action="store_true")
    parser.add_argument("--enzyme_feature", choices=["aac", "gvp", "esmc"], default="gvp")
    parser.add_argument("--stage", choices=["all", "0", "1", "2", "3"], default="all",
                        help="Which stages to run (default: all)")
    args = parser.parse_args()

    cfg = Config()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Mode: {args.mode}")
    print(f"Unified dim: {cfg.unified_dim}")
    print(f"Temperature: {cfg.temp_start} → {cfg.temp_end} (cosine annealing)")
    print(f"VICReg: var_w={cfg.vicreg_var_weight}, cov_w={cfg.vicreg_cov_weight}")
    print(f"Three-way weights: w_re={cfg.w_re}, w_em={cfg.w_em}, w_sm={cfg.w_sm}")

    # ── Stage selection ──
    if args.stage != "all":
        stage_num = int(args.stage)
        # Only run one stage: zero out other stage epochs
        cfg.epochs_stage0 = cfg.epochs_stage0 if stage_num >= 0 else 0
        cfg.epochs_stage1 = cfg.epochs_stage1 if stage_num >= 1 else 0
        cfg.epochs_stage2 = cfg.epochs_stage2 if stage_num >= 2 else 0
        cfg.epochs_stage3 = cfg.epochs_stage3 if stage_num >= 3 else 0

    # ── Load data ──
    train_idx = test_idx = None
    concept_targets = None

    if args.mode == "synthetic":
        r_feat, e_feat, s_feat, m_feat, labels, metadata, concept_targets = \
            load_synthetic_data(args.data_dir)
    elif args.mode == "enzyme_cage_300":
        result = load_enzyme_cage_300(
            args.data_dir,
            test_size=0.0 if args.no_test_split else 0.15,
            enzyme_feature=args.enzyme_feature)
        if len(result) == 9:
            r_feat, e_feat, s_feat, m_feat, labels, metadata, concept_targets, \
                train_idx, test_idx = result
        else:
            r_feat, e_feat, s_feat, m_feat, labels, metadata, concept_targets = result

    n = len(r_feat)
    print(f"\n  Total samples: {n}")
    print(f"  Dims: reaction={r_feat.shape[1]}, enzyme={e_feat.shape[1]}, "
          f"substrate={s_feat.shape[1]}, microbe={m_feat.shape[1]}")

    # ── Build enzyme→microbe inverted index ──
    enzyme2microbe = build_enzyme2microbe_index(metadata)
    print(f"  Enzyme→Microbe index: {len(enzyme2microbe)} enzymes mapped")

    # ── Dataset & loaders ──
    dataset = MultiModalDataset(r_feat, e_feat, s_feat, m_feat, labels,
                                concept_targets,
                                [m.get("assembly_accession", "") for m in metadata])

    if train_idx is not None:
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        test_dataset = torch.utils.data.Subset(dataset, test_idx)
        train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)
    else:
        train_loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    # ── Model ──
    model = UnifiedSpace(
        reaction_dim=r_feat.shape[1],
        enzyme_dim=e_feat.shape[1],
        substrate_dim=s_feat.shape[1],
        microbe_dim=m_feat.shape[1],
        cfg=cfg,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {param_count:,}")

    # ── FBA surrogate ──
    fba_surrogate = FBASurrogate(cfg.unified_dim).to(device)

    # ── Train ──
    print(f"\n{'='*60}")
    print(f"Starting 4-stage training ({cfg.total_epochs} total epochs)")
    print(f"  Stage 0 (pretrain):         {cfg.epochs_stage0} epochs")
    print(f"  Stage 1 (pairwise):         {cfg.epochs_stage1} epochs")
    print(f"  Stage 2 (triplet+anchor):   {cfg.epochs_stage2} epochs")
    print(f"  Stage 3 (self-bootstrap):   {cfg.epochs_stage3} epochs")
    print(f"{'='*60}\n")

    history = train_four_stages(model, train_loader, cfg, device,
                                fba_surrogate=fba_surrogate,
                                enzyme2microbe=enzyme2microbe)

    # ── Save checkpoint BEFORE evaluation (protect against eval OOM) ──
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("\nSaving checkpoint before evaluation...")
    torch.save({
        "model_state_dict": model.state_dict(),
        "fba_surrogate_state_dict": fba_surrogate.state_dict(),
        "config": {k: v for k, v in cfg.__class__.__dict__.items()
                   if not k.startswith('_') and not callable(v) and not isinstance(v, property)},
        "enzyme2microbe_index": enzyme2microbe,
    }, out / "model_v3.pt")
    with open(out / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"  Checkpoint saved to {out}/model_v3.pt")
    print(f"  Training history saved to {out}/training_history.json")

    # ── Evaluate ──
    print("\nEvaluating multi-modal retrieval (chunked)...")
    eval_loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)
    metrics, all_r, all_e, all_s, all_m = evaluate_multimodal(
        model, eval_loader, device, enzyme2microbe)

    print("\n  Cross-Modal Retrieval Results:")
    for k, v in metrics.items():
        print(f"    {k}: {v:.4f}")

    if args.mode == "enzyme_cage_300":
        print_ec_breakdown(labels)

    # ── Save evaluation results ──
    np.savez(out / "embeddings_v3.npz",
             reaction=all_r, enzyme=all_e, substrate=all_s, microbe=all_m)

    # Save metrics FIRST (critical, before FAISS which may OOM)
    with open(out / "metrics_v3.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out / "metadata_v3.json", "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    with open(out / "enzyme2microbe_index.json", "w") as f:
        json.dump(enzyme2microbe, f, indent=2)
    print(f"  Metrics saved to {out}/metrics_v3.json")

    # Build FAISS indices (optional) — with numpy .npz fallback
    faiss_count, fallback_count = 0, 0
    for name, emb in [("reaction", all_r), ("enzyme", all_e),
                      ("substrate", all_s), ("microbe", all_m)]:
        idx = build_faiss_index(emb)
        if idx is not None:
            try:
                faiss.write_index(idx, str(out / f"{name}_index.faiss"))
                faiss_count += 1
            except Exception as e:
                print(f"  WARNING: FAISS write failed for {name}: {e}")
                save_nn_index(emb, str(out / f"{name}_nn_index.npz"))
                fallback_count += 1
        else:
            save_nn_index(emb, str(out / f"{name}_nn_index.npz"))
            fallback_count += 1
    if faiss_count:
        print(f"  FAISS indices saved: {faiss_count}")
    if fallback_count:
        print(f"  Numpy NN fallback indices saved: {fallback_count} (*_nn_index.npz)")

    # ── Visualize ──
    visualize_four_modal(metrics, all_r, all_e, all_s, all_m,
                         labels, history, out)

    print(f"\nAll outputs saved to {out}/")
    print(f"  model_v3.pt           — model weights + config + inverted index")
    print(f"  embeddings_v3.npz     — 4-modal embeddings")
    print(f"  *_index.faiss         — FAISS HNSW indices (4 modalities)")
    print(f"  metrics_v3.json       — retrieval metrics")
    print(f"  enzyme2microbe_index  — fact retrieval layer")
    print(f"  training_history.json — loss/temp/stage per epoch")


def print_ec_breakdown(ec_labels):
    from collections import Counter
    ec_counter = Counter(ec_labels)
    print("\n  EC class distribution:")
    for ec, count in ec_counter.most_common():
        desc = {"1": "oxidoreductase", "2": "transferase", "3": "hydrolase",
                "4": "lyase", "5": "isomerase", "6": "ligase", "7": "translocase"
                }.get(ec, "unknown")
        print(f"    EC {ec} ({desc}): {count}")


if __name__ == "__main__":
    main()
