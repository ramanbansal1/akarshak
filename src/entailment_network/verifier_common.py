from __future__ import annotations

import math

import torch
import torch.nn as nn

LABELS = ["supported", "partially_supported", "not_supported"]
NOT_SUP = 2
_ALIAS = {"supported": 0, "partially_supported": 1, "partially supported": 1,
          "partial": 1, "not_supported": 2, "not supported": 2, "refuted": 2}


def parse_label(x) -> int:
    if isinstance(x, int):
        return x
    s = str(x).strip().lower()
    return int(s) if s.isdigit() else _ALIAS[s]


# ----------------------------------------------------------------- feature cache (flat, chunk-level)
def save_cache(path, states, labels, meta, extra=None):
    """states: list of [1+N, L, d]; labels: list of N-lists (or None)."""
    blob = {
        "hq": torch.stack([s[0] for s in states]),
        "he": torch.cat([s[1:] for s in states]),
        "n_ev": torch.tensor([len(s) - 1 for s in states]),
        "meta": meta, "extra": extra or {},
    }
    if labels is not None:
        blob["y"] = torch.tensor([y for ys in labels for y in ys])
    torch.save(blob, path)


class FeatureCache:
    def __init__(self, path):
        b = torch.load(path)
        self.hq, self.he, self.n_ev = b["hq"], b["he"], b["n_ev"]
        self.y, self.meta, self.extra = b.get("y"), b["meta"], b["extra"]
        self.seq = torch.repeat_interleave(torch.arange(len(self.hq)), self.n_ev)   # chunk -> sequence
        self.chunk_row = self.extra["seq_row"][self.seq]                            # chunk -> dataset row
        self.n_rows = len(self.extra["row_y"])

    def batch(self, idx):
        return self.hq[self.seq[idx]], self.he[idx][:, None], (self.y[idx] if self.y is not None else None)


@torch.inference_mode()
def predict_cache(model, cache, dev, bs=4096):
    model.eval()
    out = []
    for i in range(0, len(cache.he), bs):
        idx = torch.arange(i, min(i + bs, len(cache.he)))
        hq, he, _ = cache.batch(idx)
        out.append(model(hq.to(dev), he.to(dev))[:, 0].softmax(-1).cpu())
    return torch.cat(out)                                   # [M,3] chunk probabilities


# ----------------------------------------------------------------- chunk -> row aggregation
def row_scores(probs, chunk_row, n_rows):
    s = torch.zeros(n_rows, 3)
    if len(probs):
        s.scatter_reduce_(0, chunk_row[:, None].expand(-1, 3), probs, reduce="amax", include_self=False)
    return s                                                # max chunk prob per class


def decide(s, t_sup, t_part):
    pred = torch.full((len(s),), NOT_SUP, dtype=torch.long)
    pred[s[:, 1] >= t_part] = 1
    pred[s[:, 0] >= t_sup] = 0
    return pred


def tune_thresholds(s, y, grid=None):
    grid = grid if grid is not None else torch.linspace(0.05, 0.95, 19).tolist()
    best = (-1.0, 0.5, 0.5)
    for ts in grid:
        for tp in grid:
            f1, _ = macro_f1(decide(s, ts, tp), y)
            if f1 > best[0]:
                best = (f1, ts, tp)
    return best                                             # (macro_f1, t_sup, t_part)


# ----------------------------------------------------------------- model + metrics
def default_layers(model_id: str):
    """Layers to cache: last hidden layer only (head design decision)."""
    from transformers import AutoConfig
    n = AutoConfig.from_pretrained(model_id).num_hidden_layers
    return [n - 1]


class VerdictModel(nn.Module):
    """Low-rank bilinear claim-evidence scorer (proposal §2.2).

    forward(hq, he): hq [B, L, d] claim states, he [B, N, L, d] evidence states.
    Uses the last cached layer of each; per class k, W_k ~ U_k V_k^T (rank r):
        z_k(b, n) = sum_r (hq_b . U_k[:, r]) * (he_bn . V_k[:, r]) + bias_k
    Returns logits [B, N, K].
    """

    def __init__(self, d: int, n_layers: int, K: int = 3, r: int = 256):
        super().__init__()
        self.d, self.n_layers, self.K, self.r = d, n_layers, K, r
        scale = 1.0 / math.sqrt(d)
        self.U = nn.Parameter(torch.randn(K, d, r) * scale)
        self.V = nn.Parameter(torch.randn(K, d, r) * scale)
        self.bias = nn.Parameter(torch.zeros(K))

    def forward(self, hq: torch.Tensor, he: torch.Tensor) -> torch.Tensor:
        hq = hq[:, -1, :]            # [B, d]   last cached layer
        he = he[:, :, -1, :]         # [B, N, d]
        qu = torch.einsum("bd,kdr->kbr", hq, self.U)        # [K, B, r]
        qe = torch.einsum("bnd,kdr->kbnr", he, self.V)      # [K, B, N, r]
        z = (qu[:, :, None, :] * qe).sum(-1)                # [K, B, N]
        return z.permute(1, 2, 0) + self.bias               # [B, N, K]


def macro_f1(pred, y, K: int = 3):
    """pred, y: index tensors of any shape. Returns (mean_f1, {label: f1})."""
    pred, y = torch.as_tensor(pred).reshape(-1), torch.as_tensor(y).reshape(-1)
    per = {}
    for c in range(K):
        tp = ((pred == c) & (y == c)).sum().item()
        fp = ((pred == c) & (y != c)).sum().item()
        fn = ((pred != c) & (y == c)).sum().item()
        per[LABELS[c]] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return sum(per.values()) / K, per
