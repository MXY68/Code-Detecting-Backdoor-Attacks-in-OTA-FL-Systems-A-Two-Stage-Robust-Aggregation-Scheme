# -*- coding: utf-8 -*-
"""
OTA-FL with PS-side LGP inspection (MAD + AHC + Innocent Criterion)
Dataset: RadioML 2016.10A (RML2016.10a_dict.pkl)
Task   : Modulation Classification (11 classes)
Attack : Backdoor via IQ-trigger + target label
Goal   : Prevent malicious clients' local updates from entering global aggregation

Run:
  python otfl_radioml2016a_lgp_backdoor.py

Note:
- root_path is set to "RML2016.10a_dict.pkl" (you provided)
- This script keeps your trust + LGP + RS pipeline, but replaces CIFAR parts with RadioML
"""

import copy, math, random, warnings
from collections import defaultdict
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset, Subset

import pickle

# ---- optional sklearn (AHC) ----
_HAVE_SK = True
try:
    from sklearn.cluster import AgglomerativeClustering
except Exception:
    _HAVE_SK = False
    warnings.warn("scikit-learn not found; will use a lightweight AHC fallback.")

# ---------------- Config ----------------
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Federated setting
NUM_CLIENTS   = 20
NUM_MALICIOUS = 6
BATCH_SIZE    = 64
LOCAL_EPOCHS  = 4
LR            = 1e-3

# Non-IID partition
DIRICHLET_ALPHA  = 1  # smaller => more non-iid

# Backdoor (poisoning) setting
POISON_FRACTION  = 0.2
BACKDOOR_TARGET  = 9          # target class index in [0..9]
LAMBDA_EUC       = 0.1        # Euclidean penalty (bigger => more hidden; may reduce ASR)

# Malicious selection
RANDOMIZE_MALICIOUS = True
MALICIOUS_SEED = 12345

# RS & LGP hyperparams
LGP_WARMUP_ROUNDS = 5
LGP_PASS_FRAC     = 0.6
APPLY_RS_HARD_THRESHOLD_IN_FINAL = True

# Dataset path
root_path = "RML2016.10a_dict.pkl"
# root_path = "RML2016.10b.dat"

# ============================================================
# 0) RadioML 2016.10A dataset utilities
# ============================================================
def load_rml2016a(pkl_path: str):
    """
    RadioML 2016.10A pickle:
        data[(mod, snr)] -> np.ndarray [N, 2, 128] (common)
    Returns:
        X:   [Total, 2, 128] float32
        y:   [Total] int64 (0..C-1)
        snr: [Total] int64
        mods: list of mods in label order
        snrs: list of snrs
        mod2id: dict
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    mods = sorted(list(set(k[0] for k in data.keys())))
    snrs = sorted(list(set(k[1] for k in data.keys())))
    mod2id = {m: i for i, m in enumerate(mods)}

    X_list, y_list, snr_list = [], [], []
    for (m, s), x in data.items():
        # x: [N, 2, 128]
        x = np.asarray(x)
        if x.ndim != 3 or x.shape[1] != 2:
            raise ValueError(f"Unexpected sample shape for key {(m,s)}: {x.shape}, expect [N,2,128].")
        X_list.append(x.astype(np.float32))
        y_list.append(np.full((x.shape[0],), mod2id[m], dtype=np.int64))
        snr_list.append(np.full((x.shape[0],), s, dtype=np.int64))

    X = np.vstack(X_list).astype(np.float32)
    y = np.concatenate(y_list).astype(np.int64)
    snr = np.concatenate(snr_list).astype(np.int64)

    return X, y, snr, mods, snrs, mod2id
# def load_rml2016a(pkl_path: str):
#     with open(pkl_path, "rb") as f:
#         data = pickle.load(f, encoding="latin1")
#
#     # 10b 的 key 结构可能是嵌套dict，先探查
#     # 打印前几个key确认格式
#     sample_keys = list(data.keys())[:3]
#     print(f"[Dataset] Sample keys: {sample_keys}")
#     print(f"[Dataset] Value type: {type(data[sample_keys[0]])}")
#     print(f"[Dataset] Value shape: {np.array(data[sample_keys[0]]).shape}")
#
#     mods = sorted(list(set(k[0] for k in data.keys())))
#     snrs = sorted(list(set(k[1] for k in data.keys())))
#     mod2id = {m: i for i, m in enumerate(mods)}
#
#     X_list, y_list, snr_list = [], [], []
#     for (m, s), x in data.items():
#         x = np.asarray(x, dtype=np.float32)
#         # 10b shape 可能是 [N, 128, 2] 而非 [N, 2, 128]，做自动转置
#         if x.ndim == 3 and x.shape[2] == 2 and x.shape[1] != 2:
#             x = x.transpose(0, 2, 1)  # -> [N, 2, 128]
#         if x.ndim != 3 or x.shape[1] != 2:
#             raise ValueError(f"Unexpected shape for key {(m,s)}: {x.shape}")
#         X_list.append(x)
#         y_list.append(np.full((x.shape[0],), mod2id[m], dtype=np.int64))
#         snr_list.append(np.full((x.shape[0],), s, dtype=np.int64))
#
#     X   = np.vstack(X_list).astype(np.float32)
#     y   = np.concatenate(y_list).astype(np.int64)
#     snr = np.concatenate(snr_list).astype(np.int64)
#
#     return X, y, snr, mods, snrs, mod2id

def split_rml_by_mod_snr(X, y, snr, test_ratio=0.2, seed=42):
    """
    Split per (mod,snr) group to keep balanced distribution.
    """
    rng = np.random.default_rng(seed)
    idx_all = np.arange(len(y))
    test_mask = np.zeros(len(y), dtype=bool)

    for m in np.unique(y):
        for s in np.unique(snr):
            idx = idx_all[(y == m) & (snr == s)]
            if len(idx) == 0:
                continue
            rng.shuffle(idx)
            n_te = int(round(test_ratio * len(idx)))
            test_mask[idx[:n_te]] = True

    idx_te = np.where(test_mask)[0]
    idx_tr = np.where(~test_mask)[0]
    return idx_tr, idx_te


class RMLDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, mean=None, std=None, do_standardize: bool = True):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

        self.do_standardize = do_standardize
        if do_standardize:
            if mean is None or std is None:
                mean = self.X.mean()
                std  = self.X.std() + 1e-6
            self.mean = float(mean)
            self.std  = float(std)
            self.X = (self.X - self.mean) / self.std
        else:
            self.mean, self.std = None, None

    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx): return self.X[idx], self.y[idx]



def make_client_loaders(train_ds, client_indices, batch_size):
    return [DataLoader(Subset(train_ds, idxs), batch_size=batch_size, shuffle=True, drop_last=False)
            for idxs in client_indices]


def partition_dirichlet_balanced_by_labels(labels: np.ndarray, num_clients: int, alpha: float, seed: int = 42):
    """
    Your balanced-Dirichlet logic adapted to use label array directly.
    Returns: list of index lists into the TRAIN SET (0..len(labels)-1).
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    num_classes = int(labels.max()) + 1
    idx_by_class = [np.where(labels == c)[0] for c in range(num_classes)]
    for c in range(num_classes):
        rng.shuffle(idx_by_class[c])

    total = len(labels)
    base = total // num_clients
    rem  = total % num_clients
    quota = np.array([base + (1 if i < rem else 0) for i in range(num_clients)], dtype=int)

    remaining = quota.copy()
    client_indices = [[] for _ in range(num_clients)]

    class_order = rng.permutation(num_classes)
    for c in class_order:
        pool = idx_by_class[c]
        n_c = len(pool)
        if n_c == 0:
            continue

        prop = rng.dirichlet(np.full(num_clients, alpha))
        placed = np.zeros(num_clients, dtype=int)
        left = n_c

        mask = remaining > 0
        if not mask.any():
            break

        probs_raw = prop * remaining * mask
        probs = probs_raw / (probs_raw.sum() if probs_raw.sum() > 0 else mask.sum())
        counts = rng.multinomial(min(left, int(remaining.sum())), probs)
        counts = np.minimum(counts, remaining)
        assigned = int(counts.sum())
        left -= assigned
        placed += counts

        while left > 0:
            mask2 = (remaining - placed) > 0
            if not mask2.any():
                break
            probs2_raw = prop * (remaining - placed) * mask2
            probs2 = probs2_raw / probs2_raw.sum()
            add = rng.multinomial(left, probs2)
            add = np.minimum(add, remaining - placed)
            got = int(add.sum())
            placed += add
            left -= got
            if got == 0:
                idxs = np.where(mask2)[0]
                take = min(left, len(idxs))
                placed[idxs[:take]] += 1
                left -= take

        start = 0
        for i in range(num_clients):
            k = int(placed[i])
            if k > 0:
                client_indices[i].extend(pool[start:start+k].tolist())
                start += k
                remaining[i] -= k

    assert all(len(ci) == q for ci, q in zip(client_indices, quota)), \
        "Partition failed: some client did not meet quota."
    return client_indices


def sample_malicious_ids(num_clients, num_malicious, seed=12345, randomized=True):
    if randomized:
        rng = np.random.default_rng(seed)
        return set(int(x) for x in rng.choice(num_clients, size=num_malicious, replace=False))
    return set(range(num_malicious))


# ============================================================
# 1) Model for AMC (1D CNN with GroupNorm)
# ============================================================
class AMC_CNN1D(nn.Module):
    """
    Input: [B, 2, 128] (I/Q, length 128)
    Output: logits [B, 11]
    """
    def __init__(self, num_classes=11):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=7, padding=3, bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 128, kernel_size=5, padding=2, bias=False),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),  # 128 -> 64

            nn.Conv1d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),  # 64 -> 32

            nn.Conv1d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool1d(1),  # [B,256,1]
        )
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.net(x)
        x = x.squeeze(-1)
        return self.fc(x)


def init_global_model(num_classes: int):
    m = AMC_CNN1D(num_classes=num_classes).to(DEVICE)

    def _init(mm):
        if isinstance(mm, nn.Conv1d):
            nn.init.kaiming_normal_(mm.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(mm, nn.Linear):
            nn.init.kaiming_normal_(mm.weight, nonlinearity="relu")
            if mm.bias is not None:
                nn.init.zeros_(mm.bias)

    m.apply(_init)
    return m


# ============================================================
# 2) Trigger + Attack helpers (IQ backdoor)
# ============================================================
def add_iq_trigger(x: torch.Tensor,
                   trig_len: int = 16,
                   i_val: float = 6.0,
                   q_val: float = -3.0,
                   pos: str = "tail") -> torch.Tensor:
    """
    x: [2, 128]
    Overwrite a fixed segment with constants (patch-like trigger).
    """
    xx = x.clone()
    L = xx.shape[-1]
    trig_len = min(trig_len, L)

    if pos == "tail":
        s = L - trig_len
    elif pos == "head":
        s = 0
    else:
        s = (L - trig_len) // 2

    xx[0, s:s+trig_len] = i_val
    xx[1, s:s+trig_len] = q_val
    return xx


# ============================================================
# 3) Param helpers
# ============================================================
def model_params_to_vector(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.data.view(-1) for p in model.parameters()])

def set_model_params_from_vector(model: nn.Module, vec: torch.Tensor) -> None:
    pointer = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(vec[pointer:pointer+numel].view_as(p).to(p.device))
        pointer += numel

def split_delta_by_layer(delta_vec: torch.Tensor, model: nn.Module) -> List[torch.Tensor]:
    out, p = [], 0
    for param in model.parameters():
        n = param.numel()
        out.append(delta_vec[p:p+n].detach().cpu().view(-1))
        p += n
    assert p == delta_vec.numel()
    return out


# ============================================================
# 4) Local train (malicious gets backdoor + optional Euclidean penalty)
# ============================================================
def local_train(model, global_model, dataloader, epochs, lr, device,
                malicious=False, poison_frac=0.0, backdoor_target=0, lambda_euc=0.0,
                trig_len=16, i_val=3.0, q_val=-3.0, trig_pos="tail"):
    model.to(device)
    ce = nn.CrossEntropyLoss()
    # opt = optim.SGD(model.parameters(), lr=lr, momentum=0.85, weight_decay=1e-4)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    all_batches = list(dataloader)
    num_batches = len(all_batches)
    poison_batches = set()
    if malicious and poison_frac > 0:
        num_poison = max(1, int(math.ceil(poison_frac * num_batches)))
        poison_batches = set(random.sample(range(num_batches), num_poison))

    # 参考向量：当前全局模型参数
    global_vec = model_params_to_vector(global_model).detach().to(device)
    model.train()

    for _ in range(epochs):
        for i, (x, y) in enumerate(all_batches):
            x, y = x.to(device), y.to(device)  # x: [B,2,128]

            # ---- 保留原来的 IQ trigger backdoor 注入方式 ----
            if malicious and i in poison_batches:
                x_poison = x.clone()
                for k in range(x_poison.shape[0]):
                    x_poison[k] = add_iq_trigger(
                        x_poison[k], trig_len=trig_len, i_val=i_val, q_val=q_val, pos=trig_pos
                    )
                inputs = x_poison
                targets = torch.full_like(y, backdoor_target)
            else:
                inputs, targets = x, y

            opt.zero_grad()
            loss_cls = ce(model(inputs), targets)
            loss = loss_cls

            # --------- 从 Euc-constrained attack 改成 Cos-constrained attack ---------
            if malicious and lambda_euc > 0:
                # 当前攻击更新
                local_vec = model_params_to_vector(model).to(device)
                delta_attack = local_vec - global_vec

                theta_norm = global_vec.norm() + 1e-12
                delta_norm = delta_attack.norm() + 1e-12
                rel_l2 = delta_norm / theta_norm

                # ========== (1) rel_l2 clipping ==========
                MAX_REL_L2 = 0.08
                if rel_l2 > MAX_REL_L2:
                    with torch.no_grad():
                        scale = MAX_REL_L2 / rel_l2
                        new_vec = global_vec + delta_attack * scale
                        set_model_params_from_vector(model, new_vec)

                    # 裁剪后重新计算
                    local_vec = model_params_to_vector(model).to(device)
                    delta_attack = local_vec - global_vec
                    theta_norm = global_vec.norm() + 1e-12
                    delta_norm = delta_attack.norm() + 1e-12
                    rel_l2 = delta_norm / theta_norm

                # ========== (2) cosine penalty ==========
                cos_sim = torch.dot(delta_attack, global_vec) / (delta_norm * theta_norm + 1e-12)
                cos_loss = 1.0 - cos_sim

                # ========== (3) norm matching penalty ==========
                TARGET_REL_L2 = 0.03
                excess = torch.relu(rel_l2 - TARGET_REL_L2)
                norm_loss = excess * excess

                lambda_cos = float(lambda_euc)
                lambda_norm = 0.5 * lambda_cos

                loss = loss + lambda_cos * cos_loss + lambda_norm * norm_loss
            # ------------------------------------------------------------------------

            loss.backward()
            opt.step()

    local_vec = model_params_to_vector(model).detach().cpu()
    global_vec_cpu = model_params_to_vector(global_model).detach().cpu()
    delta = local_vec - global_vec_cpu
    return {"delta": delta, "num_samples": len(dataloader.dataset)}


# ============================================================
# 5) Client-side trust (your original)
# ============================================================
client_states = defaultdict(dict)

def client_compute_trust_local_only(theta_prev_vec, delta_i_vec, prev_delta_i_vec=None,
                                    state=None, topk=None, quantile=0.99, hist_maxlen=50, eps=1e-12):
    import numpy as _np
    import torch

    def _cos(a, b):
        na = torch.linalg.norm(a)
        nb = torch.linalg.norm(b)
        if float(na) < eps or float(nb) < eps:
            return 0.0
        return float(torch.dot(a, b) / (na * nb))

    def _push(st, key, val):
        """Store floats as float; store tensors as CPU float clones; enforce hist_maxlen."""
        if st is None:
            return
        st.setdefault(key, [])

        if torch.is_tensor(val):
            v = val.detach().cpu().float().clone()
        else:
            v = float(val)

        st[key].append(v)
        if len(st[key]) > hist_maxlen:
            st[key] = st[key][-hist_maxlen:]

    # ---- ensure CPU 1-D vectors ----
    theta = theta_prev_vec.detach().cpu().float().flatten()
    delta = delta_i_vec.detach().cpu().float().flatten()

    # ---------------- basic indicators (keep yours) ----------------
    tda   = _cos(delta, theta)
    tda01 = (tda + 1.0) / 2.0

    rel_l2 = float(torch.linalg.norm(delta) / (torch.linalg.norm(theta) + eps))

    abs_delta = torch.abs(delta)
    if delta.numel() > 0:
        thr = torch.quantile(abs_delta, torch.tensor(quantile, dtype=abs_delta.dtype))
        key_mask = (abs_delta >= thr)
    else:
        key_mask = torch.zeros_like(delta, dtype=torch.bool)

    if key_mask.any():
        spiky = float((torch.linalg.norm(delta[key_mask])**2 / (torch.linalg.norm(delta)**2 + eps)).item())
    else:
        spiky = 0.0

    temporal = float('nan')
    if prev_delta_i_vec is not None:
        prev_delta = prev_delta_i_vec.detach().cpu().float().flatten()
        # guard shape mismatch
        if prev_delta.numel() == delta.numel():
            temporal = _cos(delta, prev_delta)
    temporal01 = 0.5 if _np.isnan(temporal) else (temporal + 1.0) / 2.0

    # ---- select indices for wsign ----
    if delta.numel() > 0:
        if topk is not None:
            k = max(1, min(int(topk), delta.numel()))
            idx = torch.topk(abs_delta, k, largest=True).indices
        else:
            idx = key_mask.nonzero().flatten()
            if idx.numel() == 0:
                k = max(1, min(1000, delta.numel()))
                idx = torch.topk(abs_delta, k, largest=True).indices
    else:
        idx = torch.tensor([], dtype=torch.long)

    if idx.numel() == 0:
        wsign = 0.5
    else:
        d_sel = delta[idx]
        t_sel = theta[idx]
        w     = torch.abs(d_sel)
        mask  = (w > 1e-6) & (torch.abs(t_sel) > 1e-6)
        if mask.sum().item() == 0:
            wsign = 0.5
        else:
            sd    = torch.sign(d_sel[mask])
            st    = torch.sign(t_sel[mask])
            agree = (sd == st).float()
            wsign = float((agree * w[mask]).sum().item() / (w[mask].sum().item() + 1e-12))

    # -------------------------
    # Step 2: residual consistency (client-local history)
    # resid = ||delta - mean(prev_deltas)|| / ||delta||
    # -------------------------
    resid = 0.5  # default neutral
    if state is not None:
        hist = state.get("hist_delta_vec", [])
        M = 5
        # ensure hist tensors are compatible
        if isinstance(hist, list) and len(hist) >= M:
            try:
                H = torch.stack([h.flatten().float() for h in hist[-M:]], dim=0)  # [M, D]
                if H.shape[1] == delta.numel():
                    mu = H.mean(dim=0)
                    resid = float(torch.linalg.norm(delta - mu) / (torch.linalg.norm(delta) + eps))
            except Exception:
                resid = 0.5

    # ---------------- NEW: residual ratio (orthogonal energy) ----------------
    # resid_ratio = ||delta - proj_theta(delta)|| / ||delta||
    # bigger => more orthogonal energy => more suspicious
    theta_norm = torch.linalg.norm(theta) + eps
    delta_norm = torch.linalg.norm(delta) + eps
    theta_dir  = theta / theta_norm

    proj_scalar = torch.dot(delta, theta_dir)  # scalar
    resid_vec   = delta - proj_scalar * theta_dir
    resid_ratio = float(torch.linalg.norm(resid_vec) / delta_norm)  # [0, ~1]
    resid_score = float(max(0.0, 1.0 - resid_ratio))               # higher is better

    # ---------------- record history (optional) ----------------
    if state is not None:
        _push(state, 'hist_tda01', tda01)
        _push(state, 'hist_rel_l2', rel_l2)
        _push(state, 'hist_spiky', spiky)
        if not _np.isnan(temporal01):
            _push(state, 'hist_temporal01', temporal01)
        _push(state, 'hist_wsign', wsign)
        # IMPORTANT: store delta vector for residual consistency (bounded by hist_maxlen)
        _push(state, 'hist_delta_vec', delta)

    # ---------------- trust formula (updated weights) ----------------
    # rel_l2: keep your tanh-style shaping
    r0, alpha = 0.35, 3.0
    rel_score = 1.0 - float(_np.tanh(alpha * max(0.0, rel_l2 - r0)))

    # spiky: keep but lower weight
    s0, gamma = 0.40, 2.0
    spky_score = 1.0 - float(max(0.0, spiky - s0) ** gamma)

    # weights (you can tune ONLY these)
    w_rel   = 0.25
    w_spk   = 0.10
    w_wsign = 0.15
    w_resid = 0.40
    w_temp  = 0.10

    trust = (w_rel   * rel_score +
             w_spk   * spky_score +
             w_wsign * wsign +
             w_resid * resid_score +
             w_temp  * temporal01)

    trust = float(_np.clip(trust, 0.0, 1.0))

    return {
        "tda":         float(tda),
        "tda01":       float(tda01),
        "rel_l2":      float(rel_l2),
        "spiky":       float(spiky),
        "temporal":    float(0.0 if _np.isnan(temporal) else temporal),
        "temporal01":  float(temporal01),
        "wsign":       float(wsign),
        "resid":       float(resid),
        "resid_ratio": float(resid_ratio),
        "trust":       float(trust),
    }




# ============================================================
# 6) LGP for PS inspection (paper-faithful; your original)
# ============================================================
RS = defaultdict(int)

def _mad(v: np.ndarray, eps=1e-12):
    med = np.median(v)
    return med, np.median(np.abs(v - med)) + eps

def _kurtosis(x: np.ndarray, eps=1e-12):
    x = x.astype(np.float64)
    m = x.mean()
    s2 = ((x-m)**2).mean() + eps
    return float(((x-m)**4).mean()/ (s2**2) - 3.0)

def _skewness(x: np.ndarray, eps=1e-12):
    x = x.astype(np.float64)
    m = x.mean()
    s = np.sqrt(((x-m)**2).mean() + eps)
    return float(((x-m)**3).mean() / (s**3 + eps))

def _pairwise_mean_distance(mat: np.ndarray):
    if len(mat) <= 1: return 0.0
    from itertools import combinations
    dsum, cnt = 0.0, 0
    for i, j in combinations(range(len(mat)), 2):
        dsum += np.linalg.norm(mat[i]-mat[j])
        cnt += 1
    return float(dsum/cnt)

def _cluster_stats(vectors: List[np.ndarray], ref: np.ndarray, member_ids: List[int]):
    if len(member_ids) == 0:
        return dict(Dmean=1e9, SDmean=1e9, Devmean=1e9, RSsum=0.0)
    mat = np.stack([vectors[i] for i in member_ids], axis=0)
    Dmean  = _pairwise_mean_distance(mat)
    SDmean = float(np.std(mat, axis=0).mean())
    Devmean= float(np.abs(mat - ref[None,:]).sum(axis=1).mean())
    RSsum  = float(sum(RS[i] for i in member_ids))
    return dict(Dmean=Dmean, SDmean=SDmean, Devmean=Devmean, RSsum=RSsum)

def _innocent_choose(cluster1_ids: List[int], cluster2_ids: List[int],
                     vectors: List[np.ndarray], ref: np.ndarray):
    c1 = _cluster_stats(vectors, ref, cluster1_ids)
    c2 = _cluster_stats(vectors, ref, cluster2_ids)
    cond1 = (c2["Dmean"] > c1["Dmean"]) and (c2["SDmean"] > c1["SDmean"]) and (c2["Devmean"] < c1["Devmean"]) and (c2["RSsum"] >= c1["RSsum"])
    cond2 = (c2["Dmean"] < c1["Dmean"]) and (c2["Devmean"] < c1["Devmean"]) and (c2["RSsum"] >= c1["RSsum"])
    if cond1 or cond2: return 2, c1, c2
    return 1, c1, c2

def _ahc_split(feature_mat: np.ndarray):
    K = len(feature_mat)
    if _HAVE_SK and K >= 2:
        lab = AgglomerativeClustering(n_clusters=2, linkage='average').fit_predict(feature_mat)
        cluster1 = [i for i in range(K) if lab[i] == 0]
        cluster2 = [i for i in range(K) if lab[i] == 1]
    else:
        u, s, vh = np.linalg.svd(feature_mat - feature_mat.mean(0, keepdims=True), full_matrices=False)
        proj = (feature_mat @ vh[0])
        med = np.median(proj)
        cluster1 = [i for i in range(K) if proj[i] <= med]
        cluster2 = [i for i in range(K) if proj[i] >  med]
    return cluster1, cluster2

def ps_inspect_with_LGP(model: nn.Module,
                        suspicious_updates: List[dict],
                        ref_vec: torch.Tensor,
                        round_idx: int,
                        log_prefix: str = "[PS-LGP]"):
    susp_cids = [u["cid"] for u in suspicious_updates]
    deltas = [u["delta"].detach().cpu() for u in suspicious_updates]
    layers_per_client = [split_delta_by_layer(d, model) for d in deltas]
    ref_layers = split_delta_by_layer(ref_vec.detach().cpu(), model)

    L = len(ref_layers)
    K = len(suspicious_updates)

    layer_pass = np.zeros((K, L), dtype=bool)
    deny_details = {cid: {"mad_fail_layers": [], "ahc_non_honest_layers": []} for cid in susp_cids}

    for l in range(L):
        glist = [layers_per_client[k][l].numpy().astype(np.float32) for k in range(K)]
        norms = np.array([np.linalg.norm(g) for g in glist], dtype=np.float64)
        med, mad = _mad(norms)
        LB, UB = med - mad, med + mad

        cand1_idx = [i for i, n in enumerate(norms) if (n >= LB and n <= UB)]
        cand2_idx = [i for i in range(K) if i not in cand1_idx]

        choose_cand = cand1_idx
        if len(cand2_idx) > 0:
            winner, c1s, c2s = _innocent_choose(cand1_idx, cand2_idx, glist, ref_layers[l].numpy())
            if winner == 2:
                choose_cand = cand2_idx

        if len(choose_cand) == 0:
            for i in range(K):
                deny_details[susp_cids[i]]["mad_fail_layers"].append(l)
            continue

        feats = []
        for i in choose_cand:
            g = glist[i]
            ref = ref_layers[l].numpy()
            PC = float((g > 0).sum()); NC = float((g < 0).sum()); ZC = float((g == 0).sum())
            kurt = _kurtosis(g); skew = _skewness(g)
            others = [glist[j] for j in choose_cand if j != i]
            if len(others) == 0:
                Dmean = 0.0
            else:
                Dmean = float(np.mean([np.linalg.norm(g - o) for o in others]))
            dev = float(np.abs(ref - g).sum())
            L2  = float(np.linalg.norm(g))
            denom = (np.linalg.norm(g)*np.linalg.norm(ref) + 1e-12)
            dir_angle = float(np.arccos(np.clip((g@ref)/denom, -1.0, 1.0)))
            feats.append([PC, NC, ZC, kurt, skew, Dmean, dev, L2, dir_angle])
        feats = np.asarray(feats, dtype=np.float64)

        c1, c2 = _ahc_split(feats)
        winner, stats_c1, stats_c2 = _innocent_choose([choose_cand[i] for i in c1],
                                                      [choose_cand[i] for i in c2],
                                                      glist, ref_layers[l].numpy())
        honest_idx = [choose_cand[i] for i in (c2 if winner == 2 else c1)]

        for i in range(K):
            if i in honest_idx:
                layer_pass[i, l] = True
            else:
                if i in choose_cand:
                    deny_details[susp_cids[i]]["ahc_non_honest_layers"].append(l)
                else:
                    deny_details[susp_cids[i]]["mad_fail_layers"].append(l)

    passed_ids = []
    for i in range(K):
        frac = float(layer_pass[i].mean()) if L > 0 else 0.0
        need = 0.4 if round_idx <= LGP_WARMUP_ROUNDS else LGP_PASS_FRAC
        if frac >= need:
            passed_ids.append(susp_cids[i])
            RS[susp_cids[i]] += 1
    denied_ids = [cid for cid in susp_cids if cid not in passed_ids]

    for cid in denied_ids:
        info = deny_details[cid]
        print(f"{log_prefix} cid={cid:2d} DENY | mad_fail_layers={info['mad_fail_layers']} "
              f"| ahc_non_honest_layers={info['ahc_non_honest_layers']} "
              f"| benign_layer_frac={1.0 - (len(info['mad_fail_layers'])+len(info['ahc_non_honest_layers']))/max(1,L):.2f}")

    return passed_ids, deny_details


# ============================================================
# 7) Eval
# ============================================================
@torch.no_grad()
def evaluate(model, test_loader, device):
    model.to(device).eval()
    correct, total = 0, 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total

@torch.no_grad()
def evaluate_asr(model, test_loader, device, target, add_trigger_fn):
    model.to(device).eval()
    ok, tot = 0, 0
    for x, y in test_loader:
        x_bd = x.clone()
        for k in range(x_bd.shape[0]):
            x_bd[k] = add_trigger_fn(x_bd[k])
        x_bd, y = x_bd.to(device), y.to(device)
        pred = model(x_bd).argmax(1)
        ok += (pred == target).sum().item()
        tot += y.size(0)
    return 100.0 * ok / tot


# ============================================================
# 8) Aggregation helpers
# ============================================================
def weighted_average(updates, key="delta", weight_key="num_samples"):
    total = sum(u[weight_key] for u in updates)
    agg = None
    for u in updates:
        w = u[weight_key] / total
        agg = w * u[key] if agg is None else agg + w * u[key]
    return agg, total


# ============================================================
# 9) One-round (baseline: no defense)
# ============================================================
def one_round_no_defense(global_model, client_loaders, malicious_ids,
                         trig_len=16, i_val=3.0, q_val=-3.0, trig_pos="tail"):
    updates = []
    for cid in range(NUM_CLIENTS):
        local_model = copy.deepcopy(global_model)
        malicious = (cid in malicious_ids)
        loader = client_loaders[cid]

        dd = local_train(
            local_model, global_model, loader,
            epochs=LOCAL_EPOCHS, lr=LR, device=DEVICE,
            malicious=malicious,
            poison_frac=POISON_FRACTION,
            backdoor_target=BACKDOOR_TARGET,
            lambda_euc=(LAMBDA_EUC if malicious else 0.0),
            trig_len=trig_len, i_val=i_val, q_val=q_val, trig_pos=trig_pos
        )

        updates.append({"cid": cid, "delta": dd["delta"], "num_samples": dd["num_samples"]})

    agg_all, _ = weighted_average(updates)
    global_vec = model_params_to_vector(global_model).cpu()
    new_global = copy.deepcopy(global_model)
    set_model_params_from_vector(new_global, global_vec + agg_all)
    return new_global


# ============================================================
# 10) One-round (defended: trust -> LGP -> RS threshold -> final agg)
# ============================================================
def one_round(round_idx, global_model, client_loaders, malicious_ids,
              split=(0.50, 0.30, 0.20), verbose=True, use_ema=False, beta_ema=0.90, ref_delta_ema=None,
              trig_len=16, i_val=3.0, q_val=-3.0, trig_pos="tail"):

    import numpy as np
    import torch

    # ---------------- helpers (robust trust) ----------------
    def robust_med_mad(x, eps=1e-12):
        x = np.asarray(x, dtype=np.float64)
        med = np.median(x)
        mad = np.median(np.abs(x - med)) + eps
        return med, mad

    def score_high_is_bad(x, med, mad, k=1.0):
        # x > med 越多分越低；x <= med 不惩罚
        z = (x - med) / mad
        z = np.maximum(0.0, z)
        return np.exp(-k * z)

    def score_low_is_bad(x, med, mad, k=1.0):
        # x < med 越多分越低；x >= med 不惩罚
        z = (med - x) / mad
        z = np.maximum(0.0, z)
        return np.exp(-k * z)

    def cos_sim(a: torch.Tensor, b: torch.Tensor, eps=1e-12) -> float:
        a = a.detach().cpu().float().flatten()
        b = b.detach().cpu().float().flatten()
        na = torch.linalg.norm(a)
        nb = torch.linalg.norm(b)
        if float(na) < eps or float(nb) < eps:
            return 0.0
        return float(torch.dot(a, b) / (na * nb))

    # ---------------- main ----------------
    theta_prev_vec = model_params_to_vector(global_model).detach().cpu()

    updates = []
    for cid in range(NUM_CLIENTS):
        local_model = copy.deepcopy(global_model)
        malicious = (cid in malicious_ids)
        loader = client_loaders[cid]

        dd = local_train(
            local_model, global_model, loader,
            epochs=LOCAL_EPOCHS, lr=LR, device=DEVICE,
            malicious=malicious,
            poison_frac=POISON_FRACTION,
            backdoor_target=BACKDOOR_TARGET,
            lambda_euc=(LAMBDA_EUC if malicious else 0.0),
            trig_len=trig_len, i_val=i_val, q_val=q_val, trig_pos=trig_pos
        )

        state_i = client_states[cid]
        prev_delta = state_i.get('prev_delta')

        # 先保留你现有的 client-local 指标计算（但 trust 后面会重算）
        scores = client_compute_trust_local_only(
            theta_prev_vec, dd["delta"], prev_delta,
            state=state_i, topk=None, quantile=0.99, hist_maxlen=50
        )

        state_i['prev_delta'] = dd["delta"].clone()
        client_states[cid] = state_i

        updates.append({
            "cid": cid, "delta": dd["delta"], "num_samples": dd["num_samples"],
            "trust": scores["trust"],  # will be overwritten by robust trust below
            "tda": scores["tda"], "rel_l2": scores["rel_l2"], "spiky": scores["spiky"],
            "temporal": scores["temporal"], "wsign": scores["wsign"],
            "resid_ratio": scores["resid_ratio"],
        })

    # ---------------- (NEW) robust-population trust recompute ----------------
    # 用 agg_all 或 EMA 作为“参考方向”，计算 align_ref
    agg_all, _ = weighted_average(updates)
    ref0 = agg_all.detach().cpu()
    if use_ema and (ref_delta_ema is not None):
        ref0 = (beta_ema * ref_delta_ema + (1.0 - beta_ema) * ref0).detach().cpu()

    # 收集 per-round 标量特征
    rel_list   = [u["rel_l2"] for u in updates]               # high -> suspicious
    spk_list   = [u["spiky"] for u in updates]                # high -> suspicious
    res_list   = [u["resid_ratio"] for u in updates]          # high -> suspicious (先按你现有定义用)
    ws_list    = [u["wsign"] for u in updates]                # low  -> suspicious
    tmp01_list = [0.5 if np.isnan(u["temporal"]) else (u["temporal"] + 1.0) / 2.0 for u in updates]  # low -> suspicious
    ali_list   = [cos_sim(u["delta"], ref0) for u in updates] # low -> suspicious (NEW)

    # robust stats
    rel_med, rel_mad = robust_med_mad(rel_list)
    spk_med, spk_mad = robust_med_mad(spk_list)
    res_med, res_mad = robust_med_mad(res_list)
    ws_med,  ws_mad  = robust_med_mad(ws_list)
    tmp_med, tmp_mad = robust_med_mad(tmp01_list)
    ali_med, ali_mad = robust_med_mad(ali_list)

    # 组合权重（你可以之后再调）
    w_align = 0.30
    w_wsign = 0.20
    w_rel   = 0.15
    w_spk   = 0.10
    w_res   = 0.15
    w_tmp   = 0.10

    # 重新写回 updates[*]["trust"]
    for u, ali, tmp01 in zip(updates, ali_list, tmp01_list):
        s_rel   = score_high_is_bad(u["rel_l2"],      rel_med, rel_mad, k=1.2)
        s_spk   = score_high_is_bad(u["spiky"],       spk_med, spk_mad, k=1.0)
        s_res   = score_high_is_bad(u["resid_ratio"], res_med, res_mad, k=1.0)
        s_ws    = score_low_is_bad(u["wsign"],        ws_med,  ws_mad,  k=1.0)
        s_tmp   = score_low_is_bad(tmp01,             tmp_med, tmp_mad, k=0.8)
        s_align = score_low_is_bad(ali,               ali_med, ali_mad, k=1.5)

        trust_new = (w_align * s_align +
                     w_wsign * s_ws +
                     w_rel   * s_rel +
                     w_spk   * s_spk +
                     w_res   * s_res +
                     w_tmp   * s_tmp)

        u["trust"] = float(np.clip(trust_new, 0.0, 1.0))
        u["align_ref0"] = float(ali)
        u["temporal01"] = float(tmp01)

    # ---------------- verbose print (now prints new trust) ----------------
    if verbose:
        for u in updates:
            label = "MAL" if u["cid"] in malicious_ids else "BEN"
            print(f"CID {u['cid']:2d} [{label}] | trust={u['trust']:.3f} | align={u.get('align_ref0', 0.0):+.3f} "
                  f"| relL2={u['rel_l2']:.3f} | spiky={u['spiky']:.3f} | resid={u['resid_ratio']:.3f} "
                  f"| temporal01={u.get('temporal01', 0.5):.3f} | wsign={u['wsign']:.3f}")

    # (for logging only) keep your original baselines
    benign_updates = [u for u in updates if u["cid"] not in malicious_ids]
    agg_benign, _ = weighted_average(benign_updates)
    global_vec = model_params_to_vector(global_model).cpu()

    contaminated_global = copy.deepcopy(global_model); set_model_params_from_vector(contaminated_global, global_vec + agg_all)
    clean_global        = copy.deepcopy(global_model); set_model_params_from_vector(clean_global,        global_vec + agg_benign)

    # ----- split by trust (unchanged) -----
    t_trusted, t_susp, t_rej = split
    trusts = np.array([u["trust"] for u in updates], dtype=float)
    order  = np.argsort(-trusts)

    nT = max(1, int(round(t_trusted * NUM_CLIENTS)))
    nR = max(1, int(round(t_rej     * NUM_CLIENTS)))
    nS = max(1, NUM_CLIENTS - nT - nR)
    while nT + nS + nR > NUM_CLIENTS:
        if nS > 1: nS -= 1
        elif nR > 1: nR -= 1
        else: nT -= 1
    while nT + nS + nR < NUM_CLIENTS:
        nT += 1

    idx_trusted  = order[:nT]
    idx_susp     = order[nT:nT+nS]
    idx_rejected = order[nT+nS:]

    trusted_updates    = [updates[i] for i in idx_trusted]
    suspicious_updates = [updates[i] for i in idx_susp]
    rejected_updates   = [updates[i] for i in idx_rejected]

    if verbose:
        print(f"[Split] trusted={len(trusted_updates)}, suspicious={len(suspicious_updates)}, rejected={len(rejected_updates)}")

    # ----- trusted aggregate as reference (unchanged) -----
    agg_trusted, _ = weighted_average(trusted_updates)
    trusted_global = copy.deepcopy(global_model); set_model_params_from_vector(trusted_global, global_vec + agg_trusted)

    delta_trusted = agg_trusted.detach().cpu()
    if use_ema and ref_delta_ema is not None:
        ref_vec = (beta_ema * ref_delta_ema + (1.0 - beta_ema) * delta_trusted).detach().cpu()
    else:
        ref_vec = delta_trusted.clone()

    # ----- PS inspection on suspicious via LGP (unchanged) -----
    acc_ids, deny_details = ps_inspect_with_LGP(global_model, suspicious_updates, ref_vec, round_idx)

    groups = dict(
        trusted=sorted([u["cid"] for u in trusted_updates]),
        suspicious=sorted([u["cid"] for u in suspicious_updates]),
        rejected=sorted([u["cid"] for u in rejected_updates]),
        accepted_from_susp=sorted(acc_ids),
    )
    denied_after_ps = sorted(list(set(groups["suspicious"]) - set(groups["accepted_from_susp"])))
    groups["denied_after_ps"] = denied_after_ps
    groups["denied_overall"]  = sorted(groups["rejected"] + denied_after_ps)
    groups["final_participants"] = sorted(groups["trusted"] + groups["accepted_from_susp"])

    # ----- RS threshold & final participants (after RS) (unchanged) -----
    groups["final_participants_afterRS"] = groups["final_participants"].copy()
    groups["removed_by_rs"] = []

    if len(RS) > 0:
        rs_items = sorted([(cid, RS.get(cid, 0)) for cid in range(NUM_CLIENTS)], key=lambda x: (-x[1], x[0]))
        rs_vals = np.array([v for _, v in rs_items], dtype=float)
        rs_med, rs_mad = _mad(rs_vals) if len(rs_vals) else (0.0, 0.0)
        rs_thr = rs_med - rs_mad

        groups["rs_items"] = rs_items
        groups["rs_med"]   = rs_med
        groups["rs_mad"]   = rs_mad
        groups["rs_thr"]   = rs_thr

        if APPLY_RS_HARD_THRESHOLD_IN_FINAL and round_idx > LGP_WARMUP_ROUNDS:
            before_rs = groups["final_participants"].copy()
            after_rs = [cid for cid in groups["final_participants"] if RS[cid] >= rs_thr]
            groups["final_participants_afterRS"] = after_rs
            groups["removed_by_rs"] = sorted(list(set(before_rs) - set(after_rs)))

    # ----- Final aggregate (unchanged) -----
    final_pool = [u for u in updates if u["cid"] in groups["final_participants_afterRS"]]
    if len(final_pool) == 0:
        final_pool = trusted_updates

    agg_final, _ = weighted_average(final_pool)
    final_global = copy.deepcopy(global_model)
    set_model_params_from_vector(final_global, global_vec + agg_final)

    # ----- RS update: for participants actually aggregated (afterRS) (unchanged) -----
    for cid in groups["final_participants_afterRS"]:
        RS[cid] = RS.get(cid, 0) + 1

    if len(RS) > 0:
        rs_items = sorted([(cid, RS.get(cid, 0)) for cid in range(NUM_CLIENTS)], key=lambda x: (-x[1], x[0]))
        rs_vals = np.array([v for _, v in rs_items], dtype=float)
        rs_med, rs_mad = _mad(rs_vals) if len(rs_vals) else (0.0, 0.0)
        rs_thr = rs_med - rs_mad
        print("\n[RS] cid->RS (sorted):", rs_items)
        print(f"[RS] median={rs_med:.2f}, MAD={rs_mad:.2f}, threshold (med-MAD)={rs_thr:.2f}")

    print("\n=== Client Group Summary (this round) ===")
    print(f"Trusted (Top {int(split[0]*100)}%, OTA aggregated)        : {groups['trusted']}")
    print(f"Suspicious (middle {int(split[1]*100)}%, individual access): {groups['suspicious']}")
    print(f"Rejected initially (trust phase)                          : {groups['rejected']}")
    print(f"Suspicious accepted after inspection                      : {groups['accepted_from_susp']}")
    print(f"Suspicious denied after inspection                        : {groups['denied_after_ps']}")
    print(f"Denied overall (trust + PS)                               : {groups['denied_overall']}")
    print(f"[RS Filter] Removed by RS threshold                       : {groups['removed_by_rs']}")
    print(f"Final aggregation participants (after RS)                 : {groups['final_participants_afterRS']}")

    return dict(
        models=dict(
            contaminated=contaminated_global,
            clean=clean_global,
            trusted=trusted_global,
            final=final_global
        ),
        groups=groups,
        aux=dict(ref_delta_ema=ref_vec.clone())
    )



# ============================================================
# 11) Multi-round simulation
# ============================================================
def simulate_many_rounds(num_rounds=10, split=(0.50, 0.30, 0.20), verbose_round=True,
                         print_models_eval_each_round=True, use_ema=False,
                         trig_len=16, i_val=3.0, q_val=-3.0, trig_pos="tail",
                         test_ratio=0.2, do_standardize=True):

    # Load RadioML
    X, y, snr, mods, snrs, mod2id = load_rml2016a(root_path)
    # ------------------------------------------------
    # SNR filtering (keep only SNR >= SNR_MIN)
    # ------------------------------------------------
    SNR_MIN = -8  # 你要的门限：0 / -2 / 5 都可以
    mask = (snr >= SNR_MIN)
    X = X[mask]
    y = y[mask]
    snr = snr[mask]

    print(f"[SNR Filter] Keep SNR >= {SNR_MIN}")
    print(f"[SNR Filter] Samples kept: {mask.sum()} / {len(mask)}")
    print(f"[SNR Filter] Unique SNRs: {sorted(np.unique(snr).tolist())}")
    # ------------------------------------------------
    num_classes = int(y.max()) + 1

    print("\n[Dataset] RadioML 2016.10A loaded.")
    print(f"[Dataset] X shape={X.shape}, y shape={y.shape}, snr shape={snr.shape}")
    print(f"[Dataset] num_classes={num_classes}, mods={mods}")
    print(f"[Dataset] snrs={snrs}")
    assert 0 <= BACKDOOR_TARGET < num_classes, f"BACKDOOR_TARGET must be in [0,{num_classes-1}]"

    print(f"[Backdoor] BACKDOOR_TARGET={BACKDOOR_TARGET} => mod='{mods[BACKDOOR_TARGET]}'")

    # Split train/test per (mod,snr)
    idx_tr, idx_te = split_rml_by_mod_snr(X, y, snr, test_ratio=test_ratio, seed=SEED)
    train_raw = torch.tensor(X[idx_tr], dtype=torch.float32)
    train_mean = train_raw.mean()
    train_std = train_raw.std() + 1e-6

    train_ds = RMLDataset(X[idx_tr], y[idx_tr], mean=train_mean, std=train_std, do_standardize=True)
    test_ds = RMLDataset(X[idx_te], y[idx_te], mean=train_mean, std=train_std, do_standardize=True)

    # Client partition on train labels
    client_indices = partition_dirichlet_balanced_by_labels(
        labels=y[idx_tr],
        num_clients=NUM_CLIENTS,
        alpha=DIRICHLET_ALPHA,
        seed=SEED
    )

    client_loaders = make_client_loaders(train_ds, client_indices, BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, drop_last=False)

    # Two chains: defended and baseline
    global_model_def = init_global_model(num_classes=num_classes)
    global_model_all = copy.deepcopy(global_model_def)

    malicious_ids = sample_malicious_ids(NUM_CLIENTS, NUM_MALICIOUS, MALICIOUS_SEED, RANDOMIZE_MALICIOUS)
    print("\nMalicious client IDs (fixed across rounds):", sorted(list(malicious_ids)))

    ref_delta_ema = None
    history = dict(
        round=[],
        acc_final=[], asr_final=[],
        acc_all=[], asr_all=[],

        # ---------- 数量 ----------
        num_final_participants=[],
        num_denied_overall=[],
        num_removed_by_rs=[],
        num_selected_malicious=[],
        num_removed_benign=[],

        # ---------- 用户ID ----------
        final_participants_ids=[],
        denied_overall_ids=[],
        removed_by_rs_ids=[],
        selected_malicious_ids=[],
        removed_benign_ids=[]
    )

    for r in range(1, num_rounds+1):
        print(f"\n========== Round {r} ==========")

        # baseline (no defense)
        global_model_all = one_round_no_defense(
            global_model_all, client_loaders, malicious_ids,
            trig_len=trig_len, i_val=i_val, q_val=q_val, trig_pos=trig_pos
        )

        # defended
        result = one_round(
            r, global_model_def, client_loaders, malicious_ids,
            split=split, verbose=verbose_round,
            use_ema=use_ema, ref_delta_ema=ref_delta_ema,
            trig_len=trig_len, i_val=i_val, q_val=q_val, trig_pos=trig_pos
        )

        ref_delta_ema = result["aux"]["ref_delta_ema"]
        models = result["models"]
        groups = result["groups"]  # ← 加上这一行
        global_model_def = models["final"]

        # Evaluation
        if print_models_eval_each_round:
            print("\n=== Model Evaluation (this round) ===")

            acc_all = evaluate(global_model_all, test_loader, DEVICE)
            asr_all = evaluate_asr(global_model_all, test_loader, DEVICE, BACKDOOR_TARGET,
                                   lambda xx: add_iq_trigger(xx, trig_len=trig_len, i_val=i_val, q_val=q_val, pos=trig_pos))
            print(f"Contaminated (all clients, no defense) | Acc: {acc_all:6.2f}% | ASR: {asr_all:6.2f}%")

            acc_final = evaluate(models["final"], test_loader, DEVICE)
            asr_final = evaluate_asr(models["final"], test_loader, DEVICE, BACKDOOR_TARGET,
                                     lambda xx: add_iq_trigger(xx, trig_len=trig_len, i_val=i_val, q_val=q_val, pos=trig_pos))
            print(f"Final (defended, selected clients)     | Acc: {acc_final:6.2f}% | ASR: {asr_final:6.2f}%")

        # Store
        acc_all = evaluate(global_model_all, test_loader, DEVICE)
        asr_all = evaluate_asr(global_model_all, test_loader, DEVICE, BACKDOOR_TARGET,
                               lambda xx: add_iq_trigger(xx, trig_len=trig_len, i_val=i_val, q_val=q_val, pos=trig_pos))
        acc_final = evaluate(models["final"], test_loader, DEVICE)
        asr_final = evaluate_asr(models["final"], test_loader, DEVICE, BACKDOOR_TARGET,
                                 lambda xx: add_iq_trigger(xx, trig_len=trig_len, i_val=i_val, q_val=q_val, pos=trig_pos))

        print(f"[Round {r}] Baseline  Acc={acc_all:.2f}% | ASR={asr_all:.2f}%")
        print(f"[Round {r}] Defended  Acc={acc_final:.2f}% | ASR={asr_final:.2f}%")

        history["round"].append(r)
        history["acc_all"].append(acc_all)
        history["asr_all"].append(asr_all)
        history["acc_final"].append(acc_final)
        history["asr_final"].append(asr_final)

        # =========================
        # 额外保存：数量和ID
        # =========================
        final_ids = set(groups["final_participants_afterRS"])
        denied_ids = set(groups["denied_overall"])
        removed_by_rs_ids = set(groups["removed_by_rs"])
        mal_set = set(malicious_ids)
        benign_set = set(range(NUM_CLIENTS)) - mal_set

        # ---- 数量 ----
        history["num_final_participants"].append(len(final_ids))
        history["num_denied_overall"].append(len(denied_ids))
        history["num_removed_by_rs"].append(len(removed_by_rs_ids))
        history["num_selected_malicious"].append(len(final_ids & mal_set))
        history["num_removed_benign"].append(len(denied_ids & benign_set))

        # ---- 用户ID ----
        history["final_participants_ids"].append(sorted(list(final_ids)))
        history["denied_overall_ids"].append(sorted(list(denied_ids)))
        history["removed_by_rs_ids"].append(sorted(list(removed_by_rs_ids)))
        history["selected_malicious_ids"].append(sorted(list(final_ids & mal_set)))
        history["removed_benign_ids"].append(sorted(list(denied_ids & benign_set)))

    return dict(
        baseline_model=global_model_all,
        defended_model=global_model_def,
        history=history
    )


# ============================================================
# 12) main
# ============================================================
if __name__ == "__main__":
    import csv
    import json

    results = simulate_many_rounds(
        num_rounds=100,
        split=(0.5, 0.3, 0.20),
        verbose_round=True,
        print_models_eval_each_round=True,
        use_ema=False
    )

    FINAL_MODEL = results["defended_model"]
    LOGS = results["history"]
    BASELINE_MODEL = results["baseline_model"]

    # =========================
    # 1) 保存数值指标到 CSV
    # =========================
    csv_path = "WaveClean.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "round",
            "acc_all", "asr_all",
            "acc_final", "asr_final",
            "num_final_participants",
            "num_denied_overall",
            "num_removed_by_rs",
            "num_selected_malicious",
            "num_removed_benign"
        ])

        for i in range(len(LOGS["round"])):
            writer.writerow([
                LOGS["round"][i],
                LOGS["acc_all"][i],
                LOGS["asr_all"][i],
                LOGS["acc_final"][i],
                LOGS["asr_final"][i],
                LOGS["num_final_participants"][i],
                LOGS["num_denied_overall"][i],
                LOGS["num_removed_by_rs"][i],
                LOGS["num_selected_malicious"][i],
                LOGS["num_removed_benign"][i],
            ])

    print(f"\nSaved round metrics to: {csv_path}")

    # =========================
    # 2) 保存用户ID到 JSON
    # =========================
    json_path = "WaveClean.csv.json"
    with open(json_path, "w") as f:
        json.dump({
            "round": LOGS["round"],
            "final_participants_ids": LOGS["final_participants_ids"],
            "denied_overall_ids": LOGS["denied_overall_ids"],
            "removed_by_rs_ids": LOGS["removed_by_rs_ids"],
            "selected_malicious_ids": LOGS["selected_malicious_ids"],
            "removed_benign_ids": LOGS["removed_benign_ids"]
        }, f, indent=2)

    print(f"Saved round user IDs to: {json_path}")

    # =========================
    # 3) 控制台打印 summary
    # =========================
    print("\n=== Multi-round Summary ===")
    for i in range(len(LOGS["round"])):
        print(
            f"Round {LOGS['round'][i]:02d}: "
            f"Baseline Acc={LOGS['acc_all'][i]:6.2f}% | ASR={LOGS['asr_all'][i]:6.2f}% || "
            f"Defended Acc={LOGS['acc_final'][i]:6.2f}% | ASR={LOGS['asr_final'][i]:6.2f}% || "
            f"FinalParticipants={LOGS['num_final_participants'][i]} | "
            f"SelectedMal={LOGS['num_selected_malicious'][i]} | "
            f"RemovedBenign={LOGS['num_removed_benign'][i]}"
        )