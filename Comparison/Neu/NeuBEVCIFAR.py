# -*- coding: utf-8 -*-
"""
OTA-FL with PS-side LGP inspection (MAD + AHC + Innocent Criterion)
Reference method: LGP: Layerwise Gradient Purify for Robust Federated Learning against Poisoning Attacks
"""

import copy, math, random, warnings
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

# ---- optional sklearn (AHC) ----
_HAVE_SK = True
try:
    from sklearn.cluster import AgglomerativeClustering
except Exception:
    _HAVE_SK = False  # noqa: E999  (line split to avoid linter confusion)
    _HAVE_SK = False
    warnings.warn("scikit-learn not found; will use a lightweight AHC fallback.")

# ---------------- Config ----------------
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLIENTS   = 20
NUM_MALICIOUS = 6
BATCH_SIZE    = 64
LOCAL_EPOCHS  = 4
LR            = 0.01

DIRICHLET_ALPHA  = 0.5 # non-iid level, smaller value means higher non-iid level, reference 51, 54
POISON_FRACTION  = 0.2
BACKDOOR_TARGET  = 0
BENIGN_CLEAN_RATIO = 1.0

RANDOMIZE_MALICIOUS = True
MALICIOUS_SEED = 12345

# RS & LGP hyperparams
LGP_WARMUP_ROUNDS = 3       # warm-up 放宽层数门限
LAMBDA_EUC       = 0.1
LGP_PASS_FRAC      = 0.4     # 通过需要的“诚实层占比”，warm-up 会自动放宽
APPLY_RS_HARD_THRESHOLD_IN_FINAL = True  # 若 True，则最终参与者再加一道 RS ≥ median - MAD 过滤

# ------------- Model -------------
class SimpleCNN(nn.Module):
    """
    深一点的 CNN，用 GroupNorm 替代 BatchNorm，避免联邦场景下 BN 统计量不同步的问题。
    [64x2] -> pool -> [128x2] -> pool -> [256x2] -> pool -> GAP -> FC
    """
    def __init__(self, num_classes=10):
        super().__init__()

        def Norm(c):
            # 8 组的 GroupNorm，c 必须能被 8 整除，这里 64/128/256 都可以
            return nn.GroupNorm(num_groups=8, num_channels=c)

        # Block 1: 3x32x32 -> 64x16x16
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            Norm(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            Norm(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2)  # 32x32 -> 16x16
        )

        # Block 2: 64x16x16 -> 128x8x8
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            Norm(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            Norm(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2)  # 16x16 -> 8x8
        )

        # Block 3: 128x8x8 -> 256x4x4
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            Norm(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm(256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2)  # 8x8 -> 4x4
        )

        # 全局平均池化: 256x4x4 -> 256x1x1
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # 全连接层：256 -> 128 -> num_classes
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)             # [B, 256, 1, 1]
        x = torch.flatten(x, 1)     # [B, 256]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# -------- Data --------
def load_cifar10(root="./data"):
    # CIFAR10的标准预处理
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_ds = datasets.CIFAR10(root, train=True, download=True, transform=transform_train)
    test_ds = datasets.CIFAR10(root, train=False, download=True, transform=transform_test)
    return train_ds, test_ds

def add_box_trigger(img_tensor: torch.Tensor, box_size: int = 3, intensity: float = 1.0) -> torch.Tensor:
    img = img_tensor.clone()
    C, H, W = img.shape  # C=3 for CIFAR10
    # 在所有通道上添加白色方块
    img[:, H-box_size:H, W-box_size:W] = intensity
    return img

def partition_dirichlet_balanced(dataset: Dataset, num_clients: int, alpha: float, seed: int = 42) -> List[List[int]]:
    rng = np.random.default_rng(seed)
    labels = np.array(dataset.targets)
    num_classes = int(labels.max()) + 1
    idx_by_class = [np.where(labels == c)[0] for c in range(num_classes)]
    for c in range(num_classes): rng.shuffle(idx_by_class[c])
    total = len(labels); base = total // num_clients; rem = total % num_clients
    quota = np.array([base + (1 if i < rem else 0) for i in range(num_clients)], dtype=int)
    remaining = quota.copy(); client_indices = [[] for _ in range(num_clients)]
    class_order = rng.permutation(num_classes)
    for c in class_order:
        pool = idx_by_class[c]; n_c = len(pool)
        if n_c == 0: continue
        prop = rng.dirichlet(np.full(num_clients, alpha))
        placed = np.zeros(num_clients, dtype=int); left = n_c
        mask = remaining > 0
        if not mask.any(): break
        probs_raw = prop * remaining * mask
        probs = probs_raw / (probs_raw.sum() if probs_raw.sum() > 0 else mask.sum())
        counts = rng.multinomial(min(left, int(remaining.sum())), probs)
        counts = np.minimum(counts, remaining); assigned = int(counts.sum()); left -= assigned; placed += counts
        while left > 0:
            mask2 = (remaining - placed) > 0
            if not mask2.any(): break
            probs2_raw = prop * (remaining - placed) * mask2
            probs2 = probs2_raw / probs2_raw.sum()
            add = rng.multinomial(left, probs2)
            add = np.minimum(add, remaining - placed)
            got = int(add.sum()); placed += add; left -= got
            if got == 0:
                idxs = np.where(mask2)[0]; take = min(left, len(idxs))
                placed[idxs[:take]] += 1; left -= take
        start = 0
        for i in range(num_clients):
            k = int(placed[i])
            if k > 0:
                client_indices[i].extend(pool[start:start+k].tolist())
                start += k; remaining[i] -= k
    assert all(len(ci) == q for ci, q in zip(client_indices, quota))
    return client_indices

# -------- Param helpers --------
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

# -------- Helper: train one local model & return param vector --------
def _train_local_model(base_model: nn.Module,
                       dataloader,
                       epochs: int,
                       lr: float,
                       device,
                       poison: bool = False,
                       poison_frac: float = 0.0,
                       backdoor_target: int = 0):
    """
    从 base_model 出发做本地 SGD，返回训练完的参数向量（不做 delta 计算）。
    如果 poison=True，则对部分 batch 加触发器并改标签。
    """
    model = copy.deepcopy(base_model).to(device)
    ce = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.85, weight_decay=1e-4)

    all_batches = list(dataloader)
    num_batches = len(all_batches)
    poison_batches = set()
    if poison and poison_frac > 0:
        num_poison = max(1, int(math.ceil(poison_frac * num_batches)))
        poison_batches = set(random.sample(range(num_batches), num_poison))

    model.train()
    for _ in range(epochs):
        for i, (x, y) in enumerate(all_batches):
            x, y = x.to(device), y.to(device)
            if poison and i in poison_batches:
                x_poison = x.clone()
                for k in range(x_poison.shape[0]):
                    x_poison[k] = add_box_trigger(x_poison[k])
                inputs, targets = x_poison, torch.full_like(y, backdoor_target)
            else:
                inputs, targets = x, y

            opt.zero_grad()
            loss = ce(model(inputs), targets)
            loss.backward()
            opt.step()

    local_vec = model_params_to_vector(model).detach().cpu()
    return local_vec
# -------- Helper: Neurotoxin-style combination --------
def neurotoxin_combine(delta_benign: torch.Tensor,
                       delta_poison: torch.Tensor,
                       lcd_frac: float = 0.9,
                       alpha: float = 1,  # 新增：向 poisoned 靠拢程度
                       scale_attack: float = 1,  # 新增：整体放大系数
                       eps: float = 1e-12):
    assert delta_benign.shape == delta_poison.shape
    v = delta_benign
    numel = v.numel()

    k = max(1, int(lcd_frac * numel))
    abs_v = v.abs()

    # 仍然选低贡献维度（你如果想更“暴力”，可以 later 改成 largest=True）
    lcd_idx = torch.topk(abs_v, k, largest=False).indices

    # 原始 Neurotoxin：只在 LCD 上替换
    delta_nt = delta_benign.clone()
    delta_nt[lcd_idx] = delta_poison[lcd_idx]

    # 关键一步：整体向 poisoned 靠拢
    # alpha 越大，攻击越接近 full poison，越容易被检测
    delta_attack = (1 - alpha) * delta_benign + alpha * delta_nt

    # 不做 norm matching，而是直接放大一点，让 relL2/spiky 明显大于 benign
    delta_attack = delta_attack * scale_attack

    return delta_attack



# -------- Local train (Neurotoxin attack for malicious) --------
def local_train(model, global_model, dataloader, epochs, lr, device,
                malicious=False, poison_frac=0.0, backdoor_target=0, lambda_euc=0.0):
    global_vec_cpu = model_params_to_vector(global_model).detach().cpu()

    if not malicious:
        # benign：正常训练
        benign_vec = _train_local_model(global_model, dataloader,
                                        epochs=epochs, lr=lr, device=device,
                                        poison=False, poison_frac=0.0,
                                        backdoor_target=backdoor_target)
        delta = benign_vec - global_vec_cpu
        return {"delta": delta, "num_samples": len(dataloader.dataset)}

    # -------- malicious client --------
    epochs_clean = 1  # 干净阶段：保持不变
    epochs_poison = epochs

    benign_vec = _train_local_model(global_model, dataloader,
                                    epochs=epochs_clean, lr=lr, device=device,
                                    poison=False, poison_frac=0.0,
                                    backdoor_target=backdoor_target)
    delta_benign = benign_vec - global_vec_cpu

    poison_vec = _train_local_model(global_model, dataloader,
                                    epochs=epochs_poison, lr=lr, device=device,
                                    poison=True, poison_frac=poison_frac,
                                    backdoor_target=backdoor_target)
    delta_poison = poison_vec - global_vec_cpu

    delta_attack = neurotoxin_combine(delta_benign, delta_poison,
                                      lcd_frac=0.9,  # 你已经改过
                                      eps=1e-12)
    return {"delta": delta_attack, "num_samples": len(dataloader.dataset)}


# -------- Client-side trust (与你原版一致) --------
client_states = defaultdict(dict)
# def client_compute_trust_local_only(theta_prev_vec, delta_i_vec, prev_delta_i_vec=None,
#                                     state=None, topk=None, quantile=0.99, hist_maxlen=50, eps=1e-12):
#     import math as _m, numpy as _np
#
#     def _cos(a, b):
#         na = torch.linalg.norm(a); nb = torch.linalg.norm(b)
#         if float(na) < eps or float(nb) < eps: return 0.0
#         return float(torch.dot(a, b) / (na * nb))
#
#     def _push(st, key, val):
#         if st is None: return
#         st.setdefault(key, []).append(float(val))
#         if len(st[key]) > hist_maxlen:
#             st[key] = st[key][-hist_maxlen:]
#
#     def _conf(val, hist):
#         if hist is None or len(hist) < 5 or _m.isnan(val) or _m.isinf(val):
#             return 1.0
#         h   = _np.asarray(hist, dtype=float)
#         med = _np.median(h)
#         mad = _np.median(_np.abs(h - med)) + eps
#         z   = 0.6745 * (val - med) / mad
#         za  = abs(z)
#         return float(1.0 / (1.0 + _m.exp(_np.clip(za, -5, 5))))
#
#     theta = theta_prev_vec.detach().cpu()
#     delta = delta_i_vec.detach().cpu()
#
#     tda   = _cos(delta, theta)
#     tda01 = (tda + 1.0) / 2.0
#
#     rel_l2 = float(torch.linalg.norm(delta) / (torch.linalg.norm(theta) + eps))
#
#     abs_delta = torch.abs(delta)
#     thr = torch.quantile(abs_delta, torch.tensor(quantile, dtype=delta.dtype)) if delta.numel() > 0 else 0
#     key_mask = (abs_delta >= thr) if delta.numel() > 0 else torch.zeros_like(delta, dtype=torch.bool)
#     spiky = float((torch.linalg.norm(delta[key_mask])**2 / (torch.linalg.norm(delta)**2 + eps)).item()) if key_mask.any() else 0.0
#
#     temporal = float('nan')
#     if prev_delta_i_vec is not None:
#         temporal = _cos(delta, prev_delta_i_vec.detach().cpu())
#     temporal01 = 0.5 if _np.isnan(temporal) else (temporal + 1.0) / 2.0
#
#     if topk is not None:
#         k   = max(1, min(int(topk), delta.numel()))
#         idx = torch.topk(abs_delta, k, largest=True).indices
#     else:
#         idx = key_mask.nonzero().flatten()
#         if idx.numel() == 0:
#             k   = max(1, min(1000, delta.numel()))
#             idx = torch.topk(abs_delta, k, largest=True).indices
#
#     d_sel = delta[idx]
#     t_sel = theta[idx]
#     w     = torch.abs(d_sel)
#     mask  = (w > 1e-6) & (torch.abs(t_sel) > 1e-6)
#     if mask.sum().item() == 0:
#         wsign = 0.5
#     else:
#         sd    = torch.sign(d_sel[mask])
#         st    = torch.sign(t_sel[mask])
#         agree = (sd == st).float()
#         wsign = float((agree * w[mask]).sum().item() / (w[mask].sum().item() + 1e-12))
#
#     # 仍然可以记录历史以备后用，但 trust 不用历史版公式
#     if state is not None:
#         _push(state, 'hist_tda01',      tda01)
#         _push(state, 'hist_rel_l2',     rel_l2)
#         _push(state, 'hist_spiky',      spiky)
#         if not _np.isnan(temporal01):
#             _push(state, 'hist_temporal01', temporal01)
#             _push(state, 'hist_wsign',      wsign)
#
#
#     # ---------- New trust scoring: 无阈值、无额外超参数 ----------
#     # rel_l2 越大越可疑 → 得分越小
#     rel_score = 1.0 / (1.0 + rel_l2)   # in (0,1]
#     # spiky 越大越“正常” → 得分越大
#     spky_score = float(spiky)          # already in [0,1]
#     # wsign 越大越“正常” → 得分越大
#     wsign_score = float(wsign)         # already in [0,1]
#
#     # 为了拉开差距，可以再做一次平方（可选，但推荐）
#     rel_score  = rel_score ** 2
#     spky_score = spky_score ** 2
#
#     w_tda, w_rel, w_spk, w_w = 0.0, 0.5, 0.3, 0.2
#     trust = (w_tda * tda01 +
#              w_rel * rel_score +
#              w_spk * spky_score +
#              w_w * wsign_score)
#     # clip 到 [0,1]
#     trust = float(max(0.0, min(1.0, trust)))
#
#     return {
#         "tda":       float(tda),
#         "tda01":     float(tda01),
#         "rel_l2":    float(rel_l2),
#         "spiky":     float(spiky),
#         "temporal":  float(0.0 if _np.isnan(temporal) else temporal),
#         "temporal01":float(temporal01),
#         "wsign":     float(wsign),
#         "trust":     float(trust),
#     }
def client_compute_trust_local_only(theta_prev_vec, delta_i_vec, prev_delta_i_vec=None,
                                    state=None, topk=None, quantile=0.99,
                                    hist_maxlen=50, eps=1e-12):
    """
    计算单个客户端的一轮 trust 相关指标 + trust 分数（带 r0/s0/alpha/gamma 的非线性放大版）

    返回:
        {
            "tda", "tda01",
            "rel_l2",
            "spiky",
            "temporal", "temporal01",
            "wsign",
            "trust",
        }
    """
    import math as _m
    import numpy as _np

    # --------- 这里是从 NUM_MALICIOUS=0 的实验里估出来的一组常数 ---------
    # relL2 的“正常上界” r0，超过就快速惩罚
    R0 = 0.023        # ≈ med_rel + 1 * MAD_rel
    ALPHA = 60.0      # 放大 relL2 超出 r0 时的惩罚

    # spiky 的“正常中心值” s0，以及 benign 的 MAD_spk
    S0 = 0.541        # ≈ med_spk
    MAD_SPK = 0.0275  # ≈ benign 的 MAD(spiky)
    GAMMA = 4.0       # spiky 偏离越多，惩罚越陡

    def _cos(a, b):
        na = torch.linalg.norm(a)
        nb = torch.linalg.norm(b)
        if float(na) < eps or float(nb) < eps:
            return 0.0
        return float(torch.dot(a, b) / (na * nb))

    def _push(st, key, val):
        if st is None:
            return
        st.setdefault(key, []).append(float(val))
        if len(st[key]) > hist_maxlen:
            st[key] = st[key][-hist_maxlen:]

    def _conf(val, hist):
        # 当前版本没在 trust 里用到，但保留接口方便以后扩展
        if hist is None or len(hist) < 5 or _m.isnan(val) or _m.isinf(val):
            return 1.0
        h   = _np.asarray(hist, dtype=float)
        med = _np.median(h)
        mad = _np.median(_np.abs(h - med)) + eps
        z   = 0.6745 * (val - med) / mad
        za  = abs(z)
        return float(1.0 / (1.0 + _m.exp(_np.clip(za, -5, 5))))

    # ---------------- 基础向量与原始指标 ----------------
    theta = theta_prev_vec.detach().cpu()
    delta = delta_i_vec.detach().cpu()

    # TDA：当前增量与全局参数的 cos
    tda   = _cos(delta, theta)
    tda01 = (tda + 1.0) / 2.0

    # relL2：‖Δ‖ / ‖θ‖
    rel_l2 = float(torch.linalg.norm(delta) / (torch.linalg.norm(theta) + eps))

    # spiky：取 top-quantile 的大幅度维，能量占比
    abs_delta = torch.abs(delta)
    thr = torch.quantile(abs_delta, torch.tensor(quantile, dtype=delta.dtype)) if delta.numel() > 0 else 0
    key_mask = (abs_delta >= thr) if delta.numel() > 0 else torch.zeros_like(delta, dtype=torch.bool)
    if key_mask.any():
        spiky = float((torch.linalg.norm(delta[key_mask]) ** 2 /
                       (torch.linalg.norm(delta) ** 2 + eps)).item())
    else:
        spiky = 0.0

    # temporal：与上一次本地增量的 cos
    temporal = float('nan')
    if prev_delta_i_vec is not None:
        temporal = _cos(delta, prev_delta_i_vec.detach().cpu())
    temporal01 = 0.5 if _np.isnan(temporal) else (temporal + 1.0) / 2.0

    # wsign：Δ 与 θ 在“大权重维度上”的符号一致性（加权）
    if topk is not None:
        k   = max(1, min(int(topk), delta.numel()))
        idx = torch.topk(abs_delta, k, largest=True).indices
    else:
        idx = key_mask.nonzero().flatten()
        if idx.numel() == 0:
            k   = max(1, min(1000, delta.numel()))
            idx = torch.topk(abs_delta, k, largest=True).indices

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
        wsign = float((agree * w[mask]).sum().item() /
                      (w[mask].sum().item() + 1e-12))

    # ---------------- 记录历史（可选） ----------------
    if state is not None:
        _push(state, 'hist_tda01',      tda01)
        _push(state, 'hist_rel_l2',     rel_l2)
        _push(state, 'hist_spiky',      spiky)
        if not _np.isnan(temporal01):
            _push(state, 'hist_temporal01', temporal01)
            _push(state, 'hist_wsign',      wsign)

    # ================= 新版 trust 计算 =================
    # ---- 1) relL2: 超过 r0 的部分用 tanh 强烈惩罚 ----
    pen_rel   = max(0.0, rel_l2 - R0)
    rel_score = 1.0 - float(_np.tanh(ALPHA * pen_rel))   # in (0,1]

    # ---- 2) spiky: 以 s0 为中心，偏离越多越可疑 ----
    delta_spk = abs(spiky - S0)
    x = delta_spk / (MAD_SPK + 1e-12)
    spky_score = 1.0 / (1.0 + x ** GAMMA)                # in (0,1]

    # ---- 3) wsign: 越接近 1 越正常 ----
    wsign_score = float(wsign)                           # already in [0,1]

    # ---- 4) TDA：保留，但目前权重设为 0，可随时打开 ----
    w_tda, w_rel, w_spk, w_w = 0.0, 0.45, 0.40, 0.15
    trust = (w_tda * tda01 +
             w_rel * rel_score +
             w_spk * spky_score +
             w_w   * wsign_score)

    trust = float(max(0.0, min(1.0, trust)))

    return {
        "tda":        float(tda),
        "tda01":      float(tda01),
        "rel_l2":     float(rel_l2),
        "spiky":      float(spiky),
        "temporal":   float(0.0 if _np.isnan(temporal) else temporal),
        "temporal01": float(temporal01),
        "wsign":      float(wsign),
        "trust":      float(trust),
    }


# ================= LGP for PS inspection (paper-faithful) =================
RS = defaultdict(int)   # reliability score, persistent over rounds

def _mad(v: np.ndarray, eps=1e-12):
    med = np.median(v); return med, np.median(np.abs(v - med)) + eps

def _kurtosis(x: np.ndarray, eps=1e-12):
    x = x.astype(np.float64); m = x.mean(); s2 = ((x-m)**2).mean() + eps
    return float(((x-m)**4).mean()/ (s2**2) - 3.0)

def _skewness(x: np.ndarray, eps=1e-12):
    x = x.astype(np.float64); m = x.mean(); s = np.sqrt(((x-m)**2).mean() + eps)
    return float(((x-m)**3).mean() / (s**3 + eps))

def _pairwise_mean_distance(mat: np.ndarray):
    # mat: [K, D]
    if len(mat) <= 1: return 0.0
    # compute upper triangle distances
    from itertools import combinations
    dsum, cnt = 0.0, 0
    for i, j in combinations(range(len(mat)), 2):
        dsum += np.linalg.norm(mat[i]-mat[j])
        cnt += 1
    return float(dsum/cnt)

def _cluster_stats(vectors: List[np.ndarray], ref: np.ndarray, member_ids: List[int]):
    if len(member_ids) == 0:
        return dict(Dmean=1e9, SDmean=1e9, Devmean=1e9, RSsum=0.0)
    mat = np.stack([vectors[i] for i in member_ids], axis=0)  # [k, d]
    Dmean  = _pairwise_mean_distance(mat)
    SDmean = float(np.std(mat, axis=0).mean())
    Devmean= float(np.abs(mat - ref[None,:]).sum(axis=1).mean())  # L1 对 ref 的平均绝对偏差
    RSsum  = float(sum(RS[i] for i in member_ids))
    return dict(Dmean=Dmean, SDmean=SDmean, Devmean=Devmean, RSsum=RSsum)

def _innocent_choose(cluster1_ids: List[int], cluster2_ids: List[int],
                     vectors: List[np.ndarray], ref: np.ndarray):
    c1 = _cluster_stats(vectors, ref, cluster1_ids)
    c2 = _cluster_stats(vectors, ref, cluster2_ids)
    # Eq.(7) 两条路任选其一 => 选C2
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
        # very small fallback: project to first SVD direction, threshold by median
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
    """
    对 suspicious 集合做：逐层 MAD -> 特征提取 -> AHC -> innocent criterion -> 汇总通过/拒绝
    返回：accepted_from_susp, deny_reasons (逐层)
    """
    # 逐层拆分 suspicious 的 Δ_i，以及参考 ref 的逐层
    susp_cids = [u["cid"] for u in suspicious_updates]
    deltas = [u["delta"].detach().cpu() for u in suspicious_updates]
    layers_per_client = [split_delta_by_layer(d, model) for d in deltas]
    ref_layers = split_delta_by_layer(ref_vec.detach().cpu(), model)
    L = len(ref_layers); K = len(suspicious_updates)

    # 逐层判定：记录每个客户端在该层是否“通过”
    layer_pass = np.zeros((K, L), dtype=bool)
    deny_details = {cid: {"mad_fail_layers": [], "ahc_non_honest_layers": []} for cid in susp_cids}

    # 辅助：构建 RS list 与打印
    # （RS 增加在汇总环节进行）
    # MAD 门限：使用 med(norm) ± MAD(norm)
    for l in range(L):
        glist = [layers_per_client[k][l].numpy().astype(np.float32) for k in range(K)]  # list of [D]
        norms = np.array([np.linalg.norm(g) for g in glist], dtype=np.float64)
        med, mad = _mad(norms); LB, UB = med - mad, med + mad

        cand1_idx = [i for i, n in enumerate(norms) if (n >= LB and n <= UB)]
        cand2_idx = [i for i in range(K) if i not in cand1_idx]  # complement
        # 若 cand2 非空且满足 innocent 判据，则替换
        choose_cand = cand1_idx
        if len(cand2_idx) > 0:
            # 用“向量本身”作为聚类输入的集合；innocent 只需要集合统计
            winner, c1, c2 = _innocent_choose(cand1_idx, cand2_idx, glist, ref_layers[l].numpy())
            if winner == 2:
                choose_cand = cand2_idx
        # 对 choose_cand 做 AHC 二分
        if len(choose_cand) == 0:
            # 无人通过 MAD：全部记 mad_fail
            for i in range(K): deny_details[susp_cids[i]]["mad_fail_layers"].append(l)
            continue

        # 提取每个候选的特征（与论文一致）
        feats = []
        for i in choose_cand:
            g = glist[i]; ref = ref_layers[l].numpy()
            PC = float((g > 0).sum()); NC = float((g < 0).sum()); ZC = float((g == 0).sum())
            kurt = _kurtosis(g); skew = _skewness(g)
            # Dmean: g 到其他候选的平均欧氏距离
            others = [glist[j] for j in choose_cand if j != i]
            if len(others) == 0: Dmean = 0.0
            else: Dmean = float(np.mean([np.linalg.norm(g - o) for o in others]))
            dev = float(np.abs(ref - g).sum())      # L1 与参考的绝对偏差
            L2  = float(np.linalg.norm(g))
            denom = (np.linalg.norm(g)*np.linalg.norm(ref) + 1e-12)
            dir_angle = float(np.arccos(np.clip((g@ref)/denom, -1.0, 1.0)))  # 弧度
            feats.append([PC, NC, ZC, kurt, skew, Dmean, dev, L2, dir_angle])
        feats = np.asarray(feats, dtype=np.float64)

        c1, c2 = _ahc_split(feats)
        # innocent criterion 在二分后选“诚实簇”
        # 注意：这里是对“候选索引”的二次选择
        winner, stats_c1, stats_c2 = _innocent_choose([choose_cand[i] for i in c1],
                                                      [choose_cand[i] for i in c2],
                                                      glist, ref_layers[l].numpy())
        honest_idx = [choose_cand[i] for i in (c2 if winner == 2 else c1)]
        # 标记本层通过
        for i in range(K):
            if i in honest_idx: layer_pass[i, l] = True
            else:
                # 若 i 原本就没通过 MAD，则 mad_fail 已记录；否则属于“落到非诚实簇”
                if i in choose_cand:
                    deny_details[susp_cids[i]]["ahc_non_honest_layers"].append(l)
                else:
                    deny_details[susp_cids[i]]["mad_fail_layers"].append(l)

    # 汇总：按层比例决定该 suspicious 客户端是否通过
    passed_ids = []
    for i in range(K):
        frac = float(layer_pass[i].mean()) if L > 0 else 0.0
        need = 0.4 if round_idx <= LGP_WARMUP_ROUNDS else LGP_PASS_FRAC
        if frac >= need:
            passed_ids.append(susp_cids[i])
            RS[susp_cids[i]] += 1   # 通过者 RS+1
    denied_ids = [cid for cid in susp_cids if cid not in passed_ids]

    # 打印逐层拒绝原因
    for cid in denied_ids:
        info = deny_details[cid]
        print(f"{log_prefix} cid={cid:2d} DENY | mad_fail_layers={info['mad_fail_layers']} "
              f"| ahc_non_honest_layers={info['ahc_non_honest_layers']} "
              f"| benign_layer_frac={1.0 - (len(info['mad_fail_layers'])+len(info['ahc_non_honest_layers']))/max(1,L):.2f}")

    return passed_ids, deny_details

# ---------------- Eval ----------------
@torch.no_grad()
def evaluate(model, test_loader, device):
    model.to(device).eval(); correct=0; total=0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item(); total += y.size(0)
    return 100.0 * correct / total

@torch.no_grad()
def evaluate_asr(model, test_loader, device, target, add_trigger_fn):
    model.to(device).eval(); ok=0; tot=0
    for x, y in test_loader:
        x_bd = x.clone()
        for k in range(x_bd.shape[0]): x_bd[k] = add_trigger_fn(x_bd[k])
        x_bd, y = x_bd.to(device), y.to(device)
        pred = model(x_bd).argmax(1)
        ok += (pred == target).sum().item(); tot += y.size(0)
    return 100.0 * ok / tot

# ---------------- Utilities ----------------
def init_global_model():
    # 使用 ResNet9 替代原来的 SimpleCNN
    m = SimpleCNN(num_classes=10).to(DEVICE)

    # Kaiming 初始化：Conv 和 Linear 分别处理
    def _init(mm):
        if isinstance(mm, nn.Conv2d):
            nn.init.kaiming_normal_(mm.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(mm, nn.Linear):
            nn.init.kaiming_normal_(mm.weight, nonlinearity="relu")
            if mm.bias is not None:
                nn.init.zeros_(mm.bias)

    m.apply(_init)
    return m


def make_client_loaders(train_ds, client_indices, batch_size):
    return [DataLoader(Subset(train_ds, idxs), batch_size=batch_size, shuffle=True) for idxs in client_indices]

def sample_malicious_ids(num_clients, num_malicious, seed=12345, randomized=True):
    if randomized:
        rng = np.random.default_rng(seed)
        return set(int(x) for x in rng.choice(num_clients, size=num_malicious, replace=False))
    return set(range(num_malicious))

def weighted_average(updates, key="delta", weight_key="num_samples"):
    total = sum(u[weight_key] for u in updates)
    agg = None
    for u in updates:
        w = u[weight_key] / total
        agg = w * u[key] if agg is None else agg + w * u[key]
    return agg, total

def weighted_average_tiered(updates, trusted_ids, suspicious_ids,
                            trusted_weight=1.0, suspicious_weight=0.5,
                            key="delta", weight_key="num_samples"):
    total = 0.0
    agg = None

    for u in updates:
        cid = u["cid"]

        if cid in trusted_ids:
            tier_w = trusted_weight
        elif cid in suspicious_ids:
            tier_w = suspicious_weight
        else:
            tier_w = 0.0   # denied clients do not participate

        w = tier_w * u[weight_key]
        if w <= 0:
            continue

        agg = w * u[key] if agg is None else agg + w * u[key]
        total += w

    if agg is None:
        D = updates[0][key].numel()
        agg = torch.zeros(D)

    agg = agg / max(total, 1e-12)
    return agg, total

def _get_rs(cid: int) -> int:
    st = client_states[cid]
    return int(st.get('RS', 0))

def _bump_rs(cid: int, by: int = 1):
    st = client_states[cid]
    st['RS'] = int(st.get('RS', 0)) + by
    client_states[cid] = st

# ---------------- One Round ----------------
def one_round(round_idx, global_model, client_loaders, malicious_ids,
              split=(0.50, 0.30, 0.20), verbose=True,
              trusted_weight=1.0, suspicious_weight=0.5):

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
            lambda_euc=(LAMBDA_EUC if malicious else 0.0)
        )

        state_i = client_states[cid]
        prev_delta = state_i.get('prev_delta')

        scores = client_compute_trust_local_only(
            theta_prev_vec, dd["delta"], prev_delta,
            state=state_i, topk=None, quantile=0.99, hist_maxlen=50
        )

        state_i['prev_delta'] = dd["delta"].clone()
        client_states[cid] = state_i

        updates.append({
            "cid": cid,
            "delta": dd["delta"],
            "num_samples": dd["num_samples"],
            "trust": scores["trust"],
            "tda": scores["tda"],
            "rel_l2": scores["rel_l2"],
            "spiky": scores["spiky"],
            "temporal": scores["temporal"],
            "wsign": scores["wsign"],
        })

    if verbose:
        for u in updates:
            label = "MAL" if u["cid"] in malicious_ids else "BEN"
            print(
                f"CID {u['cid']:2d} [{label}] | trust={u['trust']:.3f} | "
                f"TDA={u['tda']:+.3f} | relL2={u['rel_l2']:.3f} | "
                f"spiky={u['spiky']:.3f} | temporal={u['temporal']:+.3f} | "
                f"wsign={u['wsign']:.3f}"
            )

    global_vec = model_params_to_vector(global_model).cpu()

    # ----- split by trust -----
    t_trusted, t_susp, t_denied = split
    trusts = np.array([u["trust"] for u in updates], dtype=float)
    order = np.argsort(-trusts)

    nT = max(1, int(round(t_trusted * NUM_CLIENTS)))
    nD = max(1, int(round(t_denied * NUM_CLIENTS)))
    nS = max(1, NUM_CLIENTS - nT - nD)

    while nT + nS + nD > NUM_CLIENTS:
        if nS > 1:
            nS -= 1
        elif nD > 1:
            nD -= 1
        else:
            nT -= 1

    while nT + nS + nD < NUM_CLIENTS:
        nT += 1

    idx_trusted = order[:nT]
    idx_susp = order[nT:nT + nS]
    idx_denied = order[nT + nS:]

    trusted_updates = [updates[i] for i in idx_trusted]
    suspicious_updates = [updates[i] for i in idx_susp]
    denied_updates = [updates[i] for i in idx_denied]

    if verbose:
        print(f"[Split] trusted={len(trusted_updates)}, suspicious={len(suspicious_updates)}, denied={len(denied_updates)}")

    groups = dict(
        trusted=sorted([u["cid"] for u in trusted_updates]),
        suspicious=sorted([u["cid"] for u in suspicious_updates]),
        denied=sorted([u["cid"] for u in denied_updates]),
    )
    groups["final_participants"] = sorted(groups["trusted"] + groups["suspicious"])

    trusted_ids = set(groups["trusted"])
    suspicious_ids = set(groups["suspicious"])

    final_pool = [u for u in updates if u["cid"] in groups["final_participants"]]

    agg_final, _ = weighted_average_tiered(
        final_pool,
        trusted_ids=trusted_ids,
        suspicious_ids=suspicious_ids,
        trusted_weight=trusted_weight,
        suspicious_weight=suspicious_weight
    )

    final_global = copy.deepcopy(global_model)
    set_model_params_from_vector(final_global, global_vec + agg_final)

    print("\n=== Client Group Summary (this round) ===")
    print(f"Trusted (Top {int(split[0]*100)}%)        : {groups['trusted']}")
    print(f"Suspicious (Middle {int(split[1]*100)}%) : {groups['suspicious']}")
    print(f"Denied (Bottom {int(split[2]*100)}%)     : {groups['denied']}")
    print(f"Final aggregation participants           : {groups['final_participants']}")

    return dict(
        models=dict(final=final_global),
        groups=groups
    )

# ---------------- Multi-round ----------------
def simulate_many_rounds(num_rounds=10, split=(0.50, 0.30, 0.20), verbose_round=True,
                         print_models_eval_each_round=True,
                         trusted_weight=1.0, suspicious_weight=0.5):

    train_ds, test_ds = load_cifar10()
    client_indices = partition_dirichlet_balanced(train_ds, NUM_CLIENTS, DIRICHLET_ALPHA)
    client_loaders = make_client_loaders(train_ds, client_indices, BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    global_model = init_global_model()

    malicious_ids = sample_malicious_ids(
        NUM_CLIENTS, NUM_MALICIOUS,
        MALICIOUS_SEED, RANDOMIZE_MALICIOUS
    )
    print("Malicious client IDs (fixed across rounds):", sorted(list(malicious_ids)))

    history = dict(
        round=[],
        acc_final=[],
        asr_final=[],

        num_trusted=[],
        num_suspicious=[],
        num_denied=[],
        num_final_participants=[],
        num_selected_malicious=[],
        num_removed_benign=[],

        trusted_ids=[],
        suspicious_ids=[],
        denied_ids=[],
        final_participants_ids=[],
        selected_malicious_ids=[],
        removed_benign_ids=[]
    )

    for r in range(1, num_rounds + 1):
        print(f"\n========== Round {r} ==========")

        result = one_round(
            r, global_model, client_loaders, malicious_ids,
            split=split,
            verbose=verbose_round,
            trusted_weight=trusted_weight,
            suspicious_weight=suspicious_weight
        )

        models = result["models"]
        groups = result["groups"]
        global_model = models["final"]

        acc_final = evaluate(models["final"], test_loader, DEVICE)
        asr_final = evaluate_asr(
            models["final"], test_loader, DEVICE,
            BACKDOOR_TARGET, add_box_trigger
        )

        if print_models_eval_each_round:
            print("\n=== Model Evaluation (this round) ===")
            print(f"Tier-aware BEV-style aggregation | Acc: {acc_final:6.2f}% | ASR: {asr_final:6.2f}%")

        print(f"[Round {r}] Tier-aware BEV-style  Acc={acc_final:.2f}% | ASR={asr_final:.2f}%")

        trusted_ids = set(groups["trusted"])
        suspicious_ids = set(groups["suspicious"])
        denied_ids = set(groups["denied"])
        final_ids = set(groups["final_participants"])

        mal_set = set(malicious_ids)
        benign_set = set(range(NUM_CLIENTS)) - mal_set

        history["round"].append(r)
        history["acc_final"].append(acc_final)
        history["asr_final"].append(asr_final)

        history["num_trusted"].append(len(trusted_ids))
        history["num_suspicious"].append(len(suspicious_ids))
        history["num_denied"].append(len(denied_ids))
        history["num_final_participants"].append(len(final_ids))
        history["num_selected_malicious"].append(len(final_ids & mal_set))
        history["num_removed_benign"].append(len(denied_ids & benign_set))

        history["trusted_ids"].append(sorted(list(trusted_ids)))
        history["suspicious_ids"].append(sorted(list(suspicious_ids)))
        history["denied_ids"].append(sorted(list(denied_ids)))
        history["final_participants_ids"].append(sorted(list(final_ids)))
        history["selected_malicious_ids"].append(sorted(list(final_ids & mal_set)))
        history["removed_benign_ids"].append(sorted(list(denied_ids & benign_set)))

    return dict(
        final_model=global_model,
        history=history
    )


# ---------------- main ----------------
if __name__ == "__main__":
    import csv
    import json

    results = simulate_many_rounds(
        num_rounds=100,
        split=(0.50, 0.30, 0.20),
        verbose_round=True,
        print_models_eval_each_round=True,
        trusted_weight=1.0,
        suspicious_weight=0.5
    )

    FINAL_MODEL = results["final_model"]
    LOGS = results["history"]

    # =========================
    # 1) 保存数值指标到 CSV
    # =========================
    csv_path = "NeuBEVCIFAR.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "round",
            "acc_final", "asr_final",
            "num_trusted",
            "num_suspicious",
            "num_denied",
            "num_final_participants",
            "num_selected_malicious",
            "num_removed_benign"
        ])

        for i in range(len(LOGS["round"])):
            writer.writerow([
                LOGS["round"][i],
                LOGS["acc_final"][i],
                LOGS["asr_final"][i],
                LOGS["num_trusted"][i],
                LOGS["num_suspicious"][i],
                LOGS["num_denied"][i],
                LOGS["num_final_participants"][i],
                LOGS["num_selected_malicious"][i],
                LOGS["num_removed_benign"][i],
            ])

    print(f"\nSaved round metrics to: {csv_path}")

    # =========================
    # 2) 保存用户ID到 JSON
    # =========================
    json_path = "NeuBEVCIFAR.json"
    with open(json_path, "w") as f:
        json.dump({
            "round": LOGS["round"],
            "trusted_ids": LOGS["trusted_ids"],
            "suspicious_ids": LOGS["suspicious_ids"],
            "denied_ids": LOGS["denied_ids"],
            "final_participants_ids": LOGS["final_participants_ids"],
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
            f"Acc={LOGS['acc_final'][i]:6.2f}% | "
            f"ASR={LOGS['asr_final'][i]:6.2f}% || "
            f"Trusted={LOGS['num_trusted'][i]} | "
            f"Suspicious={LOGS['num_suspicious'][i]} | "
            f"Denied={LOGS['num_denied'][i]} | "
            f"FinalParticipants={LOGS['num_final_participants'][i]} | "
            f"SelectedMal={LOGS['num_selected_malicious'][i]} | "
            f"RemovedBenign={LOGS['num_removed_benign'][i]}"
        )