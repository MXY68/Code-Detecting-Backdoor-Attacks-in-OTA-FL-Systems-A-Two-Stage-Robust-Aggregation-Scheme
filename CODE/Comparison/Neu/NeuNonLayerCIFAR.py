

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


_HAVE_SK = True
try:
    from sklearn.cluster import AgglomerativeClustering
except Exception:
    _HAVE_SK = False
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

DIRICHLET_ALPHA  = 0.5
POISON_FRACTION  = 0.2
BACKDOOR_TARGET  = 0
BENIGN_CLEAN_RATIO = 1.0

RANDOMIZE_MALICIOUS = True
MALICIOUS_SEED = 12345

# RS & LGP hyperparams
LGP_WARMUP_ROUNDS = 30
LAMBDA_EUC       = 0.1
LGP_PASS_FRAC      = 0.6
APPLY_RS_HARD_THRESHOLD_IN_FINAL = True

# ------------- Model -------------
class SimpleCNN(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()

        def Norm(c):

            return nn.GroupNorm(num_groups=8, num_channels=c)


        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            Norm(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            Norm(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2)  # 32x32 -> 16x16
        )


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


        self.gap = nn.AdaptiveAvgPool2d((1, 1))


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
                       alpha: float = 1,
                       scale_attack: float = 1,
                       eps: float = 1e-12):
    assert delta_benign.shape == delta_poison.shape
    v = delta_benign
    numel = v.numel()

    k = max(1, int(lcd_frac * numel))
    abs_v = v.abs()


    lcd_idx = torch.topk(abs_v, k, largest=False).indices


    delta_nt = delta_benign.clone()
    delta_nt[lcd_idx] = delta_poison[lcd_idx]

    delta_attack = (1 - alpha) * delta_benign + alpha * delta_nt


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
    epochs_clean = 1
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
                                      lcd_frac=0.9,
                                      alpha = 1,
                                      scale_attack = 1,
                                      eps=1e-12)
    return {"delta": delta_attack, "num_samples": len(dataloader.dataset)}


# -------- Client-side trust --------
client_states = defaultdict(dict)

def client_compute_trust_local_only(theta_prev_vec, delta_i_vec, prev_delta_i_vec=None,
                                    state=None, topk=None, quantile=0.99,
                                    hist_maxlen=50, eps=1e-12):

    import math as _m
    import numpy as _np


    R0 = 0.023
    ALPHA = 60.0


    S0 = 0.541
    MAD_SPK = 0.0275
    GAMMA = 4.0

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

        if hist is None or len(hist) < 5 or _m.isnan(val) or _m.isinf(val):
            return 1.0
        h   = _np.asarray(hist, dtype=float)
        med = _np.median(h)
        mad = _np.median(_np.abs(h - med)) + eps
        z   = 0.6745 * (val - med) / mad
        za  = abs(z)
        return float(1.0 / (1.0 + _m.exp(_np.clip(za, -5, 5))))


    theta = theta_prev_vec.detach().cpu()
    delta = delta_i_vec.detach().cpu()


    tda   = _cos(delta, theta)
    tda01 = (tda + 1.0) / 2.0


    rel_l2 = float(torch.linalg.norm(delta) / (torch.linalg.norm(theta) + eps))


    abs_delta = torch.abs(delta)
    thr = torch.quantile(abs_delta, torch.tensor(quantile, dtype=delta.dtype)) if delta.numel() > 0 else 0
    key_mask = (abs_delta >= thr) if delta.numel() > 0 else torch.zeros_like(delta, dtype=torch.bool)
    if key_mask.any():
        spiky = float((torch.linalg.norm(delta[key_mask]) ** 2 /
                       (torch.linalg.norm(delta) ** 2 + eps)).item())
    else:
        spiky = 0.0


    temporal = float('nan')
    if prev_delta_i_vec is not None:
        temporal = _cos(delta, prev_delta_i_vec.detach().cpu())
    temporal01 = 0.5 if _np.isnan(temporal) else (temporal + 1.0) / 2.0


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


    if state is not None:
        _push(state, 'hist_tda01',      tda01)
        _push(state, 'hist_rel_l2',     rel_l2)
        _push(state, 'hist_spiky',      spiky)
        if not _np.isnan(temporal01):
            _push(state, 'hist_temporal01', temporal01)
            _push(state, 'hist_wsign',      wsign)



    pen_rel   = max(0.0, rel_l2 - R0)
    rel_score = 1.0 - float(_np.tanh(ALPHA * pen_rel))   # in (0,1]


    delta_spk = abs(spiky - S0)
    x = delta_spk / (MAD_SPK + 1e-12)
    spky_score = 1.0 / (1.0 + x ** GAMMA)                # in (0,1]


    wsign_score = float(wsign)                           # already in [0,1]


    w_tda, w_rel, w_spk, w_w = 0.15, 0.45, 0.40, 0
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


# ================= LGP for PS inspection  =================
RS = defaultdict(int)

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

def ps_inspect_whole_model(model: nn.Module,
                           suspicious_updates: List[dict],
                           ref_vec: torch.Tensor,
                           round_idx: int,
                           log_prefix: str = "[PS-Whole]"):

    susp_cids = [u["cid"] for u in suspicious_updates]
    deltas = [u["delta"].detach().cpu().numpy().astype(np.float32) for u in suspicious_updates]
    ref = ref_vec.detach().cpu().numpy().astype(np.float32)
    K = len(suspicious_updates)


    if K == 0:
        return [], {}

    if K == 1:
        cid = susp_cids[0]
        RS[cid] += 1
        return [cid], {
            cid: {
                "mad_pass": True,
                "in_candidate_set": True,
                "cluster_honest": True,
                "reason": "single_suspicious_client_auto_pass"
            }
        }

    deny_details = {
        cid: {
            "mad_pass": False,
            "in_candidate_set": False,
            "cluster_honest": False,
            "reason": ""
        }
        for cid in susp_cids
    }

    # -------------------------------------------------
    # Step 1: no MAD filtering
    # all suspicious clients go directly into clustering
    # -------------------------------------------------
    candidate_idx = list(range(K))

    for i in candidate_idx:
        deny_details[susp_cids[i]]["mad_pass"] = True
        deny_details[susp_cids[i]]["in_candidate_set"] = True

    # -------------------------------------------------
    # Step 2: only use 2 whole-model features
    #         [dev, dir_angle]
    # -------------------------------------------------
    feats = []
    for i in candidate_idx:
        g = deltas[i]

        dev = float(np.abs(ref - g).sum())  # L1 deviation to trusted reference

        denom = (np.linalg.norm(g) * np.linalg.norm(ref) + 1e-12)
        dir_angle = float(np.arccos(np.clip((g @ ref) / denom, -1.0, 1.0)))

        feats.append([dev, dir_angle])

    feats = np.asarray(feats, dtype=np.float64)

    # -------------------------------------------------
    # Step 3: AHC split without innocent criterion
    #         choose the larger cluster as honest
    # -------------------------------------------------
    if len(candidate_idx) == 1:
        honest_idx = [candidate_idx[0]]
    else:
        c1, c2 = _ahc_split(feats)

        cluster1_global = [candidate_idx[i] for i in c1]
        cluster2_global = [candidate_idx[i] for i in c2]


        if len(cluster1_global) > len(cluster2_global):
            honest_idx = cluster1_global
        elif len(cluster2_global) > len(cluster1_global):
            honest_idx = cluster2_global
        else:
            dev1 = np.mean([feats[c1_i][0] for c1_i in c1]) if len(c1) > 0 else np.inf
            dev2 = np.mean([feats[c2_i][0] for c2_i in c2]) if len(c2) > 0 else np.inf
            honest_idx = cluster1_global if dev1 <= dev2 else cluster2_global

    # -------------------------------------------------
    # Step 4: final accept / deny
    # -------------------------------------------------
    passed_ids = []
    for i in range(K):
        cid = susp_cids[i]
        if i in honest_idx:
            deny_details[cid]["cluster_honest"] = True
            deny_details[cid]["reason"] = "accepted_in_larger_cluster"
            passed_ids.append(cid)
            RS[cid] += 1
        else:
            if not deny_details[cid]["in_candidate_set"]:
                deny_details[cid]["reason"] = "filtered_by_relaxed_mad"
            else:
                deny_details[cid]["reason"] = "assigned_to_smaller_cluster"

    denied_ids = [cid for cid in susp_cids if cid not in passed_ids]
    for cid in denied_ids:
        info = deny_details[cid]
        print(
            f"{log_prefix} cid={cid:2d} DENY | "
            f"mad_pass={info['mad_pass']} | "
            f"in_candidate_set={info['in_candidate_set']} | "
            f"cluster_honest={info['cluster_honest']} | "
            f"reason={info['reason']}"
        )

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

    m = SimpleCNN(num_classes=10).to(DEVICE)


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

def _get_rs(cid: int) -> int:
    st = client_states[cid]
    return int(st.get('RS', 0))

def _bump_rs(cid: int, by: int = 1):
    st = client_states[cid]
    st['RS'] = int(st.get('RS', 0)) + by
    client_states[cid] = st

# ---------------- One Round ----------------
def one_round(round_idx, global_model, client_loaders, malicious_ids,
              split=(0.50, 0.30, 0.20), verbose=True, use_ema=False, beta_ema=0.90, ref_delta_ema=None):
    # print(f"\n========== Round {round_idx} ==========")
    theta_prev_vec = model_params_to_vector(global_model).detach().cpu()

    updates = []
    for cid in range(NUM_CLIENTS):
        local_model = copy.deepcopy(global_model)
        malicious = (cid in malicious_ids)
        loader = client_loaders[cid]
        dd = local_train(local_model, global_model, loader,
                         epochs=LOCAL_EPOCHS, lr=LR, device=DEVICE,
                         malicious=malicious, poison_frac=POISON_FRACTION,
                         backdoor_target=BACKDOOR_TARGET, lambda_euc=(LAMBDA_EUC if malicious else 0.0))
        state_i = client_states[cid]; prev_delta = state_i.get('prev_delta')
        scores = client_compute_trust_local_only(theta_prev_vec, dd["delta"], prev_delta, state=state_i,
                                                 topk=None, quantile=0.99, hist_maxlen=50)
        state_i['prev_delta'] = dd["delta"].clone(); client_states[cid] = state_i
        updates.append({"cid": cid, "delta": dd["delta"], "num_samples": dd["num_samples"], "trust": scores["trust"],
                        "tda": scores["tda"], "rel_l2": scores["rel_l2"], "spiky": scores["spiky"],
                        "temporal": scores["temporal"], "wsign": scores["wsign"]})
    if verbose:
        for u in updates:
            label = "MAL" if u["cid"] in malicious_ids else "BEN"
            print(f"CID {u['cid']:2d} [{label}] | trust={u['trust']:.3f} | TDA={u['tda']:+.3f} | relL2={u['rel_l2']:.3f} "
                  f"| spiky={u['spiky']:.3f} | temporal={u['temporal']:+.3f} | wsign={u['wsign']:.3f}")

    agg_all, _ = weighted_average(updates)
    benign_updates = [u for u in updates if u["cid"] not in malicious_ids]
    agg_benign, _ = weighted_average(benign_updates)
    global_vec = model_params_to_vector(global_model).cpu()

    contaminated_global = copy.deepcopy(global_model); set_model_params_from_vector(contaminated_global, global_vec + agg_all)
    clean_global        = copy.deepcopy(global_model); set_model_params_from_vector(clean_global,        global_vec + agg_benign)

    # ----- split by trust -----
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
    while nT + nS + nR < NUM_CLIENTS: nT += 1

    idx_trusted  = order[:nT]
    idx_susp     = order[nT:nT+nS]
    idx_rejected = order[nT+nS:]

    trusted_updates    = [updates[i] for i in idx_trusted]
    suspicious_updates = [updates[i] for i in idx_susp]
    rejected_updates   = [updates[i] for i in idx_rejected]
    if verbose:
        print(f"[Split] trusted={len(trusted_updates)}, suspicious={len(suspicious_updates)}, rejected={len(rejected_updates)}")

    # ----- trusted aggregate as reference (current round) -----
    agg_trusted, total_trusted = weighted_average(trusted_updates)
    trusted_global = copy.deepcopy(global_model); set_model_params_from_vector(trusted_global, global_vec + agg_trusted)
    delta_trusted = agg_trusted.detach().cpu()
    if use_ema and ref_delta_ema is not None:
        ref_vec = (beta_ema * ref_delta_ema + (1.0 - beta_ema) * delta_trusted).detach().cpu()
    else:
        ref_vec = delta_trusted.clone()

    # ----- PS inspection on suspicious via LGP -----
    acc_ids, deny_details = ps_inspect_whole_model(global_model, suspicious_updates, ref_vec, round_idx)

    # group IDs for logging
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


    groups["final_participants_afterRS"] = groups["final_participants"].copy()
    groups["removed_by_rs"] = []

    if len(RS) > 0:
        rs_items = sorted(
            [(cid, RS.get(cid, 0)) for cid in range(NUM_CLIENTS)],
            key=lambda x: (-x[1], x[0])
        )
        rs_vals = np.array([v for _, v in rs_items], dtype=float)
        rs_med, rs_mad = _mad(rs_vals) if len(rs_vals) else np.array([0.0])
        rs_thr = rs_med - rs_mad


        groups["rs_items"] = rs_items
        groups["rs_med"]   = rs_med
        groups["rs_mad"]   = rs_mad
        groups["rs_thr"]   = rs_thr


        if APPLY_RS_HARD_THRESHOLD_IN_FINAL and round_idx > LGP_WARMUP_ROUNDS:
            before_rs = groups["final_participants"].copy()
            after_rs = [
                cid for cid in groups["final_participants"]
                if RS[cid] >= rs_thr
            ]
            groups["final_participants_afterRS"] = after_rs
            groups["removed_by_rs"] = sorted(list(set(before_rs) - set(after_rs)))

    # ----- Final aggregate: trusted + accepted suspicious (after optional RS cut) -----
    final_pool = [u for u in updates if u["cid"] in groups["final_participants_afterRS"]]
    if len(final_pool) == 0:
        final_pool = trusted_updates

    agg_final, _ = weighted_average(final_pool)
    final_global = copy.deepcopy(global_model)
    set_model_params_from_vector(final_global, global_vec + agg_final)


    for cid in groups["final_participants_afterRS"]:
        RS[cid] = RS.get(cid, 0) + 1

    if len(RS) > 0:
        rs_items = sorted([(cid, RS.get(cid, 0)) for cid in range(NUM_CLIENTS)],
                          key=lambda x: (-x[1], x[0]))
        rs_vals = np.array([v for _, v in rs_items], dtype=float)
        rs_med, rs_mad = _mad(rs_vals) if len(rs_vals) else np.array([0.0])
        rs_thr = rs_med - rs_mad
        print("\n[RS] cid->RS (sorted):", rs_items)
        print(f"[RS] median={rs_med:.2f}, MAD={rs_mad:.2f}, threshold (med-MAD)={rs_thr:.2f}")




    print("\n=== Client Group Summary (this round) ===")
    print(f"Trusted (Top {int(split[0]*100)}%, OTA aggregated)        : {groups['trusted']}")
    print(f"Suspicious (middle {int(split[1]*100)}%, individual access): {groups['suspicious']}")
    print(f"Rejected initially (trust phase)                      : {groups['rejected']}")
    print(f"Suspicious accepted after inspection                  : {groups['accepted_from_susp']}")
    print(f"Suspicious denied after inspection                    : {groups['denied_after_ps']}")
    print(f"Denied overall (trust + PS)                           : {groups['denied_overall']}")
    print(f"[RS Filter] Removed by RS threshold                   : {groups['removed_by_rs']}")
    print(f"Final aggregation participants (after RS)             : {groups['final_participants_afterRS']}")

    return dict(models=dict(contaminated=contaminated_global, clean=clean_global,
                            trusted=trusted_global, final=final_global),
                groups=groups,
                aux=dict(ref_delta_ema=ref_vec.clone()))

# ---------------- Multi-round ----------------
def simulate_many_rounds(num_rounds=10, split=(0.50,0.30,0.20), verbose_round=True,
                         print_models_eval_each_round=True, use_ema=False):
    train_ds, test_ds = load_cifar10()
    client_indices = partition_dirichlet_balanced(train_ds, NUM_CLIENTS, DIRICHLET_ALPHA)
    client_loaders = make_client_loaders(train_ds, client_indices, BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)


    global_model_def = init_global_model()

    malicious_ids = sample_malicious_ids(NUM_CLIENTS, NUM_MALICIOUS,
                                         MALICIOUS_SEED, RANDOMIZE_MALICIOUS)
    print("Malicious client IDs (fixed across rounds):", sorted(list(malicious_ids)))

    ref_delta_ema = None
    history = dict(
        round=[],
        acc_final=[], asr_final=[],


        num_final_participants=[],
        num_denied_overall=[],
        num_removed_by_rs=[],
        num_selected_malicious=[],
        num_removed_benign=[],


        final_participants_ids=[],
        denied_overall_ids=[],
        removed_by_rs_ids=[],
        selected_malicious_ids=[],
        removed_benign_ids=[]
    )

    for r in range(1, num_rounds+1):
        print(f"\n========== Round {r} ==========")

        result = one_round(r, global_model_def, client_loaders,
                           malicious_ids, split=split,
                           verbose=verbose_round,
                           use_ema=use_ema, ref_delta_ema=ref_delta_ema)

        ref_delta_ema = result["aux"]["ref_delta_ema"]
        models = result["models"]
        groups = result["groups"]
        global_model_def = models["final"]


        acc_final = evaluate(models["final"], test_loader, DEVICE)
        asr_final = evaluate_asr(models["final"], test_loader, DEVICE,
                                 BACKDOOR_TARGET, add_box_trigger)

        if print_models_eval_each_round:
            print("\n=== Model Evaluation (this round) ===")
            print(f"Final (whole-model inspection) | Acc: {acc_final:6.2f}% | ASR: {asr_final:6.2f}%")

        print(f"[Round {r}] Whole-model inspection | Acc={acc_final:.2f}% | ASR={asr_final:.2f}%")

        history["round"].append(r)
        history["acc_final"].append(acc_final)
        history["asr_final"].append(asr_final)


        final_ids = set(groups["final_participants_afterRS"])
        denied_ids = set(groups["denied_overall"])
        removed_by_rs_ids = set(groups["removed_by_rs"])
        mal_set = set(malicious_ids)
        benign_set = set(range(NUM_CLIENTS)) - mal_set


        history["num_final_participants"].append(len(final_ids))
        history["num_denied_overall"].append(len(denied_ids))
        history["num_removed_by_rs"].append(len(removed_by_rs_ids))
        history["num_selected_malicious"].append(len(final_ids & mal_set))
        history["num_removed_benign"].append(len(denied_ids & benign_set))


        history["final_participants_ids"].append(sorted(list(final_ids)))
        history["denied_overall_ids"].append(sorted(list(denied_ids)))
        history["removed_by_rs_ids"].append(sorted(list(removed_by_rs_ids)))
        history["selected_malicious_ids"].append(sorted(list(final_ids & mal_set)))
        history["removed_benign_ids"].append(sorted(list(denied_ids & benign_set)))

    return dict(
        defended_model=global_model_def,
        history=history
    )



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


    csv_path = "NeuCIAFRCNN556508.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "round",
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
                LOGS["acc_final"][i],
                LOGS["asr_final"][i],
                LOGS["num_final_participants"][i],
                LOGS["num_denied_overall"][i],
                LOGS["num_removed_by_rs"][i],
                LOGS["num_selected_malicious"][i],
                LOGS["num_removed_benign"][i],
            ])

    print(f"\nSaved round metrics to: {csv_path}")


    json_path = "NeuCIAFRCNN556508.json"
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


    print("\n=== Multi-round Summary ===")
    for i in range(len(LOGS["round"])):
        print("\n=== Multi-round Summary ===")
        for i in range(len(LOGS["round"])):
            print(
                f"Round {LOGS['round'][i]:02d}: "
                f"WholeModel Acc={LOGS['acc_final'][i]:6.2f}% | "
                f"ASR={LOGS['asr_final'][i]:6.2f}% || "
                f"FinalParticipants={LOGS['num_final_participants'][i]} | "
                f"SelectedMal={LOGS['num_selected_malicious'][i]} | "
                f"RemovedBenign={LOGS['num_removed_benign'][i]}"
            )