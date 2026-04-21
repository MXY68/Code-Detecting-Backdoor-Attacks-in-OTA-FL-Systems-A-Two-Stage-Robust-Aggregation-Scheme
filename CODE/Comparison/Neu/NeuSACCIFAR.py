

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


NUM_GROUPS = 5
BASELINE_WARMUP_ROUNDS = 5

GROUP_SCORE_WEIGHTS = (0.7, 0.05, 0.25)
GROUPS_TO_KEEP = 3

REP_ACCEPT_REWARD = 1.0
REP_REJECT_REWARD = 0.0

# baseline persistent states
REPUTATION = defaultdict(float)   # client-level historical reputation
GROUP_PREV_UPDATES = {}           # gid -> previous round's group update

# ------------- Model -------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        def Norm(c):

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
    L = len(ref_layers); K = len(suspicious_updates)


    layer_pass = np.zeros((K, L), dtype=bool)
    deny_details = {cid: {"mad_fail_layers": [], "ahc_non_honest_layers": []} for cid in susp_cids}


    for l in range(L):
        glist = [layers_per_client[k][l].numpy().astype(np.float32) for k in range(K)]  # list of [D]
        norms = np.array([np.linalg.norm(g) for g in glist], dtype=np.float64)
        med, mad = _mad(norms); LB, UB = med - mad, med + mad

        cand1_idx = [i for i, n in enumerate(norms) if (n >= LB and n <= UB)]
        cand2_idx = [i for i in range(K) if i not in cand1_idx]  # complement

        choose_cand = cand1_idx
        if len(cand2_idx) > 0:

            winner, c1, c2 = _innocent_choose(cand1_idx, cand2_idx, glist, ref_layers[l].numpy())
            if winner == 2:
                choose_cand = cand2_idx

        if len(choose_cand) == 0:

            for i in range(K): deny_details[susp_cids[i]]["mad_fail_layers"].append(l)
            continue


        feats = []
        for i in choose_cand:
            g = glist[i]; ref = ref_layers[l].numpy()
            PC = float((g > 0).sum()); NC = float((g < 0).sum()); ZC = float((g == 0).sum())
            kurt = _kurtosis(g); skew = _skewness(g)

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

        winner, stats_c1, stats_c2 = _innocent_choose([choose_cand[i] for i in c1],
                                                      [choose_cand[i] for i in c2],
                                                      glist, ref_layers[l].numpy())
        honest_idx = [choose_cand[i] for i in (c2 if winner == 2 else c1)]

        for i in range(K):
            if i in honest_idx: layer_pass[i, l] = True
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
            RS[susp_cids[i]] += 1   # 通过者 RS+1
    denied_ids = [cid for cid in susp_cids if cid not in passed_ids]


    for cid in denied_ids:
        info = deny_details[cid]
        print(f"{log_prefix} cid={cid:2d} DENY | mad_fail_layers={info['mad_fail_layers']} "
              f"| ahc_non_honest_layers={info['ahc_non_honest_layers']} "
              f"| benign_layer_frac={1.0 - (len(info['mad_fail_layers'])+len(info['ahc_non_honest_layers']))/max(1,L):.2f}")

    return passed_ids, deny_details


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

def make_ps_reference_loader(root="./data", batch_size=128, num_samples=1000, seed=2026):

    transform_ref = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    ref_ds = datasets.CIFAR10(root, train=True, download=True, transform=transform_ref)
    rng = np.random.default_rng(seed)
    all_idx = np.arange(len(ref_ds))
    choose = rng.choice(all_idx, size=min(num_samples, len(ref_ds)), replace=False)
    return DataLoader(Subset(ref_ds, choose.tolist()), batch_size=batch_size, shuffle=False)


def compute_reference_update(global_model, ref_loader, device, lr=LR, max_batches=1):

    model = copy.deepcopy(global_model).to(device)
    model.train()

    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.85, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()

    before = model_params_to_vector(global_model).detach().cpu()

    used = 0
    for x, y in ref_loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = ce(model(x), y)
        loss.backward()
        opt.step()
        used += 1
        if used >= max_batches:
            break

    after = model_params_to_vector(model).detach().cpu()
    return after - before


def random_grouping(num_clients, num_groups, seed=None):

    ids = list(range(num_clients))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)

    groups = []
    base = num_clients // num_groups
    rem = num_clients % num_groups
    start = 0
    for g in range(num_groups):
        size = base + (1 if g < rem else 0)
        groups.append(ids[start:start+size])
        start += size
    return groups


def reputation_grouping(num_clients, num_groups, reputation_dict):

    ids = list(range(num_clients))
    ids.sort(key=lambda cid: (-float(reputation_dict[cid]), cid))

    groups = []
    base = num_clients // num_groups
    rem = num_clients % num_groups
    start = 0
    for g in range(num_groups):
        size = base + (1 if g < rem else 0)
        groups.append(ids[start:start+size])
        start += size
    return groups


def cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps=1e-12):
    a = a.detach().cpu().view(-1)
    b = b.detach().cpu().view(-1)
    na = torch.linalg.norm(a)
    nb = torch.linalg.norm(b)
    if float(na) < eps or float(nb) < eps:
        return 0.0
    return float(torch.dot(a, b) / (na * nb + eps))


def compute_group_score(group_update: torch.Tensor,
                        ref_update: torch.Tensor,
                        prev_group_update: Optional[torch.Tensor] = None,
                        weights=(0.7, 0.05, 0.25),
                        eps=1e-12):

    w_cos, w_norm, w_temp = weights

    s_cos = cosine_similarity(group_update, ref_update)

    gnorm = float(torch.linalg.norm(group_update))
    rnorm = float(torch.linalg.norm(ref_update))
    norm_ratio = gnorm / (rnorm + eps)
    norm_dev = abs(gnorm - rnorm) / (gnorm + rnorm + eps)

    if prev_group_update is None:
        s_temp = 0.0
    else:
        s_temp = cosine_similarity(group_update, prev_group_update)

    score = w_cos * s_cos - w_norm * norm_dev + w_temp * s_temp
    return {
        "score": float(score),
        "cosine": float(s_cos),
        "norm_ratio": float(norm_ratio),
        "norm_dev": float(norm_dev),
        "temporal": float(s_temp),
    }

def _get_rs(cid: int) -> int:
    st = client_states[cid]
    return int(st.get('RS', 0))

def _bump_rs(cid: int, by: int = 1):
    st = client_states[cid]
    st['RS'] = int(st.get('RS', 0)) + by
    client_states[cid] = st

def one_round_fedsac_baseline(round_idx,
                              global_model,
                              client_loaders,
                              malicious_ids,
                              ref_loader,
                              verbose=True):

    global REPUTATION, GROUP_PREV_UPDATES

    global_vec = model_params_to_vector(global_model).detach().cpu()


    updates = []
    for cid in range(NUM_CLIENTS):
        local_model = copy.deepcopy(global_model)
        malicious = (cid in malicious_ids)
        loader = client_loaders[cid]

        dd = local_train(local_model, global_model, loader,
                         epochs=LOCAL_EPOCHS, lr=LR, device=DEVICE,
                         malicious=malicious,
                         poison_frac=POISON_FRACTION,
                         backdoor_target=BACKDOOR_TARGET,
                        )

        updates.append({
            "cid": cid,
            "delta": dd["delta"].detach().cpu(),
            "num_samples": dd["num_samples"]
        })

    if verbose:
        print(f"\n[FedSAC-baseline] Round {round_idx}")
        print("Current reputation:",
              [(cid, round(float(REPUTATION[cid]), 2)) for cid in range(NUM_CLIENTS)])

    # --------------------------------------------------
    # Step 2: grouping
    # --------------------------------------------------
    if round_idx <= BASELINE_WARMUP_ROUNDS:
        groups = random_grouping(NUM_CLIENTS, NUM_GROUPS, seed=SEED + round_idx)
        grouping_mode = "random-warmup"
    else:
        groups = reputation_grouping(NUM_CLIENTS, NUM_GROUPS, REPUTATION)
        grouping_mode = "reputation-guided"

    if verbose:
        print(f"[Grouping mode] {grouping_mode}")
        for gid, members in enumerate(groups):
            print(f"  Group {gid}: {members}")

    update_by_cid = {u["cid"]: u for u in updates}

    # --------------------------------------------------
    # Step 3: reference update at PS
    # --------------------------------------------------
    ref_delta = compute_reference_update(global_model, ref_loader, DEVICE, lr=LR, max_batches=1)

    # --------------------------------------------------
    # Step 4: group aggregation + scoring
    # --------------------------------------------------
    group_infos = []
    for gid, members in enumerate(groups):
        group_updates = [update_by_cid[cid] for cid in members]
        agg_group, _ = weighted_average(group_updates)
        agg_group = agg_group.detach().cpu()

        prev_group_update = GROUP_PREV_UPDATES.get(gid, None)
        stat = compute_group_score(
            agg_group,
            ref_delta,
            prev_group_update=prev_group_update,
            weights=GROUP_SCORE_WEIGHTS
        )

        group_infos.append({
            "gid": gid,
            "members": members,
            "group_update": agg_group,
            "accepted": False,
            "score": stat["score"],
            "cosine": stat["cosine"],
            "norm_ratio": stat["norm_ratio"],
            "norm_dev": stat["norm_dev"],
            "temporal": stat["temporal"],
        })

    # --------------------------------------------------
    # Step 5: keep top-K groups by score
    # --------------------------------------------------
    group_infos_sorted = sorted(group_infos, key=lambda x: x["score"], reverse=True)

    k_keep = min(GROUPS_TO_KEEP, len(group_infos_sorted))
    accepted_groups = group_infos_sorted[:k_keep]
    rejected_groups = group_infos_sorted[k_keep:]

    accepted_gid_set = set(g["gid"] for g in accepted_groups)
    for g in group_infos:
        g["accepted"] = (g["gid"] in accepted_gid_set)

    # --------------------------------------------------
    # Step 6: final aggregation across accepted groups
    # --------------------------------------------------
    accepted_group_updates = [
        {"delta": g["group_update"], "num_samples": 1}
        for g in accepted_groups
    ]
    agg_final, _ = weighted_average(accepted_group_updates)

    new_global = copy.deepcopy(global_model)
    set_model_params_from_vector(new_global, global_vec + agg_final)

    # --------------------------------------------------
    # Step 7: update reputation
    # --------------------------------------------------
    for g in accepted_groups:
        for cid in g["members"]:
            REPUTATION[cid] += REP_ACCEPT_REWARD

    for g in rejected_groups:
        for cid in g["members"]:
            REPUTATION[cid] += REP_REJECT_REWARD

    # --------------------------------------------------
    # Step 8: update group temporal memory
    # --------------------------------------------------
    GROUP_PREV_UPDATES = {
        g["gid"]: g["group_update"].clone().detach().cpu()
        for g in group_infos
    }

    # --------------------------------------------------
    # Step 9: logging
    # --------------------------------------------------
    if verbose:
        print("\n=== Group Summary (FedSAC-style baseline) ===")
        for g in sorted(group_infos, key=lambda x: x["gid"]):
            tag = "KEEP" if g["accepted"] else "DROP"
            print(f"Group {g['gid']} [{tag}] | members={g['members']} "
                  f"| score={g['score']:.4f} | cos={g['cosine']:.4f} "
                  f"| norm_ratio={g['norm_ratio']:.4f} | temp={g['temporal']:.4f}")

        print("Accepted groups :", [g["gid"] for g in accepted_groups])
        print("Rejected groups :", [g["gid"] for g in rejected_groups])

        selected_clients = sorted([cid for g in accepted_groups for cid in g["members"]])
        print("Selected clients:", selected_clients)

        print("[Updated reputation]",
              [(cid, round(float(REPUTATION[cid]), 2)) for cid in range(NUM_CLIENTS)])

    selected_clients = sorted([cid for g in accepted_groups for cid in g["members"]])
    rejected_clients = sorted([cid for g in rejected_groups for cid in g["members"]])

    return {
        "model": new_global,
        "groups": {
            "grouping_mode": grouping_mode,
            "accepted_group_ids": [g["gid"] for g in accepted_groups],
            "rejected_group_ids": [g["gid"] for g in rejected_groups],
            "selected_clients": selected_clients,
            "rejected_clients": rejected_clients,
            "group_infos": group_infos,
        }
    }

# ---------------- Multi-round ----------------
def simulate_many_rounds(num_rounds=10, verbose_round=True, print_models_eval_each_round=True):
    global REPUTATION, GROUP_PREV_UPDATES

    # reset persistent states
    REPUTATION = defaultdict(float)
    GROUP_PREV_UPDATES = {}

    train_ds, test_ds = load_cifar10()
    client_indices = partition_dirichlet_balanced(train_ds, NUM_CLIENTS, DIRICHLET_ALPHA)
    client_loaders = make_client_loaders(train_ds, client_indices, BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    # PS-side clean reference loader
    ps_ref_loader = make_ps_reference_loader(
        root="./data",
        batch_size=128,
        num_samples=1000,
        seed=2026
    )

    global_model_baseline = init_global_model()

    malicious_ids = sample_malicious_ids(NUM_CLIENTS, NUM_MALICIOUS,
                                         MALICIOUS_SEED, RANDOMIZE_MALICIOUS)
    print("Malicious client IDs (fixed across rounds):", sorted(list(malicious_ids)))

    history = dict(
        round=[],
        acc_baseline=[],
        asr_baseline=[],

        num_selected_clients=[],
        num_selected_malicious=[],
        num_removed_benign=[],

        accepted_group_ids=[],
        rejected_group_ids=[],
        selected_clients_ids=[],
        removed_clients_ids=[],
    )

    for r in range(1, num_rounds + 1):
        print(f"\n========== Round {r} ==========")

        result = one_round_fedsac_baseline(
            round_idx=r,
            global_model=global_model_baseline,
            client_loaders=client_loaders,
            malicious_ids=malicious_ids,
            ref_loader=ps_ref_loader,
            verbose=verbose_round
        )

        global_model_baseline = result["model"]
        groups = result["groups"]

        acc_baseline = evaluate(global_model_baseline, test_loader, DEVICE)
        asr_baseline = evaluate_asr(global_model_baseline, test_loader, DEVICE,
                                    BACKDOOR_TARGET, add_box_trigger)

        if print_models_eval_each_round:
            print("\n=== Model Evaluation (FedSAC-style baseline) ===")
            print(f"Baseline | Acc: {acc_baseline:6.2f}% | ASR: {asr_baseline:6.2f}%")

        selected_ids = set(groups["selected_clients"])
        removed_ids = set(groups["rejected_clients"])
        mal_set = set(malicious_ids)
        benign_set = set(range(NUM_CLIENTS)) - mal_set

        num_selected_malicious = len(selected_ids & mal_set)
        num_removed_benign = len(removed_ids & benign_set)

        print(f"[Round {r}] Baseline Acc={acc_baseline:.2f}% | ASR={asr_baseline:.2f}%")
        print(f"[Round {r}] SelectedClients={len(selected_ids)} | "
              f"SelectedMalicious={num_selected_malicious} | RemovedBenign={num_removed_benign}")

        history["round"].append(r)
        history["acc_baseline"].append(acc_baseline)
        history["asr_baseline"].append(asr_baseline)

        history["num_selected_clients"].append(len(selected_ids))
        history["num_selected_malicious"].append(num_selected_malicious)
        history["num_removed_benign"].append(num_removed_benign)

        history["accepted_group_ids"].append(groups["accepted_group_ids"])
        history["rejected_group_ids"].append(groups["rejected_group_ids"])
        history["selected_clients_ids"].append(sorted(list(selected_ids)))
        history["removed_clients_ids"].append(sorted(list(removed_ids)))

    return dict(
        baseline_model=global_model_baseline,
        history=history
    )


# ---------------- main ----------------
if __name__ == "__main__":
    import csv
    import json

    results = simulate_many_rounds(
        num_rounds=100,
        verbose_round=True,
        print_models_eval_each_round=True
    )

    FINAL_MODEL = results["baseline_model"]
    LOGS = results["history"]


    csv_path = "NeuSACCIFAR.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "round",
            "acc_baseline",
            "asr_baseline",
            "num_selected_clients",
            "num_selected_malicious",
            "num_removed_benign"
        ])

        for i in range(len(LOGS["round"])):
            writer.writerow([
                LOGS["round"][i],
                LOGS["acc_baseline"][i],
                LOGS["asr_baseline"][i],
                LOGS["num_selected_clients"][i],
                LOGS["num_selected_malicious"][i],
                LOGS["num_removed_benign"][i],
            ])

    print(f"\nSaved round metrics to: {csv_path}")


    json_path = "NeuSACCIFAR.json"
    with open(json_path, "w") as f:
        json.dump({
            "round": LOGS["round"],
            "accepted_group_ids": LOGS["accepted_group_ids"],
            "rejected_group_ids": LOGS["rejected_group_ids"],
            "selected_clients_ids": LOGS["selected_clients_ids"],
            "removed_clients_ids": LOGS["removed_clients_ids"],
        }, f, indent=2)

    print(f"Saved round user IDs to: {json_path}")


    print("\n=== Multi-round Summary (FedSAC-style baseline) ===")
    for i in range(len(LOGS["round"])):
        print(
            f"Round {LOGS['round'][i]:02d}: "
            f"Baseline Acc={LOGS['acc_baseline'][i]:6.2f}% | "
            f"ASR={LOGS['asr_baseline'][i]:6.2f}% || "
            f"SelectedClients={LOGS['num_selected_clients'][i]} | "
            f"SelectedMal={LOGS['num_selected_malicious'][i]} | "
            f"RemovedBenign={LOGS['num_removed_benign'][i]}"
        )