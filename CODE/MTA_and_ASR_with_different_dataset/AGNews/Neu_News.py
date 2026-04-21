

import copy, math, random, warnings
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset, Subset

# ---- torchtext for AG News ----
from torch.nn.utils.rnn import pad_sequence
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# ---- optional sklearn (AHC) ----
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
BACKDOOR_TARGET  = 3

# Neurotoxin attack hyperparameters
NT_LCD_FRAC      = 0.8
NT_ALPHA         = 1.0
NT_SCALE_ATTACK  = 1.5

BENIGN_CLEAN_RATIO = 1.0

RANDOMIZE_MALICIOUS = True
MALICIOUS_SEED = 12345

# RS & LGP hyperparams
LGP_WARMUP_ROUNDS = 30
LGP_PASS_FRAC      = 0.6
APPLY_RS_HARD_THRESHOLD_IN_FINAL = True

# ---- Text backdoor trigger token ----
TRIG_TOKEN = "cf_trigger"
PAD_IDX = 0
TRIG_ID = 1

# ---------------- Model: TextCNN ----------------
class TextCNN(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128, num_classes: int = 4,
                 kernel_sizes=(3,4,5), num_channels=128, pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, num_channels, k) for k in kernel_sizes])
        self.fc = nn.Linear(num_channels * len(kernel_sizes), num_classes)

    def forward(self, x):
        # x: [B, T]
        emb = self.embedding(x)     # [B, T, E]
        emb = emb.transpose(1, 2)   # [B, E, T]
        feats = []
        for conv in self.convs:
            h = F.relu(conv(emb))   # [B, C, T-k+1]
            h = F.max_pool1d(h, kernel_size=h.size(2)).squeeze(2)  # [B, C]
            feats.append(h)
        out = torch.cat(feats, dim=1)  # [B, C*len(K)]
        return self.fc(out)            # [B, num_classes]

# ---------------- Data: AG News ----------------
class AGNewsDataset(Dataset):
    def __init__(self, data_list):
        # data_list: List[(label(0..3), text(str))]
        self.data = data_list
        self.targets = [y for (y, _) in data_list]  # to match your partition_dirichlet_balanced

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        y, text = self.data[idx]
        return text, y

def _yield_tokens(data_list, tokenizer):
    for (y, text) in data_list:
        yield tokenizer(text)

def load_agnews(vocab_max_tokens=50000):
    tokenizer = get_tokenizer("basic_english")

    train_iter = list(AG_NEWS(split="train"))
    test_iter  = list(AG_NEWS(split="test"))

    # torchtext label is 1..4, convert to 0..3
    train_list = [(label - 1, text) for (label, text) in train_iter]
    test_list  = [(label - 1, text) for (label, text) in test_iter]

    vocab = build_vocab_from_iterator(
        _yield_tokens(train_list, tokenizer),
        specials=["<pad>", "<unk>", TRIG_TOKEN],
        max_tokens=vocab_max_tokens
    )
    vocab.set_default_index(vocab["<unk>"])

    pad_idx = vocab["<pad>"]
    trig_id = vocab[TRIG_TOKEN]

    def text_to_tensor(text: str):
        ids = vocab(tokenizer(text))
        if len(ids) == 0:
            ids = [vocab["<unk>"]]
        return torch.tensor(ids, dtype=torch.long)

    def collate_fn(batch):
        # batch: List[(text(str), y(int))]
        ys = torch.tensor([y for (text, y) in batch], dtype=torch.long)
        xs = [text_to_tensor(text) for (text, y) in batch]
        xs = pad_sequence(xs, batch_first=True, padding_value=pad_idx)  # [B, T]
        return xs, ys

    train_ds = AGNewsDataset(train_list)
    test_ds  = AGNewsDataset(test_list)
    return train_ds, test_ds, vocab, pad_idx, trig_id, collate_fn

# -------- Partition (unchanged) --------
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

# -------- Param helpers (unchanged) --------
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

# -------- Text trigger helper --------
def apply_trigger_to_batch_tokens(x: torch.Tensor) -> torch.Tensor:
    """
    x: [B, T] token ids
    Replace the last non-pad token with TRIG_ID
    """
    x_poison = x.clone()
    B = x_poison.size(0)
    for b in range(B):
        row = x_poison[b]
        nz = (row != PAD_IDX).nonzero(as_tuple=False).flatten()
        pos = int(nz[-1].item()) if nz.numel() > 0 else 0
        row[pos] = TRIG_ID
        x_poison[b] = row
    return x_poison


def _train_local_model(base_model, dataloader, epochs, lr, device,
                       poison=False, poison_frac=0.0, backdoor_target=0):

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
                inputs = apply_trigger_to_batch_tokens(x)
                targets = torch.full_like(y, backdoor_target)
            else:
                inputs, targets = x, y

            opt.zero_grad()
            loss = ce(model(inputs), targets)
            loss.backward()
            opt.step()

    local_vec = model_params_to_vector(model).detach().cpu()
    return local_vec


def neurotoxin_combine(delta_benign: torch.Tensor,
                       delta_poison: torch.Tensor,
                       lcd_frac: float = 0.8,
                       alpha: float = 1.0,
                       scale_attack: float = 1.5,
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

# -------- Local train (Neurotoxin attack for malicious clients) --------
def local_train(model, global_model, dataloader, epochs, lr, device,
                malicious=False, poison_frac=0.0, backdoor_target=0,
                nt_lcd_frac=0.8, nt_alpha=1.0, nt_scale_attack=1.5):

    global_vec_cpu = model_params_to_vector(global_model).detach().cpu()

    if not malicious:
        benign_vec = _train_local_model(
            global_model, dataloader,
            epochs=epochs, lr=lr, device=device,
            poison=False, poison_frac=0.0,
            backdoor_target=backdoor_target
        )
        delta = benign_vec - global_vec_cpu
        return {"delta": delta, "num_samples": len(dataloader.dataset)}

    # -------- malicious client: Neurotoxin-style --------
    epochs_clean = 1
    epochs_poison = epochs

    benign_vec = _train_local_model(
        global_model, dataloader,
        epochs=epochs_clean, lr=lr, device=device,
        poison=False, poison_frac=0.0,
        backdoor_target=backdoor_target
    )
    delta_benign = benign_vec - global_vec_cpu

    poison_vec = _train_local_model(
        global_model, dataloader,
        epochs=epochs_poison, lr=lr, device=device,
        poison=True, poison_frac=poison_frac,
        backdoor_target=backdoor_target
    )
    delta_poison = poison_vec - global_vec_cpu

    delta_attack = neurotoxin_combine(
        delta_benign, delta_poison,
        lcd_frac=nt_lcd_frac,
        alpha=nt_alpha,
        scale_attack=nt_scale_attack,
        eps=1e-12
    )

    return {"delta": delta_attack, "num_samples": len(dataloader.dataset)}

# -------- Client-side trust (unchanged) --------
client_states = defaultdict(dict)
# -------- Client-side trust (TextCNN/NLP-aware) --------

_TRUST_PARAM_META = None  # lazy cache: list of (name, start, end, shape)

def _build_param_meta(model: nn.Module):
    meta = []
    p = 0
    for name, param in model.named_parameters():
        n = param.numel()
        meta.append({
            "name": name,
            "start": p,
            "end": p + n,
            "shape": tuple(param.shape)
        })
        p += n
    return meta

def _pick_embedding_and_head(meta, num_classes=None):
    """
    Heuristic to find embedding weight and classifier head (weight+bias).
    Works for common TextCNN: embedding.weight [V, E], fc.weight [C, H], fc.bias [C]
    """
    emb = None
    head_w = None
    head_b = None

    # 1) embedding: prefer name contains emb/embedding and shape is 2D with large vocab dimension
    cand_emb = []
    for m in meta:
        sh = m["shape"]
        nm = m["name"].lower()
        if len(sh) == 2:
            V, E = sh
            if ("emb" in nm or "embedding" in nm) and V >= 5000 and E >= 16:
                cand_emb.append((V, m))
    if cand_emb:
        emb = sorted(cand_emb, key=lambda x: -x[0])[0][1]
    else:
        # fallback: largest 2D matrix with big first dim
        cand = []
        for m in meta:
            sh = m["shape"]
            if len(sh) == 2:
                V, E = sh
                if V >= 5000 and E >= 16:
                    cand.append((V, m))
        if cand:
            emb = sorted(cand, key=lambda x: -x[0])[0][1]

    # 2) head: prefer name contains fc/classifier and shape matches num_classes if provided
    cand_head_w = []
    for m in meta:
        sh = m["shape"]
        nm = m["name"].lower()
        if len(sh) == 2:
            C, H = sh
            if ("fc" in nm or "classifier" in nm or "out" in nm):
                cand_head_w.append((C, H, m))
    # rank candidates
    if cand_head_w:
        if num_classes is not None:
            # exact match first
            exact = [t for t in cand_head_w if t[0] == num_classes]
            if exact:
                head_w = sorted(exact, key=lambda x: -x[1])[0][2]
            else:
                head_w = sorted(cand_head_w, key=lambda x: (abs(x[0] - (num_classes or x[0])), -x[1]))[0][2]
        else:
            # choose smallest C but >1
            head_w = sorted(cand_head_w, key=lambda x: (x[0], -x[1]))[0][2]
    else:
        # fallback: any 2D with small first dim
        cand = []
        for m in meta:
            sh = m["shape"]
            if len(sh) == 2:
                C, H = sh
                if 2 <= C <= 50:
                    cand.append((C, H, m))
        if cand:
            head_w = sorted(cand, key=lambda x: (x[0], -x[1]))[0][2]

    # find matching bias for head if exists
    if head_w is not None:
        target_name = head_w["name"]
        for m in meta:
            sh = m["shape"]
            if len(sh) == 1:
                nm = m["name"].lower()
                # common: fc.bias or classifier.bias
                if (("bias" in nm) and
                    (("fc" in nm and "fc" in target_name.lower()) or
                     ("classifier" in nm and "classifier" in target_name.lower()) or
                     (target_name.lower().replace("weight", "bias") in nm)) and
                    sh[0] == head_w["shape"][0]):
                    head_b = m
                    break
        if head_b is None:
            # fallback: any 1D with length == C
            C = head_w["shape"][0]
            candb = [m for m in meta if len(m["shape"]) == 1 and m["shape"][0] == C and "bias" in m["name"].lower()]
            if candb:
                head_b = candb[0]

    return emb, head_w, head_b

def client_compute_trust_local_only(theta_prev_vec, delta_i_vec, prev_delta_i_vec=None,
                                    state=None, topk=None, quantile=0.99, hist_maxlen=50, eps=1e-12,
                                    model: Optional[nn.Module] = None,
                                    trig_id: Optional[int] = None,
                                    num_classes: Optional[int] = None,
                                    backdoor_target: Optional[int] = None):
    """
    NLP-aware trust:
      - keep your original metrics (rel_l2, spiky, wsign, temporal, tda)
      - add embedding row-concentration + head target-bias (more sensitive for AG News/TextCNN)
    """
    import math as _m, numpy as _np

    global _TRUST_PARAM_META
    if (model is not None) and (_TRUST_PARAM_META is None):
        _TRUST_PARAM_META = _build_param_meta(model)

    # allow auto from your globals if present
    if trig_id is None:
        trig_id = globals().get("TRIG_ID", None)
    if backdoor_target is None:
        backdoor_target = globals().get("BACKDOOR_TARGET", None)
    if num_classes is None:
        num_classes = globals().get("NUM_CLASSES", None) or globals().get("NUM_LABELS", None) or 4

    # ---- basic helpers ----
    def _cos(a, b):
        na = torch.linalg.norm(a); nb = torch.linalg.norm(b)
        if float(na) < eps or float(nb) < eps:
            return 0.0
        return float(torch.dot(a, b) / (na * nb))

    def _push(st, key, val):
        if st is None: return
        st.setdefault(key, []).append(float(val))
        if len(st[key]) > hist_maxlen:
            st[key] = st[key][-hist_maxlen:]

    theta = theta_prev_vec.detach().cpu()
    delta = delta_i_vec.detach().cpu()

    # ---- your original metrics (keep) ----
    tda   = _cos(delta, theta)
    tda01 = (tda + 1.0) / 2.0

    rel_l2 = float(torch.linalg.norm(delta) / (torch.linalg.norm(theta) + eps))

    abs_delta = torch.abs(delta)
    thr = torch.quantile(abs_delta, torch.tensor(quantile, dtype=delta.dtype)) if delta.numel() > 0 else 0
    key_mask = (abs_delta >= thr) if delta.numel() > 0 else torch.zeros_like(delta, dtype=torch.bool)
    spiky = float((torch.linalg.norm(delta[key_mask])**2 / (torch.linalg.norm(delta)**2 + eps)).item()) if key_mask.any() else 0.0

    temporal = float('nan')
    if prev_delta_i_vec is not None:
        temporal = _cos(delta, prev_delta_i_vec.detach().cpu())
    temporal01 = 0.5 if _np.isnan(temporal) else (temporal + 1.0) / 2.0

    # wsign uses selected coords
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
        wsign = float((agree * w[mask]).sum().item() / (w[mask].sum().item() + 1e-12))

    # ---- NEW: embedding/head sensitive metrics ----
    embed_topk_ratio = 0.0
    embed_active_rows = 0.0
    embed_trig_rankpct = 0.0

    head_bias_jump = 0.0
    head_weight_jump = 0.0

    if _TRUST_PARAM_META is not None:
        emb_meta, head_w_meta, head_b_meta = _pick_embedding_and_head(_TRUST_PARAM_META, num_classes=num_classes)

        # ---- embedding features ----
        if emb_meta is not None:
            s, e = emb_meta["start"], emb_meta["end"]
            sh = emb_meta["shape"]  # [V, E]
            dE = delta[s:e].view(sh[0], sh[1])
            row_norms = torch.linalg.norm(dE, dim=1)  # [V]
            total = float(row_norms.sum().item() + 1e-12)

            # topk_ratio: top 0.1% rows energy ratio (at least 10 rows)
            k = max(10, int(0.001 * sh[0]))
            topk = torch.topk(row_norms, k, largest=True).values
            embed_topk_ratio = float(topk.sum().item() / total)

            # active_rows: fraction above (median + 3*MAD)
            rn = row_norms.detach().cpu().numpy().astype(np.float64)
            med = np.median(rn)
            mad = np.median(np.abs(rn - med)) + 1e-12
            thr_act = med + 3.0 * mad
            embed_active_rows = float((row_norms > thr_act).float().mean().item())

            # trig row rank percentile (higher => more suspicious)
            if (trig_id is not None) and (0 <= int(trig_id) < sh[0]):
                tnorm = float(row_norms[int(trig_id)].item())
                embed_trig_rankpct = float((row_norms <= tnorm).float().mean().item())

        # ---- head features ----
        if head_w_meta is not None:
            s, e = head_w_meta["start"], head_w_meta["end"]
            C, H = head_w_meta["shape"]
            dW = delta[s:e].view(C, H)
            rowW = torch.linalg.norm(dW, dim=1)  # [C]

            if (backdoor_target is not None) and (0 <= int(backdoor_target) < C):
                t = int(backdoor_target)
                head_weight_jump = float(rowW[t].item() - rowW[torch.arange(C) != t].mean().item())
            else:
                # fallback: biggest-vs-mean
                head_weight_jump = float(rowW.max().item() - rowW.mean().item())

        if head_b_meta is not None:
            s, e = head_b_meta["start"], head_b_meta["end"]
            C = head_b_meta["shape"][0]
            db = delta[s:e].view(C)
            if (backdoor_target is not None) and (0 <= int(backdoor_target) < C):
                t = int(backdoor_target)
                head_bias_jump = float(db[t].item() - db[torch.arange(C) != t].mean().item())
            else:
                head_bias_jump = float(db.max().item() - db.mean().item())

    # ---- record history (optional) ----
    if state is not None:
        _push(state, 'hist_tda01', tda01)
        _push(state, 'hist_rel_l2', rel_l2)
        _push(state, 'hist_spiky', spiky)
        if not _np.isnan(temporal01):
            _push(state, 'hist_temporal01', temporal01)
        _push(state, 'hist_wsign', wsign)
        _push(state, 'hist_embed_topk_ratio', embed_topk_ratio)
        _push(state, 'hist_embed_active_rows', embed_active_rows)
        _push(state, 'hist_embed_trig_rankpct', embed_trig_rankpct)
        _push(state, 'hist_head_bias_jump', head_bias_jump)
        _push(state, 'hist_head_weight_jump', head_weight_jump)

    # ---- scoring: embed/head are primary, global is auxiliary ----
    # global part (keep your style)
    r0, alpha = 0.35, 3.0
    s0, gamma = 0.40, 2.0
    rel_score  = 1.0 - float(_np.tanh(alpha * max(0.0, rel_l2 - r0)))
    spky_score = 1.0 - float(max(0.0, spiky - s0) ** gamma)
    glob_score = 0.6 * rel_score + 0.4 * spky_score

    # embed score: concentrated updates => lower trust
    # (topk_ratio close to 1 is bad; active_rows small is also "concentrated" but benign may vary)
    k_top = 4.0
    score_embed = float(_np.exp(-k_top * embed_topk_ratio))

    # optional: penalize very high trig-rank percentile (trigger row becomes top)
    if embed_trig_rankpct > 0:
        score_trig = float(_np.exp(-6.0 * max(0.0, embed_trig_rankpct - 0.90)))
    else:
        score_trig = 1.0
    score_embed = 0.7 * score_embed + 0.3 * score_trig

    # head score: large target bias/weight jump => suspicious => lower trust
    # use softplus-like penalty
    t_bias = 0.02
    t_wj   = 0.02
    pen = max(0.0, head_bias_jump - t_bias) + max(0.0, head_weight_jump - t_wj)
    score_head = float(_np.exp(-8.0 * pen))

    # keep wsign weakly
    score_w = float(wsign)

    # final trust (weights tuned for NLP)
    trust = 0.55 * score_embed + 0.30 * score_head + 0.10 * glob_score + 0.05 * score_w

    return {
        "tda": float(tda),
        "tda01": float(tda01),
        "rel_l2": float(rel_l2),
        "spiky": float(spiky),
        "temporal": float(0.0 if _np.isnan(temporal) else temporal),
        "temporal01": float(temporal01),
        "wsign": float(wsign),

        # NEW fields (useful for debugging)
        "embed_topk_ratio": float(embed_topk_ratio),
        "embed_active_rows": float(embed_active_rows),
        "embed_trig_rankpct": float(embed_trig_rankpct),
        "head_bias_jump": float(head_bias_jump),
        "head_weight_jump": float(head_weight_jump),

        "trust": float(trust),
    }


# ================= LGP for PS inspection (paper-faithful) =================
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
    """
    PS-side inspection:
      - For embedding/head layers: use NLP-aware scalar features + robust z(MAD)
      - For other layers: keep your original LGP logic (MAD -> features -> AHC -> innocent)
    """
    susp_cids = [u["cid"] for u in suspicious_updates]
    deltas = [u["delta"].detach().cpu() for u in suspicious_updates]
    layers_per_client = [split_delta_by_layer(d, model) for d in deltas]
    ref_layers = split_delta_by_layer(ref_vec.detach().cpu(), model)

    # build per-layer meta (name/shape) aligned with split_delta_by_layer order
    layer_meta = []
    for name, p in model.named_parameters():
        layer_meta.append({"name": name, "shape": tuple(p.shape)})

    L = len(ref_layers)
    K = len(suspicious_updates)

    layer_pass = np.zeros((K, L), dtype=bool)
    deny_details = {cid: {"mad_fail_layers": [], "ahc_non_honest_layers": []} for cid in susp_cids}

    TRIG_ID = globals().get("TRIG_ID", None)
    BACKDOOR_TARGET = globals().get("BACKDOOR_TARGET", None)
    NUM_CLASSES = globals().get("NUM_CLASSES", None) or 4

    def _zmad(x: np.ndarray, eps=1e-12):
        med = np.median(x)
        mad = np.median(np.abs(x - med)) + eps
        z = 0.6745 * (x - med) / mad
        return med, mad, z

    def _is_embedding(meta):
        sh = meta["shape"]
        nm = meta["name"].lower()
        return (len(sh) == 2) and (sh[0] >= 5000) and (sh[1] >= 16) and ("emb" in nm or "embedding" in nm)

    def _is_head_weight(meta):
        sh = meta["shape"]
        nm = meta["name"].lower()
        return (len(sh) == 2) and (2 <= sh[0] <= 50) and ("fc" in nm or "classifier" in nm or "out" in nm)

    def _is_head_bias(meta, C):
        sh = meta["shape"]
        nm = meta["name"].lower()
        return (len(sh) == 1) and (sh[0] == C) and ("bias" in nm)

    # pre-detect embedding/head indices
    emb_idx = None
    head_w_idx = None
    head_b_idx = None

    for l in range(L):
        if emb_idx is None and _is_embedding(layer_meta[l]):
            emb_idx = l
    # head weight: prefer matches NUM_CLASSES
    cand_hw = []
    for l in range(L):
        if _is_head_weight(layer_meta[l]):
            C, H = layer_meta[l]["shape"]
            cand_hw.append((abs(C - NUM_CLASSES), -H, l))
    if cand_hw:
        head_w_idx = sorted(cand_hw)[0][2]
        C = layer_meta[head_w_idx]["shape"][0]
        # find bias with same C
        for l in range(L):
            if _is_head_bias(layer_meta[l], C):
                head_b_idx = l
                break

    # -------------- layer-wise inspection --------------
    for l in range(L):
        meta = layer_meta[l]
        is_emb = (emb_idx == l)
        is_hw  = (head_w_idx == l)
        is_hb  = (head_b_idx == l)

        # ===== (A) embedding layer: NLP-aware rule =====
        if is_emb:
            V, E = meta["shape"]
            # per-client features: topk_ratio, trig_rankpct, L2
            topk_ratio_list = []
            trig_rank_list = []
            l2_list = []

            for k in range(K):
                g = layers_per_client[k][l].numpy().astype(np.float64).reshape(V, E)
                row_norm = np.linalg.norm(g, axis=1) + 1e-12
                total = row_norm.sum() + 1e-12
                kk = max(10, int(0.001 * V))
                topk = np.partition(row_norm, -kk)[-kk:].sum()
                topk_ratio = float(topk / total)
                topk_ratio_list.append(topk_ratio)

                if TRIG_ID is not None and 0 <= int(TRIG_ID) < V:
                    tnorm = row_norm[int(TRIG_ID)]
                    trig_rank = float((row_norm <= tnorm).mean())  # percentile
                else:
                    trig_rank = 0.0
                trig_rank_list.append(trig_rank)

                l2_list.append(float(np.linalg.norm(g)))

            topk_ratio_arr = np.asarray(topk_ratio_list, dtype=np.float64)
            trig_rank_arr  = np.asarray(trig_rank_list, dtype=np.float64)
            l2_arr         = np.asarray(l2_list, dtype=np.float64)

            # robust z-score (MAD) gate
            _, _, z_top = _zmad(topk_ratio_arr)
            _, _, z_l2  = _zmad(l2_arr)

            # trig_rank only when available
            if (TRIG_ID is not None) and (trig_rank_arr.max() > 0):
                _, _, z_tr = _zmad(trig_rank_arr)
            else:
                z_tr = np.zeros_like(z_top)

            # pass if not too extreme
            zthr = 3.5
            for k in range(K):
                ok = (abs(z_top[k]) <= zthr) and (abs(z_l2[k]) <= zthr)
                if (TRIG_ID is not None) and (trig_rank_arr.max() > 0):
                    # if trigger row becomes extremely high-rank, treat as suspicious
                    ok = ok and (trig_rank_arr[k] <= 0.98)  # hard cap
                layer_pass[k, l] = bool(ok)
                if not ok:
                    deny_details[susp_cids[k]]["mad_fail_layers"].append(l)
            continue

        # ===== (B) head layers: NLP-aware rule =====
        if is_hw or is_hb:
            # we inspect head using both W and b together (if both exist)
            # compute per-client: head_bias_jump, head_weight_jump, L2
            # if only one exists, use what's available
            feat_bias = np.zeros(K, dtype=np.float64)
            feat_wj   = np.zeros(K, dtype=np.float64)
            feat_l2   = np.zeros(K, dtype=np.float64)

            # weight part
            if head_w_idx is not None:
                C, H = layer_meta[head_w_idx]["shape"]
                for k in range(K):
                    dW = layers_per_client[k][head_w_idx].numpy().astype(np.float64).reshape(C, H)
                    rowW = np.linalg.norm(dW, axis=1)
                    if BACKDOOR_TARGET is not None and 0 <= int(BACKDOOR_TARGET) < C:
                        t = int(BACKDOOR_TARGET)
                        feat_wj[k] = float(rowW[t] - rowW[np.arange(C) != t].mean())
                    else:
                        feat_wj[k] = float(rowW.max() - rowW.mean())
                    feat_l2[k] += float(np.linalg.norm(dW))

            # bias part
            if head_b_idx is not None:
                Cb = layer_meta[head_b_idx]["shape"][0]
                for k in range(K):
                    db = layers_per_client[k][head_b_idx].numpy().astype(np.float64).reshape(Cb)
                    if BACKDOOR_TARGET is not None and 0 <= int(BACKDOOR_TARGET) < Cb:
                        t = int(BACKDOOR_TARGET)
                        feat_bias[k] = float(db[t] - db[np.arange(Cb) != t].mean())
                    else:
                        feat_bias[k] = float(db.max() - db.mean())
                    feat_l2[k] += float(np.linalg.norm(db))

            # robust gate: malicious tends to have unusually large positive bias_jump / weight_jump
            _, _, z_b = _zmad(feat_bias)
            _, _, z_w = _zmad(feat_wj)
            _, _, z_l = _zmad(feat_l2)

            zthr = 3.5
            for k in range(K):
                ok = (abs(z_l[k]) <= zthr)
                # only penalize strong positive outliers (backdoor-target push)
                ok = ok and (z_b[k] <= zthr) and (z_w[k] <= zthr)
                layer_pass[k, l] = bool(ok)
                if not ok:
                    deny_details[susp_cids[k]]["mad_fail_layers"].append(l)
            continue

        # ===== (C) other layers: keep your original LGP =====
        glist = [layers_per_client[k][l].numpy().astype(np.float32) for k in range(K)]
        norms = np.array([np.linalg.norm(g) for g in glist], dtype=np.float64)
        med, mad = _mad(norms); LB, UB = med - mad, med + mad

        cand1_idx = [i for i, n in enumerate(norms) if (n >= LB and n <= UB)]
        cand2_idx = [i for i in range(K) if i not in cand1_idx]

        choose_cand = cand1_idx
        if len(cand2_idx) > 0:
            winner, c1, c2 = _innocent_choose(cand1_idx, cand2_idx, glist, ref_layers[l].numpy())
            if winner == 2:
                choose_cand = cand2_idx

        if len(choose_cand) == 0:
            for i in range(K):
                deny_details[susp_cids[i]]["mad_fail_layers"].append(l)
            continue

        feats = []
        for i in choose_cand:
            g = glist[i]; ref = ref_layers[l].numpy()
            PC = float((g > 0).sum()); NC = float((g < 0).sum()); ZC = float((g == 0).sum())
            kurt = _kurtosis(g); skew = _skewness(g)
            others = [glist[j] for j in choose_cand if j != i]
            if len(others) == 0: Dmean = 0.0
            else: Dmean = float(np.mean([np.linalg.norm(g - o) for o in others]))
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

    # -------- summarize across layers --------
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
def evaluate_asr(model, test_loader, device, target):
    model.to(device).eval(); ok=0; tot=0
    for x, y in test_loader:
        x_bd = apply_trigger_to_batch_tokens(x)
        x_bd = x_bd.to(device)
        pred = model(x_bd).argmax(1)
        ok += (pred == target).sum().item(); tot += x_bd.size(0)
    return 100.0 * ok / tot

# ---------------- Utilities ----------------
def init_global_model(vocab_size: int, num_classes: int = 4, pad_idx: int = 0):
    m = TextCNN(vocab_size=vocab_size, num_classes=num_classes, pad_idx=pad_idx).to(DEVICE)
    # simple init
    def _init(mm):
        if isinstance(mm, nn.Conv1d) or isinstance(mm, nn.Linear):
            nn.init.kaiming_normal_(mm.weight, nonlinearity="relu")
            if getattr(mm, "bias", None) is not None:
                nn.init.zeros_(mm.bias)
        if isinstance(mm, nn.Embedding):
            nn.init.normal_(mm.weight, mean=0.0, std=0.02)
            if mm.padding_idx is not None:
                with torch.no_grad():
                    mm.weight[mm.padding_idx].fill_(0.0)
    m.apply(_init)
    return m

def make_client_loaders(train_ds, client_indices, batch_size, collate_fn=None):
    return [DataLoader(Subset(train_ds, idxs), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
            for idxs in client_indices]

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

def one_round_no_defense(global_model, client_loaders, malicious_ids):
    updates = []
    for cid in range(NUM_CLIENTS):
        local_model = copy.deepcopy(global_model)
        malicious = (cid in malicious_ids)
        loader = client_loaders[cid]

        dd = local_train(local_model, global_model, loader,
                         epochs=LOCAL_EPOCHS, lr=LR, device=DEVICE,
                         malicious=malicious, poison_frac=POISON_FRACTION,
                         backdoor_target=BACKDOOR_TARGET,
                         nt_lcd_frac=NT_LCD_FRAC,
                         nt_alpha=NT_ALPHA,
                         nt_scale_attack=NT_SCALE_ATTACK)

        updates.append({
            "cid": cid,
            "delta": dd["delta"],
            "num_samples": dd["num_samples"]
        })

    agg_all, _ = weighted_average(updates)
    global_vec = model_params_to_vector(global_model).cpu()

    new_global = copy.deepcopy(global_model)
    set_model_params_from_vector(new_global, global_vec + agg_all)
    return new_global


# ---------------- One Round ----------------
def one_round(round_idx, global_model, client_loaders, malicious_ids,
              split=(0.50, 0.30, 0.20), verbose=True, use_ema=False, beta_ema=0.90, ref_delta_ema=None):
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
            nt_lcd_frac=NT_LCD_FRAC,
            nt_alpha=NT_ALPHA,
            nt_scale_attack=NT_SCALE_ATTACK
        )

        state_i = client_states[cid]
        prev_delta = state_i.get('prev_delta')

        # ---- trust (pass model so we can locate embedding/head slices) ----
        scores = client_compute_trust_local_only(
            theta_prev_vec, dd["delta"], prev_delta,
            state=state_i,
            topk=None, quantile=0.99, hist_maxlen=50,
            model=global_model,                 # << key for AG News/TextCNN
            trig_id=globals().get("TRIG_ID", None),
            num_classes=globals().get("NUM_CLASSES", None) or 4,
            backdoor_target=BACKDOOR_TARGET
        )

        state_i['prev_delta'] = dd["delta"].clone()
        client_states[cid] = state_i

        # keep whatever scores provides (compat)
        u = {
            "cid": cid,
            "delta": dd["delta"],
            "num_samples": dd["num_samples"],
            "trust": float(scores.get("trust", 0.0)),
            "tda": float(scores.get("tda", 0.0)),
            "rel_l2": float(scores.get("rel_l2", 0.0)),
            "spiky": float(scores.get("spiky", 0.0)),
            "temporal": float(scores.get("temporal", 0.0)),
            "wsign": float(scores.get("wsign", 0.5)),
        }

        # optional fields you might have added in A/B (won't break if absent)
        for k in ["rank", "alignRef",
                  "embed_topk_ratio", "embed_active_rows", "embed_trig_rankpct",
                  "head_bias_jump", "head_weight_jump"]:
            if k in scores:
                u[k] = float(scores[k])

        updates.append(u)

    # ---------------- print per-client ----------------
    if verbose:
        for u in updates:
            label = "MAL" if u["cid"] in malicious_ids else "BEN"

            # safe getters
            def g(key, default=0.0):
                return float(u.get(key, default))

            # build line with whatever exists
            parts = [
                f"CID {u['cid']:2d} [{label}]",
                f"trust={g('trust'):.3f}",
            ]
            if "rank" in u:
                parts.append(f"rank={g('rank'):.3f}")
            if "alignRef" in u:
                parts.append(f"alignRef={g('alignRef'):.3f}")

            parts += [
                f"TDA={g('tda'):+.3f}",
                f"relL2={g('rel_l2'):.3f}",
                f"spiky={g('spiky'):.3f}",
                f"temporal={g('temporal'):+.3f}",
                f"wsign={g('wsign'):.3f}",
            ]

            # NLP-aware debug signals
            if "embed_topk_ratio" in u:
                parts.append(f"embTopK={g('embed_topk_ratio'):.3f}")
            if "embed_trig_rankpct" in u:
                parts.append(f"embTrigPct={g('embed_trig_rankpct'):.3f}")
            if "head_bias_jump" in u:
                parts.append(f"hbJump={g('head_bias_jump'):+.4f}")
            if "head_weight_jump" in u:
                parts.append(f"hwJump={g('head_weight_jump'):+.4f}")

            print(" | ".join(parts))

    # ---------------- build baselines (all / benign) ----------------
    agg_all, _ = weighted_average(updates)
    benign_updates = [u for u in updates if u["cid"] not in malicious_ids]
    agg_benign, _ = weighted_average(benign_updates)
    global_vec = model_params_to_vector(global_model).cpu()

    contaminated_global = copy.deepcopy(global_model)
    set_model_params_from_vector(contaminated_global, global_vec + agg_all)

    clean_global = copy.deepcopy(global_model)
    set_model_params_from_vector(clean_global, global_vec + agg_benign)

    # ---------------- split by trust ----------------
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

    # ---------------- robust ref for PS (coordinate-wise median of ALL deltas) ----------------
    # (this is your D idea; minimal and robust)
    mat_all = torch.stack([u["delta"].detach().cpu() for u in updates], dim=0)  # [K, D]
    ref_vec_round = torch.median(mat_all, dim=0).values.detach().cpu()

    # optional EMA over ref if you want (keeps your old switch)
    if use_ema and (ref_delta_ema is not None):
        ref_vec = (beta_ema * ref_delta_ema + (1.0 - beta_ema) * ref_vec_round).detach().cpu()
    else:
        ref_vec = ref_vec_round.clone()

    # For completeness: "trusted model" (not used as ref now)
    agg_trusted, _ = weighted_average(trusted_updates)
    trusted_global = copy.deepcopy(global_model)
    set_model_params_from_vector(trusted_global, global_vec + agg_trusted)

    # ---------------- PS inspection on suspicious via LGP ----------------
    acc_ids, deny_details = ps_inspect_with_LGP(global_model, suspicious_updates, ref_vec, round_idx)

    # ---------------- groups ----------------
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

    # ---------------- RS threshold & final participants (after RS) ----------------
    groups["final_participants_afterRS"] = groups["final_participants"].copy()
    groups["removed_by_rs"] = []

    if len(RS) > 0:
        rs_items = sorted([(cid, RS.get(cid, 0)) for cid in range(NUM_CLIENTS)],
                          key=lambda x: (-x[1], x[0]))
        rs_vals = np.array([v for _, v in rs_items], dtype=float)
        rs_med, rs_mad = _mad(rs_vals) if len(rs_vals) else (0.0, 0.0)
        # rs_thr = rs_med - rs_mad
        rs_thr = rs_med


        groups["rs_items"] = rs_items
        groups["rs_med"] = float(rs_med)
        groups["rs_mad"] = float(rs_mad)
        groups["rs_thr"] = float(rs_thr)

        if APPLY_RS_HARD_THRESHOLD_IN_FINAL and round_idx > LGP_WARMUP_ROUNDS:
            before_rs = groups["final_participants"].copy()
            after_rs = [cid for cid in groups["final_participants"] if RS[cid] >= rs_thr]
            groups["final_participants_afterRS"] = after_rs
            groups["removed_by_rs"] = sorted(list(set(before_rs) - set(after_rs)))

    # ---------------- Final aggregate ----------------
    final_pool = [u for u in updates if u["cid"] in groups["final_participants_afterRS"]]
    if len(final_pool) == 0:
        final_pool = trusted_updates

    agg_final, _ = weighted_average(final_pool)
    final_global = copy.deepcopy(global_model)
    set_model_params_from_vector(final_global, global_vec + agg_final)

    # ---------------- RS update: participants after RS ----------------
    for cid in groups["final_participants_afterRS"]:
        RS[cid] = RS.get(cid, 0) + 1

    # ---------------- RS print ----------------
    if len(RS) > 0:
        rs_items2 = sorted([(cid, RS.get(cid, 0)) for cid in range(NUM_CLIENTS)],
                           key=lambda x: (-x[1], x[0]))
        rs_vals2 = np.array([v for _, v in rs_items2], dtype=float)
        rs_med2, rs_mad2 = _mad(rs_vals2) if len(rs_vals2) else (0.0, 0.0)
        # rs_thr2 = rs_med2 - rs_mad2
        rs_thr2 = rs_med2
        print("\n[RS] cid->RS (sorted):", rs_items2)
        print(f"[RS] median={rs_med2:.2f}, MAD={rs_mad2:.2f}, threshold (med-MAD)={rs_thr2:.2f}")

    # ---------------- group summary ----------------
    print("\n=== Client Group Summary (this round) ===")
    print(f"Trusted (Top {int(split[0]*100)}%, OTA aggregated)        : {groups['trusted']}")
    print(f"Suspicious (middle {int(split[1]*100)}%, individual access): {groups['suspicious']}")
    print(f"Rejected initially (trust phase)                      : {groups['rejected']}")
    print(f"Suspicious accepted after inspection                  : {groups['accepted_from_susp']}")
    print(f"Suspicious denied after inspection                    : {groups['denied_after_ps']}")
    print(f"Denied overall (trust + PS)                           : {groups['denied_overall']}")
    print(f"[RS Filter] Removed by RS threshold                   : {groups['removed_by_rs']}")
    print(f"Final aggregation participants (after RS)             : {groups['final_participants_afterRS']}")

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




# ---------------- Multi-round ----------------
def simulate_many_rounds(num_rounds=10, split=(0.50,0.30,0.20), verbose_round=True,
                         print_models_eval_each_round=True, use_ema=False):

    global PAD_IDX, TRIG_ID

    train_ds, test_ds, vocab, pad_idx, trig_id, collate_fn = load_agnews()
    PAD_IDX = int(pad_idx)
    TRIG_ID = int(trig_id)

    client_indices = partition_dirichlet_balanced(train_ds, NUM_CLIENTS, DIRICHLET_ALPHA)
    client_loaders = make_client_loaders(train_ds, client_indices, BATCH_SIZE, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate_fn)

    global_model_def = init_global_model(vocab_size=len(vocab), num_classes=4, pad_idx=PAD_IDX)
    global_model_all = copy.deepcopy(global_model_def)

    malicious_ids = sample_malicious_ids(NUM_CLIENTS, NUM_MALICIOUS, MALICIOUS_SEED, RANDOMIZE_MALICIOUS)
    print("Malicious client IDs (fixed across rounds):", sorted(list(malicious_ids)))
    print(f"[AG News] vocab_size={len(vocab)} | PAD_IDX={PAD_IDX} | TRIG_ID={TRIG_ID} | classes=4")

    ref_delta_ema = None
    history = dict(
        round=[],
        acc_final=[], asr_final=[],
        acc_all=[], asr_all=[],


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

        global_model_all = one_round_no_defense(global_model_all, client_loaders, malicious_ids)

        result = one_round(r, global_model_def, client_loaders,
                           malicious_ids, split=split,
                           verbose=verbose_round,
                           use_ema=use_ema, ref_delta_ema=ref_delta_ema)
        ref_delta_ema = result["aux"]["ref_delta_ema"]
        models = result["models"]
        groups = result["groups"]
        global_model_def = models["final"]

        if print_models_eval_each_round:
            print("\n=== Model Evaluation (this round) ===")
            acc_all = evaluate(global_model_all, test_loader, DEVICE)
            asr_all = evaluate_asr(global_model_all, test_loader, DEVICE, BACKDOOR_TARGET)
            print(f"Contaminated (all clients, no defense) | Acc: {acc_all:6.2f}% | ASR: {asr_all:6.2f}%")

            acc_final = evaluate(models["final"], test_loader, DEVICE)
            asr_final = evaluate_asr(models["final"], test_loader, DEVICE, BACKDOOR_TARGET)
            print(f"Final (defended, selected clients)     | Acc: {acc_final:6.2f}% | ASR: {asr_final:6.2f}%")

        acc_all = evaluate(global_model_all, test_loader, DEVICE)
        asr_all = evaluate_asr(global_model_all, test_loader, DEVICE, BACKDOOR_TARGET)
        acc_final = evaluate(models["final"], test_loader, DEVICE)
        asr_final = evaluate_asr(models["final"], test_loader, DEVICE, BACKDOOR_TARGET)

        print(f"[Round {r}] Baseline  Acc={acc_all:.2f}% | ASR={asr_all:.2f}%")
        print(f"[Round {r}] Defended  Acc={acc_final:.2f}% | ASR={asr_final:.2f}%")

        history["round"].append(r)
        history["acc_all"].append(acc_all)
        history["asr_all"].append(asr_all)
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

    return dict(baseline_model=global_model_all, defended_model=global_model_def, history=history)


# ---------------- main ----------------
if __name__ == "__main__":
    import csv
    import json

    results = simulate_many_rounds(
        num_rounds=100,
        split=(0.45, 0.35, 0.20),
        verbose_round=True,
        print_models_eval_each_round=True,
        use_ema=False
    )

    FINAL_MODEL = results["defended_model"]
    LOGS = results["history"]
    BASELINE_MODEL = results["baseline_model"]


    csv_path = "NeuNews.csv"
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


    json_path = "NeuNews.json"
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
        print(
            f"Round {LOGS['round'][i]:02d}: "
            f"Baseline Acc={LOGS['acc_all'][i]:6.2f}% | ASR={LOGS['asr_all'][i]:6.2f}% || "
            f"Defended Acc={LOGS['acc_final'][i]:6.2f}% | ASR={LOGS['asr_final'][i]:6.2f}% || "
            f"FinalParticipants={LOGS['num_final_participants'][i]} | "
            f"SelectedMal={LOGS['num_selected_malicious'][i]} | "
            f"RemovedBenign={LOGS['num_removed_benign'][i]}"
        )