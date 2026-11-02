"""
FedPRADA - Module 3 : Adaptive Aggregation Module
==================================================
Trains TWO global models in one run:
    1) DataCo               (clients: africa, europe, latam, pacific_asia, usca)
    2) E-Commerce Shipping  (clients: A, B, C, D, F)

Implements Algorithm 4 of the paper, per communication round t:
    client side : theta_m <- theta_g ; E local epochs (Adam, FedProx-style
                  proximal regularisation) ; local validation loss L_m^(t) ;
                  label counts c_m(c) ; sample size n_m
    server side : P_g  from the summed label counts                    (Module 2)
                  D_m  = KL(P_m || P_g)                       Eq.(5)   (Module 3)
                  S_m  = 1 / (1 + std(last w val losses))     Eq.(6)
                  w_m  ~ n_m * exp(-lambda_div * D_m) * S_m^gamma  Eq.(7)
                  normalise (uniform fallback), theta_g <- sum_m w_m theta_m

Run:
    python fedprada_module3.py                          # both datasets
    python fedprada_module3.py --dataset dataco
    python fedprada_module3.py --dataset ecommerce --rounds 60
    python fedprada_module3.py --compare_fedavg         # adds FedAvg overlay
    python fedprada_module3.py --dataset dataco --partition mixed_shard
    python fedprada_module3.py --batch_size 0 --sample_fraction 1.0
                                   # = strict paper setting (full batch, full data)

Outputs per dataset (outputs/fedprada_module3_<dataset>/):
    global_model_performance.png      global model per round
    kl_divergence_per_round.png       D_m per client per round (+heatmap)
    stability_per_round.png           S_m per client per round (+ raw val losses)
    client_validation_per_round.png   local and global-model validation per client
    weighted_fusion_weights.png       w_m per round (stack + deviation from FedAvg)
    weighted_fusion_breakdown.png     step by step: n_m -> x exp(-lambda D) -> x S^gamma
    fusion_parameter_movement.png     how far each local model is from the fused model
    fusion_log.csv                    every number used in every fusion (all rounds)
    fusion_report.txt                 human readable fusion arithmetic (selected rounds)
    global_metrics.csv, module3_summary.json
    global_model_best.pth, global_model_final.pth
"""

import os
import json
import copy
import time
import random
import argparse

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)

from secure_transport import (
    FederatedRoundCoordinator,
    TLSFederatedClient,
    TLSFederatedServer,
    add_tls_arguments,
    tls_config_from_args,
)


# ============================================================
# SETTINGS
# ============================================================

SEED = 42

# ---- Fed-FP3 Novelty hyperparameters -----------------------
BETA_FP = 1.0               # sensitivity parameter for relative feature divergence
ALPHA_FP = 1.0              # exponent for feature pertinence in aggregation weight
FP3_BASE_OUTPUT = "outputs/fed_fp3"

# ---- settings taken from the paper -------------------------
LOCAL_EPOCHS = 3            # paper: local epochs fixed to 3
VALIDATION_SIZE = 0.20      # same split as Module 2
ROUNDS = 100                # 100 rounds x 3 epochs = 300 equivalent epochs
MU_PROX = 0.01              # proximal regularisation (paper grid .001/.01/.1/.5/1)
STAB_WINDOW = 5             # w in Eq.(6)  (paper does not give a value)
EPS = 1e-8                  # smoothing epsilon of Algorithm 4
LEARNING_RATE = {"dataco": 1e-3, "ecommerce": 5e-4}   # paper Table 4

# ---- tunable (paper gives no numeric value) ----------------
GAMMA_STAB = 2.0
LAMBDA_DIV = {"natural": 100.0, "mixed_shard": 5.0}

# ---- practical accuracy / visibility improvements ----------
# (deviations from the strict paper protocol, switchable from the CLI)
LOCAL_BATCH_SIZE = 512      # 0 -> full batch (paper). Mini-batch = many more
                            # optimisation steps per round = better accuracy.
ROUND_SAMPLE_FRACTION = 0.7 # each client trains on a fresh random 70 % of its
                            # train split every round and reports the label counts
                            # of THAT sample -> n_m, P_m, D_m change per round.
                            # 1.0 = paper (D_m is then constant for every round).
GRAD_CLIP = 1.0
PATHOLOGICAL_RATIO = 0.8    # paper Sec 5.3.3 mixed-shard construction

DATASETS = {
    "dataco": {
        "name": "DataCo",
        "clients": ["africa", "europe", "latam", "pacific_asia", "usca"],
        "data_dir": "processed/clients/dataco",
        "output_dir": "outputs/fedprada_module3_dataco",
        "target": "Delivery Status",
        "drop": ["Delivery Status", "Days for shipping (real)",
                 "Late_delivery_risk", "Market"],
        "hidden_dim": 128, "num_blocks": 3, "dropout": 0.20,
        "kind": "dataco",
        "standardize": False,       # client files are already standardised
    },
    "ecommerce": {
        "name": "E-Commerce Shipping",
        "clients": ["A", "B", "C", "D", "F"],
        "data_dir": "processed/clients",
        "output_dir": "outputs/fedprada_module3_ecommerce",
        "target": "Reached.on.Time_Y.N",
        "drop": ["Reached.on.Time_Y.N"],
        "hidden_dim": 256, "num_blocks": 4, "dropout": 0.10,
        "kind": "ecommerce",
        "standardize": True,        # federated mean/std (see load_dataset)
    },
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# MODELS  (identical to Module 2)
# ============================================================

class DataCoResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):                     # Eq.(4)
        identity = x
        x = self.dropout(self.relu(self.linear(x)))
        return self.layer_norm(x + identity)


class DataCoResNetMLP(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim, num_blocks, dropout):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.input_relu = nn.ReLU()
        self.blocks = nn.ModuleList(
            [DataCoResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)])
        self.output_layer = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.input_relu(self.input_layer(x))
        for block in self.blocks:
            x = block(x)
        return self.output_layer(x)


class EcommerceResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        identity = x
        out = self.dropout(self.act(self.bn1(self.fc1(x))))
        out = self.dropout(self.act(self.bn2(self.fc2(out))))
        return out + identity


class EcommerceResNetMLP(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim, num_blocks, dropout):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim),
            nn.GELU(), nn.Dropout(dropout))
        self.blocks = nn.ModuleList(
            [EcommerceResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, num_classes))

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


def create_model(ds_key, input_dim, num_classes):
    c = DATASETS[ds_key]
    cls = DataCoResNetMLP if c["kind"] == "dataco" else EcommerceResNetMLP
    return cls(input_dim, num_classes, c["hidden_dim"],
               c["num_blocks"], c["dropout"]).to(device)


# ============================================================
# DATA (Module 1 output -> tensors on device)
# ============================================================

def read_client_csv(ds_key, client):
    c = DATASETS[ds_key]
    path = os.path.join(c["data_dir"], f"client_{client}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Client file not found: {path}")
    df = pd.read_csv(path)
    if c["target"] not in df.columns:
        raise ValueError(f"Target '{c['target']}' missing in {path}")
    y_raw = df[c["target"]].values
    order_ids = df["Order Id"].values if "Order Id" in df.columns else None
    X = df.drop(columns=[k for k in c["drop"] if k in df.columns])
    X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0.0).values.astype(np.float32)
    return X, y_raw, order_ids


def mixed_shard_partition(y, num_clients, ratio, seed):
    """Paper Sec 5.3.3: pathological (label-sorted) shards + IID background.
    Every client gets one label-skewed shard and one IID shard."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(y))
    n_path = int(ratio * len(y))
    path_idx, iid_idx = perm[:n_path], perm[n_path:]
    path_idx = path_idx[np.argsort(y[path_idx], kind="stable")]
    shards = np.array_split(path_idx, num_clients)
    iid_parts = np.array_split(iid_idx, num_clients)
    order = rng.permutation(num_clients)
    return [np.concatenate([shards[order[m]], iid_parts[m]])
            for m in range(num_clients)]


def normalised_entropy_report(y_list, num_classes):
    """Eq.(17)-(18): H and sigma_H of the client label mixes."""
    hs = []
    for y in y_list:
        q = np.bincount(y, minlength=num_classes) / max(len(y), 1)
        q = q[q > 0]
        hs.append(float(-(q * np.log(q)).sum() / np.log(num_classes)))
    return float(np.mean(hs)), float(np.std(hs))


def load_dataset(ds_key, partition, return_meta=False):
    c = DATASETS[ds_key]
    clients = c["clients"]

    raw = {cl: read_client_csv(ds_key, cl) for cl in clients}

    # one global label mapping: class index means the same everywhere
    class_values = sorted(set(np.concatenate([raw[cl][1] for cl in clients]).tolist()))
    v2c = {v: i for i, v in enumerate(class_values)}
    num_classes = len(class_values)

    data = {cl: (raw[cl][0], np.array([v2c[v] for v in raw[cl][1]], dtype=np.int64), raw[cl][2])
            for cl in clients}

    dims = {d[0].shape[1] for d in data.values()}
    if len(dims) != 1:
        raise ValueError(f"Clients have different feature sizes: {dims}")

    if partition == "mixed_shard":
        X_all = np.concatenate([data[cl][0] for cl in clients])
        y_all = np.concatenate([data[cl][1] for cl in clients])
        parts = mixed_shard_partition(y_all, len(clients), PATHOLOGICAL_RATIO, SEED)
        data = {cl: (X_all[idx], y_all[idx], None) for cl, idx in zip(clients, parts)}
        print(f"[{c['name']}] mixed-shard non-IID partition "
              f"(pathological_ratio={PATHOLOGICAL_RATIO})")

    H, sH = normalised_entropy_report([data[cl][1] for cl in clients], num_classes)
    print(f"[{c['name']}] label-mix entropy  H={H:.3f}  sigma_H={sH:.3f}  "
          f"(lower H = more non-IID)")

    # local train / validation split (same as Module 2)
    split = {}
    split_meta = {}
    for cl in clients:
        X, y, oids = data[cl]
        indices = np.arange(len(y))
        counts = np.bincount(y, minlength=num_classes)
        strat = y if counts[counts > 0].min() >= 2 else None
        Xtr, Xva, ytr, yva, idx_tr, idx_va = train_test_split(
            X, y, indices, test_size=VALIDATION_SIZE, random_state=SEED, stratify=strat)
        split[cl] = [Xtr, ytr, Xva, yva]
        va_oids = oids[idx_va] if oids is not None else None
        split_meta[cl] = dict(client=cl, val_indices=idx_va, val_order_ids=va_oids)

    # federated standardisation: clients only share sum / sum-of-squares / count
    if c["standardize"]:
        n = sum(len(s[0]) for s in split.values())
        s1 = sum(s[0].astype(np.float64).sum(0) for s in split.values())
        s2 = sum((s[0].astype(np.float64) ** 2).sum(0) for s in split.values())
        mean = s1 / n
        std = np.sqrt(np.maximum(s2 / n - mean ** 2, 0.0))
        std[std < 1e-6] = 1.0
        for cl in clients:
            split[cl][0] = ((split[cl][0] - mean) / std).astype(np.float32)
            split[cl][2] = ((split[cl][2] - mean) / std).astype(np.float32)
        print(f"[{c['name']}] features standardised with federated statistics")

    tensors = {}
    for cl in clients:
        Xtr, ytr, Xva, yva = split[cl]
        tensors[cl] = dict(
            Xtr=torch.tensor(Xtr, device=device), ytr=torch.tensor(ytr, device=device),
            Xva=torch.tensor(Xva, device=device), yva=torch.tensor(yva, device=device))
        print(f"   client {cl:<13} train={len(ytr):>6}  val={len(yva):>6}  "
              f"train label counts={np.bincount(ytr, minlength=num_classes).tolist()}")

    if return_meta:
        return tensors, class_values, num_classes, split[clients[0]][0].shape[1], (H, sH), split_meta
    return tensors, class_values, num_classes, split[clients[0]][0].shape[1], (H, sH)


# ============================================================
# CLIENT SIDE
# ============================================================

@torch.no_grad()
def evaluate(model, X, y, bs=8192):
    model.eval()
    loss_sum, preds = 0.0, []
    for i in range(0, len(X), bs):
        out = model(X[i:i + bs])
        loss_sum += F.cross_entropy(out, y[i:i + bs], reduction="sum").item()
        preds.append(out.argmax(1))
    preds = torch.cat(preds)
    return loss_sum / len(X), (preds == y).float().mean().item(), preds


def local_update(model, global_params, X, y, lr, epochs, batch_size, mu, gen):
    """Algorithm 4 lines 5-8 with the proximal (regularisation) term:
       L = CE + (mu/2) * ||theta_m - theta_g||^2"""
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(X)
    last_epoch_loss = 0.0
    for _ in range(epochs):
        if batch_size <= 0 or batch_size >= n:
            batches = [torch.arange(n, device=device)]
        else:
            perm = torch.randperm(n, generator=gen).to(device)
            batches = [perm[i:i + batch_size] for i in range(0, n, batch_size)]
        tot, cnt = 0.0, 0
        for idx in batches:
            if len(idx) < 2:                      # BatchNorm needs >= 2 samples
                continue
            opt.zero_grad(set_to_none=True)
            out = model(X[idx])
            ce = F.cross_entropy(out, y[idx])
            prox = sum(((p - g) ** 2).sum()
                       for p, g in zip(model.parameters(), global_params))
            loss = ce + 0.5 * mu * prox
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            tot += ce.item() * len(idx)
            cnt += len(idx)
        last_epoch_loss = tot / max(cnt, 1)
    return last_epoch_loss


# ============================================================
# SERVER SIDE : FedPRADA weights (Eq. 5, 6, 7) and fusion
# ============================================================

def kl_divergence(p_m, p_g):
    """Eq.(5): D_m = sum_c P_m(c) log(P_m(c)/P_g(c)), smoothed with EPS."""
    mask = p_m > 0
    d = np.sum(p_m[mask] * np.log((p_m[mask] + EPS) / (p_g[mask] + EPS)))
    return float(max(d, 0.0))


def stability_score(buffer, window):
    """Eq.(6): S_m = 1 / (1 + std(last w validation losses)); S=1 if < 2 values."""
    buf = buffer[-window:]
    if len(buf) < 2:
        return 1.0
    return float(1.0 / (1.0 + np.std(buf)))


def normalise(v):
    s = float(np.sum(v))
    if np.isfinite(s) and s > 0:
        return v / s
    return np.full(len(v), 1.0 / len(v))          # uniform fallback (Alg.4 l.24)


# ============================================================
# FED-FP3 NOVELTY: FEATURE-PERTINENCE ESTIMATION
# ============================================================

def compute_feature_statistics(X_np):
    """
    Computes local feature distribution moments:
    mean: (d,)
    std: (d,)
    """
    mean = np.mean(X_np, axis=0)
    std = np.std(X_np, axis=0)
    return mean, std


def compute_feature_pertinence(feature_stats_list, beta_fp=1.0):
    """
    Derives Feature-Pertinence score for each client (Fed-FP3 Novelty Component 1).
    
    Mathematical Formulation:
    -------------------------
    1. Federated Global Feature Distribution:
       Let client m have sample size n_m, mean vector mu_m, std vector sigma_m.
       The federated global mean is:
           mu_g = (sum_m n_m * mu_m) / (sum_m n_m)
       The federated global variance is:
           sigma_g^2 = sum_m [n_m * (sigma_m^2 + (mu_m - mu_g)^2)] / (sum_m n_m)
           sigma_g = sqrt(sigma_g^2 + EPS)
       
    2. Per-Feature Divergence:
       Standardized 2-Wasserstein distance across distribution parameters:
           delta_{m, j} = [ (mu_{m, j} - mu_{g, j})^2 + (sigma_{m, j} - sigma_{g, j})^2 ] / (sigma_{g, j}^2 + EPS)
       Safely handling constant features: if sigma_{g, j}^2 < EPS, delta_{m, j} = 0.
       
    3. Composite Feature Divergence:
           FD_m = (1 / d) * sum_{j=1}^d delta_{m, j}
       
    4. Relative Feature Divergence:
       To ensure scale invariance across datasets:
           mean_FD = mean_k(FD_k)
           rel_FD_m = FD_m / (mean_FD + EPS)
           
    5. Feature-Pertinence Score (FP_m):
           FP_m = exp(-beta_fp * rel_FD_m)
       Clients whose feature distributions are well-aligned with the global
       feature space exhibit low rel_FD_m and high FP_m in (0, 1].
    """
    M = len(feature_stats_list)
    n = np.array([s["n"] for s in feature_stats_list], dtype=np.float64)
    tot_n = np.sum(n)
    if tot_n <= 0:
        return np.zeros(M), np.ones(M), np.ones(M)
    
    means = np.stack([s["mean"] for s in feature_stats_list], axis=0) # (M, d)
    stds = np.stack([s["std"] for s in feature_stats_list], axis=0)   # (M, d)
    d = means.shape[1]
    
    # Global mean and variance
    weights_n = (n / tot_n)[:, None] # (M, 1)
    g_mean = np.sum(weights_n * means, axis=0) # (d,)
    g_var = np.sum(weights_n * (stds**2 + (means - g_mean)**2), axis=0) # (d,)
    g_std = np.sqrt(np.maximum(g_var, 0.0) + EPS)
    
    FD = np.zeros(M, dtype=np.float64)
    for m in range(M):
        diff_mean_sq = (means[m] - g_mean) ** 2
        diff_std_sq = (stds[m] - g_std) ** 2
        var_denom = g_var + EPS
        feature_deltas = np.where(g_var < 1e-7, 0.0, (diff_mean_sq + diff_std_sq) / var_denom)
        FD[m] = float(np.mean(feature_deltas))
        
    mean_fd = float(np.mean(FD))
    rel_fd = FD / (mean_fd + EPS) if mean_fd > 0 else np.ones(M)
    FP = np.exp(-beta_fp * rel_fd)
    
    # Guard against non-finite
    FP = np.nan_to_num(FP, nan=1.0, posinf=1.0, neginf=1.0)
    norm_FP = normalise(FP)
    return FD, FP, norm_FP


def compute_weights(method, n, counts, buffers, lam, gamma, window,
                    feature_stats=None, beta_fp=1.0, alpha_fp=1.0):
    n = np.asarray(n, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    P_g = counts.sum(0) / counts.sum()                         # global label distribution
    P_m = counts / np.maximum(counts.sum(1, keepdims=True), 1)
    D = np.array([kl_divergence(P_m[m], P_g) for m in range(len(n))])
    S = np.array([stability_score(buffers[m], window) for m in range(len(n))])
    div_factor = np.exp(-lam * D)
    stab_factor = S ** gamma
    w_fedavg = normalise(n)
    w_after_div = normalise(n * div_factor)
    w_raw_fedprada = n * div_factor * stab_factor              # Eq.(7)
    w_fedprada = normalise(w_raw_fedprada)
    
    # Feature Pertinence (Fed-FP3)
    if feature_stats is not None:
        FD, FP, norm_FP = compute_feature_pertinence(feature_stats, beta_fp=beta_fp)
    else:
        FD = np.zeros(len(n))
        FP = np.ones(len(n))
        norm_FP = normalise(FP)
        
    fp_factor = FP ** alpha_fp
    w_raw_fp3 = n * div_factor * stab_factor * fp_factor
    w_fed_fp3 = normalise(w_raw_fp3)
    
    if method == "fed_fp3":
        w = w_fed_fp3
        w_raw = w_raw_fp3
    elif method == "fedprada":
        w = w_fedprada
        w_raw = w_raw_fedprada
    else: # fedavg
        w = w_fedavg
        w_raw = n
        
    return dict(P_g=P_g, P_m=P_m, D=D, S=S, div_factor=div_factor,
                stab_factor=stab_factor, FD=FD, FP=FP, fp_factor=fp_factor,
                norm_FP=norm_FP, w_fedavg=w_fedavg, w_after_div=w_after_div,
                w_raw=w_raw, w_fedprada=w_fedprada, w_fed_fp3=w_fed_fp3, w=w)


def fuse_states(states, weights):
    """theta_g^{t+1} = sum_m w_m theta_m  (every float tensor, incl. BN buffers)."""
    fused = {}
    for k in states[0]:
        if states[0][k].dtype.is_floating_point:
            acc = torch.zeros_like(states[0][k])
            for w, s in zip(weights, states):
                acc += float(w) * s[k]
            fused[k] = acc
        else:
            fused[k] = states[0][k].clone()        # e.g. BatchNorm num_batches_tracked
    return fused


def flat_params(state):
    return torch.cat([v.flatten().float() for v in state.values()
                      if v.dtype.is_floating_point])


# ============================================================
# FEDERATED TRAINING LOOP (Modules 2 + 3)
# ============================================================

def run_federated(ds_key, method, tensors, num_classes, input_dim, rounds,
                  lam, gamma, mu, batch_size, sample_fraction, window,
                  beta_fp=BETA_FP, alpha_fp=ALPHA_FP, tag="",
                  secure_clients=None, coordinator=None):
    cfg = DATASETS[ds_key]
    clients = cfg["clients"]
    M = len(clients)
    lr = LEARNING_RATE[ds_key]

    set_seed(SEED)
    global_model = create_model(ds_key, input_dim, num_classes)        # server init
    local_model = create_model(ds_key, input_dim, num_classes)

    buffers = [[] for _ in range(M)]
    keys = ["n", "counts", "D", "S", "div_factor", "stab_factor", "FD", "FP",
            "fp_factor", "w_fedavg", "w_after_div", "w_raw", "w_fedprada",
            "w_fed_fp3", "w", "local_val_loss", "local_val_acc",
            "local_train_loss", "g_val_loss_c", "g_val_acc_c", "dist_to_fused"]
    H = {k: [] for k in keys}
    H.update(g_val_loss=[], g_val_acc=[], g_val_f1=[], g_step=[], round_time=[])

    best_acc, best_state, best_round = -1.0, None, 0
    val_X = torch.cat([tensors[c]["Xva"] for c in clients])
    val_y = torch.cat([tensors[c]["yva"] for c in clients])

    for t in range(1, rounds + 1):
        t0 = time.time()
        g_state = {k: v.detach().clone() for k, v in global_model.state_dict().items()}
        g_params = [p.detach().clone() for p in global_model.parameters()]
        if coordinator is not None:
            coordinator.begin_round(ds_key, t, g_state)

        states, n_list, count_list, feature_stats = [], [], [], []
        l_val_loss, l_val_acc, l_train = [], [], []

        for m, cl in enumerate(clients):
            d = tensors[cl]
            if secure_clients is not None:
                broadcast = secure_clients[cl].exchange(
                    {
                        "action": "broadcast",
                        "dataset": ds_key,
                        "round": t,
                        "metadata": {},
                    }
                )
                client_global_state = broadcast["model_state"]
            else:
                client_global_state = g_state
            local_model.load_state_dict(client_global_state)           # theta_m <- theta_g
            rs = np.random.RandomState(SEED + 1000 * t + m)
            n_all = len(d["ytr"])
            if sample_fraction < 1.0:
                idx = rs.choice(n_all, size=max(int(sample_fraction * n_all), 2),
                                replace=False)
            else:
                idx = np.arange(n_all)
            idx_t = torch.tensor(idx, device=device)
            Xr, yr = d["Xtr"][idx_t], d["ytr"][idx_t]

            torch.manual_seed(SEED + 1000 * t + m)
            gen = torch.Generator().manual_seed(SEED + 1000 * t + m)
            tr_loss = local_update(local_model, g_params, Xr, yr, lr,
                                   LOCAL_EPOCHS, batch_size, mu, gen)

            vl, va, _ = evaluate(local_model, d["Xva"], d["yva"])      # L_m^(t)
            local_state = {
                k: v.detach().clone()
                for k, v in local_model.state_dict().items()
            }
            xr_np = Xr.detach().cpu().numpy()
            m_mean, m_std = compute_feature_statistics(xr_np)

            if secure_clients is not None:
                label_counts = np.bincount(
                    yr.cpu().numpy(),
                    minlength=num_classes,
                )
                acknowledgment = secure_clients[cl].exchange(
                    {
                        "action": "update",
                        "dataset": ds_key,
                        "round": t,
                        "metadata": {
                            "num_train_samples": int(len(yr)),
                            "label_counts": label_counts.tolist(),
                            "validation_loss": float(vl),
                            "validation_accuracy": float(va),
                            "training_loss": float(tr_loss),
                            "feature_mean": m_mean.tolist(),
                            "feature_std": m_std.tolist(),
                        },
                        "model_state": local_state,
                    }
                )
                if acknowledgment.get("metadata", {}).get("accepted") is not True:
                    raise RuntimeError(
                        f"Server did not accept the update from client {cl}."
                    )
                received = coordinator.updates[cl]
                update_metadata = received["metadata"]
                states.append(received["model_state"])
                n_list.append(update_metadata["num_train_samples"])
                count_list.append(
                    np.asarray(update_metadata["label_counts"], dtype=np.int64)
                )
                feature_stats.append(
                    {
                        "mean": np.asarray(
                            update_metadata["feature_mean"], dtype=np.float64
                        ),
                        "std": np.asarray(
                            update_metadata["feature_std"], dtype=np.float64
                        ),
                        "n": update_metadata["num_train_samples"],
                    }
                )
                vl = update_metadata["validation_loss"]
                va = update_metadata["validation_accuracy"]
                tr_loss = update_metadata["training_loss"]
            else:
                states.append(local_state)
                n_list.append(len(yr))
                count_list.append(
                    np.bincount(yr.cpu().numpy(), minlength=num_classes)
                )
                feature_stats.append(dict(mean=m_mean, std=m_std, n=len(yr)))

            buffers[m].append(vl)                                      # loss buffer B_m
            l_val_loss.append(vl); l_val_acc.append(va); l_train.append(tr_loss)

        # ---------------- server ----------------
        W = compute_weights(method, n_list, count_list, buffers,
                            lam, gamma, window,
                            feature_stats=feature_stats,
                            beta_fp=beta_fp, alpha_fp=alpha_fp)
        fused = fuse_states(states, W["w"])
        old_flat = flat_params(g_state)
        global_model.load_state_dict(fused)
        new_flat = flat_params(fused)
        dist = [float(torch.norm(flat_params(s) - new_flat).item()) for s in states]
        step = float(torch.norm(new_flat - old_flat).item())

        # ---------------- evaluate fused global model ----------------
        gl, ga = [], []
        for cl in clients:
            l, a, _ = evaluate(global_model, tensors[cl]["Xva"], tensors[cl]["yva"])
            gl.append(l); ga.append(a)
        pl, pa, pp = evaluate(global_model, val_X, val_y)
        pf1 = float(f1_score(val_y.cpu().numpy(), pp.cpu().numpy(),
                             average="weighted", zero_division=0))

        for k, v in [("n", n_list), ("counts", count_list), ("D", W["D"]),
                     ("S", W["S"]), ("div_factor", W["div_factor"]),
                     ("stab_factor", W["stab_factor"]), ("FD", W["FD"]),
                     ("FP", W["FP"]), ("fp_factor", W["fp_factor"]),
                     ("w_fedavg", W["w_fedavg"]), ("w_after_div", W["w_after_div"]),
                     ("w_raw", W["w_raw"]), ("w_fedprada", W["w_fedprada"]),
                     ("w_fed_fp3", W["w_fed_fp3"]), ("w", W["w"]),
                     ("local_val_loss", l_val_loss), ("local_val_acc", l_val_acc),
                     ("local_train_loss", l_train), ("g_val_loss_c", gl),
                     ("g_val_acc_c", ga), ("dist_to_fused", dist)]:
            H[k].append(np.asarray(v, dtype=np.float64))
        H["g_val_loss"].append(pl); H["g_val_acc"].append(pa)
        H["g_val_f1"].append(pf1); H["g_step"].append(step)
        H["round_time"].append(time.time() - t0)

        if pa > best_acc:
            best_acc, best_round = pa, t
            best_state = {k: v.detach().cpu().clone()
                          for k, v in global_model.state_dict().items()}

        if t == 1 or t % 5 == 0 or t == rounds:
            fp_str = ", ".join(f"{x:.3f}" for x in W["FP"])
            print(f"[{cfg['name']}{tag}|{method}] round {t:3d}/{rounds} | "
                  f"global val acc={pa:.4f} loss={pl:.4f} f1={pf1:.4f} | "
                  f"D=[{', '.join(f'{x:.4f}' for x in W['D'])}] | "
                  f"FP=[{fp_str}] | "
                  f"w=[{', '.join(f'{x:.3f}' for x in W['w'])}] "
                  f"({time.time() - t0:.1f}s)")

    for k in keys:
        H[k] = np.stack(H[k])                    # (T, M) or (T, M, C)
    for k in ["g_val_loss", "g_val_acc", "g_val_f1", "g_step", "round_time"]:
        H[k] = np.asarray(H[k])
    final_state = {k: v.detach().cpu().clone()
                   for k, v in global_model.state_dict().items()}
    return H, best_state, best_round, final_state


# ============================================================
# PLOTS
# ============================================================

COLORS = plt.cm.tab10.colors
MARK = ["o", "s", "^", "D", "v"]


def _lines(ax, x, Y, labels, ylabel, title, **kw):
    for m in range(Y.shape[1]):
        ax.plot(x, Y[:, m], color=COLORS[m], marker=MARK[m % 5], markersize=3,
                linewidth=1.3, label=labels[m], **kw)
    ax.set_xlabel("Communication round"); ax.set_ylabel(ylabel)
    ax.set_title(title); ax.grid(alpha=0.3)


def _heat(ax, fig, Y, labels, title, cmap="viridis"):
    T, M = Y.shape
    im = ax.imshow(Y.T, aspect="auto", origin="lower", cmap=cmap,
                   extent=[0.5, T + 0.5, -0.5, M - 0.5])
    ax.set_yticks(range(M)); ax.set_yticklabels(labels)
    ax.set_xlabel("Communication round"); ax.set_title(title)
    fig.colorbar(im, ax=ax)


def plot_global(H, out, name, best_round, H_base=None):
    T = len(H["g_val_acc"]); x = np.arange(1, T + 1)
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"{name} - FedPRADA global model per round", fontsize=14)
    series = [("g_val_acc", "Validation accuracy", ax[0, 0]),
              ("g_val_loss", "Validation loss", ax[0, 1]),
              ("g_val_f1", "Validation weighted F1", ax[1, 0])]
    for key, lab, a in series:
        a.plot(x, H[key], color="crimson", marker="o", markersize=3, linewidth=1.6,
               label="FedPRADA")
        if H_base is not None:
            a.plot(x, H_base[key], color="gray", linestyle="--", linewidth=1.4,
                   label="FedAvg")
        a.axvline(best_round, color="green", linestyle=":", alpha=0.7)
        a.set_xlabel("Communication round"); a.set_ylabel(lab)
        a.set_title(lab); a.grid(alpha=0.3); a.legend()
    a = ax[1, 1]
    a.plot(x, H["g_step"], color="purple", marker="o", markersize=3,
           label="||theta_g(t+1) - theta_g(t)||")
    a.set_xlabel("Communication round"); a.set_ylabel("Parameter step norm")
    a.set_title("How much the fused global model moves each round")
    a.grid(alpha=0.3); a.legend()
    fig.tight_layout(); fig.savefig(os.path.join(out, "global_model_performance.png"), dpi=200)
    plt.close(fig)


def plot_kl(H, out, name, clients, lam):
    T = len(H["D"]); x = np.arange(1, T + 1)
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"{name} - label-distribution divergence D_m = KL(P_m || P_g)", fontsize=14)
    _lines(ax[0, 0], x, H["D"], clients, "KL divergence D_m", "D_m per client per round")
    ax[0, 0].legend(fontsize=8)
    _heat(ax[0, 1], fig, H["D"], clients, "D_m heat-map (client x round)", "magma")
    _lines(ax[1, 0], x, H["div_factor"], clients, "exp(-lambda_div * D_m)",
           f"Divergence factor (lambda_div={lam:g})")
    a = ax[1, 1]
    P = H["counts"][-1] / H["counts"][-1].sum(1, keepdims=True)
    Pg = H["counts"][-1].sum(0) / H["counts"][-1].sum()
    C = P.shape[1]; width = 0.8 / (len(clients) + 1); xs = np.arange(C)
    for m, cl in enumerate(clients):
        a.bar(xs + m * width, P[m], width, color=COLORS[m], label=cl)
    a.bar(xs + len(clients) * width, Pg, width, color="black", label="global P_g")
    a.set_xticks(xs + 0.4); a.set_xticklabels([f"class {i}" for i in range(C)])
    a.set_ylabel("Probability"); a.set_title("Reported label distributions (last round)")
    a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out, "kl_divergence_per_round.png"), dpi=200)
    plt.close(fig)


def plot_stability(H, out, name, clients, gamma, window):
    T = len(H["S"]); x = np.arange(1, T + 1)
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"{name} - training stability S_m = 1/(1+std(last {window} val losses))",
                 fontsize=14)
    _lines(ax[0, 0], x, H["S"], clients, "S_m", "Stability score S_m per round")
    ax[0, 0].legend(fontsize=8)
    _heat(ax[0, 1], fig, H["S"], clients, "S_m heat-map (client x round)", "viridis")
    _lines(ax[1, 0], x, H["local_val_loss"], clients, "Local validation loss",
           "Raw signal: local validation loss L_m(t) (feeds the buffer B_m)")
    _lines(ax[1, 1], x, H["stab_factor"], clients, f"S_m^gamma (gamma={gamma:g})",
           "Stability factor used in Eq.(7)")
    fig.tight_layout(); fig.savefig(os.path.join(out, "stability_per_round.png"), dpi=200)
    plt.close(fig)


def plot_client_validation(H, out, name, clients):
    T = len(H["S"]); x = np.arange(1, T + 1)
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"{name} - client validation per round", fontsize=14)
    _lines(ax[0, 0], x, H["local_val_loss"], clients, "Loss",
           "Local model (after local training): validation loss")
    ax[0, 0].legend(fontsize=8)
    _lines(ax[0, 1], x, H["local_val_acc"], clients, "Accuracy",
           "Local model (after local training): validation accuracy")
    _lines(ax[1, 0], x, H["g_val_acc_c"], clients, "Accuracy",
           "FUSED global model on each client's validation set")
    _lines(ax[1, 1], x, H["g_val_loss_c"], clients, "Loss",
           "FUSED global model on each client's validation set: loss")
    fig.tight_layout(); fig.savefig(os.path.join(out, "client_validation_per_round.png"), dpi=200)
    plt.close(fig)


def plot_fusion_weights(H, out, name, clients):
    T = len(H["w"]); x = np.arange(1, T + 1)
    fig, ax = plt.subplots(3, 1, figsize=(13, 13), sharex=False)
    fig.suptitle(f"{name} - weighted fusion: aggregation weight w_m per round", fontsize=14)
    ax[0].stackplot(x, H["w"].T, labels=clients, colors=COLORS[:len(clients)], alpha=0.85)
    ax[0].set_ylim(0, 1); ax[0].set_ylabel("w_m (sums to 1)")
    ax[0].set_title("FedPRADA weights (stacked)"); ax[0].legend(loc="upper right", fontsize=8)
    _lines(ax[1], x, H["w"], clients, "w_m", "FedPRADA weight of each client")
    ax[1].legend(fontsize=8)
    _lines(ax[2], x, H["w"] - H["w_fedavg"], clients, "w_m(FedPRADA) - w_m(FedAvg)",
           "Change relative to plain FedAvg (n_m-only) weights")
    ax[2].axhline(0, color="black", linewidth=0.8)
    fig.tight_layout(); fig.savefig(os.path.join(out, "weighted_fusion_weights.png"), dpi=200)
    plt.close(fig)


def selected_rounds(T):
    return [r for r in sorted(set([1, 2, max(T // 2, 1), T])) if 1 <= r <= T]


def plot_fusion_breakdown(H, out, name, clients, lam, gamma):
    T = len(H["w"]); rs = selected_rounds(T)[:4]
    fig, axes = plt.subplots(1, len(rs), figsize=(5.2 * len(rs), 5.5), squeeze=False)
    fig.suptitle(f"{name} - how the weights are built:  w ~ n_m  x exp(-{lam:g} D_m)  x S_m^{gamma:g}",
                 fontsize=13)
    M = len(clients); xs = np.arange(M); width = 0.27
    for a, r in zip(axes[0], rs):
        i = r - 1
        a.bar(xs - width, H["w_fedavg"][i], width, color="#9aa5b1", label="1) n_m only (FedAvg)")
        a.bar(xs, H["w_after_div"][i], width, color="#f4a261", label="2) x exp(-lambda D_m)")
        a.bar(xs + width, H["w"][i], width, color="#e63946", label="3) x S_m^gamma = FINAL w_m")
        a.set_xticks(xs); a.set_xticklabels(clients, rotation=20)
        a.set_title(f"Round {r}"); a.set_ylabel("normalised weight"); a.grid(alpha=0.3, axis="y")
    axes[0][0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, "weighted_fusion_breakdown.png"), dpi=200)
    plt.close(fig)


def plot_fusion_movement(H, out, name, clients):
    T = len(H["dist_to_fused"]); x = np.arange(1, T + 1)
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
    fig.suptitle(f"{name} - parameter-space view of the fusion", fontsize=14)
    _lines(ax[0], x, H["dist_to_fused"], clients, "||theta_m - theta_g(t+1)||",
           "Distance of each local model to the fused model")
    ax[0].legend(fontsize=8)
    a = ax[1]
    a.plot(x, H["g_step"], color="purple", marker="o", markersize=3)
    a.set_xlabel("Communication round"); a.set_ylabel("||theta_g(t+1) - theta_g(t)||")
    a.set_title("Global model step per round"); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fusion_parameter_movement.png"), dpi=200)
    plt.close(fig)


# ============================================================
# FILES : fusion log / report
# ============================================================

def save_fusion_files(H, out, name, clients, class_values, lam, gamma, window, mu, method,
                      beta_fp=1.0, alpha_fp=1.0):
    T = len(H["w"]); rows = []
    for t in range(T):
        for m, cl in enumerate(clients):
            rows.append(dict(
                round=t + 1, client=cl, n_m=int(H["n"][t, m]),
                label_counts=H["counts"][t, m].astype(int).tolist(),
                KL_D_m=float(H["D"][t, m]), exp_neg_lambda_D=float(H["div_factor"][t, m]),
                local_val_loss=float(H["local_val_loss"][t, m]),
                stability_S_m=float(H["S"][t, m]), S_pow_gamma=float(H["stab_factor"][t, m]),
                feature_divergence_FD_m=float(H["FD"][t, m]) if "FD" in H else 0.0,
                feature_pertinence_FP_m=float(H["FP"][t, m]) if "FP" in H else 1.0,
                fp_factor=float(H["fp_factor"][t, m]) if "fp_factor" in H else 1.0,
                raw_weight=float(H["w_raw"][t, m]),
                weight_fedavg=float(H["w_fedavg"][t, m]),
                weight_after_divergence=float(H["w_after_div"][t, m]),
                weight_fedprada=float(H["w_fedprada"][t, m]) if "w_fedprada" in H else float(H["w_raw"][t, m]),
                weight_fed_fp3=float(H["w_fed_fp3"][t, m]) if "w_fed_fp3" in H else float(H["w"][t, m]),
                weight_final=float(H["w"][t, m]),
                local_val_acc=float(H["local_val_acc"][t, m]),
                global_model_val_acc_on_client=float(H["g_val_acc_c"][t, m]),
                dist_local_to_fused=float(H["dist_to_fused"][t, m])))
    pd.DataFrame(rows).to_csv(os.path.join(out, "fusion_log.csv"), index=False)

    pd.DataFrame(dict(round=np.arange(1, T + 1), global_val_loss=H["g_val_loss"],
                      global_val_acc=H["g_val_acc"], global_val_f1=H["g_val_f1"],
                      global_step_norm=H["g_step"], round_seconds=H["round_time"])
                 ).to_csv(os.path.join(out, "global_metrics.csv"), index=False)

    report_title = "Fed-FP3" if method == "fed_fp3" else "FedPRADA"
    lines = [f"{report_title} weighted fusion report - {name}",
             f"method={method}  lambda_div={lam}  gamma_stab={gamma}  beta_fp={beta_fp}  alpha_fp={alpha_fp}  window={window}  mu_prox={mu}",
             "Formula:  w_m ~ n_m * exp(-lambda_div * D_m) * S_m^gamma * FP_m^alpha ; ",
             "where FP_m = exp(-beta_fp * (FD_m / mean_FD))",
             "theta_g(t+1) = sum_m w_m * theta_m", ""]
    for r in selected_rounds(T):
        i = r - 1
        lines.append("=" * 135)
        lines.append(f"ROUND {r}   global val acc after fusion = {H['g_val_acc'][i]:.4f}   "
                     f"loss = {H['g_val_loss'][i]:.4f}")
        lines.append(f"{'client':<13}{'n_m':>7}{'D_m':>11}{'exp(-lD)':>10}{'L_m':>9}"
                     f"{'S_m':>8}{'FD_m':>9}{'FP_m':>9}{'w FedAvg':>10}{'w FedPRADA':>12}{'w FINAL':>10}")
        for m, cl in enumerate(clients):
            fd_val = float(H["FD"][i, m]) if "FD" in H else 0.0
            fp_val = float(H["FP"][i, m]) if "FP" in H else 1.0
            w_prada = float(H["w_fedprada"][i, m]) if "w_fedprada" in H else float(H["w"][i, m])
            lines.append(f"{cl:<13}{int(H['n'][i, m]):>7}{H['D'][i, m]:>11.6f}"
                         f"{H['div_factor'][i, m]:>10.4f}{H['local_val_loss'][i, m]:>9.4f}"
                         f"{H['S'][i, m]:>8.4f}{fd_val:>9.4f}{fp_val:>9.4f}"
                         f"{H['w_fedavg'][i, m]:>10.4f}{w_prada:>12.4f}"
                         f"{H['w'][i, m]:>10.4f}")
        lines.append(f"sum of final weights = {H['w'][i].sum():.6f}")
        lines.append("global label distribution P_g = " +
                     str(np.round(H['counts'][i].sum(0) / H['counts'][i].sum(), 4).tolist()) +
                     "   classes = " + str(class_values))
        lines.append("")
    with open(os.path.join(out, "fusion_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def plot_feature_pertinence(H, out, name, clients, beta_fp):
    if "FD" not in H or "FP" not in H:
        return
    T = len(H["FP"]); x = np.arange(1, T + 1)
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"{name} - Fed-FP3 Feature-Pertinence & Distribution Discrepancy (beta={beta_fp:g})",
                 fontsize=14)
    _lines(ax[0, 0], x, H["FD"], clients, "Feature Divergence FD_m", "FD_m per client per round")
    ax[0, 0].legend(fontsize=8)
    _heat(ax[0, 1], fig, H["FD"], clients, "FD_m heat-map (client x round)", "plasma")
    _lines(ax[1, 0], x, H["FP"], clients, "Feature-Pertinence FP_m", "FP_m = exp(-beta * rel_FD_m)")
    ax[1, 0].legend(fontsize=8)
    _heat(ax[1, 1], fig, H["FP"], clients, "FP_m heat-map (client x round)", "viridis")
    fig.tight_layout(); fig.savefig(os.path.join(out, "feature_pertinence_analysis.png"), dpi=200)
    plt.close(fig)


def plot_weights_comparison(H, out, name, clients):
    if "w_fedprada" not in H:
        return
    T = len(H["w"]); x = np.arange(1, T + 1)
    fig, ax = plt.subplots(3, 1, figsize=(13, 13))
    fig.suptitle(f"{name} - Fed-FP3 Aggregation Weights & Comparison with FedPRADA", fontsize=14)
    ax[0].stackplot(x, H["w"].T, labels=clients, colors=COLORS[:len(clients)], alpha=0.85)
    ax[0].set_ylim(0, 1); ax[0].set_ylabel("w_m (sums to 1)")
    ax[0].set_title("Fed-FP3 weights (stacked)"); ax[0].legend(loc="upper right", fontsize=8)
    _lines(ax[1], x, H["w"], clients, "w_m", "Fed-FP3 weight of each client")
    ax[1].legend(fontsize=8)
    _lines(ax[2], x, H["w"] - H["w_fedprada"], clients, "w_m(Fed-FP3) - w_m(FedPRADA)",
           "Shift in aggregation weight due to Feature-Pertinence")
    ax[2].axhline(0, color="black", linewidth=0.8)
    ax[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, "weights_comparison.png"), dpi=200)
    plt.close(fig)


# ============================================================
# FED-FP3 NOVELTY: PREDICTION-BASED DELIVERY RISK PRIORITIZATION
# ============================================================

def assign_risk_priority(ds_key, predicted_label, confidence):
    """
    Assigns risk priority category and recommended operational action
    based on predicted delivery outcome and prediction confidence.
    """
    conf_pct = float(confidence) * 100.0
    if ds_key == "dataco":
        if predicted_label == "Late delivery":
            if confidence >= 0.70:
                prio = "CRITICAL"
                action = "Expedite shipment immediately via priority routing; issue proactive customer delay notification."
                risk_score = 0.50 + 0.50 * confidence
            elif confidence >= 0.50:
                prio = "HIGH"
                action = "Flag order for active transit monitoring; contact freight carrier for milestone update."
                risk_score = 0.40 + 0.40 * confidence
            else:
                prio = "MEDIUM"
                action = "Verify delivery status with local logistics hub; audit telemetry logs."
                risk_score = 0.30 + 0.30 * confidence
        elif predicted_label == "Shipping canceled":
            if confidence >= 0.60:
                prio = "HIGH"
                action = "Investigate fulfillment cancellation risk; verify warehouse inventory and order clearance."
                risk_score = 0.45 + 0.35 * confidence
            else:
                prio = "MEDIUM"
                action = "Audit order payment and inventory status with distribution coordinator."
                risk_score = 0.30 + 0.30 * confidence
        else: # "Advance shipping" or "Shipping on time"
            if confidence >= 0.70:
                prio = "LOW"
                action = "Standard dispatch schedule; shipment progressing on track."
                risk_score = 0.20 * (1.0 - confidence)
            else:
                prio = "LOW-MEDIUM"
                action = "Monitor regular milestone check due to marginal model prediction certainty."
                risk_score = 0.35 * (1.0 - confidence)
    else: # ecommerce
        is_late = str(predicted_label) in ["Not on-time", "1"]
        if is_late:
            if confidence >= 0.75:
                prio = "CRITICAL"
                action = "Upgrade carrier shipping tier immediately; prioritize warehouse pick/pack queue."
                risk_score = 0.50 + 0.50 * confidence
            elif confidence >= 0.60:
                prio = "HIGH"
                action = "Coordinate with regional fulfillment center to expedite shipping dispatch."
                risk_score = 0.40 + 0.40 * confidence
            else:
                prio = "MEDIUM"
                action = "Cross-check warehouse block dispatch queue and verify flight/truck schedule."
                risk_score = 0.30 + 0.30 * confidence
        else:
            if confidence >= 0.70:
                prio = "LOW"
                action = "Proceed with standard logistics handling; order on schedule."
                risk_score = 0.20 * (1.0 - confidence)
            else:
                prio = "LOW-MEDIUM"
                action = "Review upcoming transit milestone; low model confidence on on-time status."
                risk_score = 0.35 * (1.0 - confidence)
    return prio, action, float(risk_score)


def evaluate_and_prioritize(model, val_X, val_y, ds_key, out,
                            class_values, num_classes, split_meta=None):
    """
    Evaluates global Fed-FP3 model independently on held-out split and generates
    the delivery risk priority list.
    """
    model.eval()
    probabilities_list = []
    with torch.no_grad():
        for start in range(0, len(val_X), 4096):
            batch = val_X[start:start + 4096]
            logits = model(batch)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            probabilities_list.append(probs)
    probs = np.concatenate(probabilities_list, axis=0)
    y_pred = probs.argmax(axis=1)
    y_true = val_y.cpu().numpy()
    confidences = probs.max(axis=1)

    if ds_key == "dataco":
        class_names = ["Advance shipping", "Late delivery", "Shipping canceled", "Shipping on time"]
        label_map = {i: class_names[i] for i in range(min(num_classes, len(class_names)))}
    else:
        label_map = {0: "On-time", 1: "Not on-time"}

    acc = float(accuracy_score(y_true, y_pred))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    prec_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    rec_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_m = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_w = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    class_spec = []
    for i in range(num_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        denom = tn + fp
        class_spec.append(float(tn / denom) if denom > 0 else np.nan)
    spec_macro = float(np.nanmean(class_spec))

    try:
        if num_classes == 2:
            auc_val = float(roc_auc_score(y_true, probs[:, 1]))
        else:
            auc_val = float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro", labels=np.arange(num_classes)))
    except Exception:
        auc_val = float("nan")

    loss_val = float(F.cross_entropy(torch.tensor(probs).log().clamp(min=-100), torch.tensor(y_true)).item())

    # Save confusion matrix CSV and plot
    cm_df = pd.DataFrame(cm,
                         index=[f"Actual_{label_map.get(i, str(i))}" for i in range(num_classes)],
                         columns=[f"Pred_{label_map.get(i, str(i))}" for i in range(num_classes)])
    cm_df.to_csv(os.path.join(out, "confusion_matrix.csv"))

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    fig.colorbar(im, ax=ax)
    tick_marks = np.arange(num_classes)
    tick_labels = [label_map.get(i, str(i)) for i in range(num_classes)]
    ax.set_xticks(tick_marks); ax.set_xticklabels(tick_labels, rotation=30, ha="right")
    ax.set_yticks(tick_marks); ax.set_yticklabels(tick_labels)
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_ylabel("Actual label"); ax.set_xlabel("Predicted label")
    ax.set_title(f"{DATASETS[ds_key]['name']} - Fed-FP3 Confusion Matrix")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "confusion_matrix.png"), dpi=200)
    plt.close(fig)

    metrics_summary = {
        "dataset": DATASETS[ds_key]["name"],
        "method": "fed_fp3",
        "num_eval_samples": int(len(y_true)),
        "accuracy": acc,
        "precision_macro": prec_macro,
        "precision_weighted": prec_weighted,
        "recall_macro_sensitivity": rec_macro,
        "recall_weighted": rec_weighted,
        "f1_macro": f1_m,
        "f1_weighted": f1_w,
        "specificity_macro": spec_macro,
        "auc": auc_val,
        "validation_loss": loss_val,
        "class_specificity": class_spec,
        "class_labels": [label_map.get(i, str(i)) for i in range(num_classes)],
        "averaging_documentation": "Macro averaging computes metric independently per class and averages equally; Weighted averaging weights each class by its support.",
    }
    with open(os.path.join(out, "evaluation_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    pd.DataFrame([metrics_summary]).to_csv(os.path.join(out, "evaluation_metrics.csv"), index=False)

    # Shipment Delivery Risk Prioritization
    clients = DATASETS[ds_key]["clients"]
    rows = []
    offset = 0
    for cl in clients:
        if split_meta and cl in split_meta:
            n_cl = len(split_meta[cl]["val_indices"])
        else:
            n_cl = len(y_true) // len(clients)

        for k in range(n_cl):
            idx = offset + k
            if idx >= len(y_true):
                break
            pred_lbl = label_map.get(y_pred[idx], str(y_pred[idx]))
            act_lbl = label_map.get(y_true[idx], str(y_true[idx]))
            conf = float(confidences[idx])
            prio, action, risk_score = assign_risk_priority(ds_key, pred_lbl, conf)

            if ds_key == "dataco":
                shipment_id = f"Order_DC_{cl}_{k+1}"
                note = "Derived from client partition and order sequence"
            else:
                shipment_id = f"Ecomm_Block_{cl}_{k+1}"
                note = "Warehouse-block sample identifier (raw shipment ID was excluded in preprocessing)"

            row = {
                "Shipment_ID": shipment_id,
                "Client_Partition": cl,
                "Predicted_Status": pred_lbl,
                "Actual_Status": act_lbl,
                "Confidence": conf,
                "Confidence_Percentage": f"{conf * 100:.2f}%",
                "Risk_Priority": prio,
                "Recommended_Action": action,
                "Composite_Risk_Score": risk_score,
                "Correct_Prediction": bool(y_pred[idx] == y_true[idx]),
                "Identifier_Note": note,
            }
            for c_idx in range(num_classes):
                c_name = label_map.get(c_idx, str(c_idx))
                row[f"Prob_{c_name}"] = float(probs[idx, c_idx])
            rows.append(row)
        offset += n_cl

    prio_df = pd.DataFrame(rows).sort_values("Composite_Risk_Score", ascending=False)
    prio_df.to_csv(os.path.join(out, "shipment_risk_prioritization.csv"), index=False)
    print(f"[{DATASETS[ds_key]['name']}|fed_fp3] Evaluated {len(y_true)} samples. Accuracy: {acc:.4f}, Macro-F1: {f1_m:.4f}, AUC: {auc_val:.4f}")
    print(f"[{DATASETS[ds_key]['name']}|fed_fp3] Delivery risk priority table saved: {len(rows)} shipments.")
    return metrics_summary, prio_df


# ============================================================
# ONE DATASET
# ============================================================

def _run_dataset(ds_key, args, secure_clients=None, coordinator=None):
    cfg = DATASETS[ds_key]
    method = args.method
    methods_to_run = ["fedprada", "fed_fp3"] if method == "both" else [method]
    partition = args.partition if ds_key == "dataco" else "natural"

    tensors, class_values, num_classes, input_dim, (Hm, sHm), split_meta = load_dataset(ds_key, partition, return_meta=True)
    clients = cfg["clients"]
    lam = args.lambda_div if args.lambda_div is not None else LAMBDA_DIV[partition]
    bs = args.batch_size

    for m in methods_to_run:
        out = os.path.join(FP3_BASE_OUTPUT, ds_key) if m == "fed_fp3" else cfg["output_dir"]
        os.makedirs(out, exist_ok=True)

        print("\n" + "=" * 90)
        print(f"MODULE 3 - {cfg['name']} | Method: {m.upper()}   (partition: {partition})")
        print("=" * 90)
        print(f"classes={class_values} input_dim={input_dim} rounds={args.rounds} "
              f"local_epochs={LOCAL_EPOCHS} lr={LEARNING_RATE[ds_key]} mu={args.mu} "
              f"lambda_div={lam} gamma={args.gamma_stab} window={args.window} "
              f"batch={'full' if bs <= 0 else bs} sample_fraction={args.sample_fraction} "
              f"beta_fp={args.beta_fp} alpha_fp={args.alpha_fp}")

        H, best_state, best_round, final_state = run_federated(
            ds_key, m, tensors, num_classes, input_dim, args.rounds,
            lam, args.gamma_stab, args.mu, bs, args.sample_fraction, args.window,
            beta_fp=args.beta_fp, alpha_fp=args.alpha_fp,
            secure_clients=secure_clients, coordinator=coordinator)

        H_base = None
        if args.compare_fedavg and m != "fedavg":
            H_base, _, _, _ = run_federated(
                ds_key, "fedavg", tensors, num_classes, input_dim, args.rounds,
                lam, args.gamma_stab, args.mu, bs, args.sample_fraction, args.window,
                beta_fp=args.beta_fp, alpha_fp=args.alpha_fp,
                secure_clients=secure_clients, coordinator=coordinator)

        common = dict(input_dim=input_dim, num_classes=num_classes, class_values=class_values,
                      dataset=cfg["name"], clients=clients, method=m)
        torch.save(dict(model_state_dict=best_state, round=best_round, **common),
                   os.path.join(out, "global_model_best.pth"))
        torch.save(dict(model_state_dict=final_state, round=args.rounds, **common),
                   os.path.join(out, "global_model_final.pth"))

        plot_global(H, out, cfg["name"], best_round, H_base)
        plot_kl(H, out, cfg["name"], clients, lam)
        plot_stability(H, out, cfg["name"], clients, args.gamma_stab, args.window)
        plot_client_validation(H, out, cfg["name"], clients)
        plot_fusion_weights(H, out, cfg["name"], clients)
        plot_fusion_breakdown(H, out, cfg["name"], clients, lam, args.gamma_stab)
        plot_fusion_movement(H, out, cfg["name"], clients)
        save_fusion_files(H, out, cfg["name"], clients, class_values, lam,
                          args.gamma_stab, args.window, args.mu, m,
                          beta_fp=args.beta_fp, alpha_fp=args.alpha_fp)

        if m == "fed_fp3":
            plot_feature_pertinence(H, out, cfg["name"], clients, args.beta_fp)
            plot_weights_comparison(H, out, cfg["name"], clients)
            
            # Post-training evaluation and Delivery Risk Prioritization
            eval_model = create_model(ds_key, input_dim, num_classes)
            eval_model.load_state_dict(final_state)
            val_X = torch.cat([tensors[c]["Xva"] for c in clients])
            val_y = torch.cat([tensors[c]["yva"] for c in clients])
            eval_metrics, prio_df = evaluate_and_prioritize(
                eval_model, val_X, val_y, ds_key, out, class_values, num_classes, split_meta)

        summary = dict(
            dataset=cfg["name"], method=m, partition=partition, rounds=args.rounds,
            local_epochs=LOCAL_EPOCHS, learning_rate=LEARNING_RATE[ds_key],
            mu_prox=args.mu, lambda_div=lam, gamma_stab=args.gamma_stab,
            stability_window=args.window, batch_size=("full" if bs <= 0 else bs),
            round_sample_fraction=args.sample_fraction, beta_fp=args.beta_fp, alpha_fp=args.alpha_fp,
            label_entropy_H=Hm, label_entropy_sigma=sHm,
            best_round=best_round, best_global_val_acc=float(H["g_val_acc"][best_round - 1]),
            final_global_val_acc=float(H["g_val_acc"][-1]),
            final_global_val_loss=float(H["g_val_loss"][-1]),
            final_global_val_f1=float(H["g_val_f1"][-1]),
            D_range_over_rounds=[float(H["D"].min()), float(H["D"].max())],
            S_range_over_rounds=[float(H["S"].min()), float(H["S"].max())],
            FD_range_over_rounds=[float(H["FD"].min()), float(H["FD"].max())] if "FD" in H else [0, 0],
            FP_range_over_rounds=[float(H["FP"].min()), float(H["FP"].max())] if "FP" in H else [1, 1],
            final_weights={c: float(w) for c, w in zip(clients, H["w"][-1])},
            final_feature_pertinence={c: float(fp) for c, fp in zip(clients, H["FP"][-1])} if "FP" in H else {},
            total_train_seconds=float(H["round_time"].sum()))

        summary_name = "fed_fp3_summary.json" if m == "fed_fp3" else "module3_summary.json"
        with open(os.path.join(out, summary_name), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"\n{cfg['name']} ({m}) done. best round={best_round}  "
              f"best val acc={summary['best_global_val_acc']:.4f}  "
              f"final val acc={summary['final_global_val_acc']:.4f}")
        print("Results in:", out)


def run_dataset(ds_key, args, tls_config=None):
    if tls_config is None:
        return _run_dataset(ds_key, args)

    clients = DATASETS[ds_key]["clients"]
    missing_tokens = set(clients).difference(tls_config["client_tokens"])
    if missing_tokens:
        raise ValueError(
            f"Missing TLS client tokens for: {sorted(missing_tokens)}"
        )

    coordinator = FederatedRoundCoordinator(clients)
    with TLSFederatedServer(
        certfile=tls_config["certfile"],
        keyfile=tls_config["keyfile"],
        client_tokens=tls_config["client_tokens"],
        handler=coordinator.handle,
    ) as server:
        secure_clients = {
            client: TLSFederatedClient(
                host=server.address[0],
                port=server.address[1],
                ca_file=tls_config["ca_file"],
                client_id=client,
                token=tls_config["client_tokens"][client],
            )
            for client in clients
        }
        return _run_dataset(
            ds_key,
            args,
            secure_clients=secure_clients,
            coordinator=coordinator,
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="FedPRADA & Fed-FP3 Module 3 - adaptive aggregation")
    ap.add_argument("--dataset", choices=["dataco", "ecommerce", "both"], default="both")
    ap.add_argument("--method", choices=["fed_fp3", "fedprada", "both"], default="fed_fp3",
                    help="fed_fp3 (Fed-FP3 novelty), fedprada (original paper), or both")
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    ap.add_argument("--partition", choices=["natural", "mixed_shard"], default="natural",
                    help="DataCo only: natural market clients or paper's mixed-shard non-IID")
    ap.add_argument("--lambda_div", type=float, default=None)
    ap.add_argument("--gamma_stab", type=float, default=GAMMA_STAB)
    ap.add_argument("--beta_fp", type=float, default=BETA_FP,
                    help="Fed-FP3 feature pertinence sensitivity coefficient")
    ap.add_argument("--alpha_fp", type=float, default=ALPHA_FP,
                    help="Fed-FP3 feature pertinence exponent in aggregation weight")
    ap.add_argument("--window", type=int, default=STAB_WINDOW)
    ap.add_argument("--mu", type=float, default=MU_PROX)
    ap.add_argument("--batch_size", type=int, default=LOCAL_BATCH_SIZE,
                    help="0 = full batch (paper)")
    ap.add_argument("--sample_fraction", type=float, default=ROUND_SAMPLE_FRACTION,
                    help="fraction of local train split used per round (1.0 = paper)")
    ap.add_argument("--compare_fedavg", action="store_true")
    ap.add_argument("--data_root", default=None,
                    help="optional prefix for the processed/ folder")
    add_tls_arguments(ap)
    args = ap.parse_args()
    tls_config = tls_config_from_args(args)

    if args.data_root:
        for k in DATASETS:
            DATASETS[k]["data_dir"] = os.path.join(args.data_root, DATASETS[k]["data_dir"])

    print("Using device:", device)
    keys = ["dataco", "ecommerce"] if args.dataset == "both" else [args.dataset]
    for k in keys:
        run_dataset(k, args, tls_config=tls_config)

    print("\nALL DONE - Training completed.")