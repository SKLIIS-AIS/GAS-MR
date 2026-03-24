#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from data_partition import partition_data
from model import get_model

from torchvision import datasets, transforms


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def normalize_rows(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=1, keepdim=True) + eps)


def roc_auc_score_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    y_score = y_score.astype(np.float64)

    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(y_score), dtype=np.float64) + 1.0

    sorted_scores = y_score[order]
    i = 0
    while i < len(sorted_scores):
        j = i + 1
        while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        if j - i > 1:
            avg_rank = ranks[order[i:j]].mean()
            ranks[order[i:j]] = avg_rank
        i = j

    sum_ranks_pos = ranks[y_true == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray, fpr: float) -> float:
    y_true = y_true.astype(np.int64)
    y_score = y_score.astype(np.float64)

    out_scores = y_score[y_true == 0]
    in_scores = y_score[y_true == 1]
    if len(out_scores) == 0 or len(in_scores) == 0:
        return 0.0

    q = 1.0 - float(fpr)
    q = min(max(q, 0.0), 1.0)
    thr = np.quantile(out_scores, q)

    tpr = float((in_scores >= thr).mean())
    return tpr


def build_transforms(dataset: str):
    dataset = dataset.lower()
    if dataset == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
        return transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    if dataset == "cifar100":
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        return transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    if dataset in ("fashionmnist", "fmnist"):
        mean = (0.2860,)
        std = (0.3530,)
        return transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    raise ValueError(f"Unsupported dataset: {dataset}")


def load_dataset(dataset: str, datadir: str):
    dataset_l = dataset.lower()
    tf = build_transforms(dataset_l)

    if dataset_l == "cifar10":
        train_set = datasets.CIFAR10(root=datadir, train=True, download=False, transform=tf)
        test_set = datasets.CIFAR10(root=datadir, train=False, download=False, transform=tf)
        y_train = np.array(train_set.targets, dtype=np.int64)
        y_test = np.array(test_set.targets, dtype=np.int64)
        return train_set, test_set, y_train, y_test, 10, 3

    if dataset_l == "cifar100":
        train_set = datasets.CIFAR100(root=datadir, train=True, download=False, transform=tf)
        test_set = datasets.CIFAR100(root=datadir, train=False, download=False, transform=tf)
        y_train = np.array(train_set.targets, dtype=np.int64)
        y_test = np.array(test_set.targets, dtype=np.int64)
        return train_set, test_set, y_train, y_test, 100, 3

    if dataset_l in ("fashionmnist", "fmnist"):
        train_set = datasets.FashionMNIST(root=datadir, train=True, download=False, transform=tf)
        test_set = datasets.FashionMNIST(root=datadir, train=False, download=False, transform=tf)
        y_train = np.array(train_set.targets, dtype=np.int64)
        y_test = np.array(test_set.targets, dtype=np.int64)
        return train_set, test_set, y_train, y_test, 10, 1

    raise ValueError(f"Unsupported dataset: {dataset}")


def load_checkpoint_state_dict(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")

    if p.suffix.lower() in [".pkl", ".pickle", ".pt", ".pth"]:
        obj = torch.load(p, map_location="cpu", pickle_module=pickle)
    else:
        obj = torch.load(p, map_location="cpu")

    if isinstance(obj, dict):
        for key in ["state_dict", "model_state_dict", "net_state_dict", "global_w", "model", "net"]:
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break

    if not isinstance(obj, dict):
        raise TypeError(f"checkpoint is not state_dict: {type(obj)}")
    return obj


def build_model(model_name: str, dataset: str, num_classes: int, in_channels: int, device: torch.device):
    class ArgsLike:
        pass

    args_like = ArgsLike()
    args_like.model = model_name
    args_like.channel = int(in_channels)

    image_size = 32 if dataset.lower() in ("cifar10", "cifar100") else 28
    cfg = {"image_size": image_size}

    builder_or_model = get_model(args_like, cfg)

    if isinstance(builder_or_model, torch.nn.Module):
        model = builder_or_model
    else:
        if callable(builder_or_model):
            model = builder_or_model(num_classes)
        else:
            raise TypeError(f"invalid model builder: {type(builder_or_model)}")

    model.to(device)
    return model


def split_base_and_classifier(model: torch.nn.Module):
    m = model.module if hasattr(model, "module") else model
    base = m.base

    if hasattr(m, "classifier") and hasattr(m.classifier, "fc"):
        W = m.classifier.fc.weight.detach()
    elif hasattr(m, "classifier") and hasattr(m.classifier, "weight"):
        W = m.classifier.weight.detach()
    elif hasattr(m, "fc") and hasattr(m.fc, "weight"):
        W = m.fc.weight.detach()
    else:
        raise AttributeError("cannot find classifier weight")

    return base, W


def build_client_map(args, y_train: np.ndarray) -> Dict[int, List[int]]:
    seed = int(getattr(args, "seed", 0))
    root = getattr(args, "partition_assets_root", "ours_two/partition_assets")
    dataset = str(args.dataset).lower()
    partition = str(args.partition)

    base_dir = os.path.join(root, dataset, partition, f"seed{seed}")
    pkl_path = os.path.join(base_dir, "net_dataidx_map.pkl")
    npy_path = os.path.join(base_dir, "net_dataidx_map.npy")

    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            obj = pickle.load(f)
        return {int(k): list(map(int, v)) for k, v in obj.items()}

    if os.path.exists(npy_path):
        obj = np.load(npy_path, allow_pickle=True)
        if isinstance(obj, np.ndarray) and obj.dtype == object and obj.shape == ():
            obj = obj.item()
        return {int(k): list(map(int, v)) for k, v in obj.items()}

    n_train = int(len(y_train))
    out = partition_data(
        args.partition,
        n_train,
        int(args.n_parties),
        y_train,
        beta=float(getattr(args, "beta", 0.5)),
        skew_class=int(getattr(args, "skew_class", 2)),
    )

    net_dataidx_map = out[0] if isinstance(out, tuple) else out
    return {int(k): list(map(int, v)) for k, v in net_dataidx_map.items()}


@torch.no_grad()
def extract_features(base, ds, batch_size, device, num_workers=0):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    feats_list, labs_list = [], []
    base.eval()

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        h = base(xb).detach().float().cpu()
        feats_list.append(h)
        labs_list.append(yb.detach().cpu())

    return torch.cat(feats_list, 0), torch.cat(labs_list, 0).long()


@dataclass
class GaussianStats:
    mu: torch.Tensor
    inv_cov_diag: torch.Tensor


def fit_class_gaussian_diag(feats, labels, num_classes, eps=1e-6):
    stats = {}
    d = feats.shape[1]

    for c in range(num_classes):
        idx = (labels == c).nonzero(as_tuple=False).view(-1)
        if idx.numel() == 0:
            stats[c] = GaussianStats(torch.zeros(d), torch.ones(d) / eps)
            continue

        fc = feats[idx]
        mu = fc.mean(0)
        var = fc.var(0, unbiased=False)
        stats[c] = GaussianStats(mu, 1.0 / (var + eps))

    return stats


def score_mahalanobis_diag(feats, labels, stats):
    out = torch.empty(feats.shape[0])

    for c, st in stats.items():
        idx = (labels == c).nonzero(as_tuple=False).view(-1)
        if idx.numel() == 0:
            continue
        h = feats[idx]
        diff = h - st.mu
        out[idx] = -(diff * diff * st.inv_cov_diag).sum(1)

    return out.numpy()


def score_proto_cosine(feats, labels, W):
    feats_n = normalize_rows(feats.float())
    W_n = normalize_rows(W.float())
    return (feats_n * W_n[labels.long()]).sum(1).numpy()


def build_label_matched_out_indices(y_test, labels_in, seed, allow_replacement=True):
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(labels_in, return_counts=True)
    out_indices = []

    for c, cnt in zip(classes, counts):
        pool = np.where(y_test == c)[0]
        if len(pool) == 0:
            continue
        chosen = rng.choice(pool, size=cnt, replace=len(pool) < cnt or allow_replacement)
        out_indices.append(chosen)

    return np.concatenate(out_indices) if out_indices else np.array([], dtype=np.int64)


def parse_clients(s, n_parties):
    if s.strip().lower() == "all":
        return list(range(n_parties))
    return [int(p) for p in s.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--datadir", type=str, required=True)
    ap.add_argument("--model", type=str, default="simplecnn")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--partition", type=str, required=True)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--skew_class", type=int, default=2)
    ap.add_argument("--n_parties", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clients", type=str, default="all")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--out_json", type=str, required=True)

    args = ap.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    train_set, test_set, y_train, y_test, num_classes, in_channels = load_dataset(args.dataset, args.datadir)
    net_dataidx_map = build_client_map(args, y_train)

    model = build_model(args.model, args.dataset, num_classes, in_channels, device)
    sd = load_checkpoint_state_dict(args.checkpoint)
    model.load_state_dict(sd, strict=False)

    base, W = split_base_and_classifier(model)
    W = W.cpu()

    feats_test, labs_test = extract_features(base, test_set, args.batch_size, device)
    stats = fit_class_gaussian_diag(feats_test, labs_test, num_classes)

    clients = parse_clients(args.clients, args.n_parties)
    results = []

    for k in clients:
        idx_in = net_dataidx_map[int(k)]
        feats_in, labs_in = extract_features(base, Subset(train_set, idx_in), args.batch_size, device)

        out_idx = build_label_matched_out_indices(y_test, to_numpy(labs_in), args.seed + k)
        if out_idx.size == 0:
            continue

        feats_out = feats_test[out_idx]
        labs_out = labs_test[out_idx]

        s_in = score_mahalanobis_diag(feats_in, labs_in, stats)
        s_out = score_mahalanobis_diag(feats_out, labs_out, stats)

        y_true = np.concatenate([np.ones_like(s_in), np.zeros_like(s_out)])
        scores = np.concatenate([s_in, s_out])

        auc = roc_auc_score_binary(y_true, scores)

        results.append({"client": k, "auc": float(auc)})

    ensure_dir(str(Path(args.out_csv).parent))
    ensure_dir(str(Path(args.out_json).parent))

    import csv
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["client", "auc"])
        w.writeheader()
        for r in results:
            w.writerow(r)

    with open(args.out_json, "w") as f:
        json.dump({"results": results}, f, indent=2)


if __name__ == "__main__":
    main()
