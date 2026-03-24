import argparse
import json
import os
import pickle
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
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

    return float((in_scores >= thr).mean())


def build_transforms(dataset: str):
    if dataset == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std  = (0.2470, 0.2435, 0.2616)
    elif dataset == "cifar100":
        mean = (0.5071, 0.4867, 0.4408)
        std  = (0.2675, 0.2565, 0.2761)
    elif dataset in ("fashionmnist", "fmnist"):
        mean = (0.2860,)
        std  = (0.3530,)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])


def load_dataset(dataset: str, datadir: str):
    tf = build_transforms(dataset)

    if dataset == "cifar10":
        train = datasets.CIFAR10(datadir, train=True, download=False, transform=tf)
        test  = datasets.CIFAR10(datadir, train=False, download=False, transform=tf)
        num_classes, in_channels = 10, 3
    elif dataset == "cifar100":
        train = datasets.CIFAR100(datadir, train=True, download=False, transform=tf)
        test  = datasets.CIFAR100(datadir, train=False, download=False, transform=tf)
        num_classes, in_channels = 100, 3
    elif dataset in ("fashionmnist", "fmnist"):
        train = datasets.FashionMNIST(datadir, train=True, download=False, transform=tf)
        test  = datasets.FashionMNIST(datadir, train=False, download=False, transform=tf)
        num_classes, in_channels = 10, 1
    else:
        raise ValueError(dataset)

    y_train = np.array(train.targets, dtype=np.int64)
    y_test  = np.array(test.targets, dtype=np.int64)
    return train, test, y_train, y_test, num_classes, in_channels


def load_checkpoint_state_dict(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")

    if p.suffix.lower() in [".pkl", ".pickle", ".pt", ".pth"]:
        obj = torch.load(p, map_location="cpu", pickle_module=pickle)
    else:
        obj = torch.load(p, map_location="cpu")

    if isinstance(obj, dict):
        for k in ["state_dict", "model_state_dict", "net_state_dict", "global_w", "model", "net"]:
            if k in obj and isinstance(obj[k], dict):
                obj = obj[k]
                break

    if not isinstance(obj, dict):
        raise TypeError(f"Checkpoint is not a state_dict: {type(obj)}")
    return obj


def build_model(model_name: str, dataset: str, num_classes: int, in_channels: int, device):
    class ArgsLike: pass
    args_like = ArgsLike()
    args_like.model = model_name
    args_like.channel = in_channels

    image_size = 32 if dataset in ("cifar10", "cifar100") else 28
    cfg = {"image_size": image_size}

    builder = get_model(args_like, cfg)
    model = builder(num_classes) if callable(builder) else builder
    model.to(device)
    return model


def build_client_map(args, y_train: np.ndarray) -> Dict[int, List[int]]:
    out = partition_data(
        args.partition,
        len(y_train),
        args.n_parties,
        y_train,
        beta=args.beta,
        skew_class=args.skew_class,
    )
    net_dataidx_map = out[0] if isinstance(out, tuple) else out
    return {int(k): list(map(int, v)) for k, v in net_dataidx_map.items()}


def build_label_matched_out_indices(y_test, labels_in, seed):
    rng = np.random.default_rng(seed)
    out_idx = []
    classes, counts = np.unique(labels_in, return_counts=True)

    for c, cnt in zip(classes, counts):
        pool = np.where(y_test == c)[0]
        if len(pool) == 0:
            continue
        chosen = rng.choice(pool, size=cnt, replace=len(pool) < cnt)
        out_idx.append(chosen)

    if not out_idx:
        return np.array([], dtype=np.int64)

    out_idx = np.concatenate(out_idx)
    rng.shuffle(out_idx)
    return out_idx


@torch.no_grad()
def compute_losses(model, ds, batch_size, device, num_workers):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=False)
    criterion = nn.CrossEntropyLoss(reduction="none")
    model.eval()

    losses = []
    labels = []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        losses.append(loss.cpu())
        labels.append(yb.cpu())

    return torch.cat(losses), torch.cat(labels)


def parse_clients(s: str, n_parties: int):
    if s.lower() == "all":
        return list(range(n_parties))
    return [int(x) for x in s.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--datadir", required=True)
    ap.add_argument("--model", default="simplecnn")
    ap.add_argument("--checkpoint", required=True)

    ap.add_argument("--partition", required=True)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--skew_class", type=int, default=2)
    ap.add_argument("--n_parties", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--clients", default="all")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--device", default="cpu")

    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_json", required=True)

    args = ap.parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if args.device != "cuda" else "cpu")

    train_set, test_set, y_train, y_test, num_classes, in_channels = \
        load_dataset(args.dataset, args.datadir)

    net_dataidx_map = build_client_map(args, y_train)

    model = build_model(args.model, args.dataset, num_classes, in_channels, device)
    sd = load_checkpoint_state_dict(args.checkpoint)
    model.load_state_dict(sd, strict=False)

    clients = parse_clients(args.clients, args.n_parties)
    results = []

    for k in clients:
        idx_in = net_dataidx_map[k]
        ds_in = Subset(train_set, idx_in)

        loss_in, labs_in = compute_losses(
            model, ds_in, args.batch_size, device, args.num_workers
        )

        out_idx = build_label_matched_out_indices(
            y_test, to_numpy(labs_in), seed=args.seed * 1000 + k
        )
        if out_idx.size == 0:
            continue

        ds_out = Subset(test_set, out_idx)
        loss_out, _ = compute_losses(
            model, ds_out, args.batch_size, device, args.num_workers
        )

        scores = np.concatenate([
            -to_numpy(loss_in),
            -to_numpy(loss_out)
        ])
        y_true = np.concatenate([
            np.ones(len(loss_in), dtype=np.int64),
            np.zeros(len(loss_out), dtype=np.int64)
        ])

        auc_raw = roc_auc_score_binary(y_true, scores)
        auc = max(auc_raw, 1.0 - auc_raw)

        results.append({
            "client": k,
            "score": "loss",
            "auc_macro": auc,
            "tpr@1%_macro": tpr_at_fpr(y_true, scores, 0.01),
            "tpr@0.1%_macro": tpr_at_fpr(y_true, scores, 0.001),
            "n_classes_used": int(len(np.unique(to_numpy(labs_in))))
        })

        print(f"[OK] client {k}: IN={len(loss_in)} OUT={len(loss_out)}")

    ensure_dir(Path(args.out_csv).parent)
    ensure_dir(Path(args.out_json).parent)

    import csv
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["client", "score", "auc_macro", "tpr@1%_macro", "tpr@0.1%_macro", "n_classes_used"]
        )
        w.writeheader()
        for r in results:
            w.writerow(r)

    with open(args.out_json, "w") as f:
        json.dump({"results": results, "args": vars(args)}, f, indent=2)

    print("[DONE]")


if __name__ == "__main__":
    main()