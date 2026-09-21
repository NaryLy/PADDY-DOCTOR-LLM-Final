"""
Train one of three models (baseline / frozen / finetuned) on the rice disease dataset,
checkpoint the best-val-accuracy weights, and evaluate on the held-out test split.

Usage:
    python3 train.py --mode finetuned --epochs 10
"""
import argparse
import csv
import json
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import RiceDiseaseDataset, load_manifest, build_label_map
from models import build_model

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms():
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


def evaluate(model, loader, device, n_classes):
    model.eval()
    correct = 0
    total = 0
    confusion = [[0] * n_classes for _ in range(n_classes)]
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            for t, p in zip(y.tolist(), pred.tolist()):
                confusion[t][p] += 1
    acc = correct / total if total else 0.0
    return acc, confusion


def precision_recall_f1(confusion, class_names):
    n = len(class_names)
    per_class = {}
    for i, name in enumerate(class_names):
        tp = confusion[i][i]
        fp = sum(confusion[r][i] for r in range(n)) - tp
        fn = sum(confusion[i][c] for c in range(n)) - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        support = sum(confusion[i])
        per_class[name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }
    macro_f1 = sum(v["f1"] for v in per_class.values()) / n
    return per_class, round(macro_f1, 4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["baseline", "frozen", "finetuned"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"],
                         help="auto picks cuda, then mps (Apple Silicon), then cpu")
    args = parser.parse_args()

    torch.manual_seed(42)
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"[{args.mode}] using device: {device}", flush=True)

    rows = load_manifest()
    label_map = build_label_map(rows)
    class_names = [name for name, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
    n_classes = len(class_names)

    train_tf, eval_tf = get_transforms()
    train_ds = RiceDiseaseDataset("train", label_map, transform=train_tf)
    val_ds = RiceDiseaseDataset("val", label_map, transform=eval_tf)
    test_ds = RiceDiseaseDataset("test", label_map, transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model, trainable_params = build_model(args.mode, n_classes)
    model.to(device)
    print(f"[{args.mode}] trainable parameters: {trainable_params:,}", flush=True)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    history_path = os.path.join(RESULTS_DIR, f"{args.mode}_history.csv")
    with open(history_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_acc", "seconds"])

    best_val_acc = -1.0
    ckpt_path = os.path.join(MODELS_DIR, f"{args.mode}_best.pt")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        n_batches = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1

        val_acc, _ = evaluate(model, val_loader, device, n_classes)
        elapsed = time.time() - t0
        avg_loss = running_loss / max(n_batches, 1)
        print(f"[{args.mode}] epoch {epoch}/{args.epochs} loss={avg_loss:.4f} val_acc={val_acc:.4f} ({elapsed:.1f}s)", flush=True)

        with open(history_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, round(avg_loss, 4), round(val_acc, 4), round(elapsed, 1)])

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "mode": args.mode,
                "class_names": class_names,
                "val_acc": val_acc,
                "epoch": epoch,
                "trainable_params": trainable_params,
            }, ckpt_path)
            print(f"[{args.mode}]   -> new best checkpoint saved (val_acc={val_acc:.4f})", flush=True)

    # Final test-set evaluation using the best checkpoint
    best = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state"])
    test_acc, confusion = evaluate(model, test_loader, device, n_classes)
    per_class, macro_f1 = precision_recall_f1(confusion, class_names)

    result = {
        "mode": args.mode,
        "trainable_params": trainable_params,
        "best_val_acc": best_val_acc,
        "best_epoch": best["epoch"],
        "test_acc": round(test_acc, 4),
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": confusion,
        "class_names": class_names,
        "epochs_trained": args.epochs,
    }
    result_path = os.path.join(RESULTS_DIR, f"{args.mode}_test_eval.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[{args.mode}] DONE. test_acc={test_acc:.4f} macro_f1={macro_f1:.4f}", flush=True)


if __name__ == "__main__":
    main()
