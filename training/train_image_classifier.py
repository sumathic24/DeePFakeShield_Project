"""
train_image_classifier.py — EfficientNet-B4 + ViT-B/16 for AI Image Detection
Dataset: CIFAKE + DiffusionDB + AI Generated Images Dataset
ImageFolder alphabetical indices: 0 = AI Generated | 1 = Real Image
Usage: python train_image_classifier.py --data_dir ./data/images --model efficientnet
"""
import argparse, os, json, hashlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix)
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

LABEL_NAMES = ["AI Generated", "Real Image"]

TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(8),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.15, scale=(0.02, 0.12)),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def get_model(name: str) -> nn.Module:
    if name == "efficientnet":
        model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    elif name == "vit":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        model.heads.head = nn.Linear(model.heads.head.in_features, 2)
    else:
        raise ValueError(f"Unknown model: {name}")
    return model


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, preds_all, labels_all = 0, [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        preds_all.extend(out.argmax(-1).cpu().numpy())
        labels_all.extend(labels.cpu().numpy())
    return total_loss / len(loader), accuracy_score(labels_all, preds_all)


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, preds_all, labels_all = 0, [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            total_loss += criterion(out, labels).item()
            preds_all.extend(out.argmax(-1).cpu().numpy())
            labels_all.extend(labels.cpu().numpy())
    acc = accuracy_score(labels_all, preds_all)
    precision = precision_score(labels_all, preds_all, average="weighted", zero_division=0)
    recall = recall_score(labels_all, preds_all, average="weighted", zero_division=0)
    f1 = f1_score(labels_all, preds_all, average="weighted", zero_division=0)
    macro_f1 = f1_score(labels_all, preds_all, average="macro", zero_division=0)
    return total_loss / len(loader), acc, precision, recall, f1, macro_f1, preds_all, labels_all


def verify_dataset_splits(train_ds, val_ds, test_ds):
    expected_classes = {"ai_generated", "real"}
    for split_name, dataset in (("train", train_ds), ("validation", val_ds), ("test", test_ds)):
        if set(dataset.classes) != expected_classes:
            raise ValueError(f"{split_name} must contain exactly these class folders: {sorted(expected_classes)}.")
        if not dataset.samples:
            raise ValueError(f"The {split_name} image split is empty.")
    if not (train_ds.class_to_idx == val_ds.class_to_idx == test_ds.class_to_idx):
        raise ValueError("Train, validation, and test class-to-index mappings differ.")

    seen = {}
    for split_name, dataset in (("train", train_ds), ("validation", val_ds), ("test", test_ds)):
        for filepath, _ in dataset.samples:
            digest = hashlib.sha256()
            with Image.open(filepath) as image_file:
                image = image_file.convert("RGB")
                digest.update(str(image.size).encode("ascii"))
                digest.update(image.tobytes())
            image_hash = digest.digest()
            previous = seen.get(image_hash)
            if previous is not None and previous != split_name:
                raise ValueError(f"Exact duplicate image crosses {previous} and {split_name} splits: {filepath}")
            seen[image_hash] = split_name


def plot_cm(labels, preds, model_name, out_dir):
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax)
    ax.set_title(f"{model_name} — Confusion Matrix")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{model_name}_confusion_matrix.png"))
    plt.close()


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Image Trainer] Model: {args.model} | Device: {device}")

    # Datasets (ImageFolder structure: data/images/train/real/ data/images/train/ai_generated/)
    train_ds = datasets.ImageFolder(os.path.join(args.data_dir, "train"), transform=TRAIN_TRANSFORM)
    val_ds   = datasets.ImageFolder(os.path.join(args.data_dir, "val"),   transform=VAL_TRANSFORM)
    test_ds  = datasets.ImageFolder(os.path.join(args.data_dir, "test"),  transform=VAL_TRANSFORM)
    verify_dataset_splits(train_ds, val_ds, test_ds)

    pin_memory = device.type == "cuda"
    train_ld = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=0, pin_memory=pin_memory)
    val_ld   = DataLoader(val_ds,   args.batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory)
    test_ld  = DataLoader(test_ds,  args.batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory)

    model = get_model(args.model).to(device)
    class_counts = np.bincount([label for _, label in train_ds.samples], minlength=2)
    if np.any(class_counts == 0):
        raise ValueError("Both classes must be represented in the training split.")
    class_weights = len(train_ds) / (2 * torch.tensor(class_counts, dtype=torch.float32))
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1 = -1
    epochs_without_improvement = 0
    print(f"\nTraining {args.epochs} epochs on {len(train_ds)} images...\n")

    for epoch in range(1, args.epochs + 1):
        tl, ta = train_epoch(model, train_ld, optimizer, criterion, device)
        vl, va, vp, vr, vf1, vmacro_f1, _, _ = eval_epoch(model, val_ld, criterion, device)
        scheduler.step()
        print(f"Epoch {epoch}/{args.epochs} | Train {tl:.4f}/{ta:.4f} | "
              f"Val {vl:.4f}/{va:.4f} P={vp:.4f} R={vr:.4f} F1={vf1:.4f}")

        if vmacro_f1 > best_f1:
            best_f1 = vmacro_f1
            epochs_without_improvement = 0
            save_path = os.path.join(args.output_dir, f"{args.model}_detector.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  Best model saved (validation macro-F1={best_f1:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping after {epoch} epochs without validation improvement.")
                break

    save_path = os.path.join(args.output_dir, f"{args.model}_detector.pth")
    model.load_state_dict(torch.load(save_path, map_location=device))
    _, ta, precision, recall, tf1, macro_f1, tp, tl2 = eval_epoch(model, test_ld, criterion, device)
    cm = confusion_matrix(tl2, tp, labels=[0, 1])
    print(f"\n[Test] Accuracy: {ta:.4f} | Precision: {precision:.4f} | "
          f"Recall: {recall:.4f} | F1: {tf1:.4f}")
    print(classification_report(tl2, tp, labels=[0, 1], target_names=LABEL_NAMES, zero_division=0))
    plot_cm(tl2, tp, args.model, args.output_dir)

    metrics = {"model": args.model, "test_accuracy": round(ta, 4),
               "test_precision": round(precision, 4), "test_recall": round(recall, 4),
               "test_f1": round(tf1, 4), "test_macro_f1": round(macro_f1, 4),
               "test_confusion_matrix": cm.tolist(), "test_class_labels": LABEL_NAMES,
               "test_samples": len(tl2), "best_val_macro_f1": round(best_f1, 4)}
    with open(os.path.join(args.output_dir, f"{args.model}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[Done] {args.model} saved to {args.output_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   default="./data/images")
    p.add_argument("--output_dir", default="../backend/models/saved")
    p.add_argument("--model",      default="efficientnet", choices=["efficientnet", "vit"])
    p.add_argument("--epochs",     type=int,   default=10)
    p.add_argument("--batch_size", type=int,   default=32)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
