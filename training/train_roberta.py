"""
train_roberta.py
Fine-tune RoBERTa for AI-generated text detection.

Dataset: HC3 + GPT-Wiki-Intro + OpenAI Human vs AI Text Dataset
Label 0: Human Written
Label 1: AI Generated

Usage:
    python train_roberta.py --data_dir ./data/text --epochs 5 --batch_size 16
"""

import argparse
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from data_utils import load_text_records, split_text_indices


# ── Config ──────────────────────────────────────────────────────────────────
MODEL_NAME = "roberta-base"
NUM_LABELS = 2
MAX_LENGTH = 512
LABEL_NAMES = ["Human Written", "AI Generated"]


# ── Dataset ─────────────────────────────────────────────────────────────────
class AITextDataset(Dataset):
    """
    Expects a JSONL file with {"text": "...", "label": 0|1}
    label 0 = human, label 1 = AI-generated
    """
    def __init__(self, records, tokenizer, max_length=MAX_LENGTH):
        self.samples = [(item["text"], int(item["label"])) for item in records]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


# ── Training Loop ────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss, total_correct, total = 0, 0, 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = outputs.logits.argmax(dim=-1)
        total_correct += (preds == labels).sum().item()
        total += len(labels)

    return total_loss / len(loader), total_correct / total


def eval_epoch(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
            preds = outputs.logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    return total_loss / len(loader), acc, f1, all_preds, all_labels


def plot_confusion_matrix(labels, preds, output_dir):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("RoBERTa — Confusion Matrix")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "roberta_confusion_matrix.png"))
    plt.close()


def plot_training_curves(train_losses, val_losses, val_accs, output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, label="Train Loss", color="#22d3ee")
    ax1.plot(val_losses, label="Val Loss", color="#fb7185")
    ax1.set_title("Loss Curves")
    ax1.legend()
    ax2.plot(val_accs, label="Val Accuracy", color="#a78bfa")
    ax2.set_title("Validation Accuracy")
    ax2.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "roberta_training_curves.png"))
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────────────
def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[RoBERTa Trainer] Device: {device}")

    # Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    model.to(device)

    # Dataset
    records = load_text_records(args.data_path)
    splits = split_text_indices(records, args.seed)
    dataset = AITextDataset(records, tokenizer, args.max_length)
    train_ds = Subset(dataset, splits["train"])
    val_ds = Subset(dataset, splits["validation"])
    test_ds = Subset(dataset, splits["test"])
    train_size = len(train_ds)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, num_workers=0)

    # Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps // 10,
                                                num_training_steps=total_steps)

    best_f1 = -1
    epochs_without_improvement = 0
    train_losses, val_losses, val_accs = [], [], []

    print(f"\n[RoBERTa] Training {args.epochs} epochs on {train_size} samples...\n")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss, val_acc, val_f1, _, _ = eval_epoch(model, val_loader, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch {epoch}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            epochs_without_improvement = 0
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            print(f"  ✓ Best model saved (F1={best_f1:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping after {epoch} epochs without validation improvement.")
                break

    # Final test evaluation
    model = AutoModelForSequenceClassification.from_pretrained(args.output_dir).to(device)
    _, test_acc, test_f1, test_preds, test_labels = eval_epoch(model, test_loader, device)
    test_precision = precision_score(test_labels, test_preds, average="weighted", zero_division=0)
    test_recall = recall_score(test_labels, test_preds, average="weighted", zero_division=0)
    test_cm = confusion_matrix(test_labels, test_preds, labels=[0, 1])
    print(f"\n[Test Results] Accuracy: {test_acc:.4f} | Precision: {test_precision:.4f} | "
          f"Recall: {test_recall:.4f} | F1: {test_f1:.4f}")
    print(classification_report(test_labels, test_preds, target_names=LABEL_NAMES))

    plot_confusion_matrix(test_labels, test_preds, args.output_dir)
    plot_training_curves(train_losses, val_losses, val_accs, args.output_dir)

    # Save metrics
    metrics = {
        "test_accuracy": round(test_acc, 4),
        "test_f1": round(test_f1, 4),
        "test_precision": round(test_precision, 4),
        "test_recall": round(test_recall, 4),
        "best_val_f1": round(best_f1, 4),
        "test_confusion_matrix": test_cm.tolist(),
        "test_class_labels": LABEL_NAMES,
        "test_samples": len(test_labels),
    }
    with open(os.path.join(args.output_dir, "roberta_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[Done] Model saved to {args.output_dir}")
    print(f"Metrics: {metrics}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data/text/ai_human.jsonl")
    parser.add_argument("--output_dir", type=str, default="../backend/models/saved/roberta_ai_detector")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()
    main(args)
