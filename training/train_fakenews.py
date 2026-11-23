"""
train_fakenews.py — BERT fake news / misinformation classifier.
Dataset: LIAR (binary) + custom corpus.
Label 0: Credible | Label 1: Fake News
Usage: python train_fakenews.py --data_path ./data/text/fakenews.jsonl
"""
import argparse, os, json
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix)
from data_utils import load_text_records, split_text_indices

MODEL_NAME = "bert-base-uncased"
NUM_LABELS = 2
LABEL_NAMES = ["Real / Credible", "Fake News"]


class FakeNewsDataset(Dataset):
    def __init__(self, records, tokenizer):
        self.data = records
        self.tok  = tokenizer
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        enc  = self.tok(item["text"], max_length=256, truncation=True,
                        padding="max_length", return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "label": torch.tensor(int(item["label"]), dtype=torch.long)}


def run_epoch(model, loader, optimizer=None, scheduler=None, device="cpu", train=True):
    model.train() if train else model.eval()
    preds_all, labels_all, total_loss = [], [], 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for b in loader:
            ids  = b["input_ids"].to(device)
            mask = b["attention_mask"].to(device)
            lbl  = b["label"].to(device)
            if train: optimizer.zero_grad()
            out = model(input_ids=ids, attention_mask=mask, labels=lbl)
            if train:
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); scheduler.step()
            total_loss += out.loss.item()
            preds_all.extend(out.logits.argmax(-1).cpu().numpy())
            labels_all.extend(lbl.cpu().numpy())
    return total_loss/len(loader), accuracy_score(labels_all, preds_all), \
           f1_score(labels_all, preds_all, average="weighted"), preds_all, labels_all


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS).to(device)
    records = load_text_records(args.data_path)
    splits = split_text_indices(records, args.seed)
    ds = FakeNewsDataset(records, tokenizer)
    tr_ds = Subset(ds, splits["train"])
    val_ds = Subset(ds, splits["validation"])
    test_ds = Subset(ds, splits["test"])
    tr_ld  = DataLoader(tr_ds, args.batch_size, shuffle=True, num_workers=0)
    val_ld = DataLoader(val_ds, args.batch_size, num_workers=0)
    te_ld  = DataLoader(test_ds, args.batch_size, num_workers=0)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, len(tr_ld)*args.epochs//10, len(tr_ld)*args.epochs)
    best = -1
    epochs_without_improvement = 0
    for ep in range(1, args.epochs+1):
        tl,ta,_,_,_  = run_epoch(model, tr_ld,  opt, sch, device, True)
        vl,va,vf1,_,_ = run_epoch(model, val_ld, device=device, train=False)
        print(f"Epoch {ep} | Train {tl:.4f}/{ta:.4f} | Val {vl:.4f}/{va:.4f} F1={vf1:.4f}")
        if vf1 > best:
            best = vf1
            epochs_without_improvement = 0
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            print(f"  ✓ Saved (F1={best:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping after {ep} epochs without validation improvement.")
                break
    model = AutoModelForSequenceClassification.from_pretrained(args.output_dir).to(device)
    _, ta, tf1, tp, tl2 = run_epoch(model, te_ld, device=device, train=False)
    precision = precision_score(tl2, tp, average="weighted", zero_division=0)
    recall = recall_score(tl2, tp, average="weighted", zero_division=0)
    cm = confusion_matrix(tl2, tp, labels=[0, 1])
    print(f"\n[Test] Accuracy={ta:.4f} Precision={precision:.4f} Recall={recall:.4f} F1={tf1:.4f}")
    print(classification_report(tl2, tp, labels=[0, 1], target_names=LABEL_NAMES, zero_division=0))
    metrics = {"test_accuracy": round(ta, 4), "test_precision": round(precision, 4),
               "test_recall": round(recall, 4), "test_f1": round(tf1, 4),
               "test_confusion_matrix": cm.tolist(), "test_class_labels": LABEL_NAMES,
               "test_samples": len(tl2), "best_val_f1": round(best, 4)}
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as output:
        json.dump(metrics, output, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_path",  default="./data/text/fakenews.jsonl")
    p.add_argument("--output_dir", default="../backend/models/saved/fakenews_classifier")
    p.add_argument("--epochs",     type=int,   default=5)
    p.add_argument("--batch_size", type=int,   default=32)
    p.add_argument("--lr",         type=float, default=2e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=3)
    main(p.parse_args())
