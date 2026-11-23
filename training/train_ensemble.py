"""
train_ensemble.py — RoBERTa + LinguaForge-X Ensemble Training
Loads both trained models, runs inference on validation/test set,
learns optimal weights via grid search or logistic regression meta-learner.
Usage: python train_ensemble.py --roberta_dir ... --lingua_dir ... --data_path ...
"""
import argparse, os, json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix)
from sklearn.model_selection import StratifiedShuffleSplit
from train_linguaforge import LinguaForgeX, extract_linguistic_features
from data_utils import load_text_records, split_text_indices

LABEL_NAMES = ["Human Written", "AI Generated"]


class EnsembleDataset(Dataset):
    def __init__(self, filepath, roberta_tok, lingua_tok, max_len=512):
        self.data = load_text_records(filepath)
        self.roberta_tok = roberta_tok
        self.lingua_tok  = lingua_tok
        self.max_len     = max_len

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        re = self.roberta_tok(text, max_length=self.max_len, truncation=True,
                              padding="max_length", return_tensors="pt")
        le = self.lingua_tok(text, max_length=self.max_len, truncation=True,
                             padding="max_length", return_tensors="pt")
        return {
            "r_ids":  re["input_ids"].squeeze(0),
            "r_mask": re["attention_mask"].squeeze(0),
            "l_ids":  le["input_ids"].squeeze(0),
            "l_mask": le["attention_mask"].squeeze(0),
            "ling":   torch.tensor(extract_linguistic_features(text)),
            "label":  torch.tensor(int(item["label"]), dtype=torch.long),
        }


def get_probabilities(roberta, lingua_model, loader, device):
    roberta.eval(); lingua_model.eval()
    r_probs, l_probs, all_labels = [], [], []
    with torch.no_grad():
        for b in loader:
            # RoBERTa
            r_out = roberta(input_ids=b["r_ids"].to(device),
                            attention_mask=b["r_mask"].to(device))
            r_p = torch.softmax(r_out.logits, dim=-1)[:, 1].cpu().numpy()
            # LinguaForge
            _, logits = lingua_model(b["l_ids"].to(device), b["l_mask"].to(device),
                                     b["ling"].to(device))
            l_p = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            r_probs.extend(r_p)
            l_probs.extend(l_p)
            all_labels.extend(b["label"].numpy())
    return np.array(r_probs), np.array(l_probs), np.array(all_labels)


def grid_search_weights(r_probs, l_probs, labels):
    """Choose fusion weights on calibration data only."""
    best_f1, best_w = -1, 0.5
    for w in np.arange(0.1, 1.0, 0.05):
        ensemble = w * r_probs + (1 - w) * l_probs
        preds = (ensemble > 0.5).astype(int)
        f1 = f1_score(labels, preds, average="weighted")
        if f1 > best_f1:
            best_f1 = f1
            best_w = round(w, 2)
    return best_w, round(1 - best_w, 2), best_f1


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Ensemble] Device: {device}")

    # Load models
    roberta_tok = AutoTokenizer.from_pretrained(args.roberta_dir)
    roberta     = AutoModelForSequenceClassification.from_pretrained(args.roberta_dir).to(device)
    lingua_tok  = AutoTokenizer.from_pretrained(args.lingua_dir)
    lingua      = LinguaForgeX().to(device)
    lingua.load_state_dict(torch.load(
        os.path.join(args.lingua_dir, "linguaforge_model.pth"), map_location=device))

    dataset = EnsembleDataset(args.data_path, roberta_tok, lingua_tok)
    splits = split_text_indices(dataset.data, args.seed)
    val_ld = DataLoader(torch.utils.data.Subset(dataset, splits["validation"]),
                        batch_size=args.batch_size, num_workers=0)
    test_ld = DataLoader(torch.utils.data.Subset(dataset, splits["test"]),
                         batch_size=args.batch_size, num_workers=0)
    print("Extracting probabilities from held-out validation and test samples...")
    r_val, l_val, labels_val = get_probabilities(roberta, lingua, val_ld, device)
    r_test, l_test, labels_test = get_probabilities(roberta, lingua, test_ld, device)

    calibration_idx, selection_idx = next(StratifiedShuffleSplit(
        n_splits=1, test_size=0.5, random_state=args.seed
    ).split(np.zeros(len(labels_val)), labels_val))
    r_cal, l_cal, y_cal = r_val[calibration_idx], l_val[calibration_idx], labels_val[calibration_idx]
    r_select, l_select, y_select = r_val[selection_idx], l_val[selection_idx], labels_val[selection_idx]

    r_weight, l_weight, _ = grid_search_weights(r_cal, l_cal, y_cal)
    weighted_select = (r_weight * r_select + l_weight * l_select > 0.5).astype(int)
    weighted_val_f1 = f1_score(y_select, weighted_select, average="weighted", zero_division=0)

    # Fit the meta-learner on calibration samples and choose it on separate validation samples.
    X_cal = np.column_stack([r_cal, l_cal])
    meta = LogisticRegression(max_iter=500)
    meta.fit(X_cal, y_cal)
    meta_select = meta.predict(np.column_stack([r_select, l_select]))
    meta_val_f1 = f1_score(y_select, meta_select, average="weighted", zero_division=0)

    if meta_val_f1 > weighted_val_f1:
        strategy = "meta_learner"
        selected_val_f1 = meta_val_f1
        import pickle
        with open(os.path.join(args.output_dir, "meta_learner.pkl"), "wb") as f:
            pickle.dump(meta, f)
    else:
        strategy = "weighted_average"
        selected_val_f1 = weighted_val_f1

    if strategy == "meta_learner":
        test_preds = meta.predict(np.column_stack([r_test, l_test]))
    else:
        test_preds = (r_weight * r_test + l_weight * l_test > 0.5).astype(int)
    test_metrics = {
        "test_accuracy": round(accuracy_score(labels_test, test_preds), 4),
        "test_precision": round(precision_score(labels_test, test_preds, average="weighted", zero_division=0), 4),
        "test_recall": round(recall_score(labels_test, test_preds, average="weighted", zero_division=0), 4),
        "test_f1": round(f1_score(labels_test, test_preds, average="weighted", zero_division=0), 4),
        "test_confusion_matrix": confusion_matrix(labels_test, test_preds, labels=[0, 1]).tolist(),
        "test_class_labels": LABEL_NAMES,
        "test_samples": len(labels_test),
    }
    print(f"[Validation] Meta F1={meta_val_f1:.4f}; weighted F1={weighted_val_f1:.4f}")
    print("[Unseen test set]")
    print(classification_report(labels_test, test_preds, target_names=LABEL_NAMES, zero_division=0))

    config = {
        "strategy": strategy,
        "roberta_weight": r_weight,
        "linguaforge_weight": l_weight,
        "meta_validation_f1": round(meta_val_f1, 4),
        "weighted_validation_f1": round(weighted_val_f1, 4),
        "selected_validation_f1": round(selected_val_f1, 4),
        **test_metrics,
    }
    with open(os.path.join(args.output_dir, "ensemble_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(args.output_dir, "ensemble_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"\n[Done] Ensemble config and held-out test metrics saved.")
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--roberta_dir", default="../backend/models/saved/roberta_ai_detector")
    p.add_argument("--lingua_dir",  default="../backend/models/saved/linguaforge_detector")
    p.add_argument("--data_path",   default="./data/text/ai_human.jsonl")
    p.add_argument("--output_dir",  default="../backend/models/saved/ensemble")
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
