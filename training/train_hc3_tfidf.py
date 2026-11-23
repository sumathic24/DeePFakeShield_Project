"""Train and evaluate an HC3 TF-IDF baseline with source-grouped data splits."""
import json
import os
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.pipeline import make_pipeline

from data_utils import load_text_records, split_text_indices

DATA = "./data/text/ai_human.jsonl"
OUT = "../backend/models/saved/hc3_detector.pkl"


def main():
    records = load_text_records(DATA)
    splits = split_text_indices(records, seed=42)
    train_records = [records[index] for index in splits["train"]]
    test_records = [records[index] for index in splits["test"]]
    train_texts = [row["text"] for row in train_records]
    train_labels = [row["label"] for row in train_records]
    test_texts = [row["text"] for row in test_records]
    test_labels = [row["label"] for row in test_records]

    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=300_000, sublinear_tf=True),
        LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced"),
    )
    print(f"Training on {len(train_records)} unique texts; evaluating on {len(test_records)} held-out texts.")
    model.fit(train_texts, train_labels)
    predictions = model.predict(test_texts)
    matrix = confusion_matrix(test_labels, predictions, labels=[0, 1])
    metrics = {
        "test_accuracy": round(accuracy_score(test_labels, predictions), 4),
        "test_precision": round(precision_score(test_labels, predictions, average="weighted", zero_division=0), 4),
        "test_recall": round(recall_score(test_labels, predictions, average="weighted", zero_division=0), 4),
        "test_f1": round(f1_score(test_labels, predictions, average="weighted", zero_division=0), 4),
        "test_confusion_matrix": matrix.tolist(),
        "test_class_labels": ["Human", "AI"],
        "test_samples": len(test_labels),
    }
    print(classification_report(test_labels, predictions, labels=[0, 1],
                                target_names=["Human", "AI"], zero_division=0))
    print(json.dumps(metrics, indent=2))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "wb") as output:
        pickle.dump(model, output)
    with open(os.path.splitext(OUT)[0] + "_metrics.json", "w", encoding="utf-8") as output:
        json.dump(metrics, output, indent=2)


if __name__ == "__main__":
    main()
