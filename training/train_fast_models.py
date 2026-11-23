"""Train the lightweight TF-IDF text model from a real JSONL corpus."""
import argparse
import json
import os
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.pipeline import Pipeline

from data_utils import load_text_records, split_text_indices


def train_text_model(data_path, output_file, seed=42):
    records = load_text_records(data_path)
    splits = split_text_indices(records, seed)
    train_records = [records[index] for index in splits["train"]]
    test_records = [records[index] for index in splits["test"]]

    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100_000)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    model.fit([row["text"] for row in train_records], [row["label"] for row in train_records])
    predictions = model.predict([row["text"] for row in test_records])
    labels = [row["label"] for row in test_records]
    metrics = {
        "test_accuracy": round(accuracy_score(labels, predictions), 4),
        "test_precision": round(precision_score(labels, predictions, average="weighted", zero_division=0), 4),
        "test_recall": round(recall_score(labels, predictions, average="weighted", zero_division=0), 4),
        "test_f1": round(f1_score(labels, predictions, average="weighted", zero_division=0), 4),
        "test_confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
        "test_samples": len(labels),
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "wb") as output:
        pickle.dump(model, output)
    with open(os.path.splitext(output_file)[0] + "_metrics.json", "w", encoding="utf-8") as output:
        json.dump(metrics, output, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="./data/text/ai_human.jsonl")
    parser.add_argument("--output_file", default="../backend/models/saved/fast_text_model.pkl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_text_model(args.data_path, args.output_file, args.seed)
