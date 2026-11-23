"""Evaluate the configured video detector on a labeled, held-out video directory."""
import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path

from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)

BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from services.video_service import VideoAnalysisService

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
LABELS = ["Real", "Fake / manipulated"]


async def evaluate(data_dir: Path):
    samples = []
    seen_hashes = set()
    for folder, label in (("real", 0), ("fake", 1)):
        class_dir = data_dir / folder
        if not class_dir.is_dir():
            raise ValueError(f"Expected labeled video folder: {class_dir}")
        for path in sorted(class_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
                continue
            content_hash = hashlib.sha256()
            with path.open("rb") as video_file:
                for chunk in iter(lambda: video_file.read(1024 * 1024), b""):
                    content_hash.update(chunk)
            digest = content_hash.digest()
            if digest in seen_hashes:
                raise ValueError(f"Duplicate video content in evaluation set: {path}")
            seen_hashes.add(digest)
            samples.append((path, label))

    labels = [label for _, label in samples]
    if not labels or set(labels) != {0, 1}:
        raise ValueError("The evaluation set must contain videos in both real/ and fake/ folders.")

    service = VideoAnalysisService()
    if service.detector is None:
        raise RuntimeError("The configured video model is unavailable. Download it and restart the backend.")

    scores = []
    statuses = []
    for path, _ in samples:
        result = await service.analyze_video(str(path))
        scores.append(float(result["ai_confidence"]))
        statuses.append(result["status"])

    predictions = [int(score > 0.5) for score in scores]
    confident = [index for index, status in enumerate(statuses) if status == "classified"]
    selective_accuracy = (
        accuracy_score([labels[index] for index in confident],
                       [predictions[index] for index in confident])
        if confident else None
    )
    metrics = {
        "samples": len(samples),
        "accuracy": round(accuracy_score(labels, predictions), 4),
        "precision_weighted": round(precision_score(labels, predictions, average="weighted", zero_division=0), 4),
        "recall_weighted": round(recall_score(labels, predictions, average="weighted", zero_division=0), 4),
        "f1_weighted": round(f1_score(labels, predictions, average="weighted", zero_division=0), 4),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
        "class_labels": LABELS,
        "coverage": round(len(confident) / len(samples), 4),
        "selective_accuracy": round(selective_accuracy, 4) if selective_accuracy is not None else None,
    }
    print(classification_report(labels, predictions, labels=[0, 1], target_names=LABELS, zero_division=0))
    print(json.dumps(metrics, indent=2))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, required=True,
                        help="Directory containing real/ and fake/ held-out video folders.")
    parser.add_argument("--output", type=Path, default=Path("video_evaluation.json"))
    args = parser.parse_args()
    metrics = asyncio.run(evaluate(args.data_dir))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Metrics saved to {args.output}")


if __name__ == "__main__":
    main()
