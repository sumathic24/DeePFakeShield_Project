"""Shared text-data validation and deterministic leakage-resistant splitting."""
import json
import unicodedata
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def load_text_records(filepath: str) -> List[Dict[str, object]]:
    source_records = []
    raw_count = 0
    with Path(filepath).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            text = item.get("text")
            label = item.get("label")
            if not isinstance(text, str) or not text.strip() or type(label) is not int or label not in (0, 1):
                raise ValueError(f"Invalid text or binary label on line {line_number} of {filepath}.")
            source_records.append({
                "text": text.strip(),
                "label": label,
                "source_group": item.get("source_group"),
                "source_groups": item.get("source_groups", []),
            })
            raw_count += 1
    records = deduplicate_text_records(source_records)
    print(f"[Data] {raw_count} rows -> {len(records)} unique texts; duplicate rows removed.")
    return records


def deduplicate_text_records(source_records: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped = {}
    for item in source_records:
        key = normalize_text(str(item["text"]))
        entry = grouped.setdefault(key, {
            "text": str(item["text"]).strip(),
            "labels": set(),
            "source_groups": set(),
        })
        entry["labels"].add(int(item["label"]))
        source_groups = list(item.get("source_groups", []))
        if item.get("source_group"):
            source_groups.append(item["source_group"])
        entry["source_groups"].update(str(group) for group in source_groups if group)
    records = [
        {"text": item["text"], "label": next(iter(item["labels"])),
         "source_groups": sorted(item["source_groups"])}
        for item in grouped.values() if len(item["labels"]) == 1
    ]
    excluded = len(grouped) - len(records)
    if excluded:
        print(f"[Data] Excluded {excluded} normalized texts with conflicting labels.")
    return records


def split_text_indices(records: List[Dict[str, object]], seed: int = 42):
    labels = np.asarray([int(row["label"]) for row in records])
    class_counts = np.bincount(labels, minlength=2)
    if min(class_counts) < 10:
        raise ValueError(
            "At least 10 unique, non-conflicting samples per class are required for grouped "
            f"train/validation/test splits; found human={class_counts[0]}, AI={class_counts[1]}."
        )

    parent = list(range(len(records)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    source_group_indices = {}
    for index, row in enumerate(records):
        for source_group in row.get("source_groups", []):
            if source_group in source_group_indices:
                parent[find(source_group_indices[source_group])] = find(index)
            else:
                source_group_indices[source_group] = index

    groups = np.asarray([find(index) for index in range(len(records))])
    grouped_counts = [len(set(groups[labels == class_id])) for class_id in (0, 1)]
    if min(grouped_counts) < 10:
        raise ValueError(
            "At least 10 independent source groups per class are required for leakage-resistant "
            f"splits; found human={grouped_counts[0]}, AI={grouped_counts[1]}."
        )
    splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    folds = [test_indices for _, test_indices in splitter.split(np.zeros(len(labels)), labels, groups)]
    test_indices = folds[0]
    val_indices = np.concatenate((folds[1], folds[2]))
    excluded = set(test_indices.tolist()) | set(val_indices.tolist())
    train_indices = np.asarray([i for i in range(len(records)) if i not in excluded], dtype=np.int64)

    for split_name, indices in (("train", train_indices), ("validation", val_indices), ("test", test_indices)):
        if set(labels[indices].tolist()) != {0, 1}:
            raise ValueError(f"The grouped {split_name} split does not contain both classes.")
    return {"train": train_indices, "validation": val_indices, "test": test_indices}
