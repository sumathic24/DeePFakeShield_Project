import json
import tempfile
import unittest
from pathlib import Path

from training.data_utils import deduplicate_text_records, load_text_records, split_text_indices


class TextDataUtilsTests(unittest.TestCase):
    def test_normalized_duplicates_and_conflicting_labels_are_removed(self):
        records = deduplicate_text_records([
            {"text": "Same   sample", "label": 0},
            {"text": " same sample ", "label": 0},
            {"text": "Conflicting sample", "label": 0},
            {"text": "conflicting sample", "label": 1},
        ])
        self.assertEqual(records, [{"text": "Same   sample", "label": 0, "source_groups": []}])

    def test_source_groups_do_not_cross_splits(self):
        rows = []
        for group in range(20):
            for label in (0, 1):
                for sample in range(2):
                    rows.append({
                        "text": f"Unique source {group} label {label} sample {sample}",
                        "label": label,
                        "source_group": f"source-{group}",
                    })
        records = deduplicate_text_records(rows)
        splits = split_text_indices(records, seed=7)
        memberships = {
            index: split_name
            for split_name, indices in splits.items()
            for index in indices
        }
        source_splits = {}
        for index, row in enumerate(records):
            source_splits.setdefault(row["source_groups"][0], set()).add(memberships[index])
        self.assertTrue(all(len(split_names) == 1 for split_names in source_splits.values()))

    def test_small_duplicate_heavy_corpus_is_not_treated_as_evaluable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.jsonl"
            path.write_text(
                "\n".join(json.dumps({"text": text, "label": label}) for text, label in [
                    ("human one", 0), ("human one", 0), ("human two", 0),
                    ("ai one", 1), ("ai one", 1), ("ai two", 1),
                ]),
                encoding="utf-8",
            )
            records = load_text_records(str(path))
            with self.assertRaisesRegex(ValueError, "10 unique"):
                split_text_indices(records)


if __name__ == "__main__":
    unittest.main()
