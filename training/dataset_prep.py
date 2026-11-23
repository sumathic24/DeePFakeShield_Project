"""
dataset_prep.py — Download and prepare all datasets for DeepFakeShield.

Datasets:
  Text:  HC3, LIAR, GPT-Wiki-Intro, OpenAI Human vs AI
  Image: CIFAKE (HuggingFace), DiffusionDB subset

Output format:
  data/text/ai_human.jsonl    — {"text": "...", "label": 0|1}
  data/text/fakenews.jsonl    — {"text": "...", "label": 0|1}
  data/images/train/real/     — real images
  data/images/train/ai_generated/  — AI images
  data/images/val/  ...
  data/images/test/ ...

Usage:
    python dataset_prep.py --output_dir ./data
"""
import argparse, csv, os, json, hashlib, zipfile
from io import BytesIO, TextIOWrapper
from pathlib import Path
from urllib.request import Request, urlopen

from data_utils import deduplicate_text_records
from data_utils import normalize_text


def make_dirs(base):
    for split in ["train", "val", "test"]:
        for cls in ["real", "ai_generated"]:
            Path(f"{base}/images/{split}/{cls}").mkdir(parents=True, exist_ok=True)
    Path(f"{base}/text").mkdir(parents=True, exist_ok=True)
    print(f"[✓] Directories created under {base}/")


def prepare_hc3(output_dir):
    """HC3: Human ChatGPT Comparison Corpus — download from HuggingFace."""
    print("\n[HC3] Preparing dataset...")
    try:
        from datasets import load_dataset
        ds = load_dataset("Hello-SimpleAI/HC3", "all", split="train")
        records = []
        for item in ds:
            question_value = item.get("question")
            question = normalize_text(str(question_value)) if question_value else ""
            source_group = hashlib.sha256(question.encode("utf-8")).hexdigest() if question else None
            if item.get("human_answers"):
                for ans in item["human_answers"]:
                    if ans and len(ans.strip()) > 50:
                        records.append({"text": ans.strip(), "label": 0, "source_group": source_group})
            if item.get("chatgpt_answers"):
                for ans in item["chatgpt_answers"]:
                    if ans and len(ans.strip()) > 50:
                        records.append({"text": ans.strip(), "label": 1, "source_group": source_group})
        records = deduplicate_text_records(records)
        out = os.path.join(output_dir, "text", "ai_human.jsonl")
        with open(out, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[HC3] Saved {len(records)} samples → {out}")
    except Exception as e:
        raise RuntimeError(f"HC3 preparation failed; synthetic fallback data is disabled: {e}") from e


def prepare_liar(output_dir):
    """LIAR: 6-class → binary (true/mostly-true=0, others=1)"""
    print("\n[LIAR] Preparing dataset...")
    try:
        TRUE_LABELS = {"true", "mostly-true", "half-true"}
        records = []
        valid_labels = TRUE_LABELS | {"false", "barely-true", "pants-fire"}
        request = Request(
            "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip",
            headers={"User-Agent": "Mozilla/5.0 (compatible; DeepFakeShield/1.0)"},
        )
        with urlopen(request, timeout=60) as response:
            archive = zipfile.ZipFile(BytesIO(response.read()))

        train_file = next(
            (name for name in archive.namelist() if Path(name).name == "train.tsv"),
            None,
        )
        if train_file is None:
            raise ValueError("The LIAR archive does not contain train.tsv.")

        with archive.open(train_file) as source:
            reader = csv.reader(TextIOWrapper(source, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE)
            for line_number, row in enumerate(reader, start=1):
                if len(row) < 5:
                    raise ValueError(f"Invalid LIAR record on line {line_number}.")
                text = row[2].strip()
                label_str = row[1].casefold().replace("_", "-")
                if label_str not in valid_labels:
                    raise ValueError(f"Unknown LIAR label on line {line_number}: {label_str}.")
                speaker = normalize_text(row[4]) if row[4] else ""
                source_group = speaker or hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()
                label = 0 if label_str in TRUE_LABELS else 1
                if len(text) > 20:
                    records.append({"text": text, "label": label, "source_group": source_group})

        records = deduplicate_text_records(records)
        out = os.path.join(output_dir, "text", "fakenews.jsonl")
        with open(out, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[LIAR] Saved {len(records)} samples -> {out}")
    except Exception as e:
        raise RuntimeError(f"LIAR preparation failed; synthetic fallback data is disabled: {e}") from e


def prepare_cifake(output_dir):
    """CIFAKE: 60K real (CIFAR-10) + 60K AI-generated images from HuggingFace."""
    print("\n[CIFAKE] Preparing image dataset...")
    try:
        from datasets import load_dataset
        ds = load_dataset("mfeklix/cifake", split="train")
        counts = {"real": 0, "ai_generated": 0}
        seen_images = {}
        MAX_PER_CLASS = 5000

        for item in ds:
            label     = item["label"]  # 0=real, 1=AI
            class_dir = "real" if label == 0 else "ai_generated"
            if counts[class_dir] >= MAX_PER_CLASS:
                continue

            image = item["image"].convert("RGB")
            image_hash = hashlib.sha256(image.tobytes()).hexdigest()
            if image_hash in seen_images:
                if seen_images[image_hash] != label:
                    print(f"[CIFAKE] Skipping duplicate image with conflicting labels: {image_hash[:12]}")
                continue
            seen_images[image_hash] = label
            bucket = int(image_hash[:8], 16) % 100
            split = "train" if bucket < 70 else "val" if bucket < 85 else "test"
            out_path = os.path.join(output_dir, "images", split, class_dir,
                                    f"cifake_{counts[class_dir]:05d}.jpg")
            image.save(out_path, format="JPEG", quality=95)
            counts[class_dir] += 1

            if sum(counts.values()) % 1000 == 0:
                print(f"  Processed {sum(counts.values())} images...")

            if all(v >= MAX_PER_CLASS for v in counts.values()):
                break

        print(f"[CIFAKE] Real: {counts['real']}  AI: {counts['ai_generated']}")
    except Exception as e:
        raise RuntimeError(f"CIFAKE preparation failed: {e}") from e


def print_summary(output_dir):
    print("\n" + "="*50)
    print("DATASET PREPARATION SUMMARY")
    print("="*50)
    for f in ["text/ai_human.jsonl", "text/fakenews.jsonl"]:
        path = os.path.join(output_dir, f)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as source:
                lines = sum(1 for _ in source)
            print(f"  {f}: {lines:,} samples")
    for split in ["train", "val", "test"]:
        for cls in ["real", "ai_generated"]:
            d = os.path.join(output_dir, "images", split, cls)
            if os.path.exists(d):
                n = len(os.listdir(d))
                print(f"  images/{split}/{cls}: {n} images")
    print("="*50)


def main(args):
    if args.liar_only:
        Path(args.output_dir, "text").mkdir(parents=True, exist_ok=True)
        prepare_liar(args.output_dir)
        print_summary(args.output_dir)
        return

    make_dirs(args.output_dir)
    prepare_hc3(args.output_dir)
    prepare_liar(args.output_dir)
    prepare_cifake(args.output_dir)
    print_summary(args.output_dir)
    print("\n[✓] Dataset preparation complete!")
    print("Next steps:")
    print("  python train_roberta.py --data_path ./data/text/ai_human.jsonl")
    print("  python train_linguaforge.py --data_path ./data/text/ai_human.jsonl")
    print("  python train_fakenews.py --data_path ./data/text/fakenews.jsonl")
    print("  python train_image_classifier.py --data_dir ./data/images --model efficientnet")
    print("  python train_image_classifier.py --data_dir ./data/images --model vit")
    print("  python train_ensemble.py")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="./data")
    p.add_argument("--liar-only", action="store_true", help="prepare only LIAR text data for the misinformation classifier")
    main(p.parse_args())
