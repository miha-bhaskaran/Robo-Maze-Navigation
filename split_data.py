

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path


def load_labels(label_file: Path):
    """Returns list of (filename, label) tuples."""
    entries = []
    with open(label_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            entries.append((parts[0], parts[1]))
    return entries


def split_and_copy(data_path, train_path, test_path, val_split, seed):
    random.seed(seed)

    data_path  = Path(data_path)
    train_path = Path(train_path)
    test_path  = Path(test_path)

    label_file = data_path / "labels.txt"
    if not label_file.exists():
        raise FileNotFoundError(f"No labels.txt found in {data_path}")

    entries = load_labels(label_file)
    print(f"Total entries in labels.txt: {len(entries)}")


    by_class = defaultdict(list)
    for fname, label in entries:
        by_class[label].append((fname, label))

    train_entries, test_entries = [], []
    for label, items in sorted(by_class.items()):
        random.shuffle(items)
        split = max(1, int(len(items) * (1 - val_split)))
        train_entries.extend(items[:split])
        test_entries.extend(items[split:])

    print(f"Train: {len(train_entries)} | Test: {len(test_entries)}")

  
    for out_dir, entries_subset in [(train_path, train_entries),
                                    (test_path,  test_entries)]:
        out_dir.mkdir(parents=True, exist_ok=True)
        skipped = 0

        with open(out_dir / "labels.txt", "w") as lf:
            for fname, label in entries_subset:
                src = data_path / fname
                if not src.exists():
                    print(f"  [warn] Missing: {src}, skipping.")
                    skipped += 1
                    continue
                shutil.copy2(src, out_dir / fname)
                stem = fname.replace(".png", "")
                lf.write(f"{stem}, {label}\n")

        copied = len(entries_subset) - skipped
        print(f"  → {out_dir}: {copied} images + labels.txt written"
              + (f" ({skipped} skipped)" if skipped else ""))

    print("\n✅ Done!")
    print(f"   Train folder : {train_path}")
    print(f"   Test folder  : {test_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stratified train/test split.")
    parser.add_argument("--data_path",  default="./merged_output")
    parser.add_argument("--train_path", default="./train_data")
    parser.add_argument("--test_path",  default="./test_data")
    parser.add_argument("--val_split",  type=float, default=0.2)
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    split_and_copy(
        data_path  = args.data_path,
        train_path = args.train_path,
        test_path  = args.test_path,
        val_split  = args.val_split,
        seed       = args.seed,
    )