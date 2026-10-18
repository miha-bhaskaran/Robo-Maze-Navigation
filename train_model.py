import argparse
import os
import pickle
import random
from pathlib import Path

import cv2
import numpy as np



def extract_features(img_bgr: np.ndarray, img_size: int = 64) -> np.ndarray:
    resized = cv2.resize(img_bgr, (img_size, img_size), interpolation=cv2.INTER_AREA)
    gray    = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    hog = cv2.HOGDescriptor(
        _winSize=(img_size, img_size),
        _blockSize=(24, 24),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9,
    )
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    color_hist = np.concatenate([
        np.histogram(hsv[:, :, ch], bins=16, range=(0, 256))[0]
        for ch in range(3)
    ]).astype(np.float32)
    color_hist = color_hist / (color_hist.sum() + 1e-6)
    return hog.compute(gray).flatten()



def load_dataset(data_path: Path, img_size: int = 64):
    """
    Reads labels.txt and loads all images as feature vectors.
    Returns (X, y) as numpy arrays.
    """
    label_file = data_path / "labels.txt"
    if not label_file.exists():
        raise FileNotFoundError(f"No labels.txt found in {data_path}")

    X, y = [], []
    skipped = 0

    with open(label_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Format: "filename.png, label"  (comma separated)
            parts = line.split(",")
            if len(parts) < 2:
                continue
            filename, label = parts[0].strip(), int(parts[1].strip())

            img_path = data_path / (filename + ".png")
            if not img_path.exists():
                skipped += 1
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                skipped += 1
                continue

            features = extract_features(img, img_size)
            X.append(features)
            y.append(label)

    if skipped:
        print(f"  [warn] Skipped {skipped} missing/unreadable images.")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def train_knn(X_train: np.ndarray, y_train: np.ndarray, k: int = 21):
    knn = cv2.ml.KNearest_create()
    knn.setDefaultK(k)
    knn.setIsClassifier(True)
    knn.train(X_train, cv2.ml.ROW_SAMPLE, y_train)
    return knn


def evaluate(knn, X_val: np.ndarray, y_val: np.ndarray):
    CLASS_NAMES = {
        0: "empty wall",
        1: "left",
        2: "right",
        3: "do not enter",
        4: "stop",
        5: "goal",
    }

    _, results, _, _ = knn.findNearest(X_val, knn.getDefaultK())
    preds = results.flatten().astype(int)

    correct = (preds == y_val).sum()
    acc = correct / len(y_val) * 100
    print(f"\n  Validation accuracy: {correct}/{len(y_val)} = {acc:.1f}%")

    # Per-class breakdown
    print("\n  Per-class accuracy:")
    for cls_id, cls_name in CLASS_NAMES.items():
        mask = y_val == cls_id
        if mask.sum() == 0:
            continue
        cls_acc = (preds[mask] == y_val[mask]).sum() / mask.sum() * 100
        print(f"    Class {cls_id} ({cls_name:>12s}): {cls_acc:.1f}%  "
              f"({mask.sum()} samples)")

    return acc


def main():
    parser = argparse.ArgumentParser(description="Train KNN sign classifier.")
    parser.add_argument("--data_path",  required=True,
                        help="Path to merged+augmented dataset folder.")
    parser.add_argument("--model_path", default="./model.pkl",
                        help="Where to save the trained model (default: ./model.pkl).")
    parser.add_argument("--k",          type=int, default=3,
                        help="Number of neighbours for KNN (default: 5).")
    parser.add_argument("--img_size",   type=int, default=64,
                        help="Resize images to img_size x img_size (default: 64).")
    parser.add_argument("--val_split",  type=float, default=0.2,
                        help="Fraction of data to use for validation (default: 0.2).")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    data_path  = Path(args.data_path)
    model_path = Path(args.model_path)


    print(f"\nLoading dataset from: {data_path}")
    X, y = load_dataset(data_path, img_size=args.img_size)

    indices = np.arange(len(X))
    np.random.shuffle(indices)


    train_idx, val_idx = [], []
    for cls in np.unique(y):
        cls_idx = indices[y[indices] == cls]
        split   = max(1, int(len(cls_idx) * (1 - args.val_split)))
        train_idx.extend(cls_idx[:split].tolist())
        val_idx.extend(cls_idx[split:].tolist())

    X_train, y_train = X[train_idx], y[train_idx]
    X_val,   y_val   = X[val_idx],   y[val_idx]
    print(f"  Train: {len(X_train)}  |  Val: {len(X_val)}  "
          f"({args.val_split*100:.0f}% val split)")


    print(f"\nTraining KNN (k={args.k}) ...")
    knn = train_knn(X_train, y_train, k=args.k)

    evaluate(knn, X_val, y_val)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    knn_path  = str(model_path).replace(".pkl", ".xml")
    meta_path = str(model_path)

    knn.save(knn_path)
    with open(meta_path, "wb") as f:
        pickle.dump({"k": args.k, "img_size": args.img_size, "knn_path": knn_path}, f)

    print(f"\n✅ Model saved:")
    print(f"   KNN weights : {knn_path}")
    print(f"   Metadata    : {meta_path}")


if __name__ == "__main__":
    main()