# produce a folder called data with concat 2025F_Gimgs, 2025F_imgs, 2026S_imgs with crops, lighting, 

"""
merge_and_augment.py
--------------------
Merges multiple image datasets (each with a label.txt) into a single flat
output folder, resolves filename collisions by prefixing with the source
dataset index, and applies augmentations suitable for arrow images.

Augmentations applied (originals are always kept):
  - Brightness / contrast jitter  (lighting)
  - Small rotations (±15°)        (with reflection padding to avoid black corners)
  - Zoom / random crop            (zooms in 10-20%, then resizes back)
  - Gaussian blur + noise         (combined or separate)

NOTE: Horizontal flip is intentionally skipped since left/right arrows would
      get their labels swapped.

Usage
-----
  python merge_and_augment.py \
      --datasets /2025F_Gimgs /2025F_imgs /2026S_imgs \
      --output   /data \
      [--augments-per-image 4]   

Dependencies
------------
  pip install Pillow numpy
"""

import argparse
import os
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


# ── Augmentation helpers ─────────────────────────────────────────────────────

def augment_brightness_contrast(img: Image.Image) -> Image.Image:
    """Random brightness and contrast jitter."""
    brightness = random.uniform(0.6, 1.4)
    contrast   = random.uniform(0.7, 1.3)
    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    return img


def augment_rotation(img: Image.Image, max_degrees: float = 15.0) -> Image.Image:
    """Small random rotation with reflection padding to avoid black corners."""
    angle = random.uniform(-max_degrees, max_degrees)
    # Use BICUBIC resampling; expand=False keeps original canvas size
    return img.rotate(angle, resample=Image.BICUBIC, expand=False)


def augment_zoom_crop(img: Image.Image, zoom_range=(0.10, 0.20)) -> Image.Image:
    """Zoom in by cropping a random fraction then resizing back to original."""
    w, h = img.size
    zoom = random.uniform(*zoom_range)
    left   = int(w * zoom * random.uniform(0, 1))
    top    = int(h * zoom * random.uniform(0, 1))
    right  = w - int(w * zoom * (1 - (left / max(w * zoom, 1))))
    bottom = h - int(h * zoom * (1 - (top  / max(h * zoom, 1))))
    # Clamp
    left, top     = max(0, left), max(0, top)
    right, bottom = min(w, max(left + 1, right)), min(h, max(top + 1, bottom))
    cropped = img.crop((left, top, right, bottom))
    return cropped.resize((w, h), Image.BICUBIC)


def augment_blur_noise(img: Image.Image) -> Image.Image:
    """Apply a mild Gaussian blur followed by additive Gaussian noise."""
    # Blur
    radius = random.uniform(0.5, 1.5)
    img = img.filter(ImageFilter.GaussianBlur(radius=radius))
    # Noise
    arr   = np.array(img).astype(np.int16)
    noise = np.random.normal(0, random.uniform(5, 20), arr.shape).astype(np.int16)
    arr   = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


AUGMENT_POOL = [
    augment_brightness_contrast,
    augment_rotation,
    augment_zoom_crop,
    augment_blur_noise,
]


def apply_random_augmentations(img: Image.Image, n: int = 2) -> Image.Image:
    """
    Pick n augmentations at random and chain them.
    Always includes lighting; picks remaining from the pool.
    """
    chosen = [augment_brightness_contrast]  # lighting always included
    extras = random.sample(AUGMENT_POOL[1:], k=min(n - 1, len(AUGMENT_POOL) - 1))
    chosen += extras
    random.shuffle(chosen)
    for fn in chosen:
        img = fn(img)
    return img


# ── Core logic ────────────────────────────────────────────────────────────────

def load_labels(label_path: Path) -> dict[str, str]:
    labels = {}
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")  # split on comma instead of space
            if len(parts) < 2:
                print(f"  [warn] Skipping malformed line: {line!r}")
                continue
            filename = parts[0].strip() + ".png"  # add .png extension
            label = parts[1].strip()
            labels[filename] = label
    return labels


def merge_and_augment(
    dataset_dirs: list[Path],
    output_dir: Path,
    augments_per_image: int = 4,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_labels: dict[str, str] = {}   # output_filename → label

    for ds_idx, ds_dir in enumerate(dataset_dirs):
        ds_dir = ds_dir.resolve()
        label_file = ds_dir / "labels.txt"
        if not label_file.exists():
            print(f"[warn] No labels.txt in {ds_dir}, skipping.")
            continue

        labels = load_labels(label_file)
        print(f"\n[Dataset {ds_idx + 1}] {ds_dir}  ({len(labels)} images)")

        for orig_name, label in labels.items():
            src_path = ds_dir / orig_name
            if not src_path.exists():
                print(f"  [warn] Missing image: {src_path}, skipping.")
                continue

            # ── Rename to avoid collisions: ds0_img1.jpg ──────────────────
            stem      = src_path.stem
            suffix    = src_path.suffix
            base_name = f"ds{ds_idx}_{stem}"

            # Copy original
            out_name = f"{base_name}{suffix}"
            shutil.copy2(src_path, output_dir / out_name)
            combined_labels[out_name] = label

            # Augmented copies
            try:
                img = Image.open(src_path).convert("RGB")
            except Exception as e:
                print(f"  [warn] Could not open {src_path}: {e}")
                continue

            for aug_i in range(augments_per_image):
                aug_img  = apply_random_augmentations(img, n=random.randint(2, 3))
                aug_name = f"{base_name}_aug{aug_i}{suffix}"
                aug_img.save(output_dir / aug_name, quality=95)
                combined_labels[aug_name] = label

        print(f"  → Copied + augmented {len(labels)} images "
              f"({len(labels) * (1 + augments_per_image)} total from this dataset)")

    # Write combined label.txt
    label_out = output_dir / "labels.txt"
    with open(label_out, "w") as f:
        for fname, lbl in sorted(combined_labels.items()):
            f.write(f"{fname} {lbl}\n")

    print(f"\n✅ Done!")
    print(f"   Output folder : {output_dir}")
    print(f"   Total images  : {len(combined_labels)}")
    print(f"   labels.txt     : {label_out}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge datasets + augment images (arrow-safe)."
    )
    parser.add_argument(
        "--datasets", nargs="+", required=True,
        help="Paths to dataset folders (each must contain a labels.txt)."
    )
    parser.add_argument(
        "--output", required=True,
        help="Path to output folder (will be created if it doesn't exist)."
    )
    parser.add_argument(
        "--augments-per-image", type=int, default=4,
        help="Number of augmented copies to generate per original image (default: 4)."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    dataset_paths = [Path(p) for p in args.datasets]
    output_path   = Path(args.output)

    merge_and_augment(
        dataset_dirs=dataset_paths,
        output_dir=output_path,
        augments_per_image=args.augments_per_image,
    )