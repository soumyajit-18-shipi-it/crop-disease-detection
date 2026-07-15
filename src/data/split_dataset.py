from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _image_count(path: Path) -> int:
    return sum(1 for item in path.rglob("*") if item.suffix.lower() in IMAGE_EXTENSIONS)


def detect_dataset_root(raw_dir: str | Path = "data/raw") -> Path:
    """Find the PlantVillage class-folder root under a downloaded Kaggle tree."""
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data directory does not exist: {raw_path}")

    candidates: list[tuple[int, int, Path]] = []
    for path in [raw_path, *[p for p in raw_path.rglob("*") if p.is_dir()]]:
        class_dirs = []
        for child in path.iterdir():
            if not child.is_dir():
                continue
            direct_images = [p for p in child.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
            if direct_images:
                class_dirs.append(child)
        if class_dirs:
            candidates.append((len(class_dirs), sum(_image_count(p) for p in class_dirs), path))

    if not candidates:
        raise FileNotFoundError(f"No class folders with images found under {raw_path}")
    candidates.sort(key=lambda item: (item[1], item[0], -len(item[2].parts)), reverse=True)
    return candidates[0][2]


def _class_images(raw_dir: Path) -> dict[str, list[Path]]:
    classes: dict[str, list[Path]] = {}
    for class_dir in sorted(path for path in raw_dir.iterdir() if path.is_dir()):
        images = [p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
        if images:
            classes[class_dir.name] = images
    if not classes:
        raise FileNotFoundError(f"No class folders with images found under {raw_dir}")
    return classes


def _copy_split(items: list[tuple[Path, str]], output_dir: Path, split: str) -> None:
    for image_path, class_name in items:
        dest_dir = output_dir / split / class_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, dest_dir / image_path.name)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stratified_duplicate_safe_split(
    classes: dict[str, list[Path]],
    seed: int,
) -> tuple[list[tuple[Path, str]], list[tuple[Path, str]], list[tuple[Path, str]]]:
    rng = random.Random(seed)
    train_items: list[tuple[Path, str]] = []
    val_items: list[tuple[Path, str]] = []
    test_items: list[tuple[Path, str]] = []

    for class_name, images in sorted(classes.items()):
        groups_by_hash: dict[str, list[Path]] = {}
        for image in images:
            groups_by_hash.setdefault(_file_hash(image), []).append(image)

        groups = list(groups_by_hash.values())
        rng.shuffle(groups)
        total = sum(len(group) for group in groups)
        targets = {
            "train": round(total * 0.70),
            "val": round(total * 0.15),
            "test": total - round(total * 0.70) - round(total * 0.15),
        }
        split_groups = {"train": [], "val": [], "test": []}
        split_counts = {"train": 0, "val": 0, "test": 0}

        for group in sorted(groups, key=len, reverse=True):
            split = min(
                ("train", "val", "test"),
                key=lambda name: split_counts[name] / max(targets[name], 1),
            )
            split_groups[split].extend(group)
            split_counts[split] += len(group)

        train_items.extend((image, class_name) for image in split_groups["train"])
        val_items.extend((image, class_name) for image in split_groups["val"])
        test_items.extend((image, class_name) for image in split_groups["test"])

    return train_items, val_items, test_items


def _write_mapping(class_names: list[str], mapping_path: Path) -> None:
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
    idx_to_class = {str(idx): class_name for class_name, idx in class_to_idx.items()}
    payload = {
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
    }
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    with mapping_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def split_dataset(
    raw_dir: str | Path = "data/raw",
    processed_dir: str | Path = "data/processed",
    mapping_path: str | Path = "data/class_mapping.json",
    seed: int = 42,
) -> None:
    raw_path = detect_dataset_root(raw_dir)
    print(f"Detected dataset root: {raw_path}")
    output_path = Path(processed_dir)
    classes = _class_images(raw_path)
    class_names = sorted(classes)

    train_items, val_items, test_items = _stratified_duplicate_safe_split(classes, seed)

    if output_path.exists():
        shutil.rmtree(output_path)
    for split in ("train", "val", "test"):
        (output_path / split).mkdir(parents=True, exist_ok=True)

    _copy_split(train_items, output_path, "train")
    _copy_split(val_items, output_path, "val")
    _copy_split(test_items, output_path, "test")
    _write_mapping(class_names, Path(mapping_path))

    print("Class distribution summary")
    for split_name, items in [("train", train_items), ("val", val_items), ("test", test_items)]:
        counts = Counter(class_name for _, class_name in items)
        print(f"\n{split_name}: {sum(counts.values())} images")
        for class_name in class_names:
            marker = "  <50 images" if counts[class_name] < 50 else ""
            print(f"  {class_name}: {counts[class_name]}{marker}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Split PlantVillage into train/val/test.")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--mapping-path", default="data/class_mapping.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    split_dataset(args.raw_dir, args.processed_dir, args.mapping_path, args.seed)


if __name__ == "__main__":
    main()
