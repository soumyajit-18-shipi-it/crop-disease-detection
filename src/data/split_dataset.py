from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _class_images(raw_dir: Path) -> dict[str, list[Path]]:
    classes: dict[str, list[Path]] = {}
    for class_dir in sorted(path for path in raw_dir.iterdir() if path.is_dir()):
        images = [p for p in class_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
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
    raw_dir: str | Path = "data/raw/PlantVillage",
    processed_dir: str | Path = "data/processed",
    mapping_path: str | Path = "data/class_mapping.json",
    seed: int = 42,
) -> None:
    raw_path = Path(raw_dir)
    output_path = Path(processed_dir)
    classes = _class_images(raw_path)
    class_names = sorted(classes)

    all_items = [(image, class_name) for class_name, images in classes.items() for image in images]
    labels = [class_name for _, class_name in all_items]
    train_items, temp_items, train_labels, temp_labels = train_test_split(
        all_items,
        labels,
        test_size=0.30,
        random_state=seed,
        stratify=labels,
    )
    val_items, test_items = train_test_split(
        temp_items,
        test_size=0.50,
        random_state=seed,
        stratify=temp_labels,
    )

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
    parser.add_argument("--raw-dir", default="data/raw/PlantVillage")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--mapping-path", default="data/class_mapping.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    split_dataset(args.raw_dir, args.processed_dir, args.mapping_path, args.seed)


if __name__ == "__main__":
    main()
