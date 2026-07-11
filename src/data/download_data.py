"""Download or unpack the PlantVillage dataset.

Expected final layout:
    data/raw/PlantVillage/<class_name>/<image files>

The Kaggle API path used here is the common PlantVillage mirror
`emmarex/plantdisease`. If your Kaggle source differs, pass `--dataset`.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import zipfile
from pathlib import Path


DEFAULT_DATASET = "emmarex/plantdisease"
RAW_DIR = Path("data/raw")
PLANTVILLAGE_DIR = RAW_DIR / "PlantVillage"


def _find_zip(raw_dir: Path) -> Path | None:
    zips = sorted(raw_dir.glob("*.zip"))
    return zips[0] if zips else None


def _normalize_extracted_tree(raw_dir: Path) -> None:
    """Move nested PlantVillage-style folders to data/raw/PlantVillage."""
    candidates = [
        raw_dir / "PlantVillage",
        raw_dir / "plantvillage",
        raw_dir / "Plant_leave_diseases_dataset_without_augmentation",
        raw_dir / "color",
    ]
    source = next((path for path in candidates if path.exists() and path.is_dir()), None)
    if source is None:
        for path in raw_dir.rglob("*"):
            if path.is_dir() and any(child.is_dir() for child in path.iterdir()):
                if any(child.suffix.lower() in {".jpg", ".jpeg", ".png"} for child in path.rglob("*")):
                    source = path
                    break

    if source is None:
        raise FileNotFoundError("Could not find extracted PlantVillage class folders.")

    if source.resolve() == PLANTVILLAGE_DIR.resolve():
        return

    if PLANTVILLAGE_DIR.exists():
        shutil.rmtree(PLANTVILLAGE_DIR)
    PLANTVILLAGE_DIR.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(PLANTVILLAGE_DIR))


def download_with_kaggle(dataset: str, raw_dir: Path) -> bool:
    """Download using Kaggle CLI. Returns False if CLI/credentials are unavailable."""
    if not (os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY")):
        return False
    if shutil.which("kaggle") is None:
        return False

    raw_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", dataset, "-p", str(raw_dir)],
        check=True,
    )
    return True


def extract_dataset(raw_dir: Path = RAW_DIR) -> Path:
    """Extract a PlantVillage zip already present in data/raw."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = _find_zip(raw_dir)
    if zip_path is None and not PLANTVILLAGE_DIR.exists():
        raise FileNotFoundError(
            "No Kaggle zip found in data/raw. Manual fallback:\n"
            "1. Download PlantVillage from Kaggle, for example emmarex/plantdisease.\n"
            "2. Place the downloaded .zip file in data/raw/.\n"
            "3. Run: python -m src.data.download_data --skip-download"
        )

    if zip_path is not None:
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(raw_dir)
    _normalize_extracted_tree(raw_dir)
    return PLANTVILLAGE_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Download/extract PlantVillage dataset.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Kaggle dataset slug.")
    parser.add_argument("--raw-dir", default=str(RAW_DIR), help="Raw data directory.")
    parser.add_argument("--skip-download", action="store_true", help="Only extract an existing zip.")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    if not args.skip_download:
        downloaded = download_with_kaggle(args.dataset, raw_dir)
        if not downloaded:
            print(
                "Kaggle CLI or credentials were not available. Place the PlantVillage zip in "
                f"{raw_dir} and rerun with --skip-download."
            )

    dataset_dir = extract_dataset(raw_dir)
    print(f"PlantVillage dataset ready at {dataset_dir}")


if __name__ == "__main__":
    main()
