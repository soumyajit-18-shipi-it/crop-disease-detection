from pathlib import Path


def download_dataset(output_dir: str | Path = "data/raw") -> Path:
    """Placeholder for dataset download logic."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    raise NotImplementedError("Add dataset source and download/extraction logic here.")


if __name__ == "__main__":
    download_dataset()
