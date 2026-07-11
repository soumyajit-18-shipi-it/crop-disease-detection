from pathlib import Path


def split_dataset(raw_dir: str | Path, processed_dir: str | Path, seed: int = 42) -> None:
    """Split raw images into train/val/test folders.

    TODO: implement stratified image copy/move once dataset conventions are known.
    """
    Path(processed_dir).mkdir(parents=True, exist_ok=True)
    raise NotImplementedError("Implement dataset splitting after raw dataset format is finalized.")
