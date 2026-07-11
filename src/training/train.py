import torch

from src.training.config import TrainConfig
from src.utils.seed import set_seed


def main() -> None:
    """Training entry point placeholder."""
    config = TrainConfig()
    set_seed(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    raise NotImplementedError(f"Training pipeline will be wired once data is available. Device: {device}")


if __name__ == "__main__":
    main()
