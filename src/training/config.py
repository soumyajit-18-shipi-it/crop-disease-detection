from dataclasses import dataclass


@dataclass
class TrainConfig:
    data_dir: str = "data/processed"
    model_name: str = "efficientnet_b0"
    num_classes: int = 6
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 1e-3
    seed: int = 42
    checkpoint_dir: str = "models/checkpoints"
