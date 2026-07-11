from PIL import Image


def load_image(image_path: str) -> Image.Image:
    """Load an image from disk in RGB format."""
    return Image.open(image_path).convert("RGB")


def preprocess_for_model(image: Image.Image):
    """Preprocess PIL image for model inference.

    TODO: reuse the exact validation transforms and normalization used in training.
    """
    return image
