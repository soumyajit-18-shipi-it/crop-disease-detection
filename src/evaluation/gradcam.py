from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def generate_gradcam(model, image_tensor: torch.Tensor, target_layer) -> np.ndarray:
    """Return an RGB Grad-CAM overlay for a single normalized image tensor."""
    model.eval()
    input_tensor = image_tensor.unsqueeze(0) if image_tensor.ndim == 3 else image_tensor
    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        grayscale_cam = cam(input_tensor=input_tensor)[0]

    image_np = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_np = np.clip((image_np * std) + mean, 0, 1)
    overlay = show_cam_on_image(image_np.astype(np.float32), grayscale_cam, use_rgb=True)
    return overlay


def save_gradcam_examples(examples: list[tuple[str, np.ndarray]], output_dir: str | Path = "docs/gradcam_examples") -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, overlay in examples:
        cv2.imwrite(str(output / f"{name}.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
