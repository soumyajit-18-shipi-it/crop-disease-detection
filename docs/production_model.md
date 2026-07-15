# Production Model

The active deployed application release remains EfficientNetV2-S v1 at `models/releases/efficientnetv2_s_v1/`.

- ONNX SHA-256: `bd0af61cba3bcc83a59d93348e6e43a539c6b60069203d7ee9d4ee746810beaa`.
- Input: 300x300 RGB.
- Classes: 15.
- Release manifest and checksum verification remain unchanged.
- Rollback path: the immutable EfficientNetV2-S v1 release remains the active/default bundle.

## Phase 2.5 comparative-selection status

EfficientNetV2-S and ConvNeXt-Tiny have completed training, evaluation, calibration, ONNX export, parity validation, and common-condition benchmarking on the same split. ConvNeXt-Tiny tied test accuracy and had a 0.0000485884341928 macro-F1 increase, but had slightly lower macro ROC-AUC, per-class regressions on Tomato Late blight and Septoria leaf spot, and a 38.1% larger ONNX file.

Formal production selection is intentionally blocked because `selection.require_all_candidates` is true and ConvNeXt-Base has not been trained. Running:

```powershell
.\.venv\Scripts\python.exe -m src.training.benchmark --config configs/training/phase2_5.yaml
```

returned `selected=none`. ConvNeXt-Tiny was not promoted, no active-model default changed, and no EfficientNet ONNX bytes were modified.

## Next required action

Train ConvNeXt-Base on the unchanged split, then rerun selection:

```powershell
.\.venv\Scripts\python.exe -m src.training.train --config configs/training/phase2_5.yaml --architecture convnext_base
.\.venv\Scripts\python.exe -m src.training.benchmark --config configs/training/phase2_5.yaml
```

ConvNeXt-Base was not started during the ConvNeXt-Tiny task. A later winner must still satisfy ONNX parity, macro-F1 tolerance, calibration, deployment latency/size, and per-class regression review before a new versioned release may be created.
