# Production Model

No production model is selected yet.

Selection remains intentionally blocked until EfficientNetV2-S, ConvNeXt-Tiny, and ConvNeXt-Base all finish using the unchanged split and pass ONNX parity. This avoids presenting a partial benchmark winner as the production choice.

Run or resume the benchmark with:

```powershell
.\.venv\Scripts\python.exe -m src.training.benchmark --config configs/training/phase2_5.yaml --train
```
