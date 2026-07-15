# Active Application Model

Leaflight currently serves the completed Phase 2.5 EfficientNetV2-S bundle as release `v1`:

- ONNX: `models/releases/efficientnetv2_s_v1/model.onnx`
- Metadata: `models/releases/efficientnetv2_s_v1/model.json`
- Metrics: `models/releases/efficientnetv2_s_v1/metrics.json`
- Checksums: `models/releases/efficientnetv2_s_v1/checksum.sha256`

## Evaluated contract

| Field | Value |
|---|---|
| Architecture | EfficientNetV2-S (`efficientnetv2_s`) |
| Input | 300×300 RGB, NCHW float32 |
| Classes | 15, in the canonical `data/class_mapping.json` order |
| Resize/crop | Bicubic shortest-side resize followed by center crop |
| Normalization | mean `[0.5, 0.5, 0.5]`, std `[0.5, 0.5, 0.5]` |
| Calibration temperature | 0.05 |
| Test accuracy | 0.9987071752 |
| Test macro F1 | 0.9989051608 |
| Calibrated test ECE | 0.0012922681 |
| ONNX CPU median latency | 29.8146 ms/image |
| ONNX parity | Passed; max absolute error `2.5034e-06` |
| ONNX SHA-256 | `bd0af61cba3bcc83a59d93348e6e43a539c6b60069203d7ee9d4ee746810beaa` |

This is the active deployment release, not a claim of broad field or production readiness. Evaluation is dominated by PlantVillage-style imagery. Results require expert interpretation and are not a replacement for advice from a qualified agricultural professional.
