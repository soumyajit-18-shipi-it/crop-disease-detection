# Production Model Comparison

Every candidate uses the persisted split manifest. Missing values are shown as `-`; no metric is estimated or backfilled.

| Architecture | Status | Input px | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | Balanced Acc. | MCC | Kappa | ROC-AUC | Val ECE | ONNX img/s | ONNX MB | Peak GPU GB | Train min | Score |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| efficientnetv2_s | active v1 | 300 | 0.998707 | 0.998752 | 0.999063 | 0.998905 | 0.998708 | 0.999063 | 0.998588 | 0.998587 | 0.999953 | 0.000130 | 33.541 | 76.94 | 2.32 | 269.30 | - |
| convnext_tiny | no completed evaluated release | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| convnext_base | no completed evaluated release | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |

## Selection

EfficientNetV2-S `v1` is the active application release because its completed artifact is available and passed ONNX parity. This does not assert that it is a final multi-candidate benchmark winner or that it is field-ready.
