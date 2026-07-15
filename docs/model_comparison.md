# Production Model Comparison

EfficientNetV2-S and ConvNeXt-Tiny use the same immutable split (`f41148f...a372`) and 15-class order. Full-precision stored values—not rounded display values—govern conclusions. ConvNeXt-Base has not been trained, so formal selection is blocked.

## Quality, calibration, size, and deployment latency

The latency columns below come from `artifacts/training/crop_disease_phase2_5/fair_benchmark.json`: one process and host, batch 1, 10 warm-ups, 50 iterations, PyTorch CPU thread count 1, and ONNX Runtime `CPUExecutionProvider` in sequential mode with one intra-op and one inter-op thread. Each model uses its recorded native input size (300 for EfficientNetV2-S; 224 for ConvNeXt-Tiny).

| Metric | EfficientNetV2-S | ConvNeXt-Tiny |
|---|---:|---:|
| Test accuracy | 0.9987071751777634 | 0.9987071751777634 |
| Balanced accuracy | 0.9990629024344951 | 0.9990951646618792 |
| Macro precision | 0.9987523264281355 | 0.9988176556279121 |
| Macro recall | 0.9990629024344951 | 0.9990951646618792 |
| Macro F1 | 0.9989051607670526 | 0.9989537492012454 |
| Macro ROC-AUC | 0.9999530005208085 | 0.9999377573570158 |
| Test ECE before calibration | 0.16125731483849584 | 0.17098740433012255 |
| Test ECE after calibration | 0.001292268100826166 | 0.0005153383287462277 |
| ONNX median CPU latency | 734.7240500021144 ms | 518.2955500022217 ms |
| ONNX P95 CPU latency | 858.7892949992238 ms | 595.9242750002886 ms |
| Model size | 80,679,586 bytes (76.94 MiB) | 111,404,469 bytes (106.24 MiB) |
| Parameters | 20,196,703 | 27,831,663 |
| Peak training GPU memory | 2,486,524,928 bytes | 1,451,293,184 bytes |
| ONNX parity | Pass; max abs error 2.503395e-06 | Pass; max abs error 1.335144e-05 |

EfficientNetV2-S's previously fitted temperature is exactly the lower optimization bound (`0.05`), so that calibration result warrants boundary review. ConvNeXt-Tiny's fitted temperature (`0.34827497061965157`) is not at a boundary. ConvNeXt has better calibrated test ECE but worse uncalibrated ECE.

## Additional common-condition latency

| Runtime | EfficientNetV2-S median / P95 | ConvNeXt-Tiny median / P95 |
|---|---:|---:|
| PyTorch CUDA | 82.070950 / 104.781050 ms | 36.741900 / 60.548200 ms |
| PyTorch CPU | 743.472800 / 842.142370 ms | 766.330100 / 867.394645 ms |
| ONNX Runtime CPU | 734.724050 / 858.789295 ms | 518.295550 / 595.924275 ms |
| PyTorch CUDA inference peak | 106,641,920 bytes | 136,382,464 bytes |

These single-thread results must not be mixed with the original default-thread export measurements (EfficientNet 29.8146 ms and ConvNeXt 39.5464 ms median). Both sets are retained and labelled with their provider/thread methodology.

## Per-class regressions

Most class F1 values are identical. The non-zero ConvNeXt-minus-EfficientNet differences are:

| Class | Support | EfficientNet F1 | ConvNeXt F1 | Delta |
|---|---:|---:|---:|---:|
| Tomato_Late_blight | 287 | 0.9982547993019197 | 0.9965034965034965 | -0.0017513027984232 |
| Tomato_Septoria_leaf_spot | 265 | 1.0000000000000000 | 0.9981167608286252 | -0.0018832391713748 |
| Tomato_Spider_mites_Two_spotted_spider_mite | 252 | 0.9980119284294234 | 1.0000000000000000 | +0.0019880715705766 |
| Tomato__Target_Spot | 210 | 0.9976247030878860 | 1.0000000000000000 | +0.0023752969121140 |

The smallest-support classes, Potato healthy (23) and Tomato mosaic virus (56), remain at F1 1.0 for both models. ConvNeXt's tiny aggregate macro-F1 gain therefore does not constitute an unambiguous win: it includes regressions on Late blight and Septoria, while using 30,724,883 more ONNX bytes and 7,634,960 more parameters.

## Selection outcome

`configs/training/phase2_5.yaml` has `selection.require_all_candidates: true`. Running the existing selection command after both completed models returned `selected=none` because ConvNeXt-Base is not started. The requirements were not modified and no partial two-model score was promoted.

The active production model remains immutable EfficientNetV2-S v1. ConvNeXt-Tiny remains a parity-verified candidate under its training artifact directory; no production release directory or backend default was changed.
