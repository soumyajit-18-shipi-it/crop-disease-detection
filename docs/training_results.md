# Phase 2.5 Training Results

This report contains measured artifacts only. ConvNeXt-Base has not been trained and no value is imputed for it.

## Reproducibility contract

- Hardware: NVIDIA GeForce RTX 3050 6GB Laptop GPU; driver 581.86; CUDA 13.0.
- Training environment: Python 3.14.5, PyTorch 2.12.1+cu130, timm 1.0.28.
- Persisted split: `data/splits/phase1_split.json`.
- Split SHA-256: `f41148f03b29fe0cda41ea8b3caecf35d7c2d2299afd401013b137dab4b8a372`.
- ConvNeXt-Tiny training signature: `3e07b07fc1cb1d65bd7e883f7157b846ae05a36913857b09faf272e83c8a3e93`.
- Dataset: 14,447 train, 3,097 validation, and 3,094 test images across 15 ordered classes.
- ConvNeXt-Tiny input: 224x224 RGB using timm bicubic shortest-side resize and 0.95 center crop.
- Batch: 8 physical, 4 gradient-accumulation steps, effective batch 32. No resource-driven deviation was made.

## ConvNeXt-Tiny resume audit

The original process stopped after checkpointed epoch 6. Windows System event 6006 records an orderly event-log shutdown at 2026-07-14 22:01:22 local time, followed by service start event 6005 at 22:02:34. There was no OOM, Python exception, or application crash event. PID 34444 was no longer running; only its stale `.train.lock` was removed.

- Resume checkpoint: `artifacts/training/crop_disease_phase2_5/convnext_tiny/last.pt`.
- Confirmed checkpoint epoch: 6; next epoch: 7.
- Initial resume checkpoint SHA-256: `398b7cb2e7f5d30cfc56a447860c6632f240f3f5379d6494c20d9421bf07da27`.
- Initial best-checkpoint SHA-256: `af203a9cce8fa2cd395a33dbef62d056dddb8f2384e6b601eab4d5bccc132de2`.
- Restored state: 182 optimizer entries, scheduler step 2705, EMA update 2705, AMP scale 1024, Python/NumPy/Torch/CUDA RNG, and DataLoader generator state.
- Compatibility: architecture, 15-class order, split hash, preprocessing, image size, batch/accumulation, AdamW/cosine schedule, weighted loss, augmentation, seed, EMA, early stopping, and training signature all matched.

The isolated smoke test ran one real augmented train batch and one validation batch without writing the run bundle. Losses and gradients were finite, peak GPU memory was 1,059,860,480 bytes, and `last.pt`/`best.pt` hashes were unchanged.

## Training outcome

Training resumed at epoch 7. Epoch 13 established the final best validation macro F1. Eight consecutive later epochs did not exceed it by the configured 0.0001 minimum delta, so patience-8 early stopping ended the run naturally after epoch 21 rather than forcing epoch 40.

| Item | Result |
|---|---:|
| Starting checkpoint epoch | 6 |
| Resume starting epoch | 7 |
| Last completed epoch | 21 |
| Best epoch | 13 |
| Best validation macro F1 | 0.9991414531660916 |
| Early-stopping result | Triggered after epoch 21 |
| Total recorded training time | 6,150.948080800015 s |
| Peak training GPU memory | 1,451,293,184 bytes |

During the first attempt at epoch 20, explicit finite-gradient protection detected a transient non-finite gradient before the optimizer step and stopped with the atomic epoch-19 checkpoint intact. The handler was corrected so CUDA AMP overflow safely skips the optimizer step, lowers the scaler, and does not advance scheduler or EMA; non-AMP non-finite gradients are also refused. A regression test and a second epoch-19 smoke test passed. The exact resume then completed epochs 20 and 21 with 452 optimizer steps and zero skipped steps in each epoch; the transient overflow did not recur.

Epoch rows 7-21 include train loss/accuracy, validation loss/accuracy, macro precision/recall/F1, learning rate, epoch duration, GPU peak, gradient norm, optimizer steps, and skipped steps. Epochs 1-6 retain their original fields; metrics that were not recorded by the earlier code are explicitly null rather than reconstructed.

## Final ConvNeXt-Tiny test evaluation

The epoch-13 EMA checkpoint was evaluated once on the held-out test split after training ended.

| Metric | Value |
|---|---:|
| Test loss | 0.9710628667722306 |
| Accuracy | 0.9987071751777634 |
| Balanced accuracy | 0.9990951646618792 |
| Macro precision | 0.9988176556279121 |
| Macro recall | 0.9990951646618792 |
| Macro F1 | 0.9989537492012454 |
| Weighted precision | 0.9987129072933358 |
| Weighted recall | 0.9987071751777634 |
| Weighted F1 | 0.9987069469842674 |
| Macro ROC-AUC (OvR) | 0.9999377573570158 |
| Misclassified images | 4 of 3,094 |

Machine-readable per-class metrics, the confusion matrix, misclassified paths/top-3 probabilities, confidence distributions, and raw validation/test logits are stored with the candidate. These artifacts are not copied over EfficientNetV2-S results.

## Calibration

Temperature scaling was fit only on 3,097 validation logits. Temperature `0.34827497061965157` is inside the configured `[0.05, 20.0]` bounds and did not produce a boundary warning.

| Dataset | ECE before | ECE after |
|---|---:|---:|
| Validation | 0.17122959652544198 | 0.001161748528805685 |
| Test | 0.17098740433012255 | 0.0005153383287462277 |

## Candidate bundle

Bundle directory: `artifacts/training/crop_disease_phase2_5/convnext_tiny/`.

- ONNX: `model.onnx`, 111,404,469 bytes, SHA-256 `6e3b25e3600e4e8f641e6a5fdc35547160a77ca2b30ec6c33d177874702c9394`.
- Metadata: `model.json` with ordered classes, preprocessing, calibration, ONNX names/opset, and source-checkpoint hash.
- Source checkpoint: `best.pt`, epoch 13, SHA-256 `91369dcf48e42aaab42d5af8fe21cdc33db44febb520c1024e6e78601f1493c9`.
- Evaluation: `metrics.json`, `classification_report.json`, `classification_report.txt`, `per_class_metrics.csv`, `confusion_matrix.json`, `confusion_matrix.png`, `misclassified_images.json`, `confidence_distribution.json`, and `evaluation_report.md`.
- Reproducibility: `calibration.json`, `evaluation_logits.npz`, `checksum.sha256`, histories, and resume logs.

## Exact commands used

```powershell
python scripts/download_model.py --verify-only
.\.venv\Scripts\python.exe scripts\resume_smoke_test.py
.\.venv\Scripts\python.exe -m src.training.train --config configs/training/phase2_5.yaml --architecture convnext_tiny
.\.venv\Scripts\python.exe scripts\benchmark_candidates.py
.\.venv\Scripts\python.exe -m src.training.benchmark --config configs/training/phase2_5.yaml
```

Do not pass `--force-split` to resume these runs.

## Remaining benchmark candidate

Formal selection remains blocked by `selection.require_all_candidates: true`. ConvNeXt-Base is the only required candidate not started. The next training command is:

```powershell
.\.venv\Scripts\python.exe -m src.training.train --config configs/training/phase2_5.yaml --architecture convnext_base
```

This command was documented but not run during the ConvNeXt-Tiny task.
