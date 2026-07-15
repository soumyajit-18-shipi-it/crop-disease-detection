# Phase 2.5 Training Results

> This file reports only artifacts currently present on disk. A blank table cell is never interpreted as a benchmark result.

- Platform: `Windows-11-10.0.26200-SP0`
- PyTorch: `2.12.1+cu130`; CUDA available: `true`
- GPU: NVIDIA GeForce RTX 3050 6GB Laptop GPU, 6.0 GB VRAM.
- Persisted split: `data/splits/phase1_split.json`
- Production selection: pending

## Pipeline Audit and Implemented Policy

- AdamW excludes bias and normalization parameters from weight decay; per-step cosine decay includes linear warmup and a non-zero minimum LR.
- Full pretrained fine-tuning uses each timm backbone's native resolution, interpolation, mean, std, and evaluation crop contract.
- EMA, CUDA AMP, gradient accumulation, overflow-aware scheduler stepping, clipping, label smoothing, effective-number class weighting, and optional balanced sampling are configurable.
- Field augmentations cover restrained geometry, illumination/color, CLAHE, blur/defocus, shadow/light fog, JPEG degradation, and partial occlusion. Artificial rain is supported but disabled pending field evidence.
- MixUp and CutMix are batch-level and mutually sampled. Weighted loss is enabled; balanced sampling is disabled in the preset to avoid double-correcting class imbalance.
- Deterministic RNG and DataLoader generator states are checkpointed. `last.pt` is an atomic resume checkpoint; `best.pt` is a compact EMA inference checkpoint selected by validation macro F1.
- Temperature scaling is fit on validation logits. The test split is evaluated only after model selection within each run, and raw/calibrated ECE are both retained.

## Candidate Status

### efficientnetv2_s

- Status: not started

### convnext_tiny

- Status: not started

### convnext_base

- Status: not started

## Run or Resume Exactly

All candidates resume `last.pt` by default and reuse the existing split:

```powershell
.\.venv\Scripts\python.exe -m src.training.benchmark --config configs/training/phase2_5.yaml --train
```

Resume one candidate:

```powershell
.\.venv\Scripts\python.exe -m src.training.train --config configs/training/phase2_5.yaml --architecture convnext_base
```

Do not pass `--force-split` for these runs. Resume validation rejects a changed split hash, architecture, preprocessing contract, or optimization signature.
