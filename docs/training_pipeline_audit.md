# Phase 2.5 Training Pipeline Audit

This audit distinguishes implemented engineering changes from hypotheses that still require measured ablations. No accuracy value is inferred from a configuration change.

## Findings and Decisions

| Area | Audit finding | Phase 2.5 decision |
|---|---|---|
| Optimizer | AdamW existed, but weight decay was also applied to bias and normalization parameters. | Keep AdamW; use timm parameter grouping to exclude bias/norm from decay. Make betas, epsilon, decay, LR, and optional layer-wise decay configurable. |
| Scheduler | Per-step cosine warmup existed. AMP overflow could skip an optimizer update while the scheduler still advanced, causing the observed scheduler warning and state drift. | Retain cosine warmup, add configurable warmup start/minimum LR, and advance scheduler/EMA only after a real optimizer update. |
| Transfer learning | All backbones were fully fine-tuned from pretrained weights, which is appropriate for the available dataset size. Model dropout configuration was not passed to timm. | Retain full fine-tuning and apply configured head dropout/drop-path. ConvNeXt presets use configurable layer-wise LR decay. |
| Normalization | A single ImageNet mean/std was hard-coded. The installed timm EfficientNetV2-S pretrained contract uses 300 px, bicubic interpolation, mean/std 0.5/0.5; ConvNeXt uses a different contract. | Resolve and persist the selected backbone's timm preprocessing metadata. Use the identical contract in train/eval, PyTorch prediction, ONNX prediction, and the FastAPI service. |
| Preprocessing | Evaluation stretched every input to a square, while pretrained models expect aspect-preserving resize and center crop. Phone EXIF orientation was not explicitly normalized. | Native pretrained evaluation resize/crop and EXIF orientation correction are used for Phase 2.5; legacy metadata still retains stretch behavior for backward compatibility. |
| Augmentation | The previous policy only cropped, flipped, rotated, and changed brightness/contrast. It did not represent field blur, shadows, camera compression, perspective, or occlusion. | Add restrained grouped Albumentations policies for geometry, color/illumination, CLAHE, motion/Gaussian/defocus blur, shadow/light fog, JPEG compression, and coarse dropout. MixUp/CutMix are batch-level. |
| Weather effects | Fog/rain can represent field capture conditions, but aggressive overlays can cover the small color/texture cues that define disease. | Use light fog with low probability. Support artificial drizzle but set its production weight to zero until a field-set ablation shows benefit. |
| DataLoader | Shuffle was seeded, but worker NumPy/Python RNG state and resume sample-order state were not explicitly controlled. Class-balanced sampling was absent. | Seed every worker, disable persistent workers in deterministic mode, checkpoint the train generator, and support a source-aware class-balanced sampler. It is disabled while weighted loss is enabled to avoid double correction. |
| Loss | Weighted, label-smoothed cross entropy existed with inverse-frequency weights. The persisted training split ranges from 106 to 2,246 samples per class (21.2:1), so macro-sensitive imbalance handling is justified, but raw inverse weighting can be extreme. | Default to effective-number class weights plus label smoothing. Keep inverse frequency configurable. Do not add focal loss without an ablation because it often worsens calibration and can over-focus label noise. |
| EMA | EMA existed but duplicated the model in every best/resume payload and had no early-step bias handling. | Keep configurable EMA, use a bias-reducing warm start, persist update count, and store only the state needed for each checkpoint type. |
| AMP/accumulation | AMP and accumulation existed. The final partial accumulation group was divided by the full accumulation count. | Scale by the actual final group size and prevent scheduler/EMA updates after GradScaler overflow. |
| Metrics | Most requested aggregate metrics existed. Per-class ROC-AUC, JSON classification reports, calibrated metrics, and reliability diagrams were absent. | Add all requested aggregate/per-class metrics, raw and calibrated ECE/NLL/Brier, temperature scaling, and reliability diagrams. |
| Checkpointing | `best.pt` duplicated training model, EMA model, and optimizer state. Writes were non-atomic; a disk error produced a corrupt partial save. Best model selection used validation loss, not the production accuracy objective. | Write atomic compact `best.pt` and resumable `last.pt`; persist RNG/sampler state; validate architecture/split/preprocessing/optimization signatures; monitor validation macro F1. |
| ONNX | Export checked one random batch and reported CPU median latency. | Export atomically, validate dynamic batches, compare logits and calibrated probabilities, and report median/mean/p90 CPU latency. |
| Production selection | The old score used 50/20/20/10 and omitted memory. | Implement the required 40/20/15/15/10 score. Require complete parity-verified candidates on one split hash. |

## Intentionally Not Added

- Focal loss, heavy color distortion, aggressive fog/rain, and simultaneous balanced sampling plus weighted loss are not production defaults without an ablation on validated field images.
- Test-time augmentation is not enabled because it increases serving latency and must earn that cost in a deployment-matched benchmark.
- Ensemble inference is not introduced in this phase because the requested production score and bundle describe a single deployable model.
- Dataset expansion, segmentation, severity, explainability, active learning, and continual learning remain roadmap work and are not mixed into this training-pipeline phase.
