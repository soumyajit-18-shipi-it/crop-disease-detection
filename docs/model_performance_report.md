# Model Performance Report

Training and evaluation have not been run in this environment because the PlantVillage dataset and trained checkpoint are not present.

After running:

```bash
python -m src.training.train --config configs/base.yaml
python -m src.evaluation.evaluate --checkpoint models/checkpoints/best_model.pth
```

this file will be replaced with real accuracy, macro F1, per-class metrics, confusion matrix location, and top confused pairs.
