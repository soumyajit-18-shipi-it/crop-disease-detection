#!/usr/bin/env bash
set -euo pipefail

python -m src.training.benchmark --config configs/training/phase2_5.yaml --train
