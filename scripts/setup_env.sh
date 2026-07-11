#!/usr/bin/env bash
set -euo pipefail

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r backend/requirements.txt
python backend/db/seed_disease_data.py
