#!/usr/bin/env bash
set -euo pipefail

echo "Backend: set LEAFLIGHT_MODEL_URL, run python scripts/download_model.py, then build backend/Dockerfile from the repository root."
echo "The Docker build verifies the local release again and never receives the model URL."
echo "Frontend: deploy frontend/ to Vercel with VITE_API_URL set to the backend URL."
echo "For Vercel CLI:"
echo "  cd frontend && vercel --prod"
echo "For Railway CLI:"
echo "  railway up"
