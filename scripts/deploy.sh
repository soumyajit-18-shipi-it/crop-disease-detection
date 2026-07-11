#!/usr/bin/env bash
set -euo pipefail

echo "Backend: deploy the repository root with backend/Dockerfile to Railway or Render."
echo "Frontend: deploy frontend/ to Vercel with VITE_API_URL set to the backend URL."
echo "For Vercel CLI:"
echo "  cd frontend && vercel --prod"
echo "For Railway CLI:"
echo "  railway up"
