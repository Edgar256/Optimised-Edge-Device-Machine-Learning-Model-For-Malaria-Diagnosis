#!/usr/bin/env bash
set -euo pipefail

pip install -r requirements.txt

cd frontend
if [ -f package-lock.json ]; then
  npm ci
else
  npm install
fi
npm run build
