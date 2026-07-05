#!/usr/bin/env bash
set -euo pipefail

exec uvicorn src.deployment.api:app --host 0.0.0.0 --port "${PORT:?PORT must be set by Render}"
