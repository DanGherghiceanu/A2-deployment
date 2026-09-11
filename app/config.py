"""Runtime configuration, read from environment variables.

Every value has a sensible default so the container runs with no configuration at
all, but each one can be overridden at deploy time (Cloud Run --set-env-vars,
`docker run -e ...`, or a local .env file).
"""

from __future__ import annotations

import os
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = APP_ROOT.parent


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value or default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# --- Model -----------------------------------------------------------------
# Path to the state_dict saved in A1 (torch.save(cnn_model.state_dict(), ...)).
MODEL_PATH = Path(_env_str("MODEL_PATH", str(PROJECT_ROOT / "models" / "resnet18_finetuned.pt")))

# A1 concluded that the val-optimal threshold (0.52) gave results identical to
# the default on the test set, so 0.50 ships as the operating point. Exposed as
# an env var so the threshold can be moved at deploy time without a rebuild.
DECISION_THRESHOLD = _env_float("DECISION_THRESHOLD", 0.5)

MODEL_VERSION = _env_str("MODEL_VERSION", "resnet18-finetuned-a1")

# Torch intra-op threads. Cloud Run containers are usually 1-2 vCPU; letting
# torch spawn a thread per detected core causes contention, not speed.
TORCH_NUM_THREADS = _env_int("TORCH_NUM_THREADS", 1)

# --- Server ----------------------------------------------------------------
# Cloud Run injects PORT and expects the container to listen on it.
PORT = _env_int("PORT", 8080)
LOG_LEVEL = _env_str("LOG_LEVEL", "INFO").upper()

# Comma-separated list, or "*" for any origin. The web UI is served from the
# same origin as the API, so this only matters if the page is hosted elsewhere.
ALLOWED_ORIGINS = [o.strip() for o in _env_str("ALLOWED_ORIGINS", "*").split(",") if o.strip()]

# --- Image intake ----------------------------------------------------------
MAX_IMAGE_BYTES = _env_int("MAX_IMAGE_BYTES", 10 * 1024 * 1024)  # 10 MB
IMAGE_FETCH_TIMEOUT = _env_float("IMAGE_FETCH_TIMEOUT", 10.0)  # seconds

# Refuse URLs pointing at private / internal addresses. Keep this on for any
# publicly reachable deployment; turning it off makes the endpoint a proxy into
# whatever network the container sits in.
BLOCK_PRIVATE_ADDRESSES = _env_str("BLOCK_PRIVATE_ADDRESSES", "true").lower() != "false"
