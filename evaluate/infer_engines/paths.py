"""Portable paths shared by model inference engines."""

import os
import re
from pathlib import Path


PUBLIC_CODE_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = Path(
    os.environ.get("MVEI_MODEL_ROOT", PUBLIC_CODE_ROOT / "models")
).expanduser().resolve()


def _env_suffix(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").upper()


def model_path(name: str) -> str:
    """Return a model location, allowing a per-model environment override."""
    env_name = f"MVEI_MODEL_{_env_suffix(name)}"
    return os.environ.get(env_name, str(MODEL_ROOT / name))


def require_env(*names: str) -> None:
    """Raise a clear error when required runtime configuration is missing."""
    missing = [name for name in names if not os.environ.get(name)]
    if missing:
        raise RuntimeError(
            "Missing required environment variables: " + ", ".join(missing)
        )
