"""API settings and model IDs in one place.

Everything can be overridden from the environment or the .env file.
"""

from __future__ import annotations

import os

try:  # Load MODEL_API_KEY (and any overrides) from ./.env if python-dotenv is installed.
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - optional dependency
    pass

API_BASE_URL = os.environ.get("META_API_BASE_URL", "https://api.meta.ai/v1")
SAM_MODEL = os.environ.get("SAM_MODEL", "sam-3.1")
SPARK_MODEL = os.environ.get("SPARK_MODEL", "muse-spark-1.3")
WONDER_MODEL = os.environ.get("WONDER_MODEL", "muse-wonder-1.0")

# Standard gravity in cm/s^2, used as the reference value in fits.
G_STANDARD_CM = 980.665
