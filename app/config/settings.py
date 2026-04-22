"""Environment variables, filesystem paths, and shared constants."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import numpy as np

# Project root (parent of `app/`)
CONFIG_DIR = Path(__file__).resolve().parent
APP_DIR = CONFIG_DIR.parent
BASE_DIR = APP_DIR.parent

STATIC_DIR = str(BASE_DIR / "static")
TEMPLATES_DIR = str(BASE_DIR / "templates")
UPLOADED_FILES_DIR = str(BASE_DIR / "uploaded_files")
HISTORY_FILE = str(BASE_DIR / "classification_history.json")
CLASSIFICATION_METHODS_DIR = str(BASE_DIR / "classification_methods")

OUTPUT_PREVIEW_DIR = os.path.join(STATIC_DIR, "generated")
OUTPUT_TIMESERIES_DIR = os.path.join(OUTPUT_PREVIEW_DIR, "lulc_timeseries")
OUTPUT_EVENT_DIR = os.path.join(OUTPUT_PREVIEW_DIR, "lulc_event_change")
OUTPUT_CACHE_DIR = str(BASE_DIR / "cache_lulc")
DOWNLOADS_STATIC_DIR = str(BASE_DIR / "static" / "downloads")

for _d in (
    OUTPUT_PREVIEW_DIR,
    OUTPUT_TIMESERIES_DIR,
    OUTPUT_EVENT_DIR,
    OUTPUT_CACHE_DIR,
    UPLOADED_FILES_DIR,
    DOWNLOADS_STATIC_DIR,
):
    os.makedirs(_d, exist_ok=True)

DEFAULT_EE_PROJECT = os.environ.get("EE_PROJECT_ID", "ee-laxminarayan090503")
GCS_BUCKET = os.environ.get("GCS_BUCKET", "").strip()

# Append Google Cloud SDK bin directory to PATH (original project behavior)
os.environ["PATH"] += os.pathsep + r"C:\Users\sharm\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin"

# Sentinel-2 Harmonized starts 2022-03-29. For earlier dates use standard S2_SR (from 2015-06).
S2_HARMONIZED_START = "2022-03-29"
S2_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
NEARBY_DAYS_WINDOW = 30
YEARLY_MAX_CLOUD = 30  # for annual composites

# Canonical class order/codes used for fast rendering and change transitions.
CANONICAL_CLASSES: List[str] = ["Water", "BuiltUp", "Vegetation", "BarrenLand", "Agricultural"]
CLASS_TO_CODE: Dict[str, int] = {c: i + 1 for i, c in enumerate(CANONICAL_CLASSES)}
CODE_TO_CLASS: Dict[int, str] = {v: k for k, v in CLASS_TO_CODE.items()}

CLASS_TO_STAT_KEY: Dict[str, str] = {
    "BuiltUp": "urban",
    "Vegetation": "vegetation",
    "Water": "water",
    "BarrenLand": "barren",
    "Agricultural": "agricultural",
}

CLASS_TO_RGB: Dict[str, np.ndarray] = {
    "Water": np.array([33, 150, 243], dtype=np.uint8),
    "BuiltUp": np.array([244, 67, 54], dtype=np.uint8),
    "Vegetation": np.array([46, 125, 50], dtype=np.uint8),
    "BarrenLand": np.array([141, 110, 99], dtype=np.uint8),
    "Agricultural": np.array([255, 235, 59], dtype=np.uint8),
}
