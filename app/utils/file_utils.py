"""Classification history and small JSON helpers."""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict

from app.config.settings import HISTORY_FILE

logger = logging.getLogger(__name__)


def load_classification_history():
    """Load list of past classifications from disk."""
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Could not load classification history: %s", e)
        return []


def append_classification_history(
    file_name, method, image_url, new_classified_url, classified_tiff_url, center_lat=None, center_lon=None
):
    """Append one classification record to history and save to disk."""
    history = load_classification_history()
    record = {
        "file_name": file_name,
        "location": os.path.splitext(file_name)[0] if file_name else "",
        "method": method,
        "image_url": image_url,
        "new_classified_url": new_classified_url,
        "classified_tiff_url": classified_tiff_url,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    if center_lat is not None and center_lon is not None:
        record["center_lat"] = center_lat
        record["center_lon"] = center_lon
    history.append(record)
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        logger.warning("Could not save classification history: %s", e)


def load_stats_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
