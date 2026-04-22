"""Google Earth Engine initialization (lazy, thread-safe)."""
from __future__ import annotations

import logging
import threading
from typing import Any, Optional

import ee
from fastapi import HTTPException

from app.config.settings import DEFAULT_EE_PROJECT

logger = logging.getLogger(__name__)

_EE_INIT_LOCK = threading.Lock()
_EE_ACTIVE_PROJECT: str = DEFAULT_EE_PROJECT
_EE_INITIALIZED: bool = False


def initialize_ee(project: Optional[str] = None, credentials: Any = None) -> None:
    """Initialize Google Earth Engine once per process (project/credentials may be updated)."""
    global _EE_ACTIVE_PROJECT, _EE_INITIALIZED
    with _EE_INIT_LOCK:
        _EE_ACTIVE_PROJECT = (project or _EE_ACTIVE_PROJECT or DEFAULT_EE_PROJECT).strip()
        if not _EE_INITIALIZED:
            if credentials is not None:
                ee.Initialize(project=_EE_ACTIVE_PROJECT, credentials=credentials)
            else:
                ee.Initialize(project=_EE_ACTIVE_PROJECT)
            _EE_INITIALIZED = True
        else:
            if credentials is not None:
                try:
                    ee.Initialize(project=_EE_ACTIVE_PROJECT, credentials=credentials)
                except Exception:
                    logger.warning("EE re-initialization with new credentials failed; keeping existing session.")
            else:
                try:
                    ee.Initialize(project=_EE_ACTIVE_PROJECT)
                except Exception:
                    logger.warning("EE re-initialization with new project failed; keeping existing session.")


def ensure_ee_initialized() -> None:
    """Initialize Earth Engine lazily (prevents uvicorn import-time crashes)."""
    global _EE_INITIALIZED
    if _EE_INITIALIZED:
        return
    try:
        initialize_ee(DEFAULT_EE_PROJECT)
    except Exception as e:
        raise HTTPException(
            status_code=401,
            detail=(
                "Earth Engine is not initialized. Please authenticate first by running "
                "`earthengine authenticate` in your environment, or use `/gee-connect` "
                "with a service account JSON. Original error: " + str(e)
            ),
        )
