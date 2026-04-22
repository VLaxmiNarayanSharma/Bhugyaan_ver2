"""Thread-safe cached loading of pickled classifiers."""
from __future__ import annotations

import os
import pickle
import threading
from typing import Any, Dict, Tuple

from app.config.settings import CLASSIFICATION_METHODS_DIR

_MODEL_CACHE: Dict[str, Any] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def get_ann_bundle() -> Tuple[Any, Any, Any]:
    """Return (ANN_clf, ANN_scaler, label_encoder) from cache or disk."""
    cache_key = "ANN"
    with _MODEL_CACHE_LOCK:
        cached = _MODEL_CACHE.get(cache_key)
        if cached is None:
            model_file = os.path.join(CLASSIFICATION_METHODS_DIR, "ANN_model-3.pkl")
            scaler_file = os.path.join(CLASSIFICATION_METHODS_DIR, "Scaler_ANN_model-3.pkl")
            label_encoder_file = os.path.join(CLASSIFICATION_METHODS_DIR, "label_encoder-3.pkl")
            if not (os.path.exists(model_file) and os.path.exists(scaler_file) and os.path.exists(label_encoder_file)):
                raise FileNotFoundError("Model, scaler, or label encoder file not found.")
            with open(model_file, "rb") as model_f:
                ANN_clf = pickle.load(model_f)
            with open(scaler_file, "rb") as scaler_f:
                ANN_scaler = pickle.load(scaler_f)
            with open(label_encoder_file, "rb") as label_f:
                label_encoder = pickle.load(label_f)
            cached = (ANN_clf, ANN_scaler, label_encoder)
            _MODEL_CACHE[cache_key] = cached
        return cached


def get_sklearn_classifier(method: str) -> Any:
    """Load sklearn (or other) .pkl model for the given method name."""
    cache_key = f"SK:{method}"
    with _MODEL_CACHE_LOCK:
        cached = _MODEL_CACHE.get(cache_key)
        if cached is None:
            model_file = os.path.join(CLASSIFICATION_METHODS_DIR, f"{method}.pkl")
            if not os.path.exists(model_file):
                raise FileNotFoundError(f"Model file not found: {model_file}")
            with open(model_file, "rb") as mdl:
                clf = pickle.load(mdl)
            cached = clf
            _MODEL_CACHE[cache_key] = cached
        return cached
