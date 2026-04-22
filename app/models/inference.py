"""Raster classification inference (same logic as original main.py)."""
from __future__ import annotations

import numpy as np

from app.models.model_loader import get_ann_bundle, get_sklearn_classifier
from app.utils.raster_utils import tiff_to_csv


def run_classification_with_proba(image, method: str):
    """
    Like run_classification_on_raster but also returns per-class probabilities
    (if the model supports it) and the original class names.

    Returns: (labels, image_dataset, image_shape, proba or None, class_names or None)
    """
    image_dataset = tiff_to_csv(image)
    image_shape = image.read(1).shape

    if method == "ANN":
        ANN_clf, ANN_scaler, label_encoder = get_ann_bundle()
        x_new1 = ANN_scaler.transform(image_dataset)
        y_proba = ANN_clf.predict(x_new1)
        predicted_classes = np.argmax(y_proba, axis=1)
        labels = label_encoder.inverse_transform(predicted_classes)
        return labels, image_dataset, image_shape, y_proba, list(label_encoder.classes_)

    clf = get_sklearn_classifier(method)

    if hasattr(clf, "predict_proba"):
        y_proba = clf.predict_proba(image_dataset)
        labels = clf.classes_[np.argmax(y_proba, axis=1)]
        return labels, image_dataset, image_shape, y_proba, list(clf.classes_)

    labels = clf.predict(image_dataset)
    return labels, image_dataset, image_shape, None, None


def run_classification_on_raster(image, method: str):
    """Run the selected classifier on an opened rasterio dataset. Returns (labels, image_dataset, image_shape)."""
    image_dataset = tiff_to_csv(image)
    image_shape = image.read(1).shape

    if method == "ANN":
        ANN_clf, ANN_scaler, label_encoder = get_ann_bundle()
        x_new1 = ANN_scaler.transform(image_dataset)
        y_pred1 = ANN_clf.predict(x_new1)
        predicted_classes = np.argmax(y_pred1, axis=1)
        labels = label_encoder.inverse_transform(predicted_classes)
        return labels, image_dataset, image_shape

    clf = get_sklearn_classifier(method)
    labels = clf.predict(image_dataset)
    return labels, image_dataset, image_shape
