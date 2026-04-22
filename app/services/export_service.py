"""Earth Engine export to disk, label rendering, and static URL helpers."""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict

import ee
import geemap
import matplotlib.pyplot as plt
import numpy as np
import rasterio

from app.config.settings import (
    CANONICAL_CLASSES,
    CLASS_TO_CODE,
    CLASS_TO_RGB,
    CLASS_TO_STAT_KEY,
    OUTPUT_CACHE_DIR,
    STATIC_DIR,
)
from app.models.inference import run_classification_on_raster
from app.utils.cache_keys import safe_filename
from app.utils.file_utils import load_stats_json
from app.utils.geo_utils import normalize_class_label

logger = logging.getLogger(__name__)


def export_ee_image_to_local_geotiff(ee_image, geom, local_tif_path: str, scale: int = 30):
    """Export an Earth Engine image to a local GeoTIFF file path via geemap."""
    os.makedirs(os.path.dirname(local_tif_path), exist_ok=True)
    geemap.ee_export_image(
        ee_image,
        filename=local_tif_path,
        scale=int(scale) if scale else 30,
        region=geom,
        file_per_band=False,
    )
    if not os.path.exists(local_tif_path):
        raise RuntimeError(
            "Failed to export image from Google Earth Engine. "
            "The AOI may be too large for the requested resolution. "
            "Try drawing a smaller AOI or increasing the scale (e.g. 60–120 m)."
        )


def _labels_flat_to_code_array(labels_flat: np.ndarray) -> np.ndarray:
    labels_flat = np.asarray(labels_flat)
    code = np.zeros(labels_flat.shape[0], dtype=np.uint8)
    unique_labels = np.unique(labels_flat)
    for ul in unique_labels.tolist():
        cname = normalize_class_label(ul)
        code_val = CLASS_TO_CODE.get(cname, 0)
        code[labels_flat == ul] = code_val
    return code


def _compute_area_stats_from_codes(code_arr: np.ndarray, pixel_area_m2: float) -> Dict[str, float]:
    pixel_area_km2 = float(pixel_area_m2) / 1e6
    out: Dict[str, float] = {}
    for cname in CANONICAL_CLASSES:
        cc = CLASS_TO_CODE[cname]
        cnt = int(np.sum(code_arr == cc))
        out[CLASS_TO_STAT_KEY[cname]] = float(cnt * pixel_area_km2)
    return out


def render_codes_to_png_and_tif(
    *,
    code_arr: np.ndarray,
    raster_handle,
    png_path: str,
    label_tif_path: str,
):
    """Render code array to a PNG and store a 1-band label GeoTIFF."""
    h, w = code_arr.shape
    lut = np.zeros((len(CANONICAL_CLASSES) + 1, 3), dtype=np.uint8)
    lut[0] = np.array([255, 255, 255], dtype=np.uint8)
    for cname in CANONICAL_CLASSES:
        lut[CLASS_TO_CODE[cname]] = CLASS_TO_RGB[cname]

    rgb = lut[code_arr]
    plt.imsave(png_path, rgb)

    with rasterio.open(
        label_tif_path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=1,
        dtype=code_arr.dtype,
        crs=raster_handle.crs,
        transform=raster_handle.transform,
    ) as dst:
        dst.write(code_arr, 1)


def export_classify_render_codes(
    *,
    ee_image: ee.Image,
    geom: ee.Geometry,
    method: str,
    location_key: str,
    step_label: str,
    scale_m: int,
    out_png_path: str,
    out_label_tif_path: str,
    out_stats_json_path: str,
) -> Dict[str, Any]:
    """
    Export EE composite -> local GeoTIFF -> classify -> render + stats.
    Returns { "stats": {...}, "png": ..., "tif": ... }.
    """
    step_safe = safe_filename(step_label)
    export_tif_path = os.path.join(OUTPUT_CACHE_DIR, f"gee_export_{location_key}_{step_safe}_{scale_m}.tif")

    if not os.path.exists(export_tif_path):
        try:
            export_ee_image_to_local_geotiff(ee_image, geom, export_tif_path, scale=scale_m)
        except RuntimeError as e:
            fallback_scale = min(scale_m * 2, 120)
            fallback_path = os.path.join(OUTPUT_CACHE_DIR, f"gee_export_{location_key}_{step_safe}_{fallback_scale}.tif")
            if fallback_scale > scale_m:
                try:
                    export_ee_image_to_local_geotiff(ee_image, geom, fallback_path, scale=fallback_scale)
                    export_tif_path = fallback_path
                except Exception:
                    raise RuntimeError(
                        f"Failed for step '{step_label}': {e}. Try a smaller AOI or scale ≥ {fallback_scale}m."
                    ) from e
            else:
                raise RuntimeError(f"Failed for step '{step_label}': {e}") from e

    with rasterio.open(export_tif_path) as src:
        labels, _image_dataset, image_shape = run_classification_on_raster(src, method)
        h, w = image_shape
        band1 = src.read(1, masked=True)
        valid_mask = ~band1.mask

        code_flat = _labels_flat_to_code_array(np.asarray(labels))
        code_2d = code_flat.reshape(h, w)
        code_2d = np.where(valid_mask, code_2d, 0).astype(np.uint8)

        pixel_area_m2 = float(abs(src.res[0] * src.res[1]))
        stats_area_km2 = _compute_area_stats_from_codes(code_2d, pixel_area_m2=pixel_area_m2)

        render_codes_to_png_and_tif(
            code_arr=code_2d,
            raster_handle=src,
            png_path=out_png_path,
            label_tif_path=out_label_tif_path,
        )

    payload = {"stats": stats_area_km2}
    with open(out_stats_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return {"stats": stats_area_km2}


def static_url_from_abs_path(abs_path: str) -> str:
    rel = os.path.relpath(abs_path, STATIC_DIR).replace("\\", "/")
    return "/static/" + rel


def load_label_codes_from_tif(label_tif_path: str) -> Dict[str, Any]:
    """Load 1-band uint8 code GeoTIFF as a 2D numpy array + pixel area in m²."""
    with rasterio.open(label_tif_path) as src:
        code_arr = src.read(1)
        pixel_area_m2 = float(abs(src.res[0] * src.res[1]))
    return {"code_arr": code_arr.astype(np.uint8), "pixel_area_m2": pixel_area_m2}


def ensure_classification_for_step(
    *,
    ee_image: ee.Image,
    geom: ee.Geometry,
    method: str,
    location_key: str,
    step_label: str,
    scale_m: int,
    out_png_path: str,
    out_label_tif_path: str,
    out_stats_json_path: str,
) -> Dict[str, Any]:
    """
    Ensure classification outputs for a single step exist.
    Returns:
      { "stats": {...}, "png_url": "...", "label_tif_url": "...", "code_arr": 2D uint8, "pixel_area_m2": float }
    """
    if os.path.exists(out_label_tif_path) and os.path.exists(out_stats_json_path) and os.path.exists(out_png_path):
        stats_payload = load_stats_json(out_stats_json_path)
        loaded = load_label_codes_from_tif(out_label_tif_path)
        return {
            "stats": stats_payload.get("stats", {}),
            "png_url": static_url_from_abs_path(out_png_path),
            "label_tif_url": static_url_from_abs_path(out_label_tif_path),
            "code_arr": loaded["code_arr"],
            "pixel_area_m2": loaded["pixel_area_m2"],
        }

    export_classify_render_codes(
        ee_image=ee_image,
        geom=geom,
        method=method,
        location_key=location_key,
        step_label=step_label,
        scale_m=scale_m,
        out_png_path=out_png_path,
        out_label_tif_path=out_label_tif_path,
        out_stats_json_path=out_stats_json_path,
    )
    stats_payload = load_stats_json(out_stats_json_path)
    loaded = load_label_codes_from_tif(out_label_tif_path)
    return {
        "stats": stats_payload.get("stats", {}),
        "png_url": static_url_from_abs_path(out_png_path),
        "label_tif_url": static_url_from_abs_path(out_label_tif_path),
        "code_arr": loaded["code_arr"],
        "pixel_area_m2": loaded["pixel_area_m2"],
    }
