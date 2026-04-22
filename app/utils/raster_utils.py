"""Raster I/O and per-pixel statistics helpers."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform as transform_coords

from app.utils.geo_utils import normalize_class_label

logger = logging.getLogger(__name__)


def tiff_to_csv(image):
    bands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
    data = [image.read(i + 1).flatten() for i in range(image.count)]
    dataset = pd.DataFrame(np.array(data).T, columns=bands)
    dataset.fillna(0.0001, inplace=True)
    return dataset


def compute_class_stats_from_labels(labels: np.ndarray, image_shape, pixel_area_m2: float, valid_mask: np.ndarray = None):
    """
    Compute per-class area and percentage statistics.

    labels: 1D array of predicted class labels (strings / encodable)
    image_shape: (height, width)
    pixel_area_m2: area of one pixel in m^2
    """
    total_pixels = int(labels.size)
    if total_pixels == 0:
        return []
    if valid_mask is not None:
        vm_flat = valid_mask.reshape(-1)
    else:
        vm_flat = np.ones_like(labels, dtype=bool)
    pixel_area_km2 = pixel_area_m2 / 1e6
    counts = {}
    for v, is_valid in zip(labels.tolist(), vm_flat.tolist()):
        if not is_valid:
            continue
        cname = normalize_class_label(v)
        counts[cname] = counts.get(cname, 0) + 1
    stats = []
    for cname, count in counts.items():
        area_km2 = count * pixel_area_km2
        percent = (count * 100.0) / int(vm_flat.sum())
        stats.append(
            {
                "class": cname,
                "pixels": int(count),
                "area_km2": area_km2,
                "percent": percent,
            }
        )
    stats.sort(key=lambda x: -x["area_km2"])
    return stats


def get_tiff_center_lat_lon(raster_handle):
    """Compute center latitude and longitude of a raster in WGS84. Returns (lat, lon) or (None, None) on error."""
    try:
        b = raster_handle.bounds
        cx = (b.left + b.right) / 2
        cy = (b.bottom + b.top) / 2
        wgs84 = CRS.from_epsg(4326)
        lons, lats = transform_coords(raster_handle.crs, wgs84, [cx], [cy])
        return round(lats[0], 6), round(lons[0], 6)
    except Exception as e:
        logger.warning("Could not compute TIFF center lat/lon: %s", e)
        return None, None
