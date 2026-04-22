"""Plotting and GEE→raster classification orchestration."""
from __future__ import annotations

import logging
import os
from random import randint

import earthpy.plot as ep
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.transform import from_origin

from app.config.settings import UPLOADED_FILES_DIR
from app.models.inference import run_classification_on_raster
from app.services.export_service import export_ee_image_to_local_geotiff
from app.services.gee_service import get_image_for_date_by_source, get_s2_image_for_year
from app.utils.raster_utils import get_tiff_center_lat_lon

logger = logging.getLogger(__name__)


def plot_classified_map(image,labels,image_shape,title,temp_number,original_image,save_dir="static",file_name=None,center_lat=None,center_lon=None,valid_mask: np.ndarray = None,):
    color_map = {"Water": [0.53, 0.81, 0.98],"BuiltUp": [1, 0, 0],"Vegetation": [0.13, 0.55, 0.13],"BarrenLand": [1, 0.65, 0],"Agricultural": [1, 1, 0],}
    fig, axs = plt.subplots(1, 3, figsize=(20, 7))
    title_parts = []
    if file_name:
        title_parts.append(f"File: {file_name}")
        title_parts.append(f"Location: {os.path.splitext(file_name)[0]}")
    if center_lat is not None and center_lon is not None:
        title_parts.append(f"Center: {center_lat:.6f}°N, {center_lon:.6f}°E")
    if title_parts:
        fig.suptitle("  |  ".join(title_parts), fontsize=12, y=1.02)

    image_vis_432 = np.stack([image.read(b) for b in [4, 3, 2]])
    ep.plot_rgb(image_vis_432, ax=axs[0], stretch=True)
    axs[0].set_title("RGB (Bands 4, 3, 2)")
    axs[0].axis("off")

    image_vis_843 = np.stack([image.read(b) for b in [8, 4, 3]])
    ep.plot_rgb(image_vis_843, ax=axs[1], stretch=True)
    axs[1].set_title("FCC (Bands 8, 4, 3)")
    axs[1].axis("off")

    land_cover = labels.reshape(image_shape[0], image_shape[1])
    rgb_array = np.array([[color_map.get(value, [0.0, 0.0, 0.0]) for value in row] for row in land_cover], dtype="float32")
    axs[2].imshow(rgb_array, aspect="equal")

    if valid_mask is not None:
        flat_vals = land_cover[valid_mask]
    else:
        flat_vals = land_cover.reshape(-1)
    unique, counts = np.unique(flat_vals, return_counts=True)
    total_pixels = flat_vals.size if flat_vals.size > 0 else 1
    percentages = {label: (count / total_pixels) * 100 for label, count in zip(unique, counts)}

    colors = [color_map[key] for key in color_map]
    labels_list = [f"{key}: {percentages.get(key, 0):.2f}%" for key in color_map]
    patches = [
        plt.plot([], [], marker="s", ms=10, ls="", mec=None, color=colors[i], label=labels_list[i])[0]
        for i in range(len(labels_list))
    ]

    if valid_mask is not None:
        vm = valid_mask.astype(bool)
        rgb_array[~vm] = [1.0, 1.0, 1.0]

    axs[2].legend(handles=patches, bbox_to_anchor=(1, 1), loc="upper left", frameon=False, title="Land Cover Classes")
    axs[2].set_title(title)
    axs[2].axis("off")

    classified_map_filename = f"classified_map_{temp_number}.png"
    new_classified_image_filename = f"new-classified_{temp_number}.png"
    classified_tiff_filename = f"new-classified_{temp_number}.tiff"

    classified_map_path = os.path.join(save_dir, classified_map_filename)
    new_classified_image_path = os.path.join(save_dir, new_classified_image_filename)
    classified_tiff_path = os.path.join(save_dir, classified_tiff_filename)

    plt.savefig(classified_map_path, bbox_inches="tight", pad_inches=0.1)
    plt.imsave(new_classified_image_path, rgb_array)

    height, width, _ = rgb_array.shape
    transform = from_origin(original_image.bounds.left, original_image.bounds.top, original_image.res[0], original_image.res[1])

    with rasterio.open(
        classified_tiff_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype=rgb_array.dtype,
        crs=original_image.crs,
        transform=transform,
    ) as dst:
        dst.write(rgb_array[:, :, 0], 1)
        dst.write(rgb_array[:, :, 1], 2)
        dst.write(rgb_array[:, :, 2], 3)

    plt.close(fig)

    logger.info(
        "Classified map saved to %s, new classified image saved as %s, and TIFF saved as %s",
        classified_map_path,
        new_classified_image_path,
        classified_tiff_path,
    )

    return classified_map_path, new_classified_image_path, classified_tiff_path


def classify_ee_image_for_year(geom, year: int, method: str, location_name: str, scale: int = 30):
    """
    Fetch yearly Sentinel-2 composite from GEE, export to local GeoTIFF, and classify.
    Returns dict with: labels, image_shape, raster_path, raster_handle, center_lat, center_lon, file_name.
    """
    ee_image, err = get_s2_image_for_year(geom, year)
    if err:
        return None, err
    temp_number = str(randint(1000, 9999))
    display_file_name = f"{location_name}_{year}.tif"
    local_tif_path = os.path.join(UPLOADED_FILES_DIR, f"gee_{location_name}_{year}_{temp_number}.tif")
    try:
        export_ee_image_to_local_geotiff(ee_image, geom, local_tif_path, scale=scale)
    except Exception as e:
        logger.exception("Yearly GEE export failed")
        return None, f"Failed to fetch/export Sentinel-2 for year {year}: {e}"
    try:
        image = rasterio.open(local_tif_path)
    except Exception as e:
        logger.exception("Could not open exported yearly GeoTIFF")
        return None, f"Could not open exported GeoTIFF for year {year}: {e}"
    try:
        labels, _image_dataset, image_shape = run_classification_on_raster(image, method)
    except Exception as e:
        logger.exception("Classification failed for yearly composite")
        return None, f"Classification failed for year {year}: {e}"
    center_lat, center_lon = get_tiff_center_lat_lon(image)
    return {
        "labels": labels,
        "image_shape": image_shape,
        "raster_path": local_tif_path,
        "raster_handle": image,
        "center_lat": center_lat,
        "center_lon": center_lon,
        "file_name": display_file_name,
    }, None


def classify_ee_image_for_date(geom, date_str: str, method: str, location_name: str, scale: int = 30):
    """
    Fetch image for a specific date (with nearby clear-day fallback) for the default source (Sentinel-2),
    export to local GeoTIFF, and classify.
    Returns dict with: labels, image_shape, raster_path, raster_handle, center_lat, center_lon, file_name.
    """
    ee_image, err = get_image_for_date_by_source("sentinel2", geom, date_str)
    if err:
        return None, err
    safe_date = date_str.replace(":", "-")
    temp_number = str(randint(1000, 9999))
    display_file_name = f"{location_name}_{date_str}.tif"
    local_tif_path = os.path.join(UPLOADED_FILES_DIR, f"gee_{location_name}_{safe_date}_{temp_number}.tif")
    try:
        export_ee_image_to_local_geotiff(ee_image, geom, local_tif_path, scale=scale)
    except Exception as e:
        logger.exception("Dated GEE export failed")
        return None, f"Failed to fetch/export Sentinel-2 for {date_str}: {e}"
    try:
        image = rasterio.open(local_tif_path)
    except Exception as e:
        logger.exception("Could not open exported dated GeoTIFF")
        return None, f"Could not open exported GeoTIFF for {date_str}: {e}"
    try:
        labels, _image_dataset, image_shape = run_classification_on_raster(image, method)
    except Exception as e:
        logger.exception("Classification failed for dated image")
        return None, f"Classification failed for {date_str}: {e}"
    center_lat, center_lon = get_tiff_center_lat_lon(image)
    return {
        "labels": labels,
        "image_shape": image_shape,
        "raster_path": local_tif_path,
        "raster_handle": image,
        "center_lat": center_lat,
        "center_lon": center_lon,
        "file_name": display_file_name,
    }, None
