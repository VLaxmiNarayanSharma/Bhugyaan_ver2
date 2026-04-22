"""Change detection pages and APIs."""
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from random import randint

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

from app.api.schemas import LulcEventChangeRequest
from app.config.settings import CLASS_TO_CODE, OUTPUT_EVENT_DIR, STATIC_DIR, UPLOADED_FILES_DIR
from app.dependencies.gee_init import ensure_ee_initialized
from app.models.inference import run_classification_on_raster
from app.services.change_detection_service import (
    apply_pixel_area_to_change_stats,
    compute_change_stats_from_codes,
    parse_iso_date,
)
from app.services.classification_service import plot_classified_map
from app.services.export_service import ensure_classification_for_step, export_ee_image_to_local_geotiff, static_url_from_abs_path
from app.services.gee_service import get_image_for_date_by_source, get_s2_masked_composite_for_range
from app.services.timeseries_service import resolve_method_default
from app.templating import templates
from app.utils.cache_keys import hash_key, stable_json_dumps
from app.utils.file_utils import load_stats_json
from app.utils.geo_utils import build_geometry, normalize_class_label
from app.utils.raster_utils import get_tiff_center_lat_lon

router = APIRouter(tags=["change"])
logger = logging.getLogger(__name__)


@router.get("/change-detection", response_class=HTMLResponse)
async def change_detection_page(request: Request):
    """Dedicated page for LULC change detection between two years."""
    return templates.TemplateResponse("change_detection.html", {"request": request})


@router.post("/lulc-change", response_class=HTMLResponse)
async def lulc_change(
    request: Request,
    geojson: str = Form(...),
    location_name: str = Form(...),
    date1: str = Form(..., description="First date (YYYY-MM-DD)"),
    date2: str = Form(..., description="Second date (YYYY-MM-DD)"),
    method: str = Form(...),
    scale: int = Form(30),
    source: str = Form("sentinel2"),
):
    """
    Change detection between two dates:
    - Fetch Sentinel-2 images (with nearby clear-day fallback)
    - Classify both dates
    - Produce change_map.tif/png and change_stats.json
    """
    if not location_name or not str(location_name).strip():
        return templates.TemplateResponse("change_detection.html", {"request": request, "error": "Location name is required."})
    location_name = str(location_name).strip()

    ensure_ee_initialized()

    try:
        geojson_obj = json.loads(geojson)
    except Exception:
        return templates.TemplateResponse("change_detection.html", {"request": request, "error": "Invalid GeoJSON provided for AOI."})
    try:
        geom = build_geometry(geojson_obj)
    except Exception as e:
        return templates.TemplateResponse("change_detection.html", {"request": request, "error": f"Error processing AOI: {e}"})

    d1 = str(date1).strip()
    d2 = str(date2).strip()
    if d1 == d2:
        return templates.TemplateResponse("change_detection.html", {"request": request, "error": "Date 1 and Date 2 must be different."})

    ee1, err1 = get_image_for_date_by_source(source, geom, d1)
    if err1:
        return templates.TemplateResponse("change_detection.html", {"request": request, "error": err1})
    ee2, err2 = get_image_for_date_by_source(source, geom, d2)
    if err2:
        return templates.TemplateResponse("change_detection.html", {"request": request, "error": err2})

    safe_d1 = d1.replace(":", "-")
    safe_d2 = d2.replace(":", "-")
    tmp1 = os.path.join(UPLOADED_FILES_DIR, f"change_{location_name}_{safe_d1}_{randint(1000,9999)}.tif")
    tmp2 = os.path.join(UPLOADED_FILES_DIR, f"change_{location_name}_{safe_d2}_{randint(1000,9999)}.tif")
    try:
        export_ee_image_to_local_geotiff(ee1, geom, tmp1, scale=scale)
        export_ee_image_to_local_geotiff(ee2, geom, tmp2, scale=scale)
    except Exception as e:
        logger.exception("Change detection export failed")
        return templates.TemplateResponse("change_detection.html", {"request": request, "error": f"Failed to export images: {e}"})

    try:
        img1 = rasterio.open(tmp1)
        img2 = rasterio.open(tmp2)
    except Exception as e:
        logger.exception("Change detection open failed")
        return templates.TemplateResponse("change_detection.html", {"request": request, "error": f"Failed to open exported images: {e}"})

    try:
        labels1, _ds1, shape1 = run_classification_on_raster(img1, method)
        labels2, _ds2, shape2 = run_classification_on_raster(img2, method)
    except Exception as e:
        logger.exception("Change classification failed")
        return templates.TemplateResponse("change_detection.html", {"request": request, "error": f"Classification failed: {e}"})

    labels1 = np.array(labels1)
    labels2 = np.array(labels2)
    if labels1.shape != labels2.shape:
        return templates.TemplateResponse(
            "change_detection.html",
            {"request": request, "error": "Internal error: classified rasters have different shapes. Try a smaller AOI or same scale."},
        )

    total = labels1.size
    labels1_norm = np.vectorize(normalize_class_label)(labels1)
    labels2_norm = np.vectorize(normalize_class_label)(labels2)

    hidden_mask = (labels1_norm == "BuiltUp") & (labels2_norm != "BuiltUp")
    same_mask = labels1_norm == labels2_norm
    display_no_change_mask = same_mask | hidden_mask

    transitions = {}
    for frm, to in zip(labels1_norm.tolist(), labels2_norm.tolist()):
        if frm == to:
            continue
        if frm == "BuiltUp" and to != "BuiltUp":
            continue
        key = (frm, to)
        transitions[key] = transitions.get(key, 0) + 1

    sorted_keys = sorted(transitions.keys(), key=lambda k: (-transitions[k], k[0], k[1]))
    trans_to_code = {k: i + 1 for i, k in enumerate(sorted_keys)}

    change_codes = np.zeros_like(labels1, dtype=np.uint16)
    for k, code in trans_to_code.items():
        frm, to = k
        mask = (labels1_norm == frm) & (labels2_norm == to)
        change_codes[mask] = code

    src = img1
    pixel_area_m2 = float(abs(src.res[0] * src.res[1]))
    pixel_area_km2 = pixel_area_m2 / 1e6

    urban_pixels = sum(count for (frm, to), count in transitions.items() if to == "BuiltUp" and frm != "BuiltUp")
    veg_loss_pixels = sum(count for (frm, to), count in transitions.items() if frm == "Vegetation" and to != "Vegetation")

    change_stats = {
        "location": location_name,
        "method": method,
        "date1": d1,
        "date2": d2,
        "total_pixels": int(total),
        "no_change_pixels": int(np.sum(display_no_change_mask)),
        "no_change_percent": float(np.sum(display_no_change_mask) * 100.0 / total),
        "urban_expansion_pixels": int(urban_pixels),
        "urban_expansion_area_km2": float(urban_pixels * pixel_area_km2),
        "urban_expansion_percent": float(urban_pixels * 100.0 / total),
        "vegetation_loss_pixels": int(veg_loss_pixels),
        "vegetation_loss_area_km2": float(veg_loss_pixels * pixel_area_km2),
        "vegetation_loss_percent": float(veg_loss_pixels * 100.0 / total),
        "transitions": [],
    }
    for k in sorted_keys:
        count = transitions[k]
        change_stats["transitions"].append(
            {"from": k[0], "to": k[1], "pixels": int(count), "percent": float(count * 100.0 / total)}
        )

    temp_number = str(randint(1000, 9999))
    change_tif_name = f"change_map_{location_name}_{d1}_to_{d2}_{temp_number}.tif"
    change_png_name = f"change_map_{location_name}_{d1}_to_{d2}_{temp_number}.png"
    change_stats_name = f"change_stats_{location_name}_{d1}_to_{d2}_{temp_number}.json"

    change_tif_path = os.path.join(STATIC_DIR, change_tif_name)
    change_png_path = os.path.join(STATIC_DIR, change_png_name)
    change_stats_path = os.path.join(STATIC_DIR, change_stats_name)

    temp_num1 = str(randint(1000, 9999))
    temp_num2 = str(randint(1000, 9999))
    h, w = shape1
    labels1_2d = labels1.reshape(h, w)
    labels2_2d = labels2.reshape(h, w)

    mask1 = ~img1.read(1, masked=True).mask
    mask2 = ~img2.read(1, masked=True).mask

    c1_lat, c1_lon = get_tiff_center_lat_lon(img1)
    c2_lat, c2_lon = get_tiff_center_lat_lon(img2)
    file1 = f"{location_name}_{d1}.tif"
    file2 = f"{location_name}_{d2}.tif"
    class_png1_path, class_rgb1_tif_path, class_color1_tif_path = plot_classified_map(
        img1,
        labels1_2d.flatten(),
        shape1,
        f"Classified {d1}",
        temp_number=temp_num1,
        original_image=img1,
        save_dir=STATIC_DIR,
        file_name=file1,
        center_lat=c1_lat,
        center_lon=c1_lon,
        valid_mask=mask1,
    )
    class_png2_path, class_rgb2_tif_path, class_color2_tif_path = plot_classified_map(
        img2,
        labels2_2d.flatten(),
        shape2,
        f"Classified {d2}",
        temp_number=temp_num2,
        original_image=img2,
        save_dir=STATIC_DIR,
        file_name=file2,
        center_lat=c2_lat,
        center_lon=c2_lon,
        valid_mask=mask2,
    )

    class_png1_name = os.path.basename(class_png1_path)
    class_png2_name = os.path.basename(class_png2_path)
    class_tif1_name = os.path.basename(class_color1_tif_path)
    class_tif2_name = os.path.basename(class_color2_tif_path)

    src = img1
    try:
        with rasterio.open(
            change_tif_path,
            "w",
            driver="GTiff",
            height=src.height,
            width=src.width,
            count=1,
            dtype=change_codes.dtype,
            crs=src.crs,
            transform=src.transform,
        ) as dst:
            dst.write(change_codes.reshape(src.height, src.width), 1)
    except Exception as e:
        logger.exception("Failed to write change GeoTIFF")
        return templates.TemplateResponse("change_detection.html", {"request": request, "error": f"Failed to write change_map.tif: {e}"})

    try:
        from matplotlib.colors import ListedColormap

        display_codes = (change_codes > 0).astype(np.uint8)
        cmap = ListedColormap(["black", "red"])
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.imshow(display_codes.reshape(src.height, src.width), cmap=cmap, vmin=0, vmax=1)
        ax.set_title(f"Change Map (red = change): {location_name} ({d1} → {d2})")
        ax.axis("off")
        plt.savefig(change_png_path, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
    except Exception:
        logger.exception("Failed to write change PNG")

    try:
        with open(change_stats_path, "w", encoding="utf-8") as f:
            json.dump(change_stats, f, indent=2)
    except Exception:
        logger.exception("Failed to write change stats json")

    return templates.TemplateResponse(
        "change_detection.html",
        {
            "request": request,
            "location_name": location_name,
            "date1": d1,
            "date2": d2,
            "method": method,
            "change_png_url": change_png_name,
            "change_tif_url": change_tif_name,
            "change_stats_url": change_stats_name,
            "change_stats": change_stats,
            "class_png1_url": class_png1_name,
            "class_png2_url": class_png2_name,
            "class_tif1_url": class_tif1_name,
            "class_tif2_url": class_tif2_name,
        },
    )


@router.post("/lulc-event-change", response_class=JSONResponse)
async def lulc_event_change(req: LulcEventChangeRequest):
    """
    Disaster/event-based change detection using cloud-masked Sentinel-2 composites:
    BEFORE (event_date - window_days) vs AFTER (event_date + window_days),
    then LULC classification and transition-based change raster/stats.
    """
    ensure_ee_initialized()
    geom = build_geometry(req.aoi)
    dt = parse_iso_date(req.event_date)
    window_days = int(req.window_days)
    if window_days <= 0 or window_days > 30:
        raise HTTPException(status_code=400, detail="window_days must be between 1 and 30")

    method = resolve_method_default(req.method)

    aoi_hash = hash_key(stable_json_dumps(req.aoi))
    cache_prefix = hash_key(
        stable_json_dumps(
            {"aoi_hash": aoi_hash, "event_date": req.event_date, "window_days": window_days, "method": method, "scale_m": req.scale_m}
        )
    )

    before_paths = {
        "png": os.path.join(OUTPUT_EVENT_DIR, f"ec_{cache_prefix}_before.png"),
        "tif": os.path.join(OUTPUT_EVENT_DIR, f"ec_{cache_prefix}_before_labels.tif"),
        "stats": os.path.join(OUTPUT_EVENT_DIR, f"ec_{cache_prefix}_before_stats.json"),
    }
    after_paths = {
        "png": os.path.join(OUTPUT_EVENT_DIR, f"ec_{cache_prefix}_after.png"),
        "tif": os.path.join(OUTPUT_EVENT_DIR, f"ec_{cache_prefix}_after_labels.tif"),
        "stats": os.path.join(OUTPUT_EVENT_DIR, f"ec_{cache_prefix}_after_stats.json"),
    }

    change_png_path = os.path.join(OUTPUT_EVENT_DIR, f"ec_{cache_prefix}_change.png")
    change_tif_path = os.path.join(OUTPUT_EVENT_DIR, f"ec_{cache_prefix}_change_labels.tif")
    change_stats_json_path = os.path.join(OUTPUT_EVENT_DIR, f"ec_{cache_prefix}_change_stats.json")

    if (
        os.path.exists(change_png_path)
        and os.path.exists(change_stats_json_path)
        and os.path.exists(before_paths["png"])
        and os.path.exists(after_paths["png"])
    ):
        change_stats_payload = load_stats_json(change_stats_json_path)
        return JSONResponse(
            {
                "before_map": static_url_from_abs_path(before_paths["png"]),
                "after_map": static_url_from_abs_path(after_paths["png"]),
                "change_map": static_url_from_abs_path(change_png_path),
                "change_stats": change_stats_payload.get("change_stats", {}),
            }
        )

    before_start = (dt - timedelta(days=window_days)).strftime("%Y-%m-%d")
    before_end = dt.strftime("%Y-%m-%d")
    after_start = dt.strftime("%Y-%m-%d")
    after_end = (dt + timedelta(days=window_days + 1)).strftime("%Y-%m-%d")

    try:
        # Event mode: tolerate heavy storms/floods by allowing up to 100% cloud metadata
        # and skipping cloud masks for this endpoint only.
        before_ee = get_s2_masked_composite_for_range(
            geom, before_start, before_end, max_cloud=100, apply_cloud_mask=False
        )
        after_ee = get_s2_masked_composite_for_range(
            geom, after_start, after_end, max_cloud=100, apply_cloud_mask=False
        )

        max_workers = 2
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            f1 = ex.submit(
                ensure_classification_for_step,
                ee_image=before_ee,
                geom=geom,
                method=method,
                location_key=cache_prefix,
                step_label="before",
                scale_m=req.scale_m,
                out_png_path=before_paths["png"],
                out_label_tif_path=before_paths["tif"],
                out_stats_json_path=before_paths["stats"],
            )
            f2 = ex.submit(
                ensure_classification_for_step,
                ee_image=after_ee,
                geom=geom,
                method=method,
                location_key=cache_prefix,
                step_label="after",
                scale_m=req.scale_m,
                out_png_path=after_paths["png"],
                out_label_tif_path=after_paths["tif"],
                out_stats_json_path=after_paths["stats"],
            )
            before_res = f1.result()
            after_res = f2.result()
    except Exception as e:
        logger.exception("lulc-event-change failed during composite/export/classification")
        return JSONResponse(
            status_code=400,
            content={
                "error": str(e),
                "before_window": {"start": before_start, "end": before_end},
                "after_window": {"start": after_start, "end": after_end},
            },
        )

    before_code = before_res["code_arr"]
    after_code = after_res["code_arr"]

    if before_code.shape != after_code.shape:
        raise HTTPException(status_code=500, detail="Classified rasters have different shapes. Try a smaller AOI or different scale.")

    valid_mask = (before_code != 0) & (after_code != 0)
    pixel_area_m2 = float(before_res["pixel_area_m2"])

    change_stats_struct = compute_change_stats_from_codes(
        before_code=before_code,
        after_code=after_code,
        valid_mask=valid_mask,
        hidden_from_builtup_to_other=True,
    )
    change_stats_struct = apply_pixel_area_to_change_stats(change_stats_struct, pixel_area_m2=pixel_area_m2)
    change_stats_area = change_stats_struct.get("area_by_transition_key", {})

    hidden_mask = (before_code == CLASS_TO_CODE["BuiltUp"]) & (after_code != CLASS_TO_CODE["BuiltUp"])
    change_mask = valid_mask & (before_code != after_code) & (~hidden_mask)
    if np.any(change_mask):
        pairs = np.stack([before_code[change_mask], after_code[change_mask]], axis=1)
        uniq_pairs, inverse, counts = np.unique(pairs, axis=0, return_counts=True, return_inverse=True)
        change_code = np.zeros(before_code.shape, dtype=np.uint16)
        change_code[change_mask] = (inverse.astype(np.uint16) + 1)
    else:
        change_code = np.zeros(before_code.shape, dtype=np.uint16)

    h, w = change_code.shape
    display = np.zeros((h, w, 3), dtype=np.uint8)
    display[:] = np.array([0, 0, 0], dtype=np.uint8)
    display[change_mask] = np.array([255, 0, 0], dtype=np.uint8)
    plt.imsave(change_png_path, display)

    with rasterio.open(before_paths["tif"]) as src:
        crs = src.crs
        transform = src.transform
    with rasterio.open(
        change_tif_path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=1,
        dtype=change_code.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(change_code, 1)

    with open(change_stats_json_path, "w", encoding="utf-8") as f:
        json.dump({"change_stats": change_stats_area}, f, indent=2)

    # Build per-day classified maps and per-day transition maps across [-window_days, +window_days].
    daily_series = []
    skipped_dates = []
    prev_item = None
    for offset in range(-window_days, window_days + 1):
        day_dt = dt + timedelta(days=offset)
        day_label = day_dt.strftime("%Y-%m-%d")
        day_start = (day_dt - timedelta(days=2)).strftime("%Y-%m-%d")
        day_end = (day_dt + timedelta(days=3)).strftime("%Y-%m-%d")
        day_paths = {
            "png": os.path.join(OUTPUT_EVENT_DIR, f"ec_{cache_prefix}_day_{day_label}.png"),
            "tif": os.path.join(OUTPUT_EVENT_DIR, f"ec_{cache_prefix}_day_{day_label}_labels.tif"),
            "stats": os.path.join(OUTPUT_EVENT_DIR, f"ec_{cache_prefix}_day_{day_label}_stats.json"),
        }
        try:
            day_ee = get_s2_masked_composite_for_range(
                geom, day_start, day_end, max_cloud=100, apply_cloud_mask=False
            )
            day_res = ensure_classification_for_step(
                ee_image=day_ee,
                geom=geom,
                method=method,
                location_key=cache_prefix,
                step_label=f"day_{day_label}",
                scale_m=req.scale_m,
                out_png_path=day_paths["png"],
                out_label_tif_path=day_paths["tif"],
                out_stats_json_path=day_paths["stats"],
            )
        except Exception:
            skipped_dates.append(day_label)
            continue

        item = {
            "date": day_label,
            "map": day_res["png_url"],
            "change_map": None,
            "change_stats": {},
        }
        if prev_item is not None:
            before_day_code = prev_item["code_arr"]
            after_day_code = day_res["code_arr"]
            if before_day_code.shape == after_day_code.shape:
                valid_day_mask = (before_day_code != 0) & (after_day_code != 0)
                hidden_day_mask = (before_day_code == CLASS_TO_CODE["BuiltUp"]) & (after_day_code != CLASS_TO_CODE["BuiltUp"])
                day_change_mask = valid_day_mask & (before_day_code != after_day_code) & (~hidden_day_mask)

                day_change_png = os.path.join(OUTPUT_EVENT_DIR, f"ec_{cache_prefix}_day_change_{day_label}.png")
                day_display = np.zeros((after_day_code.shape[0], after_day_code.shape[1], 3), dtype=np.uint8)
                day_display[:] = np.array([0, 0, 0], dtype=np.uint8)
                day_display[day_change_mask] = np.array([255, 0, 0], dtype=np.uint8)
                plt.imsave(day_change_png, day_display)
                item["change_map"] = static_url_from_abs_path(day_change_png)

                day_stats_struct = compute_change_stats_from_codes(
                    before_code=before_day_code,
                    after_code=after_day_code,
                    valid_mask=valid_day_mask,
                    hidden_from_builtup_to_other=True,
                )
                day_stats_struct = apply_pixel_area_to_change_stats(
                    day_stats_struct, pixel_area_m2=float(day_res["pixel_area_m2"])
                )
                item["change_stats"] = day_stats_struct.get("area_by_transition_key", {})

        daily_series.append(item)
        prev_item = day_res

    return JSONResponse(
        {
            "before_map": static_url_from_abs_path(before_paths["png"]),
            "after_map": static_url_from_abs_path(after_paths["png"]),
            "change_map": static_url_from_abs_path(change_png_path),
            "change_stats": change_stats_area,
            "daily_series": daily_series,
            "skipped_dates": skipped_dates,
        }
    )
