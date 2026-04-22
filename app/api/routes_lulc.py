"""Core LULC pages, upload classification, GEE pipelines, and batch."""
import json
import logging
import os
import shutil
import zipfile
from random import randint
from typing import List, Optional

import geemap
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from app.config.settings import STATIC_DIR, UPLOADED_FILES_DIR
from app.dependencies.gee_init import ensure_ee_initialized
from app.models.inference import run_classification_on_raster, run_classification_with_proba
from app.services.classification_service import classify_ee_image_for_date, plot_classified_map
from app.services.export_service import export_ee_image_to_local_geotiff
from app.services.gee_service import get_image_for_date_by_source
from app.templating import templates
from app.utils.file_utils import append_classification_history, load_classification_history
from app.utils.geo_utils import build_geometry, normalize_class_label
from app.utils.raster_utils import compute_class_stats_from_labels, get_tiff_center_lat_lon

router = APIRouter(tags=["lulc"])
logger = logging.getLogger(__name__)


@router.get("/", response_class=HTMLResponse)
async def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "image_url": None})


@router.get("/history", response_class=HTMLResponse)
async def classification_history(request: Request):
    """Page showing all previous classifications done by the user."""
    history = load_classification_history()
    history = list(reversed(history))
    return templates.TemplateResponse("history.html", {"request": request, "history": history})


@router.get("/batch", response_class=HTMLResponse)
async def batch_page(request: Request):
    """Page for batch classification of multiple AOIs from a GeoJSON file."""
    return templates.TemplateResponse("batch.html", {"request": request})


@router.post("/process_shapefile/")
async def process_shapefile(file: UploadFile = File(...)):
    temp_dir = os.path.join(UPLOADED_FILES_DIR, "shapefile_temp")
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    if file.filename.endswith(".zip"):
        try:
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                zip_ref.extractall(temp_dir)
        except Exception as e:
            shutil.rmtree(temp_dir)
            return JSONResponse(content={"error": f"Error extracting ZIP file: {str(e)}"}, status_code=400)
        shp_files = [f for f in os.listdir(temp_dir) if f.endswith(".shp")]
        if not shp_files:
            shutil.rmtree(temp_dir)
            return JSONResponse(content={"error": "No .shp file found in ZIP"}, status_code=400)
        shp_file_path = os.path.join(temp_dir, shp_files[0])
    elif file.filename.endswith(".shp"):
        shp_file_path = file_path
    else:
        shutil.rmtree(temp_dir)
        return JSONResponse(content={"error": "Unsupported file format"}, status_code=400)

    try:
        gdf = gpd.read_file(shp_file_path)
        geojson = gdf.to_json()
        shutil.rmtree(temp_dir)
        return JSONResponse(content=json.loads(geojson))
    except Exception as e:
        shutil.rmtree(temp_dir)
        return JSONResponse(content={"error": f"Error processing shapefile: {str(e)}"}, status_code=400)


@router.post("/classify/")
async def classify_image(request: Request, file: UploadFile = File(...), method: str = Form(...)):
    file_name = file.filename
    temp_number = str(randint(1, 1000))
    temp_file_path = os.path.join(UPLOADED_FILES_DIR, f"temp_image_{temp_number}.tif")
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        image = rasterio.open(temp_file_path)
    except Exception as e:
        logger.error("Error processing TIFF image: %s", e)
        raise HTTPException(status_code=400, detail="Error processing TIFF image")

    try:
        labels, _image_dataset, image_shape = run_classification_on_raster(image, method)
    except FileNotFoundError as e:
        return templates.TemplateResponse("index.html", {"request": request, "image_url": None, "error": str(e)})
    except Exception as e:
        logger.exception("Classification failed")
        raise HTTPException(status_code=500, detail=f"Classification failed: {e}")

    center_lat, center_lon = get_tiff_center_lat_lon(image)
    band1_mask = image.read(1, masked=True).mask
    valid_mask = ~band1_mask
    pixel_area_m2 = float(abs(image.res[0] * image.res[1]))
    class_stats = compute_class_stats_from_labels(labels, image_shape, pixel_area_m2, valid_mask=valid_mask)
    aoi_area_km2 = sum((row.get("area_km2") or 0.0) for row in class_stats) if class_stats else None

    saved_image_path, new_classified_image_path, classified_tiff_path = plot_classified_map(
        image,
        labels,
        image_shape,
        "Classified Land Cover Map",
        temp_number=temp_number,
        original_image=image,
        save_dir=STATIC_DIR,
        file_name=file_name,
        center_lat=center_lat,
        center_lon=center_lon,
        valid_mask=valid_mask,
    )
    image_url = os.path.basename(saved_image_path)
    new_classified_url = os.path.basename(new_classified_image_path)
    classified_tiff_url = os.path.basename(classified_tiff_path)
    location_from_file = os.path.splitext(file_name)[0] if file_name else ""
    append_classification_history(
        file_name=file_name,
        method=method,
        image_url=image_url,
        new_classified_url=new_classified_url,
        classified_tiff_url=classified_tiff_url,
        center_lat=center_lat,
        center_lon=center_lon,
    )
    stats_csv_name = f"class_stats_{location_from_file}_{temp_number}.csv"
    stats_csv_path = os.path.join(STATIC_DIR, stats_csv_name)
    try:
        import csv

        with open(stats_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["class", "area_km2", "percent", "pixels"])
            for row in class_stats:
                writer.writerow(
                    [
                        row["class"],
                        f"{row['area_km2']:.4f}",
                        f"{row['percent']:.2f}",
                        row["pixels"],
                    ]
                )
        class_stats_csv_url = stats_csv_name
    except Exception as e:
        logger.warning("Failed to write class stats CSV: %s", e)
        class_stats_csv_url = None

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "file_name": file_name,
            "location_from_file": location_from_file,
            "method": method,
            "image_url": image_url,
            "new_classified_url": new_classified_url,
            "classified_tiff_url": classified_tiff_url,
            "center_lat": center_lat,
            "center_lon": center_lon,
            "class_stats": class_stats,
            "class_stats_csv_url": class_stats_csv_url,
            "aoi_area_km2": aoi_area_km2,
            "aoi_date": None,
            "aoi_data_source": None,
        },
    )


@router.post("/generate-lulc", response_class=HTMLResponse)
async def generate_lulc(
    request: Request,
    geojson: str = Form(..., description="AOI GeoJSON (Feature/FeatureCollection/Polygon/MultiPolygon)"),
    method: str = Form(...),
    date: str = Form(..., description="Requested date (YYYY-MM-DD)"),
    location_name: str = Form(..., description="Location name (used for naming/display)"),
    scale: int = Form(30, description="Pixel scale (meters)"),
    data_source: str = Form("sentinel2", description="Data source (sentinel2, landsat9, sentinel1, planet)"),
):
    """
    Fully automated pipeline:
    AOI + date -> fetch Sentinel-2 from GEE -> export GeoTIFF locally -> classify -> show result.
    """
    if not location_name or not location_name.strip():
        return templates.TemplateResponse("index.html", {"request": request, "image_url": None, "error": "Location name is required."})
    location_name = location_name.strip()

    ensure_ee_initialized()

    try:
        geojson_obj = json.loads(geojson)
    except Exception:
        return templates.TemplateResponse("index.html", {"request": request, "image_url": None, "error": "Invalid GeoJSON provided for AOI."})

    try:
        geom = build_geometry(geojson_obj)
    except Exception as e:
        return templates.TemplateResponse("index.html", {"request": request, "image_url": None, "error": f"Error processing AOI: {e}"})

    ee_image, err = get_image_for_date_by_source(data_source, geom, date)
    if err:
        return templates.TemplateResponse("index.html", {"request": request, "image_url": None, "error": err})

    temp_number = str(randint(1000, 9999))
    display_file_name = f"{location_name}_{date}.tif"
    local_tif_path = os.path.join(UPLOADED_FILES_DIR, f"gee_{location_name}_{date}_{temp_number}.tif")

    try:
        geemap.ee_export_image(
            ee_image,
            filename=local_tif_path,
            scale=int(scale) if scale else 30,
            region=geom,
            file_per_band=False,
        )
    except Exception as e:
        logger.exception("GEE export to local GeoTIFF failed")
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "image_url": None, "error": f"Failed to fetch/export Sentinel-2 image automatically: {e}"},
        )

    try:
        image = rasterio.open(local_tif_path)
    except Exception as e:
        logger.exception("Could not open exported GeoTIFF")
        return templates.TemplateResponse("index.html", {"request": request, "image_url": None, "error": f"Could not open exported GeoTIFF: {e}"})

    try:
        labels, _image_dataset, image_shape = run_classification_on_raster(image, method)
    except FileNotFoundError as e:
        return templates.TemplateResponse("index.html", {"request": request, "image_url": None, "error": str(e)})
    except Exception as e:
        logger.exception("Automated classification failed")
        return templates.TemplateResponse("index.html", {"request": request, "image_url": None, "error": f"Classification failed: {e}"})

    center_lat, center_lon = get_tiff_center_lat_lon(image)
    band1_mask = image.read(1, masked=True).mask
    valid_mask = ~band1_mask
    pixel_area_m2 = float(abs(image.res[0] * image.res[1]))
    class_stats = compute_class_stats_from_labels(labels, image_shape, pixel_area_m2, valid_mask=valid_mask)
    aoi_area_km2 = sum((row.get("area_km2") or 0.0) for row in class_stats) if class_stats else None
    saved_image_path, new_classified_image_path, classified_tiff_path = plot_classified_map(
        image,
        labels,
        image_shape,
        "Classified Land Cover Map",
        temp_number=temp_number,
        original_image=image,
        save_dir=STATIC_DIR,
        file_name=display_file_name,
        center_lat=center_lat,
        center_lon=center_lon,
        valid_mask=valid_mask,
    )

    image_url = os.path.basename(saved_image_path)
    new_classified_url = os.path.basename(new_classified_image_path)
    classified_tiff_url = os.path.basename(classified_tiff_path)
    location_from_file = os.path.splitext(display_file_name)[0]

    append_classification_history(
        file_name=display_file_name,
        method=method,
        image_url=image_url,
        new_classified_url=new_classified_url,
        classified_tiff_url=classified_tiff_url,
        center_lat=center_lat,
        center_lon=center_lon,
    )

    stats_csv_name = f"class_stats_{location_from_file}_{temp_number}.csv"
    stats_csv_path = os.path.join(STATIC_DIR, stats_csv_name)
    try:
        import csv

        with open(stats_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["class", "area_km2", "percent", "pixels"])
            for row in class_stats:
                writer.writerow(
                    [
                        row["class"],
                        f"{row['area_km2']:.4f}",
                        f"{row['percent']:.2f}",
                        row["pixels"],
                    ]
                )
        class_stats_csv_url = stats_csv_name
    except Exception as e:
        logger.warning("Failed to write class stats CSV: %s", e)
        class_stats_csv_url = None

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "file_name": display_file_name,
            "location_from_file": location_from_file,
            "method": method,
            "image_url": image_url,
            "new_classified_url": new_classified_url,
            "classified_tiff_url": classified_tiff_url,
            "center_lat": center_lat,
            "center_lon": center_lon,
            "class_stats": class_stats,
            "class_stats_csv_url": class_stats_csv_url,
            "aoi_area_km2": aoi_area_km2,
            "aoi_date": date,
            "aoi_data_source": data_source,
        },
    )


@router.post("/probability-maps", response_class=HTMLResponse)
async def probability_maps(
    request: Request,
    geojson: str = Form(...),
    method: str = Form(...),
    date: str = Form(...),
    location_name: str = Form(...),
    scale: int = Form(30),
):
    """
    Generate class probability maps (per canonical class) for a given AOI, date and method.
    Produces per-class GeoTIFFs and PNG heatmaps.
    """
    if not location_name or not location_name.strip():
        return templates.TemplateResponse(
            "index.html", {"request": request, "image_url": None, "error": "Location name is required."}
        )
    location_name = location_name.strip()

    ensure_ee_initialized()

    try:
        geojson_obj = json.loads(geojson)
    except Exception:
        return templates.TemplateResponse(
            "index.html", {"request": request, "image_url": None, "error": "Invalid GeoJSON provided for AOI."}
        )

    try:
        geom = build_geometry(geojson_obj)
    except Exception as e:
        return templates.TemplateResponse(
            "index.html", {"request": request, "image_url": None, "error": f"Error processing AOI: {e}"}
        )

    ee_image, err = get_image_for_date_by_source("sentinel2", geom, date)
    if err:
        return templates.TemplateResponse("index.html", {"request": request, "image_url": None, "error": err})

    temp_number = str(randint(1000, 9999))
    safe_date = date.replace(":", "-")
    local_tif_path = os.path.join(UPLOADED_FILES_DIR, f"gee_prob_{location_name}_{safe_date}_{temp_number}.tif")

    try:
        export_ee_image_to_local_geotiff(ee_image, geom, local_tif_path, scale=scale)
    except Exception as e:
        logger.exception("GEE export for probability maps failed")
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "image_url": None, "error": f"Failed to fetch/export Sentinel-2 image automatically: {e}"},
        )

    try:
        image = rasterio.open(local_tif_path)
    except Exception as e:
        logger.exception("Could not open exported GeoTIFF for probability maps")
        return templates.TemplateResponse(
            "index.html", {"request": request, "image_url": None, "error": f"Could not open exported GeoTIFF: {e}"}
        )

    try:
        _labels, _image_dataset, image_shape, proba, class_names = run_classification_with_proba(image, method)
    except FileNotFoundError as e:
        return templates.TemplateResponse("index.html", {"request": request, "image_url": None, "error": str(e)})
    except Exception as e:
        logger.exception("Probability classification failed")
        return templates.TemplateResponse(
            "index.html", {"request": request, "image_url": None, "error": f"Probability classification failed: {e}"}
        )

    if proba is None or class_names is None:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "image_url": None,
                "error": "Selected model does not provide probability outputs (predict_proba / softmax).",
            },
        )

    h, w = image_shape
    proba = np.asarray(proba)

    canonical_targets = ["Water", "BuiltUp", "Vegetation", "BarrenLand", "Agricultural"]
    canon_for_model_classes = [normalize_class_label(c) for c in class_names]

    prob_maps = []

    for canon in canonical_targets:
        idxs = [i for i, cn in enumerate(canon_for_model_classes) if cn == canon]
        if not idxs:
            continue
        class_proba = proba[:, idxs].sum(axis=1) if len(idxs) > 1 else proba[:, idxs[0]]
        class_proba = class_proba.reshape(h, w).astype("float32")

        prob_tif_name = f"prob_{canon}_{location_name}_{safe_date}_{temp_number}.tif"
        prob_png_name = f"prob_{canon}_{location_name}_{safe_date}_{temp_number}.png"
        prob_tif_path = os.path.join(STATIC_DIR, prob_tif_name)
        prob_png_path = os.path.join(STATIC_DIR, prob_png_name)

        try:
            with rasterio.open(
                prob_tif_path,
                "w",
                driver="GTiff",
                height=h,
                width=w,
                count=1,
                dtype="float32",
                crs=image.crs,
                transform=image.transform,
            ) as dst:
                dst.write(class_proba, 1)
        except Exception:
            logger.exception("Failed to write probability GeoTIFF for %s", canon)
            continue

        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(class_proba, cmap="viridis", vmin=0.0, vmax=1.0)
            ax.set_title(f"{canon} probability ({location_name}, {date})")
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Probability")
            plt.savefig(prob_png_path, bbox_inches="tight", pad_inches=0.1)
            plt.close(fig)
        except Exception:
            logger.exception("Failed to write probability PNG for %s", canon)
            continue

        prob_maps.append(
            {
                "class": canon,
                "png": prob_png_name,
                "tif": prob_tif_name,
            }
        )

    if not prob_maps:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "image_url": None,
                "error": "No probability maps could be generated for the selected model/classes.",
            },
        )

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "image_url": None,
            "prob_maps": prob_maps,
            "prob_location": location_name,
            "prob_date": date,
            "prob_method": method,
        },
    )


@router.post("/batch-generate-lulc", response_class=HTMLResponse)
async def batch_generate_lulc(
    request: Request,
    file: Optional[UploadFile] = File(None),
    tiff_files: Optional[List[UploadFile]] = File(None),
    geojson: Optional[str] = Form(None),
    location_prefix: str = Form("AOI"),
    date: str = Form(...),
    method: str = Form(...),
    scale: int = Form(30),
):
    """
    Batch classification:
    - Multiple AOIs: from uploaded GeoJSON FeatureCollection or drawn AOIs (geojson form field)
    - Multiple TIFFs: uploaded directly
    """
    ensure_ee_initialized()
    results = []

    fc = None
    if geojson:
        try:
            fc = json.loads(geojson)
        except Exception:
            return templates.TemplateResponse(
                "batch.html", {"request": request, "error": "Invalid AOI GeoJSON. Please draw again or upload a valid file."}
            )
    elif file is not None and file.filename:
        try:
            content = await file.read()
            fc = json.loads(content.decode("utf-8"))
        except Exception:
            return templates.TemplateResponse(
                "batch.html", {"request": request, "error": "Invalid GeoJSON file. Please upload a valid FeatureCollection."}
            )

    if fc is not None:
        if fc.get("type") == "Feature":
            features = [fc]
        elif fc.get("type") == "FeatureCollection":
            features = fc.get("features") or []
        else:
            return templates.TemplateResponse(
                "batch.html", {"request": request, "error": "GeoJSON must be a Feature or FeatureCollection of polygons."}
            )

        location_prefix = (location_prefix or "AOI").strip()

        for idx, feat in enumerate(features):
            name = feat.get("properties", {}).get("name") or f"{location_prefix}_{idx+1}"
            try:
                geom = build_geometry({"type": "FeatureCollection", "features": [feat]})
            except Exception as e:
                results.append(
                    {"name": name, "status": f"Failed (AOI error: {e})", "image_url": None, "tif_url": None, "csv_url": None}
                )
                continue

            res, err = classify_ee_image_for_date(geom, date, method, name, scale=scale)
            if err:
                results.append({"name": name, "status": f"Failed ({err})", "image_url": None, "tif_url": None, "csv_url": None})
                continue

            labels = np.array(res["labels"])
            band1_mask = res["raster_handle"].read(1, masked=True).mask
            valid_mask = ~band1_mask
            pixel_area_m2 = float(abs(res["raster_handle"].res[0] * res["raster_handle"].res[1]))
            class_stats = compute_class_stats_from_labels(labels, res["image_shape"], pixel_area_m2, valid_mask=valid_mask)

            temp_number = str(randint(1000, 9999))
            saved_image_path, _rgb_path, classified_tiff_path = plot_classified_map(
                res["raster_handle"],
                labels,
                res["image_shape"],
                "Classified Land Cover Map",
                temp_number=temp_number,
                original_image=res["raster_handle"],
                save_dir=STATIC_DIR,
                file_name=res["file_name"],
                center_lat=res["center_lat"],
                center_lon=res["center_lon"],
                valid_mask=valid_mask,
            )
            image_url = os.path.basename(saved_image_path)
            tif_url = os.path.basename(classified_tiff_path)

            stats_csv_name = f"class_stats_{name}_{temp_number}.csv"
            stats_csv_path = os.path.join(STATIC_DIR, stats_csv_name)
            csv_url = None
            try:
                import csv

                with open(stats_csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["class", "area_km2", "percent", "pixels"])
                    for row in class_stats:
                        writer.writerow(
                            [
                                row["class"],
                                f"{row['area_km2']:.4f}",
                                f"{row['percent']:.2f}",
                                row["pixels"],
                            ]
                        )
                csv_url = stats_csv_name
            except Exception as e:
                logger.warning("Failed to write batch stats CSV for %s: %s", name, e)

            results.append(
                {
                    "name": name,
                    "status": "Processed AOI",
                    "image_url": image_url,
                    "tif_url": tif_url,
                    "csv_url": csv_url,
                }
            )

    if tiff_files:
        for up in tiff_files:
            if not up.filename:
                continue
            name = up.filename
            temp_number = str(randint(1000, 9999))
            temp_path = os.path.join(UPLOADED_FILES_DIR, f"batch_{temp_number}_{name}")
            try:
                with open(temp_path, "wb") as buffer:
                    shutil.copyfileobj(up.file, buffer)
            except Exception as e:
                results.append({"name": name, "status": f"Failed (save error: {e})", "image_url": None, "tif_url": None, "csv_url": None})
                continue

            try:
                image = rasterio.open(temp_path)
            except Exception as e:
                results.append({"name": name, "status": f"Failed (TIFF open error: {e})", "image_url": None, "tif_url": None, "csv_url": None})
                continue

            try:
                labels, _image_dataset, image_shape = run_classification_on_raster(image, method)
            except Exception as e:
                logger.exception("Batch TIFF classification failed")
                results.append({"name": name, "status": f"Failed (classification error: {e})", "image_url": None, "tif_url": None, "csv_url": None})
                continue

            band1_mask = image.read(1, masked=True).mask
            valid_mask = ~band1_mask
            pixel_area_m2 = float(abs(image.res[0] * image.res[1]))
            class_stats = compute_class_stats_from_labels(labels, image_shape, pixel_area_m2, valid_mask=valid_mask)

            center_lat, center_lon = get_tiff_center_lat_lon(image)
            temp_plot_num = str(randint(1000, 9999))
            saved_image_path, _rgb_path, classified_tiff_path = plot_classified_map(
                image,
                labels,
                image_shape,
                "Classified Land Cover Map",
                temp_number=temp_plot_num,
                original_image=image,
                save_dir=STATIC_DIR,
                file_name=name,
                center_lat=center_lat,
                center_lon=center_lon,
                valid_mask=valid_mask,
            )
            image_url = os.path.basename(saved_image_path)
            tif_url = os.path.basename(classified_tiff_path)

            stats_csv_name = f"class_stats_{name}_{temp_plot_num}.csv"
            stats_csv_path = os.path.join(STATIC_DIR, stats_csv_name)
            csv_url = None
            try:
                import csv

                with open(stats_csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["class", "area_km2", "percent", "pixels"])
                    for row in class_stats:
                        writer.writerow(
                            [
                                row["class"],
                                f"{row['area_km2']:.4f}",
                                f"{row['percent']:.2f}",
                                row["pixels"],
                            ]
                        )
                csv_url = stats_csv_name
            except Exception as e:
                logger.warning("Failed to write batch stats CSV for TIFF %s: %s", name, e)

            results.append(
                {
                    "name": name,
                    "status": "Processed TIFF",
                    "image_url": image_url,
                    "tif_url": tif_url,
                    "csv_url": csv_url,
                }
            )

    if not results:
        return templates.TemplateResponse(
            "batch.html",
            {"request": request, "error": "No valid AOIs or TIFF files provided for batch processing."},
        )

    return templates.TemplateResponse("batch.html", {"request": request, "results": results})
