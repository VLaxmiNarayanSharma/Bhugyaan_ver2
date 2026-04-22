"""GeoTIFF export endpoints using Cloud Storage + auto-download to app static."""
import json
import logging
import os
import threading
import time
from random import randint
from typing import Dict

import ee
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from app.config.settings import DEFAULT_EE_PROJECT, DOWNLOADS_STATIC_DIR, GCS_BUCKET
from app.dependencies.gee_init import ensure_ee_initialized
from app.services.gee_service import get_s2_image_for_date
from app.utils.geo_utils import build_geometry

router = APIRouter(tags=["download"])
logger = logging.getLogger(__name__)

try:
    from google.cloud import storage
except Exception:  # pragma: no cover
    storage = None

_TASK_LOCK = threading.Lock()
_DOWNLOAD_TASKS: Dict[str, Dict[str, str]] = {}


def _set_task(task_id: str, **kwargs) -> None:
    with _TASK_LOCK:
        obj = _DOWNLOAD_TASKS.get(task_id, {})
        obj.update(kwargs)
        _DOWNLOAD_TASKS[task_id] = obj


def _monitor_export_and_download(task, bucket_name: str, prefix: str, local_name: str) -> None:
    """Poll EE task, then copy output from GCS to static/downloads."""
    task_id = str(task.id)
    try:
        deadline = time.time() + 1800  # 30 min
        while time.time() < deadline:
            st = task.status() or {}
            state = st.get("state", "UNKNOWN")
            _set_task(task_id, state=state)
            if state == "COMPLETED":
                break
            if state in ("FAILED", "CANCELLED"):
                _set_task(task_id, error=str(st.get("error_message", "Export failed")))
                return
            time.sleep(5)
        else:
            _set_task(task_id, state="FAILED", error="Export timed out")
            return

        if storage is None:
            _set_task(task_id, state="FAILED", error="google-cloud-storage package not available on server.")
            return

        client = storage.Client(project=DEFAULT_EE_PROJECT)
        bucket = client.bucket(bucket_name)
        blobs = list(client.list_blobs(bucket, prefix=prefix))
        if not blobs:
            _set_task(task_id, state="FAILED", error=f"No file found in gs://{bucket_name}/{prefix}")
            return
        blob = sorted(blobs, key=lambda b: b.name)[0]
        local_path = os.path.join(DOWNLOADS_STATIC_DIR, local_name + ".tif")
        blob.download_to_filename(local_path)
        download_url = "/static/downloads/" + os.path.basename(local_path)
        _set_task(task_id, state="READY", download_url=download_url, gcs_path=f"gs://{bucket_name}/{blob.name}")
    except Exception as e:
        logger.exception("Export monitor failed")
        _set_task(task_id, state="FAILED", error=str(e))


def _start_cloud_export(image, region, unique_file_name: str, scale: int = 30):
    if not GCS_BUCKET:
        raise HTTPException(
            status_code=400,
            detail="GCS_BUCKET is not configured. Set environment variable GCS_BUCKET to your bucket name.",
        )
    prefix = f"bhugyaan_exports/{unique_file_name}"
    task = ee.batch.Export.image.toCloudStorage(
        image=image,
        description=unique_file_name,
        bucket=GCS_BUCKET,
        fileNamePrefix=prefix,
        scale=scale,
        region=region,
        fileFormat="GeoTIFF",
    )
    task.start()
    task_id = str(task.id)
    _set_task(task_id, state="RUNNING", file_name=unique_file_name)
    t = threading.Thread(
        target=_monitor_export_and_download,
        args=(task, GCS_BUCKET, prefix, unique_file_name),
        daemon=True,
    )
    t.start()
    return task_id, prefix


@router.post("/download/")
async def download_tiff(
    coordinates: str = Form(...),
    file_name: str = Form(...),
    date: str = Form(..., description="Date for Sentinel-2 image (YYYY-MM-DD)"),
):
    ensure_ee_initialized()
    try:
        geom = ee.Geometry.Polygon(eval(coordinates))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid coordinates.")
    image, err = get_s2_image_for_date(geom, date)
    if err:
        logger.info("POST /download/ 400: %s (date=%s)", err, date)
        raise HTTPException(status_code=400, detail=err)
    try:
        unique_file_name = f"{file_name}_{randint(1000, 9999)}"
        region = geom.getInfo()["coordinates"]
        task_id, prefix = _start_cloud_export(image, region, unique_file_name, scale=30)
        return {
            "status": "Export started to Cloud Storage",
            "file_name": unique_file_name,
            "task_id": task_id,
            "gcs_prefix": f"gs://{GCS_BUCKET}/{prefix}",
        }
    except Exception as e:
        logger.error("Error during download: %s", e)
        raise HTTPException(status_code=500, detail=f"Download failed: {e}")


@router.post("/download_tiff_from_upload")
async def download_tiff_from_upload(request: Request):
    ensure_ee_initialized()
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    geojson = data.get("geojson")
    file_name = data.get("file_name")
    date = data.get("date")
    if not geojson:
        raise HTTPException(status_code=400, detail="No GeoJSON provided.")
    if not file_name or file_name.strip() == "":
        raise HTTPException(status_code=400, detail="No file name provided.")
    if not date or not str(date).strip():
        raise HTTPException(status_code=400, detail="Date is required (YYYY-MM-DD).")

    try:
        geom = build_geometry(geojson)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing GeoJSON: {e}")

    try:
        region = geom.getInfo()["coordinates"]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error retrieving geometry info: {e}")

    image, err = get_s2_image_for_date(geom, str(date).strip())
    if err:
        raise HTTPException(status_code=400, detail=err)

    unique_file_name = f"{file_name}_{randint(1000, 9999)}"
    task_id, prefix = _start_cloud_export(image, region, unique_file_name, scale=30)
    return JSONResponse(
        content={
            "status": "Export started to Cloud Storage",
            "file_name": unique_file_name,
            "task_id": task_id,
            "gcs_prefix": f"gs://{GCS_BUCKET}/{prefix}",
        }
    )


@router.post("/download_tiff_from_geojson_file")
async def download_tiff_from_geojson_file(
    file: UploadFile = File(...),
    file_name: str = Form(...),
    date: str = Form(..., description="Date for Sentinel-2 image (YYYY-MM-DD)"),
):
    ensure_ee_initialized()
    if not file.filename.endswith(".geojson"):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload a .geojson file.")
    try:
        content = await file.read()
        geojson = json.loads(content.decode("utf-8"))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid GeoJSON file")

    try:
        geom = build_geometry(geojson)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing GeoJSON: {e}")

    try:
        region = geom.getInfo()["coordinates"]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error retrieving geometry info: {e}")

    image, err = get_s2_image_for_date(geom, date)
    if err:
        raise HTTPException(status_code=400, detail=err)

    unique_file_name = f"{file_name}_{randint(1000, 9999)}"
    task_id, prefix = _start_cloud_export(image, region, unique_file_name, scale=30)
    return JSONResponse(
        content={
            "status": "Export started to Cloud Storage",
            "file_name": unique_file_name,
            "task_id": task_id,
            "gcs_prefix": f"gs://{GCS_BUCKET}/{prefix}",
        }
    )


@router.get("/download-task-status/{task_id}")
async def download_task_status(task_id: str):
    with _TASK_LOCK:
        st = _DOWNLOAD_TASKS.get(task_id)
    if not st:
        raise HTTPException(status_code=404, detail="Task not found")
    return JSONResponse(st)
