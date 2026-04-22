"""Time-series UI and JSON API."""
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

from app.api.schemas import LulcTimeSeriesRequest
from app.config.settings import OUTPUT_TIMESERIES_DIR
from app.dependencies.gee_init import ensure_ee_initialized
from app.services.export_service import ensure_classification_for_step
from app.services.gee_service import get_s2_masked_composite_for_range
from app.services.timeseries_service import resolve_method_default, step_windows
from app.templating import templates
from app.utils.cache_keys import hash_key, safe_filename, stable_json_dumps
from app.utils.geo_utils import build_geometry

router = APIRouter(tags=["timeseries"])


@router.get("/time-series", response_class=HTMLResponse)
async def time_series_page(request: Request):
    """Dedicated page for long-term LULC time-series analysis."""
    return templates.TemplateResponse("timeseries.html", {"request": request})


@router.post("/lulc-timeseries", response_class=JSONResponse)
async def lulc_timeseries(req: LulcTimeSeriesRequest):
    """
    Long-term LULC time-series (yearly/quarterly) using cloud-masked Sentinel-2 composites.
    Returns JSON with per-step classified map PNG URLs + class area statistics.
    """
    ensure_ee_initialized()
    geom = build_geometry(req.aoi)

    interval_windows = step_windows(req.start_year, req.end_year, req.interval)

    aoi_hash = hash_key(stable_json_dumps(req.aoi))
    cache_prefix = hash_key(
        stable_json_dumps(
            {
                "aoi_hash": aoi_hash,
                "interval": req.interval,
                "method": req.method,
                "scale_m": req.scale_m,
                "windows": [w["label"] for w in interval_windows],
            }
        )
    )

    max_cloud = 30 if req.interval == "yearly" else 25

    def step_paths(step_label: str) -> Dict[str, str]:
        step_safe = safe_filename(step_label)
        base = f"ts_{cache_prefix}_{step_safe}"
        out_png = os.path.join(OUTPUT_TIMESERIES_DIR, base + ".png")
        out_label_tif = os.path.join(OUTPUT_TIMESERIES_DIR, base + "_labels.tif")
        out_stats_json = os.path.join(OUTPUT_TIMESERIES_DIR, base + "_stats.json")
        return {"png": out_png, "tif": out_label_tif, "stats": out_stats_json}

    step_work: List[Dict[str, Any]] = []
    for w in interval_windows:
        label = w["label"]
        ee_image = get_s2_masked_composite_for_range(geom, w["start_date"], w["end_date"], max_cloud=max_cloud)
        paths = step_paths(label)
        step_work.append(
            {
                "label": label,
                "ee_image": ee_image,
                "paths": paths,
                "start_date": w["start_date"],
                "end_date": w["end_date"],
            }
        )

    max_workers = min(2, len(step_work)) if len(step_work) > 0 else 1
    results_by_label: Dict[str, Any] = {}
    method = resolve_method_default(req.method)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures_map: Dict[str, Any] = {}
        for item in step_work:
            futures_map[item["label"]] = ex.submit(
                ensure_classification_for_step,
                ee_image=item["ee_image"],
                geom=geom,
                method=method,
                location_key=cache_prefix,
                step_label=item["label"],
                scale_m=req.scale_m,
                out_png_path=item["paths"]["png"],
                out_label_tif_path=item["paths"]["tif"],
                out_stats_json_path=item["paths"]["stats"],
            )

        for label, fut in futures_map.items():
            results_by_label[label] = fut.result()

    years_out: List[Any] = []
    maps_out: List[str] = []
    stats_out: Dict[str, Any] = {}
    for w in interval_windows:
        label = w["label"]
        years_out.append(int(label) if req.interval == "yearly" and label.isdigit() else label)
        maps_out.append(results_by_label[label]["png_url"])
        stats_out[str(label)] = results_by_label[label]["stats"]

    return JSONResponse({"years": years_out, "maps": maps_out, "stats": stats_out})
