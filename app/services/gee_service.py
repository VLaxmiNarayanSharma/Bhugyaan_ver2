"""Google Earth Engine collections, composites, and image selection."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

import ee

from app.config.settings import NEARBY_DAYS_WINDOW, S2_BANDS, YEARLY_MAX_CLOUD
from app.utils.geo_utils import validate_and_parse_date

logger = logging.getLogger(__name__)


def _get_collection_for_date_range(collection_id: str, geom, start_date: str, end_date: str, max_cloud: int):
    """Build filtered S2 collection for a date range and max cloud percentage."""
    return (
        ee.ImageCollection(collection_id)
        .filterBounds(geom)
        .filterMetadata("CLOUDY_PIXEL_PERCENTAGE", "less_than", max_cloud)
        .filterDate(start_date, end_date)
        .select(S2_BANDS)
    )


def _pick_collection_id(start_date: str, end_date: str) -> str:
    """Use S2_SR_HARMONIZED when available (2017+); S2_SR for older dates."""
    if end_date < "2017-03-01":
        return "COPERNICUS/S2_SR"
    return "COPERNICUS/S2_SR_HARMONIZED"


def get_landsat9_image_for_date(geom, date_str: str):
    """
    Landsat-9 SR image for a given date (±1 day window), harmonized to 12 bands named B1..B12.
    """
    ok, err = validate_and_parse_date(date_str)
    if not ok:
        return None, err
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        start = dt.strftime("%Y-%m-%d")
        end = (dt + timedelta(days=1)).strftime("%Y-%m-%d")
        coll = (
            ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
            .filterBounds(geom)
            .filterDate(start, end)
            .filterMetadata("CLOUD_COVER", "less_than", 30)
        )
        if coll.size().getInfo() == 0:
            return None, "No Landsat-9 data available for this date in Google Earth Engine."
        img = coll.median().clip(geom)
        img = img.multiply(0.0000275).add(-0.2)
        bands = ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]
        img = img.select(bands)
        b1 = img.select("SR_B1")
        b2 = img.select("SR_B2")
        b3 = img.select("SR_B3")
        b4 = img.select("SR_B4")
        b5 = img.select("SR_B5")
        b6 = img.select("SR_B6")
        b7 = img.select("SR_B7")
        stacked = ee.Image.cat(
            [
                b1,
                b2,
                b3,
                b4,
                b5,
                b6,
                b7,
                b5,
                b5,
                b1,
                b6,
                b7,
            ]
        ).rename(S2_BANDS)
        return stacked.multiply(1.0), None
    except Exception:
        logger.exception("get_landsat9_image_for_date failed")
        return None, "No Landsat-9 data available for this date in Google Earth Engine."


def get_sentinel1_image_for_date(geom, date_str: str):
    """
    Sentinel-1 GRD composite for a given date window, converted to a 12-band stack.
    Bands are derived from VV and VH backscatter (in dB).
    """
    ok, err = validate_and_parse_date(date_str)
    if not ok:
        return None, err
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        start = (dt - timedelta(days=6)).strftime("%Y-%m-%d")
        end = (dt + timedelta(days=6)).strftime("%Y-%m-%d")
        coll = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(geom)
            .filterDate(start, end)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING"))
            .filter(ee.Filter.eq("resolution_meters", 10))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        )
        if coll.size().getInfo() == 0:
            return None, "No Sentinel-1 SAR data available for this date window in Google Earth Engine."
        img = coll.median().clip(geom)
        vv = img.select("VV")
        vh = img.select("VH")
        vv_db = vv.log10().multiply(10.0)
        vh_db = vh.log10().multiply(10.0)
        ratio = vv_db.subtract(vh_db)
        stacked = ee.Image.cat(
            [
                vv_db,
                vh_db,
                ratio,
                vv_db,
                vh_db,
                ratio,
                vv_db,
                vh_db,
                ratio,
                vv_db,
                vh_db,
                ratio,
            ]
        ).rename(S2_BANDS)
        return stacked.multiply(1.0), None
    except Exception:
        logger.exception("get_sentinel1_image_for_date failed")
        return None, "No Sentinel-1 SAR data available for this date window in Google Earth Engine."


def get_s2_image_for_date(geom, date_str: str):
    """
    Get Sentinel-2 image for the given geometry and date.
    Prefers the requested date with cloud < 1%. If none, searches nearby dates (±30 days)
    for a scene with cloud < 1% and uses the date closest to the requested one.
    Returns (image, None) if data exists, or (None, error_message) if no data or invalid date.
    """
    ok, err = validate_and_parse_date(date_str)
    if not ok:
        return None, err
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        exact_end = (dt + timedelta(days=1)).strftime("%Y-%m-%d")
        collection_id = _pick_collection_id(date_str, exact_end)

        collection_exact = _get_collection_for_date_range(
            collection_id, geom, date_str, exact_end, max_cloud=1
        )
        count_exact = collection_exact.size().getInfo()
        if count_exact > 0:
            image = collection_exact.median().clip(geom).multiply(0.0001)
            return image, None

        start_near = (dt - timedelta(days=NEARBY_DAYS_WINDOW)).strftime("%Y-%m-%d")
        end_near = (dt + timedelta(days=NEARBY_DAYS_WINDOW + 1)).strftime("%Y-%m-%d")
        collection_id_near = _pick_collection_id(start_near, end_near)
        collection_near = _get_collection_for_date_range(
            collection_id_near, geom, start_near, end_near, max_cloud=1
        )
        count_near = collection_near.size().getInfo()
        if count_near == 0:
            msg = (
                "No Sentinel-2 image with cloud cover < 1%% for this location on the requested date "
                "or within ±%d days. Try a different date or region."
            ) % NEARBY_DAYS_WINDOW
            logger.warning("get_s2_image_for_date: no clear image for date=%s (exact or ±%dd)", date_str, NEARBY_DAYS_WINDOW)
            return None, msg

        target_millis = ee.Date(date_str).millis()

        def add_diff(img):
            t = ee.Number(img.get("system:time_start"))
            return img.set("_diff", t.subtract(target_millis).abs())

        sorted_coll = collection_near.map(add_diff).sort("_diff")
        closest = sorted_coll.first()
        image = closest.clip(geom).multiply(0.0001)
        logger.debug(
            "get_s2_image_for_date: requested %s had no cloud<1%% scene; used nearest clear date in ±%dd",
            date_str,
            NEARBY_DAYS_WINDOW,
        )
        return image, None
    except Exception:
        logger.exception("get_s2_image_for_date failed")
        return None, "Invalid date. No Sentinel-2 data available for this date in Google Earth Engine."


def get_s2_image_for_year(geom, year: int):
    """
    Get a yearly Sentinel-2 composite (median) for the given geometry and year.
    Returns (ee_image, None) on success or (None, error_message) if invalid year/no data.
    """
    try:
        year = int(year)
    except Exception:
        return None, "Invalid year. Use YYYY (e.g. 2018)."
    if year < 2015 or year > datetime.utcnow().year:
        return None, "Invalid year. Use a year between 2015 and the current year."

    start_date = f"{year}-01-01"
    end_date = f"{year+1}-01-01"
    collection_id = _pick_collection_id(start_date, end_date)
    try:
        collection = _get_collection_for_date_range(
            collection_id, geom, start_date, end_date, max_cloud=YEARLY_MAX_CLOUD
        )
        count = collection.size().getInfo()
        if count == 0:
            return None, f"No Sentinel-2 data available for year {year} in this AOI."
        image = collection.median().clip(geom).multiply(0.0001)
        return image, None
    except Exception:
        logger.exception("get_s2_image_for_year failed")
        return None, f"No Sentinel-2 data available for year {year} in this AOI."


def get_image_for_date_by_source(source: str, geom, date_str: str):
    """
    General image selector by source string.
    Supported: 'sentinel2', 'landsat9', 'sentinel1'.
    'planet' is stubbed and must be implemented locally with PLANET_API_KEY.
    """
    src = (source or "sentinel2").lower()
    if src in ("sentinel2", "s2"):
        return get_s2_image_for_date(geom, date_str)
    if src in ("landsat9", "landsat-9", "l9"):
        return get_landsat9_image_for_date(geom, date_str)
    if src in ("sentinel1", "sentinel-1", "s1", "sar"):
        return get_sentinel1_image_for_date(geom, date_str)
    if src == "planet":
        return None, "Planet data source is not fully configured. Please implement fetch_planet_image() with your PLANET_API_KEY."
    return None, f"Unknown data source: {source}"


def mask_s2_sr_clouds_qa60(img: ee.Image) -> ee.Image:
    """Mask clouds/cirrus using QA60 bits (S2_SR only)."""
    qa = img.select("QA60")
    cloud_bit = 1 << 10
    cirrus_bit = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit).eq(0).And(qa.bitwiseAnd(cirrus_bit).eq(0))
    return img.updateMask(mask)


def mask_s2_sr_harmonized_clouds(img: ee.Image) -> ee.Image:
    """Mask clouds using MSK_CLDPRB / SCL (S2_SR_HARMONIZED)."""
    cld_prb = img.select("MSK_CLDPRB")
    cloud_mask = cld_prb.lte(20)
    scl = img.select("SCL")
    scl_clear = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
    mask = cloud_mask.And(scl_clear)
    return img.updateMask(mask)


def get_s2_masked_composite_for_range(
    geom: ee.Geometry,
    start_date: str,
    end_date: str,
    max_cloud: int = 30,
    apply_cloud_mask: bool = True,
) -> ee.Image:
    """Return a median Sentinel-2 composite for [start_date, end_date)."""
    collection_id = _pick_collection_id(start_date, end_date)
    if collection_id == "COPERNICUS/S2_SR_HARMONIZED":
        base = (
            ee.ImageCollection(collection_id)
            .filterBounds(geom)
            .filterMetadata("CLOUDY_PIXEL_PERCENTAGE", "less_than", max_cloud)
            .filterDate(start_date, end_date)
        )
        collection = base.map(mask_s2_sr_harmonized_clouds) if apply_cloud_mask else base
        collection = collection.select(S2_BANDS)
    else:
        base = (
            ee.ImageCollection(collection_id)
            .filterBounds(geom)
            .filterMetadata("CLOUDY_PIXEL_PERCENTAGE", "less_than", max_cloud)
            .filterDate(start_date, end_date)
        )
        collection = base.map(mask_s2_sr_clouds_qa60) if apply_cloud_mask else base
        collection = collection.select(S2_BANDS)
    count = int(collection.size().getInfo())
    if count == 0:
        raise ValueError(
            f"No Sentinel-2 data available for {start_date} to {end_date} "
            f"with cloud threshold <= {max_cloud}% for this AOI."
        )
    return collection.median().clip(geom).multiply(0.0001)
