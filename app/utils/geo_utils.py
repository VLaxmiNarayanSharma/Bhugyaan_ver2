"""Geometry and date helpers for AOIs."""
from __future__ import annotations

import re
from datetime import datetime

import ee


def normalize_class_label(label):
    """
    Map raw classifier output labels to canonical class names
    used in the legend, change statistics, and spatial stats.
    """
    s = str(label).strip()
    key = s.lower()
    mapping = {
        "water": "Water",
        "waterbody": "Water",
        "water body": "Water",
        "builtup": "BuiltUp",
        "built_up": "BuiltUp",
        "built-up": "BuiltUp",
        "urban": "BuiltUp",
        "settlement": "BuiltUp",
        "vegetation": "Vegetation",
        "forest": "Vegetation",
        "barrenland": "BarrenLand",
        "barren": "BarrenLand",
        "wasteland": "BarrenLand",
        "agricultural": "Agricultural",
        "cropland": "Agricultural",
        "crop": "Agricultural",
    }
    return mapping.get(key, s)


def validate_and_parse_date(date_str: str):
    """Validate YYYY-MM-DD format and return (True, None) or (False, error_message)."""
    if not date_str or not date_str.strip():
        return False, "Date is required."
    date_str = date_str.strip()
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
        return False, "Invalid date format. Use YYYY-MM-DD."
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        if dt.year < 2015 or dt.date() > datetime.utcnow().date():
            return False, "Invalid date. Use a date between 2015 and today."
        return True, None
    except ValueError:
        return False, "Invalid date."


def build_geometry(geojson: dict) -> ee.Geometry:
    """
    Converts a GeoJSON dictionary to an Earth Engine geometry.
    """
    if geojson.get("type") == "FeatureCollection":
        features = geojson.get("features", [])
        if not features:
            raise Exception("Empty FeatureCollection")
        if len(features) == 1:
            return ee.Geometry(features[0]["geometry"])
        else:
            polygons = []
            for feature in features:
                geom_type = feature.get("geometry", {}).get("type")
                coords = feature.get("geometry", {}).get("coordinates")
                if geom_type == "Polygon":
                    polygons.append([coords])
                elif geom_type == "MultiPolygon":
                    polygons.append(coords)
            if len(polygons) == 1:
                return ee.Geometry.Polygon(polygons[0])
            elif polygons:
                return ee.Geometry.MultiPolygon(polygons)
            else:
                raise Exception("No valid polygon geometries found")
    elif geojson.get("type") == "Feature":
        return ee.Geometry(geojson.get("geometry"))
    elif geojson.get("type") in ["Polygon", "MultiPolygon"]:
        return ee.Geometry(geojson)
    else:
        raise Exception("Unsupported GeoJSON type: " + str(geojson.get("type")))
