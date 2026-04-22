"""Transition statistics from classified code rasters."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

import numpy as np

from app.config.settings import CLASS_TO_CODE, CLASS_TO_STAT_KEY, CODE_TO_CLASS


def parse_iso_date(date_str: str) -> datetime:
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return dt


def compute_change_stats_from_codes(before_code: np.ndarray,after_code: np.ndarray,valid_mask: np.ndarray,hidden_from_builtup_to_other: bool = True,) -> Dict[str, Any]:
    """
    before_code/after_code: 2D uint8, 0 means invalid.
    valid_mask: 2D bool where both sides have valid pixels.
    """
    frm = before_code[valid_mask].astype(np.uint8)
    to = after_code[valid_mask].astype(np.uint8)
    mask_change = frm != to
    if hidden_from_builtup_to_other:
        mask_hidden = (frm == CLASS_TO_CODE["BuiltUp"]) & (to != CLASS_TO_CODE["BuiltUp"])
        mask_change = mask_change & (~mask_hidden)

    frm2 = frm[mask_change]
    to2 = to[mask_change]

    if frm2.size == 0:
        return {"transitions": [], "area_by_transition_key": {}}

    pairs = np.stack([frm2, to2], axis=1)
    uniq_pairs, counts = np.unique(pairs, axis=0, return_counts=True)

    transitions = []
    area_by_transition_key: Dict[str, float] = {}
    for (fcode, tcode), cnt in zip(uniq_pairs.tolist(), counts.tolist()):
        fc = int(fcode)
        tc = int(tcode)
        from_class = CODE_TO_CLASS.get(fc, None)
        to_class = CODE_TO_CLASS.get(tc, None)
        if not from_class or not to_class:
            continue
        transitions.append({"from": from_class, "to": to_class, "pixels": int(cnt)})
        key = f"{CLASS_TO_STAT_KEY[from_class]}_to_{CLASS_TO_STAT_KEY[to_class]}"
        area_by_transition_key[key] = float(cnt)
    return {"transitions": transitions, "area_by_transition_key": area_by_transition_key}


def apply_pixel_area_to_change_stats(change_stats: Dict[str, Any], pixel_area_m2: float) -> Dict[str, Any]:
    pixel_area_km2 = float(pixel_area_m2) / 1e6
    for t in change_stats["transitions"]:
        t["area_km2"] = float(t["pixels"]) * pixel_area_km2
    out: Dict[str, float] = {}
    for k, v in change_stats.get("area_by_transition_key", {}).items():
        out[k] = float(v) * pixel_area_km2
    change_stats["area_by_transition_key"] = out
    return change_stats
