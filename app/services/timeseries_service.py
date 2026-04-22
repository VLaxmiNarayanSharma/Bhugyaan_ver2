"""Time-series windowing and method string normalization."""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List


def step_windows(start_year: int, end_year: int, interval: str) -> List[Dict[str, str]]:
    if start_year > end_year:
        raise ValueError("start_year must be <= end_year")
    if interval not in ("yearly", "quarterly"):
        raise ValueError("interval must be 'yearly' or 'quarterly'")
    if start_year < 2015 or end_year > datetime.utcnow().year:
        raise ValueError("start_year/end_year out of supported range (2015..current year).")

    windows: List[Dict[str, str]] = []
    if interval == "yearly":
        for y in range(start_year, end_year + 1):
            windows.append(
                {
                    "label": str(y),
                    "start_date": f"{y}-01-01",
                    "end_date": f"{y + 1}-01-01",
                }
            )
        return windows

    for y in range(start_year, end_year + 1):
        for q in (1, 2, 3, 4):
            start_month = (q - 1) * 3 + 1
            end_month = start_month + 3
            start_date = f"{y}-{start_month:02d}-01"
            if end_month <= 12:
                end_date = f"{y}-{end_month:02d}-01"
            else:
                end_date = f"{y + 1}-01-01"
            windows.append(
                {
                    "label": f"{y}-Q{q}",
                    "start_date": start_date,
                    "end_date": end_date,
                }
            )
    return windows


def resolve_method_default(method: str) -> str:
    m = (method or "").strip()
    if m.endswith(".pkl"):
        m = m[: -len(".pkl")]
    return m
