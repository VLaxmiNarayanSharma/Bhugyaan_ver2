"""Pydantic request bodies for JSON API endpoints."""
from typing import Any, Dict, Literal

from pydantic import BaseModel


class LulcTimeSeriesRequest(BaseModel):
    aoi: Dict[str, Any]
    start_year: int
    end_year: int
    interval: Literal["yearly", "quarterly"] = "yearly"
    method: str = "Random-Forest-Random-search"
    scale_m: int = 30


class LulcEventChangeRequest(BaseModel):
    aoi: Dict[str, Any]
    event_date: str
    window_days: int = 7
    method: str = "Random-Forest-Random-search"
    scale_m: int = 30
