"""
FastAPI application entry: app instance, static files, and router registration only.
"""
import logging

import colorama
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

import app.bootstrap_subprocess  # noqa: F401 — patch gcloud before EE/oauth
from app.api import routes_auth, routes_change, routes_download, routes_lulc, routes_timeseries
from app.config.settings import STATIC_DIR

colorama.init()
logging.basicConfig(level=logging.INFO)

app = FastAPI()

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

app.include_router(routes_lulc.router)
app.include_router(routes_change.router)
app.include_router(routes_timeseries.router)
app.include_router(routes_download.router)
app.include_router(routes_auth.router)
