"""Shared Jinja2 templates directory."""
from fastapi.templating import Jinja2Templates

from app.config.settings import TEMPLATES_DIR

templates = Jinja2Templates(directory=TEMPLATES_DIR)
