"""Earth Engine / service-account connection."""
import json

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from google.oauth2 import service_account

from app.dependencies.gee_init import initialize_ee

router = APIRouter(tags=["auth"])


@router.post("/gee-connect", response_class=JSONResponse)
async def gee_connect(
    project_id: str = Form(...),
    service_account_json: UploadFile = File(...),
):
    """
    Connect a GEE/GCP project for Earth Engine + Drive using a service account JSON.
    This updates the active EE project for subsequent requests (process-wide).
    """
    if not project_id or not project_id.strip():
        raise HTTPException(status_code=400, detail="project_id is required")

    try:
        raw = await service_account_json.read()
        info = json.loads(raw.decode("utf-8"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid service account JSON: {e}")

    scopes = [
        "https://www.googleapis.com/auth/cloud-platform",
        "https://www.googleapis.com/auth/drive",
    ]
    try:
        creds = service_account.Credentials.from_service_account_info(info, scopes=scopes)
        initialize_ee(project=project_id.strip(), credentials=creds)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"EE initialization failed: {e}")

    return JSONResponse({"status": "connected", "project_id": project_id.strip()})
