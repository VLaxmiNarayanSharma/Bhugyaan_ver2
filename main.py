"""
Root shim: preserves `uvicorn main:app` and `from main import app` (e.g. test_api.py).
Full application lives under `app/`.
"""
from app.main import app

__all__ = ["app"]

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
