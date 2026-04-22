from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_timeseries_integration():
    payload = {
        "aoi": {
            "type": "Polygon",
            "coordinates": [[[75.0, 24.0], [75.1, 24.0], [75.1, 24.1], [75.0, 24.1], [75.0, 24.0]]]
        },
        "start_year": 2020,
        "end_year": 2022,
        "interval": "yearly"
    }

    res = client.post("/lulc-timeseries", json=payload)

    assert res.status_code in [200, 422]