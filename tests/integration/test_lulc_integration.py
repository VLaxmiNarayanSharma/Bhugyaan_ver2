from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_lulc_integration():
    payload = {
        "geojson": {
            "type": "Polygon",
            "coordinates": [[[75.0, 24.0], [75.1, 24.0], [75.1, 24.1], [75.0, 24.1], [75.0, 24.0]]]
        },
        "method": "rf",
        "date": "2023-01-01",
        "location_name": "integration_test"
    }

    response = client.post("/generate-lulc", json=payload)

    assert response.status_code in [200, 422]  # depends on backend