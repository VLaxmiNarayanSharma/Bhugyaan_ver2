import requests

def test_change_detection(base_url):
    payload = {
        "aoi": {
            "type": "Polygon",
            "coordinates": [[[75.0, 24.0], [75.1, 24.0], [75.1, 24.1], [75.0, 24.1], [75.0, 24.0]]]
        },
        "year1": 2020,
        "year2": 2024
    }

    res = requests.post(f"{base_url}/lulc-change", json=payload)

    assert res.status_code == 200
    assert "change_map" in res.json()