import requests

def test_timeseries(base_url):
    payload = {
        "aoi": {
            "type": "Polygon",
            "coordinates": [[[75.0, 24.0], [75.1, 24.0], [75.1, 24.1], [75.0, 24.1], [75.0, 24.0]]]
        },
        "start_year": 2020,
        "end_year": 2023,
        "interval": "yearly",
        "scale_m": 60
    }

    res = requests.post(f"{base_url}/lulc-timeseries", json=payload)

    assert res.status_code == 200
    assert "years" in res.json()
    assert "maps" in res.json()