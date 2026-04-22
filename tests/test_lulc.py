import requests


def test_generate_lulc_valid(base_url):
    payload = {
        "geojson": {
            "type": "Polygon",
            "coordinates": [
                [[75.0, 24.0], [75.1, 24.0], [75.1, 24.1], [75.0, 24.1], [75.0, 24.0]]
            ]
        },
        "method": "rf",              # or whatever your API supports
        "date": "2023-01-01",        # correct format (string date)
        "location_name": "test_area"
    }

    res = requests.post(f"{base_url}/generate-lulc", json=payload)

    print(res.status_code, res.text)

    assert res.status_code == 200
    assert "map_url" in res.json()


def test_generate_lulc_invalid_aoi(base_url):
    payload = {
        "geojson": {},
        "method": "rf",
        "date": "2023-01-01",
        "location_name": "test_area"
    }

    res = requests.post(f"{base_url}/generate-lulc", json=payload)

    print(res.status_code, res.text)

    assert res.status_code != 200