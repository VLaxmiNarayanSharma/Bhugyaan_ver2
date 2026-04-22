import requests

def test_large_payload(base_url):
    payload = {
        "aoi": {
            "type": "Polygon",
            "coordinates": [[[i, i] for i in range(10000)]]
        }
    }

    res = requests.post(f"{base_url}/generate-lulc", json=payload)

    assert res.status_code != 200
    
def test_invalid_endpoint(base_url):
    res = requests.get(f"{base_url}/invalid-api")
    assert res.status_code == 404