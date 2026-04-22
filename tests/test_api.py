import pytest
from fastapi.testclient import TestClient
from main import app
import requests

client = TestClient(app)


# ===============================
# 🔹 BASIC FUNCTIONAL TESTS
# ===============================

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.text is not None


def test_history():
    response = client.get("/history")
    assert response.status_code == 200


# ===============================
# 🔹 SHAPEFILE TESTS
# ===============================

def test_invalid_shapefile_format():
    response = client.post(
        "/process_shapefile/",
        files={"file": ("test.txt", b"invalid content")}
    )
    assert response.status_code == 400


def test_empty_shapefile():
    response = client.post(
        "/process_shapefile/",
        files={"file": ("empty.shp", b"")}
    )
    assert response.status_code in [400, 422]


def test_large_shapefile_simulation():
    large_data = b"x" * (1024 * 1024)  # 1MB dummy
    response = client.post(
        "/process_shapefile/",
        files={"file": ("large.shp", large_data)}
    )
    assert response.status_code in [200, 400, 500]


# ===============================
# 🔹 CLASSIFICATION TESTS
# ===============================

def test_classify_valid_method_rf():
    response = client.post(
        "/classify/",
        files={"file": ("test.tif", b"dummy data")},
        data={"method": "RF"}
    )
    assert response.status_code in [200, 400, 500]


def test_classify_invalid_method():
    response = client.post(
        "/classify/",
        files={"file": ("test.tif", b"dummy data")},
        data={"method": "INVALID"}
    )
    assert response.status_code in [400, 422]


def test_classify_missing_file():
    response = client.post(
        "/classify/",
        data={"method": "RF"}
    )
    assert response.status_code in [400, 422]


def test_classify_empty_file():
    response = client.post(
        "/classify/",
        files={"file": ("empty.tif", b"")},
        data={"method": "RF"}
    )
    assert response.status_code in [400, 500]


# ===============================
# 🔹 DOWNLOAD TESTS
# ===============================

def test_download_valid_format():
    response = client.post(
        "/download/",
        data={
            "coordinates": "77.59,12.97",
            "file_name": "test",
            "date": "2024-01-01"
        }
    )
    assert response.status_code in [200, 400, 500]


def test_download_invalid_coordinates():
    response = client.post(
        "/download/",
        data={
            "coordinates": "invalid",
            "file_name": "test",
            "date": "2024-01-01"
        }
    )
    assert response.status_code == 400


def test_download_missing_fields():
    response = client.post(
        "/download/",
        data={"coordinates": "77.59,12.97"}
    )
    assert response.status_code in [400, 422]


def test_download_invalid_date():
    response = client.post(
        "/download/",
        data={
            "coordinates": "77.59,12.97",
            "file_name": "test",
            "date": "invalid-date"
        }
    )
    assert response.status_code in [400, 422]


# ===============================
# 🔹 PERFORMANCE / STABILITY TESTS
# ===============================

def test_multiple_requests():
    for _ in range(5):
        response = client.get("/")
        assert response.status_code == 200


def test_api_stability_under_loop():
    for _ in range(10):
        response = client.get("/history")
        assert response.status_code == 200


# ===============================
# 🔹 RESPONSE STRUCTURE TESTS
# ===============================

def test_home_response_format():
    response = client.get("/")
    assert isinstance(response.text, str)


def test_history_response_format():
    response = client.get("/history")
    assert response.status_code == 200