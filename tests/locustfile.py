from locust import HttpUser, task, between
import random
import requests



class BhuGyaanUser(HttpUser):
    wait_time = between(1, 3)

    # ===============================
    # 🔹 BASIC ENDPOINT TESTS
    # ===============================

    @task(4)
    def test_home(self):
        self.client.get("/")

    @task(3)
    def test_history(self):
        self.client.get("/history")

    # ===============================
    # 🔹 DOWNLOAD API TESTS
    # ===============================

    @task(2)
    def test_download_valid(self):
        self.client.post(
            "/download/",
            data={
                "coordinates": "[[77.1,28.6],[77.2,28.6],[77.2,28.7],[77.1,28.7]]",
                "file_name": "test_area",
                "date": "2024-01-01"
            }
        )

    @task(1)
    def test_download_invalid_coordinates(self):
        self.client.post(
            "/download/",
            data={
                "coordinates": "invalid",
                "file_name": "test",
                "date": "2024-01-01"
            }
        )

    @task(1)
    def test_download_missing_fields(self):
        self.client.post(
            "/download/",
            data={
                "coordinates": "[[77.1,28.6]]"
            }
        )

    @task(1)
    def test_download_invalid_date(self):
        self.client.post(
            "/download/",
            data={
                "coordinates": "[[77.1,28.6],[77.2,28.6]]",
                "file_name": "test",
                "date": "wrong-date"
            }
        )

    # ===============================
    # 🔹 CLASSIFICATION API TESTS
    # ===============================

    @task(2)
    def test_classify_valid(self):
        self.client.post(
            "/classify/",
            files={"file": ("test.tif", b"dummy data")},
            data={"method": "RF"}
        )

    @task(1)
    def test_classify_invalid_method(self):
        self.client.post(
            "/classify/",
            files={"file": ("test.tif", b"dummy data")},
            data={"method": "INVALID"}
        )

    @task(1)
    def test_classify_empty_file(self):
        self.client.post(
            "/classify/",
            files={"file": ("empty.tif", b"")},
            data={"method": "RF"}
        )

    # ===============================
    # 🔹 RANDOMIZED LOAD TEST
    # ===============================

    @task(2)
    def random_requests(self):
        endpoints = [
            ("/", "GET"),
            ("/history", "GET"),
        ]

        endpoint, method = random.choice(endpoints)

        if method == "GET":
            self.client.get(endpoint)

    # ===============================
    # 🔹 STRESS TEST SIMULATION
    # ===============================

    @task(1)
    def burst_requests(self):
        for _ in range(3):
            self.client.get("/")

    # ===============================
    # 🔹 RESPONSE VALIDATION
    # ===============================

    @task(1)
    def validate_home_response(self):
        response = self.client.get("/")
        assert response.status_code == 200
