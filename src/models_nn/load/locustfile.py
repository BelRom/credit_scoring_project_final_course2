from __future__ import annotations
import random
from locust import HttpUser, task, between

BASE = {
    "LIMIT_BAL": 20000,
    "SEX": 2,
    "EDUCATION": 2,
    "MARRIAGE": 1,
    "AGE": 24,
}


class PredictUser(HttpUser):
    wait_time = between(0.05, 0.2)

    @task
    def predict(self):
        x = dict(BASE)
        x["LIMIT_BAL"] = random.choice([10000, 20000, 50000, 80000, 120000])
        self.client.post("/predict", json={"features": x}, timeout=10)
