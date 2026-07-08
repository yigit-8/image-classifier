import io
import os

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from src.model import ImageClassifier
from src.serve import MODEL_PATH, app


def create_dummy_model():
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    m = ImageClassifier()
    torch.save(m.state_dict(), MODEL_PATH)


def make_image_bytes(color: tuple = (128, 64, 32), size: tuple = (32, 32)) -> bytes:
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def client():
    create_dummy_model()
    with TestClient(app) as c:
        yield c


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_classes_returns_10(client):
    response = client.get("/classes")
    assert response.status_code == 200
    assert len(response.json()["classes"]) == 10


def test_class_info(client):
    response = client.get("/classes/dog")
    assert response.status_code == 200
    assert response.json()["class"] == "dog"
    assert "description" in response.json()


def test_class_info_not_found(client):
    response = client.get("/classes/unicorn")
    assert response.status_code == 404


def test_predict_returns_valid_class(client):
    image_bytes = make_image_bytes()
    response = client.post("/predict", files={"file": ("test.jpg", image_bytes, "image/jpeg")})
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "confidence" in data
    assert "top_k" in data
    assert len(data["top_k"]) == 3


def test_predict_custom_top_k(client):
    image_bytes = make_image_bytes()
    response = client.post(
        "/predict?top_k=5",
        files={"file": ("test.jpg", image_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    assert len(response.json()["top_k"]) == 5


def test_predict_high_min_confidence_returns_uncertain(client):
    image_bytes = make_image_bytes()
    response = client.post(
        "/predict?min_confidence=0.99",
        files={"file": ("test.jpg", image_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    assert response.json()["prediction"] == "uncertain"


def test_predict_batch(client):
    image_bytes = make_image_bytes()
    response = client.post(
        "/predict/batch",
        files=[
            ("files", ("img1.jpg", io.BytesIO(image_bytes), "image/jpeg")),
            ("files", ("img2.jpg", io.BytesIO(image_bytes), "image/jpeg")),
        ],
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert len(data["results"]) == 2


def test_predict_invalid_file_returns_400(client):
    response = client.post("/predict", files={"file": ("test.txt", b"not an image", "text/plain")})
    assert response.status_code == 400
