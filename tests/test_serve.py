import io
import os

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

import src.serve as serve_module
from src.model import ImageClassifier
from src.serve import MAX_BATCH_FILES, MAX_UPLOAD_BYTES, app


def create_dummy_model(model_path):
    """Write a bare state_dict checkpoint to ``model_path``."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    m = ImageClassifier()
    torch.save(m.state_dict(), model_path)
    return str(model_path)


def make_image_bytes(color: tuple = (128, 64, 32), size: tuple = (32, 32)) -> bytes:
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """A TestClient serving a throwaway checkpoint written under tmp_path.

    ``MODEL_PATH`` is redirected so the suite never writes ``data/model.pt``
    into the repository.
    """
    model_path = create_dummy_model(tmp_path_factory.mktemp("model") / "model.pt")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(serve_module, "MODEL_PATH", model_path)
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


def test_predict_batch_over_limit_returns_422(client):
    image_bytes = make_image_bytes()
    files = [
        ("files", (f"img{i}.jpg", io.BytesIO(image_bytes), "image/jpeg"))
        for i in range(MAX_BATCH_FILES + 1)
    ]
    response = client.post("/predict/batch", files=files)
    assert response.status_code == 422


def test_predict_oversized_upload_returns_413(client):
    oversized = b"\0" * (MAX_UPLOAD_BYTES + 1)
    response = client.post("/predict", files={"file": ("big.jpg", oversized, "image/jpeg")})
    assert response.status_code == 413


def test_load_model_accepts_bundled_checkpoint(client, tmp_path, monkeypatch):
    """train.py saves weights + params + seed + metrics; serving must read it."""
    bundle_path = tmp_path / "model.pt"
    torch.save(
        {
            "state_dict": ImageClassifier().state_dict(),
            "params": {"epochs": 1, "lr": 0.001, "batch_size": 128, "seed": 42},
            "seed": 42,
            "metrics": {"best_val_acc": 0.5},
            "classes": ImageClassifier.CLASSES,
        },
        bundle_path,
    )
    monkeypatch.setattr(serve_module, "MODEL_PATH", str(bundle_path))
    try:
        serve_module.load_model()
        assert serve_module.model is not None
        assert serve_module.model_metadata["seed"] == 42
        assert serve_module.model_metadata["metrics"]["best_val_acc"] == 0.5
    finally:
        # Restores MODEL_PATH to the ``client`` fixture's tmp checkpoint, which
        # is why this test depends on that fixture: the reload needs a file to
        # read even when this test runs on its own.
        monkeypatch.undo()
        serve_module.load_model()
