"""
Image classification API.

Accepts image uploads and returns the predicted CIFAR-10 class.

Endpoints:
    GET  /health               readiness probe
    GET  /classes              list all supported classes
    GET  /classes/{name}       details for a specific class
    POST /predict              classify a single image
    POST /predict/batch        classify multiple images at once
"""

import io
import os
from contextlib import asynccontextmanager

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image
from torchvision import transforms

from src.model import ImageClassifier

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "model.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = None

TRANSFORM = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

CLASS_DESCRIPTIONS = {
    "airplane": "Fixed-wing aircraft",
    "automobile": "Car or similar road vehicle",
    "bird": "Any species of bird",
    "cat": "Domestic cat",
    "deer": "Deer or similar deer-like animal",
    "dog": "Domestic dog",
    "frog": "Frog or toad",
    "horse": "Horse or similar equine",
    "ship": "Large watercraft",
    "truck": "Large road vehicle for cargo",
}


def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError("Model not found. Run src/train.py first.")
    m = ImageClassifier()
    m.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    m.eval()
    model = m
    print(f"Model loaded on {DEVICE}.")


def classify_image(image_bytes: bytes, top_k: int, min_confidence: float) -> dict:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]

    top_prob, top_idx = probs.topk(top_k)
    top_predictions = [
        {"class": ImageClassifier.CLASSES[i], "probability": round(p, 4)}
        for i, p in zip(top_idx.tolist(), top_prob.tolist())
    ]

    top_class = top_predictions[0]["class"]
    top_confidence = top_predictions[0]["probability"]
    confident = top_confidence >= min_confidence

    return {
        "prediction": top_class if confident else "uncertain",
        "confidence": top_confidence,
        "confident": confident,
        "top_k": top_predictions,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="Image Classifier API",
    description="Classifies images into CIFAR-10 categories using a custom CNN.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {"message": "Image Classifier API is running. Visit /docs for usage."}


@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return {"status": "ok", "device": DEVICE}


@app.get("/classes")
def get_classes():
    return {"classes": ImageClassifier.CLASSES}


@app.get("/classes/{name}")
def get_class_info(name: str):
    if name not in CLASS_DESCRIPTIONS:
        raise HTTPException(status_code=404, detail=f"Class '{name}' not found.")
    return {"class": name, "description": CLASS_DESCRIPTIONS[name]}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    top_k: int = Query(default=3, ge=1, le=10, description="Number of top predictions to return"),
    min_confidence: float = Query(
        default=0.0, ge=0.0, le=1.0, description="Minimum confidence to return a prediction"
    ),
):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    contents = await file.read()
    return classify_image(contents, top_k, min_confidence)


@app.post("/predict/batch")
async def predict_batch(
    files: list[UploadFile] = File(...),
    top_k: int = Query(default=3, ge=1, le=10),
    min_confidence: float = Query(default=0.0, ge=0.0, le=1.0),
):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    results = []
    for file in files:
        contents = await file.read()
        result = classify_image(contents, top_k, min_confidence)
        results.append({"filename": file.filename, **result})

    return {
        "total": len(results),
        "results": results,
    }
