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

from src.model import IMAGE_SIZE, NORM_MEAN, NORM_STD, ImageClassifier

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "model.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Uploads are read fully into memory, so they need a ceiling.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_BATCH_FILES = 20

model = None
model_metadata = {}

# Must match the eval-time transform in train.py; the constants are shared.
TRANSFORM = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
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
    global model, model_metadata
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError("Model not found. Run src/train.py first.")

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)

    # train.py writes a bundle (weights + params + seed + metrics). Older
    # checkpoints are a bare state_dict, so both are accepted.
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        model_metadata = {
            "params": checkpoint.get("params", {}),
            "seed": checkpoint.get("seed"),
            "metrics": checkpoint.get("metrics", {}),
        }
    else:
        state_dict = checkpoint
        model_metadata = {}

    m = ImageClassifier()
    m.load_state_dict(state_dict)
    m.eval()
    model = m
    print(f"Model loaded on {DEVICE}. Metadata: {model_metadata or 'none (bare state_dict)'}")


def read_upload(file: UploadFile) -> bytes:
    """Read an upload, refusing anything over the size ceiling."""
    contents = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.",
        )
    return contents


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


# /predict and /predict/batch are plain `def`, not `async def`: Torch inference
# is synchronous and CPU-bound, so running it in the event loop would block
# every other request. FastAPI hands sync endpoints to a threadpool instead,
# which also lets several batch requests make progress at once.
@app.post("/predict")
def predict(
    file: UploadFile = File(...),
    top_k: int = Query(default=3, ge=1, le=10, description="Number of top predictions to return"),
    min_confidence: float = Query(
        default=0.0, ge=0.0, le=1.0, description="Minimum confidence to return a prediction"
    ),
):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    contents = read_upload(file)
    return classify_image(contents, top_k, min_confidence)


@app.post("/predict/batch")
def predict_batch(
    files: list[UploadFile] = File(...),
    top_k: int = Query(default=3, ge=1, le=10),
    min_confidence: float = Query(default=0.0, ge=0.0, le=1.0),
):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")
    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(
            status_code=422,
            detail=f"Batch of {len(files)} exceeds the limit of {MAX_BATCH_FILES} files.",
        )

    results = []
    for file in files:
        contents = read_upload(file)
        result = classify_image(contents, top_k, min_confidence)
        results.append({"filename": file.filename, **result})

    return {
        "total": len(results),
        "results": results,
    }
