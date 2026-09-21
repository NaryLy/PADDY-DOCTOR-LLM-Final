"""
FastAPI backend for the rice disease detection app.

Endpoints:
  GET  /api/health              -- liveness check + model info
  POST /api/predict             -- upload an image, get disease prediction + treatment advice
  GET  /api/history             -- list past predictions (most recent first)
  DELETE /api/history           -- clear history
"""
import io
import json
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from torchvision import models, transforms

from treatment_db import get_treatment

BASE_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.join(BASE_DIR, "..")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "finetuned_best.pt")
DB_PATH = os.path.join(BASE_DIR, "history.db")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploaded_images")
os.makedirs(UPLOAD_DIR, exist_ok=True)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

app = FastAPI(title="Rice Disease Doctor API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

_eval_transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

_model = None
_class_names = None
_checkpoint_meta = {}


def load_model():
    global _model, _class_names, _checkpoint_meta
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}. Run training/train.py --mode finetuned first."
        )
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    class_names = checkpoint["class_names"]

    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    _model = model
    _class_names = class_names
    _checkpoint_meta = {
        "mode": checkpoint.get("mode"),
        "val_acc": checkpoint.get("val_acc"),
        "epoch": checkpoint.get("epoch"),
        "trainable_params": checkpoint.get("trainable_params"),
    }
    return model, class_names


@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id TEXT PRIMARY KEY,
                filename TEXT,
                image_path TEXT,
                predicted_label TEXT,
                confidence REAL,
                top3_json TEXT,
                created_at REAL
            )
        """)


@app.on_event("startup")
def on_startup():
    init_db()
    try:
        load_model()
        print(f"Model loaded: {_checkpoint_meta}")
    except FileNotFoundError as e:
        # Server can still start (e.g. for frontend dev) but /api/predict will 503 until trained.
        print(f"WARNING: {e}")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "classes": _class_names,
        "checkpoint": _checkpoint_meta,
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet -- train the finetuned model first.")

    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file.")

    x = _eval_transform(image).unsqueeze(0)
    with torch.no_grad():
        logits = _model(x)
        probs = F.softmax(logits, dim=1)[0]

    top_probs, top_idx = torch.topk(probs, k=min(3, len(_class_names)))
    top3 = [
        {"label": _class_names[i], "confidence": round(p.item(), 4)}
        for p, i in zip(top_probs, top_idx)
    ]
    best_label = top3[0]["label"]
    best_conf = top3[0]["confidence"]
    treatment = get_treatment(best_label)

    # Save the uploaded image + a history row so the app has a record log.
    record_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename or "")[1] or ".jpg"
    saved_name = f"{record_id}{ext}"
    saved_path = os.path.join(UPLOAD_DIR, saved_name)
    with open(saved_path, "wb") as f:
        f.write(raw)

    with get_db() as conn:
        conn.execute(
            "INSERT INTO history (id, filename, image_path, predicted_label, confidence, top3_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (record_id, file.filename, f"/uploads/{saved_name}", best_label, best_conf, json.dumps(top3), time.time()),
        )

    return {
        "id": record_id,
        "predicted_label": best_label,
        "confidence": best_conf,
        "top3": top3,
        "treatment": treatment,
        "image_url": f"/uploads/{saved_name}",
    }


@app.get("/api/history")
def history(limit: int = 50):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM history ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    result = []
    for r in rows:
        treatment = get_treatment(r["predicted_label"])
        result.append({
            "id": r["id"],
            "filename": r["filename"],
            "image_url": r["image_path"],
            "predicted_label": r["predicted_label"],
            "confidence": r["confidence"],
            "top3": json.loads(r["top3_json"]),
            "created_at": r["created_at"],
            "display_name_en": treatment["display_name_en"] if treatment else r["predicted_label"],
            "display_name_km": treatment["display_name_km"] if treatment else "",
        })
    return result


@app.delete("/api/history")
def clear_history():
    with get_db() as conn:
        conn.execute("DELETE FROM history")
    return {"status": "cleared"}


# Mounted LAST so it doesn't shadow the /api/* and /uploads routes above --
# Starlette matches routes in registration order, and a "/" mount would otherwise
# catch every request first.
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
