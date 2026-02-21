"""
==============================================================
FastAPI Inference Server for ELITE AutoML Platform
==============================================================
Endpoints:
  GET  /                    - Health check + links to all endpoints
  GET  /models              - List saved models with full metrics
  GET  /models/best         - Best model details + metrics
  GET  /models/download     - Download best model .pkl file
  GET  /models/{id}/download - Download specific model .pkl file
  GET  /metrics             - Full metrics for every trained model
  POST /train               - Upload CSV to train a new AutoML run
  POST /train/kaggle        - Provide a Kaggle dataset link to train
  POST /predict             - JSON rows → predictions
  POST /predict/csv         - Upload CSV → predictions
  GET  /predict/example     - Example payload + curl command
==============================================================

Run locally:
    uvicorn app:app --host 0.0.0.0 --port 8000

Run in Google Colab (with ngrok):
    from pyngrok import ngrok
    import subprocess, time
    proc = subprocess.Popen(['uvicorn', 'app:app', '--host', '0.0.0.0', '--port', '8000'])
    time.sleep(3)
    public_url = ngrok.connect(8000)
    print(f"API live at: {public_url}")
"""

import os
import io
import re
import json
import pickle
import shutil
import subprocess
import tempfile
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Form, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ==============================================================
# APP SETUP
# ==============================================================

app = FastAPI(
    title="ELITE AutoML Inference API",
    description=(
        "Production-grade API for the ELITE AutoML Platform.\n\n"
        "**Upload a dataset (CSV or Kaggle link)** → train models automatically → "
        "**download the best `.pkl` file** → **run predictions** via JSON or CSV upload."
    ),
    version="2.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = Path("models")
UPLOAD_DIR = Path("uploads")
MODEL_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)


# ==============================================================
# PYDANTIC SCHEMAS
# ==============================================================

class PredictRequest(BaseModel):
    """JSON body for single / batch prediction."""
    model_id: str = "latest"
    data: List[Dict[str, Any]]  # list of row dicts


class PredictResponse(BaseModel):
    model_id: str
    model_name: str
    task: str
    predictions: List[Any]
    count: int


class ModelInfo(BaseModel):
    model_id: str
    model_name: str
    task: str
    cv_score_mean: float
    cv_score_std: float
    test_score: float
    latency_ms: float
    size_mb: float
    final_score: float
    hyperparameters: Dict[str, Any]
    dataset_hash: str
    timestamp: str
    feature_count: int
    train_samples: int


class TrainRequest(BaseModel):
    """Body for /train/kaggle endpoint."""
    kaggle_url: str
    target_column: str


class TrainResponse(BaseModel):
    status: str
    message: str
    best_model_id: Optional[str] = None
    best_model_name: Optional[str] = None
    test_score: Optional[float] = None
    final_score: Optional[float] = None
    pickle_download: Optional[str] = None


# ==============================================================
# HELPERS
# ==============================================================

def _find_latest_model_id() -> Optional[str]:
    """Return model_id of the best (highest final_score) saved model."""
    best_id, best_score = None, -float("inf")
    for path in MODEL_DIR.glob("*_metadata.json"):
        with open(path) as f:
            meta = json.load(f)
        if meta.get("final_score", -999) > best_score:
            best_score = meta["final_score"]
            best_id = meta["model_id"]
    return best_id


def _load_bundle(model_id: str) -> dict:
    """Load model bundle (try pickle first, then joblib)."""
    pkl_path = MODEL_DIR / f"{model_id}.pkl"
    jbl_path = MODEL_DIR / f"{model_id}.joblib"

    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
    elif jbl_path.exists():
        return joblib.load(jbl_path)
    else:
        raise FileNotFoundError(f"No model file found for id: {model_id}")


def _resolve_model_id(model_id: str) -> str:
    """Resolve 'latest' to the actual best model id."""
    if model_id.lower() == "latest":
        resolved = _find_latest_model_id()
        if resolved is None:
            raise HTTPException(404, "No models found in the registry. Train a model first.")
        return resolved
    return model_id


def _preprocess_and_predict(bundle: dict, df: pd.DataFrame) -> np.ndarray:
    """Apply stored preprocessor and run model.predict."""
    model = bundle["model"]
    preprocessor = bundle["preprocessor"]
    feature_cols = bundle["feature_cols"]

    # Normalise column names the same way as training
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Keep only the features the model was trained on
    missing = set(feature_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in input data: {missing}")

    df = df[feature_cols]
    X_processed = preprocessor.transform(df)
    return model.predict(X_processed)


def _get_all_metadata() -> List[dict]:
    """Load all model metadata files."""
    records = []
    for path in MODEL_DIR.glob("*_metadata.json"):
        with open(path) as f:
            records.append(json.load(f))
    records.sort(key=lambda r: r.get("final_score", 0), reverse=True)
    return records


def _download_kaggle_dataset(kaggle_url: str) -> str:
    """
    Download a Kaggle dataset given a URL like:
      https://www.kaggle.com/datasets/<owner>/<dataset-name>
      or just <owner>/<dataset-name>

    Requires `kaggle` CLI to be installed and KAGGLE_USERNAME +
    KAGGLE_KEY env vars (or ~/.kaggle/kaggle.json) to be set.

    Returns path to the first CSV found.
    """
    # Normalise URL → slug  (owner/dataset-name)
    slug = kaggle_url.strip().rstrip("/")
    # Handle full URLs
    match = re.search(r"kaggle\.com/datasets/([^/]+/[^/?#]+)", slug)
    if match:
        slug = match.group(1)
    # Handle competition URLs
    comp_match = re.search(r"kaggle\.com/competitions?/([^/?#]+)", slug)
    if comp_match:
        slug = comp_match.group(1)

    dest = UPLOAD_DIR / slug.replace("/", "_")
    dest.mkdir(parents=True, exist_ok=True)

    # Try as dataset first
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", slug, "-p", str(dest), "--unzip"],
            check=True, capture_output=True, text=True, timeout=300,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Try as competition
        try:
            comp_name = slug.split("/")[-1]
            subprocess.run(
                ["kaggle", "competitions", "download", "-c", comp_name, "-p", str(dest), "--unzip"],  
                check=True, capture_output=True, text=True, timeout=300,
            )
        except FileNotFoundError:
            raise HTTPException(
                500,
                "Kaggle CLI not found. Install with: pip install kaggle  "
                "and set KAGGLE_USERNAME + KAGGLE_KEY env vars."
            )
        except subprocess.CalledProcessError as e:
            raise HTTPException(400, f"Kaggle download failed: {e.stderr.strip()}")

    # Find CSVs
    csvs = list(dest.rglob("*.csv"))
    if not csvs:
        raise HTTPException(400, "No CSV files found in the downloaded Kaggle dataset.")

    # Pick the largest CSV (likely the main file)
    csv_path = max(csvs, key=lambda p: p.stat().st_size)
    return str(csv_path)


def _run_training(csv_path: str, target_column: str) -> dict:
    """Import main module and run AutoML pipeline. Returns result dict."""
    # Import here to avoid circular imports at module level
    from main import run_automl, list_all_models, ModelRegistry

    score = run_automl(csv_path=csv_path, target_column=target_column)

    # Get best model info
    best_id = _find_latest_model_id()
    meta = {}
    if best_id:
        meta_path = MODEL_DIR / f"{best_id}_metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)

    return {
        "status": "success",
        "message": f"Training complete. Best score: {score:.4f}",
        "best_model_id": best_id,
        "best_model_name": meta.get("model_name"),
        "test_score": meta.get("test_score"),
        "final_score": meta.get("final_score"),
        "pickle_download": f"/models/{best_id}/download" if best_id else None,
    }


# ==============================================================
# ENDPOINTS — HEALTH
# ==============================================================

@app.get("/", tags=["Health"])
def health_check():
    """Health check with links to every endpoint."""
    return {
        "status": "ok",
        "service": "ELITE AutoML Inference API v2.0",
        "endpoints": {
            "GET  /":                    "This health check",
            "GET  /models":             "List all models with metrics",
            "GET  /models/best":        "Best model details",
            "GET  /models/download":    "Download best model .pkl",
            "GET  /models/{id}/download": "Download specific model .pkl",
            "GET  /metrics":            "Full metrics for all models",
            "POST /train":              "Upload CSV + target column → train",
            "POST /train/kaggle":       "Provide Kaggle URL + target → train",
            "POST /predict":            "JSON rows → predictions",
            "POST /predict/csv":        "Upload CSV → predictions",
            "GET  /predict/example":    "Example predict payload & curl",
        },
    }


# ==============================================================
# ENDPOINTS — MODELS & METRICS
# ==============================================================

@app.get("/models", response_model=List[ModelInfo], tags=["Models"])
def list_models():
    """Return metadata for all saved models, sorted by final_score descending."""
    records = _get_all_metadata()
    if not records:
        return []
    return records


@app.get("/models/best", tags=["Models"])
def best_model():
    """Return full details of the best model."""
    records = _get_all_metadata()
    if not records:
        raise HTTPException(404, "No models trained yet.")
    best = records[0]
    best["pickle_download"] = f"/models/{best['model_id']}/download"
    return best


@app.get("/models/download", tags=["Models"])
def download_best_model():
    """Download the best model's .pkl file."""
    model_id = _find_latest_model_id()
    if model_id is None:
        raise HTTPException(404, "No models found. Train a model first.")

    pkl_path = MODEL_DIR / f"{model_id}.pkl"
    if not pkl_path.exists():
        # Fallback: create pkl from joblib
        jbl_path = MODEL_DIR / f"{model_id}.joblib"
        if jbl_path.exists():
            bundle = joblib.load(jbl_path)
            with open(pkl_path, "wb") as f:
                pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            raise HTTPException(404, f"Model file not found for {model_id}")

    return FileResponse(
        path=str(pkl_path),
        filename=f"{model_id}.pkl",
        media_type="application/octet-stream",
    )


@app.get("/models/{model_id}/download", tags=["Models"])
def download_model_by_id(model_id: str):
    """Download a specific model's .pkl file by its model_id."""
    model_id = _resolve_model_id(model_id)

    pkl_path = MODEL_DIR / f"{model_id}.pkl"
    if not pkl_path.exists():
        jbl_path = MODEL_DIR / f"{model_id}.joblib"
        if jbl_path.exists():
            bundle = joblib.load(jbl_path)
            with open(pkl_path, "wb") as f:
                pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            raise HTTPException(404, f"Model file not found for {model_id}")

    return FileResponse(
        path=str(pkl_path),
        filename=f"{model_id}.pkl",
        media_type="application/octet-stream",
    )


@app.get("/metrics", tags=["Metrics"])
def get_metrics():
    """
    Full metrics dashboard for every trained model.
    Includes CV scores, test scores, latency, cost-aware final score, and hyperparameters.
    """
    records = _get_all_metadata()
    if not records:
        raise HTTPException(404, "No models trained yet.")

    best = records[0]

    return {
        "total_models": len(records),
        "best_model": {
            "model_id": best["model_id"],
            "model_name": best["model_name"],
            "task": best["task"],
            "cv_score_mean": best["cv_score_mean"],
            "cv_score_std": best["cv_score_std"],
            "test_score": best["test_score"],
            "final_score": best["final_score"],
            "latency_ms": best["latency_ms"],
            "hyperparameters": best["hyperparameters"],
            "pickle_download": f"/models/{best['model_id']}/download",
        },
        "leaderboard": [
            {
                "rank": i + 1,
                "model_id": r["model_id"],
                "model_name": r["model_name"],
                "cv_score_mean": round(r["cv_score_mean"], 4),
                "cv_score_std": round(r["cv_score_std"], 4),
                "test_score": round(r["test_score"], 4),
                "final_score": round(r["final_score"], 4),
                "latency_ms": round(r["latency_ms"], 4),
                "size_mb": r["size_mb"],
                "feature_count": r["feature_count"],
                "train_samples": r["train_samples"],
                "hyperparameters": r["hyperparameters"],
                "timestamp": r["timestamp"],
            }
            for i, r in enumerate(records)
        ],
    }


# ==============================================================
# ENDPOINTS — TRAINING (Upload CSV or Kaggle link)
# ==============================================================

@app.post("/train", response_model=TrainResponse, tags=["Training"])
async def train_from_upload(
    file: UploadFile = File(..., description="CSV dataset file"),
    target_column: str = Form(..., description="Name of the target column"),
):
    """
    Upload a CSV dataset and kick off the full AutoML pipeline.

    The best model will be saved as `.pkl` and `.joblib` in the `models/` folder.
    After training completes, use the returned `pickle_download` URL to download
    the model file, or hit `/metrics` to see the full leaderboard.
    """
    # Save uploaded file
    csv_path = UPLOAD_DIR / file.filename
    contents = await file.read()
    with open(csv_path, "wb") as f:
        f.write(contents)

    try:
        result = _run_training(str(csv_path), target_column)
    except Exception as e:
        raise HTTPException(500, f"Training failed: {traceback.format_exc()}")

    return TrainResponse(**result)


@app.post("/train/kaggle", response_model=TrainResponse, tags=["Training"])
def train_from_kaggle(req: TrainRequest):
    """
    Provide a Kaggle dataset URL (or slug like `owner/dataset-name`) and a
    target column.  The server will download the dataset via the Kaggle CLI,
    then run the full AutoML pipeline.

    **Prerequisites (one-time setup):**
    ```bash
    pip install kaggle
    export KAGGLE_USERNAME=your_username
    export KAGGLE_KEY=your_api_key
    ```

    Example body:
    ```json
    {
        "kaggle_url": "https://www.kaggle.com/datasets/nikhil1e9/loan-default",
        "target_column": "Default"
    }
    ```
    """
    csv_path = _download_kaggle_dataset(req.kaggle_url)

    try:
        result = _run_training(csv_path, req.target_column)
    except Exception as e:
        raise HTTPException(500, f"Training failed: {traceback.format_exc()}")

    return TrainResponse(**result)


# ==============================================================
# ENDPOINTS — PREDICTION
# ==============================================================

@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(req: PredictRequest):
    """
    Run predictions for one or more rows.

    Set `model_id` to `"latest"` (default) to auto-select the best model.

    Example body:
    ```json
    {
        "model_id": "latest",
        "data": [
            {"feature1": 1.0, "feature2": "A"},
            {"feature1": 2.0, "feature2": "B"}
        ]
    }
    ```
    """
    model_id = _resolve_model_id(req.model_id)

    try:
        bundle = _load_bundle(model_id)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))

    df = pd.DataFrame(req.data)

    try:
        preds = _preprocess_and_predict(bundle, df)
    except Exception as e:
        raise HTTPException(400, f"Prediction failed: {e}")

    record = bundle.get("record", {})

    return PredictResponse(
        model_id=model_id,
        model_name=record.get("model_name", "unknown"),
        task=record.get("task", "unknown"),
        predictions=preds.tolist(),
        count=len(preds),
    )


@app.post("/predict/csv", tags=["Prediction"])
async def predict_csv(
    file: UploadFile = File(...),
    model_id: str = Query("latest", description="Model ID or 'latest'"),
):
    """
    Upload a CSV file and receive predictions as JSON.

    Use `model_id=latest` (default) to auto-select the best model.
    """
    model_id = _resolve_model_id(model_id)

    try:
        bundle = _load_bundle(model_id)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))

    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception:
        raise HTTPException(400, "Failed to parse uploaded CSV.")

    try:
        preds = _preprocess_and_predict(bundle, df)
    except Exception as e:
        raise HTTPException(400, f"Prediction failed: {e}")

    record = bundle.get("record", {})

    return {
        "model_id": model_id,
        "model_name": record.get("model_name", "unknown"),
        "task": record.get("task", "unknown"),
        "predictions": preds.tolist(),
        "count": len(preds),
    }


@app.get("/predict/example", tags=["Prediction"])
def predict_example():
    """
    Returns an example prediction payload built from the best model's
    feature columns, plus ready-to-paste `curl` commands you can run
    to test the `/predict` or `/predict/csv` endpoint.
    """
    model_id = _find_latest_model_id()
    if model_id is None:
        raise HTTPException(
            404,
            "No models trained yet. Train a model first via POST /train or POST /train/kaggle."
        )

    try:
        bundle = _load_bundle(model_id)
    except FileNotFoundError:
        raise HTTPException(404, f"Model file not found for {model_id}")

    feature_cols = bundle.get("feature_cols", [])
    record = bundle.get("record", {})

    # Build a dummy row with placeholder values
    example_row = {}
    for col in feature_cols:
        example_row[col] = 0.0  # numeric placeholder

    example_payload = {
        "model_id": "latest",
        "data": [example_row],
    }

    payload_json = json.dumps(example_payload, indent=2)

    return {
        "info": "Copy the curl command below to test predictions against your trained model.",
        "model_id": model_id,
        "model_name": record.get("model_name", "unknown"),
        "task": record.get("task", "unknown"),
        "feature_columns": feature_cols,
        "example_payload": example_payload,
        "curl_json": (
            "curl -X POST http://localhost:8000/predict "
            "-H \"Content-Type: application/json\" "
            f"-d '{payload_json}'"
        ),
        "curl_csv": (
            "curl -X POST http://localhost:8000/predict/csv "
            "-F \"file=@your_data.csv\" "
            "-F \"model_id=latest\""
        ),
        "python_example": (
            "import requests\n"
            f"resp = requests.post('http://localhost:8000/predict', json={payload_json})\n"
            "print(resp.json())"
        ),
    }


# ==============================================================
# MAIN (for quick local testing)
# ==============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
