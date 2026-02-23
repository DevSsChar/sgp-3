"""
==============================================================
FastAPI Inference Server for ELITE AutoML Platform
==============================================================
Endpoints:
  GET  /                       - Health check + links to all endpoints
  GET  /models                 - List ALL models from every training run
  GET  /models/best            - Best model details + metrics
  GET  /models/candidates      - Full list of model candidates available for training
  GET  /models/download        - Download best model .pkl file
  GET  /models/{id}/download   - Download specific model .pkl file
  GET  /metrics                - Full metrics for every trained model
  POST /train                  - Upload CSV to train a new AutoML run
  POST /train/kaggle           - Provide a Kaggle dataset link to train
  POST /predict                - JSON rows → predictions
  POST /predict/csv            - Upload CSV → predictions
  GET  /predict/example        - Example payload + curl command
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
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Form, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
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
    version="3.0.0",
)

MODEL_DIR  = Path("models")
UPLOAD_DIR = Path("uploads")
MODEL_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

# Global registry that persists ALL trained models across runs
GLOBAL_REGISTRY_PATH = MODEL_DIR / "global_registry.json"


# ==============================================================
# EXPANDED MODEL CANDIDATES
# These are passed to main.py's run_automl() so every run
# automatically tries the full suite of algorithms.
#
# Classification models: 12 algorithms
# Regression models:      9 algorithms  (auto-selected by task type)
# ==============================================================

CLASSIFICATION_CANDIDATES = {
    # ── Linear / Probabilistic ──────────────────────────────────
    "LogisticRegression": {
        "class": "sklearn.linear_model.LogisticRegression",
        "param_grid": {
            "C": (1e-3, 10.0),          # log-uniform
            "penalty": ["l2", "none"],   # NOTE: only l2/none are safe with lbfgs
            "solver": ["lbfgs"],
            "max_iter": [500],
        },
        "tags": ["linear", "fast", "interpretable"],
    },
    "RidgeClassifier": {
        "class": "sklearn.linear_model.RidgeClassifier",
        "param_grid": {"alpha": (1e-3, 10.0)},
        "tags": ["linear", "fast"],
    },
    "SGDClassifier": {
        "class": "sklearn.linear_model.SGDClassifier",
        "param_grid": {
            "loss": ["hinge", "log_loss", "modified_huber"],
            "alpha": (1e-5, 1e-1),
            "penalty": ["l2", "l1", "elasticnet"],
        },
        "tags": ["linear", "online", "fast"],
    },

    # ── Tree-based ───────────────────────────────────────────────
    "DecisionTree": {
        "class": "sklearn.tree.DecisionTreeClassifier",
        "param_grid": {
            "max_depth": (2, 20),
            "min_samples_split": (2, 20),
            "criterion": ["gini", "entropy"],
        },
        "tags": ["tree", "interpretable", "fast"],
    },
    "RandomForest": {
        "class": "sklearn.ensemble.RandomForestClassifier",
        "param_grid": {
            "n_estimators": (50, 300),
            "max_depth": (3, 20),
            "min_samples_split": (2, 20),
            "min_samples_leaf": (1, 10),
            "max_features": ["sqrt", "log2", None],
        },
        "tags": ["ensemble", "robust"],
    },
    "ExtraTrees": {
        "class": "sklearn.ensemble.ExtraTreesClassifier",
        "param_grid": {
            "n_estimators": (50, 300),
            "max_depth": (3, 20),
            "min_samples_split": (2, 20),
        },
        "tags": ["ensemble", "fast", "low-variance"],
    },

    # ── Gradient Boosting ────────────────────────────────────────
    "GradientBoosting": {
        "class": "sklearn.ensemble.GradientBoostingClassifier",
        "param_grid": {
            "n_estimators": (50, 300),
            "learning_rate": (0.01, 0.3),
            "max_depth": (2, 6),
            "min_samples_split": (2, 20),
            "subsample": (0.6, 1.0),
        },
        "tags": ["ensemble", "boosting", "accurate"],
    },
    "HistGradientBoosting": {
        "class": "sklearn.ensemble.HistGradientBoostingClassifier",
        "param_grid": {
            "max_iter": (50, 300),
            "learning_rate": (0.01, 0.3),
            "max_depth": (2, 8),
            "l2_regularization": (0.0, 1.0),
        },
        "tags": ["ensemble", "boosting", "fast", "native-nan"],
    },

    # ── Distance / Kernel ────────────────────────────────────────
    "KNN": {
        "class": "sklearn.neighbors.KNeighborsClassifier",
        "param_grid": {
            "n_neighbors": (1, 30),
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan"],
        },
        "tags": ["instance-based", "no-training"],
    },
    "SVC": {
        "class": "sklearn.svm.SVC",
        "param_grid": {
            "C": (0.01, 100.0),
            "kernel": ["rbf", "poly", "linear"],
            "gamma": ["scale", "auto"],
        },
        "tags": ["kernel", "accurate", "slow-on-large"],
    },

    # ── Probabilistic ────────────────────────────────────────────
    "GaussianNB": {
        "class": "sklearn.naive_bayes.GaussianNB",
        "param_grid": {"var_smoothing": (1e-9, 1e-2)},
        "tags": ["naive-bayes", "fast", "probabilistic"],
    },

    # ── Discriminant Analysis ────────────────────────────────────
    "LDA": {
        "class": "sklearn.discriminant_analysis.LinearDiscriminantAnalysis",
        "param_grid": {"solver": ["svd", "lsqr"]},
        "tags": ["linear", "interpretable", "fast"],
    },
    "QDA": {
        "class": "sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis",
        "param_grid": {"reg_param": (0.0, 1.0)},
        "tags": ["quadratic", "probabilistic"],
    },
}


REGRESSION_CANDIDATES = {
    # ── Linear ───────────────────────────────────────────────────
    "LinearRegression": {
        "class": "sklearn.linear_model.LinearRegression",
        "param_grid": {},
        "tags": ["linear", "fast", "interpretable"],
    },
    "Ridge": {
        "class": "sklearn.linear_model.Ridge",
        "param_grid": {"alpha": (1e-3, 100.0)},
        "tags": ["linear", "regularized"],
    },
    "Lasso": {
        "class": "sklearn.linear_model.Lasso",
        "param_grid": {"alpha": (1e-3, 10.0)},
        "tags": ["linear", "sparse", "feature-selection"],
    },
    "ElasticNet": {
        "class": "sklearn.linear_model.ElasticNet",
        "param_grid": {
            "alpha": (1e-3, 10.0),
            "l1_ratio": (0.0, 1.0),
        },
        "tags": ["linear", "regularized", "balanced"],
    },

    # ── Tree-based ───────────────────────────────────────────────
    "DecisionTree": {
        "class": "sklearn.tree.DecisionTreeRegressor",
        "param_grid": {
            "max_depth": (2, 20),
            "min_samples_split": (2, 20),
        },
        "tags": ["tree", "interpretable"],
    },
    "RandomForest": {
        "class": "sklearn.ensemble.RandomForestRegressor",
        "param_grid": {
            "n_estimators": (50, 300),
            "max_depth": (3, 20),
            "min_samples_split": (2, 20),
        },
        "tags": ["ensemble", "robust"],
    },
    "GradientBoosting": {
        "class": "sklearn.ensemble.GradientBoostingRegressor",
        "param_grid": {
            "n_estimators": (50, 300),
            "learning_rate": (0.01, 0.3),
            "max_depth": (2, 6),
            "subsample": (0.6, 1.0),
        },
        "tags": ["ensemble", "boosting"],
    },
    "HistGradientBoosting": {
        "class": "sklearn.ensemble.HistGradientBoostingRegressor",
        "param_grid": {
            "max_iter": (50, 300),
            "learning_rate": (0.01, 0.3),
            "max_depth": (2, 8),
        },
        "tags": ["ensemble", "boosting", "fast", "native-nan"],
    },
    "SVR": {
        "class": "sklearn.svm.SVR",
        "param_grid": {
            "C": (0.01, 100.0),
            "kernel": ["rbf", "linear"],
            "epsilon": (0.01, 1.0),
        },
        "tags": ["kernel", "accurate"],
    },
}


# ==============================================================
# PYDANTIC SCHEMAS
# ==============================================================

class PredictRequest(BaseModel):
    """JSON body for single / batch prediction."""
    model_id: str = "latest"
    data: List[Dict[str, Any]]


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
    all_models_trained: Optional[int] = None


# ==============================================================
# GLOBAL REGISTRY  –  persists ALL trained models across runs
# ==============================================================

def _load_registry() -> List[dict]:
    """Load the global registry JSON. Returns [] if missing."""
    if GLOBAL_REGISTRY_PATH.exists():
        with open(GLOBAL_REGISTRY_PATH) as f:
            return json.load(f)
    return []


def _save_registry(records: List[dict]) -> None:
    """Overwrite the global registry JSON."""
    with open(GLOBAL_REGISTRY_PATH, "w") as f:
        json.dump(records, f, indent=2, default=str)


def _upsert_registry(new_records: List[dict]) -> None:
    """
    Merge new_records into the global registry.
    Deduplicates on model_id; new records win on conflict.
    """
    existing = _load_registry()
    existing_by_id = {r["model_id"]: r for r in existing}
    for rec in new_records:
        existing_by_id[rec["model_id"]] = rec
    merged = sorted(existing_by_id.values(),
                    key=lambda r: r.get("final_score", 0), reverse=True)
    _save_registry(merged)


def _get_all_models() -> List[dict]:
    """
    Return every trained model ever, sorted by final_score.
    Primary source: global_registry.json (all runs, all models).
    Fallback: individual *_metadata.json files (legacy / best-only saves).
    """
    registry = _load_registry()

    # Supplement with any individual metadata files not yet in the registry
    known_ids = {r["model_id"] for r in registry}
    extra = []
    for path in MODEL_DIR.glob("*_metadata.json"):
        with open(path) as f:
            rec = json.load(f)
        if rec.get("model_id") not in known_ids:
            extra.append(rec)

    all_records = registry + extra
    all_records.sort(key=lambda r: r.get("final_score", 0), reverse=True)
    return all_records


# ==============================================================
# HELPERS
# ==============================================================

def _find_best_model_id() -> Optional[str]:
    """Return model_id of the best (highest final_score) saved model."""
    records = _get_all_models()
    # Only return IDs that have a corresponding .pkl or .joblib on disk
    for rec in records:
        mid = rec.get("model_id")
        if mid and (
            (MODEL_DIR / f"{mid}.pkl").exists() or
            (MODEL_DIR / f"{mid}.joblib").exists()
        ):
            return mid
    return None


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
    if model_id.lower() == "latest":
        resolved = _find_best_model_id()
        if resolved is None:
            raise HTTPException(404, "No models found in the registry. Train a model first.")
        return resolved
    return model_id


def _preprocess_and_predict(bundle: dict, df: pd.DataFrame) -> np.ndarray:
    model       = bundle["model"]
    preprocessor = bundle["preprocessor"]
    feature_cols = bundle["feature_cols"]
    df.columns   = [c.strip().lower().replace(" ", "_") for c in df.columns]
    missing = set(feature_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in input data: {missing}")
    X_processed = preprocessor.transform(df[feature_cols])
    return model.predict(X_processed)


def _download_kaggle_dataset(kaggle_url: str) -> str:
    slug = kaggle_url.strip().rstrip("/")
    match = re.search(r"kaggle\.com/datasets/([^/]+/[^/?#]+)", slug)
    if match:
        slug = match.group(1)
    comp_match = re.search(r"kaggle\.com/competitions?/([^/?#]+)", slug)
    if comp_match:
        slug = comp_match.group(1)

    dest = UPLOAD_DIR / slug.replace("/", "_")
    dest.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", slug, "-p", str(dest), "--unzip"],
            check=True, capture_output=True, text=True, timeout=300,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            comp_name = slug.split("/")[-1]
            subprocess.run(
                ["kaggle", "competitions", "download", "-c", comp_name, "-p", str(dest), "--unzip"],
                check=True, capture_output=True, text=True, timeout=300,
            )
        except FileNotFoundError:
            raise HTTPException(
                500,
                "Kaggle CLI not found. Install with: pip install kaggle "
                "and set KAGGLE_USERNAME + KAGGLE_KEY env vars."
            )
        except subprocess.CalledProcessError as e:
            raise HTTPException(400, f"Kaggle download failed: {e.stderr.strip()}")

    csvs = list(dest.rglob("*.csv"))
    if not csvs:
        raise HTTPException(400, "No CSV files found in the downloaded Kaggle dataset.")
    return str(max(csvs, key=lambda p: p.stat().st_size))


def _run_training(csv_path: str, target_column: str) -> dict:
    """
    Run the AutoML pipeline and register ALL trained models
    (not just the best) into the global registry.
    """
    from main import run_automl, ModelRegistry

    # Pass the expanded candidate list so main.py trains all algorithms
    score = run_automl(
        csv_path=csv_path,
        target_column=target_column,
        model_candidates=CLASSIFICATION_CANDIDATES,   # main.py should accept this kwarg
        regression_candidates=REGRESSION_CANDIDATES,
    )

    # ── Collect results for every model trained in this run ───────
    # main.py should save *_metadata.json for every model, not just the best.
    # We sweep the models/ folder and upsert anything not yet in the registry.
    run_records = []
    for path in sorted(MODEL_DIR.glob("*_metadata.json"),
                       key=lambda p: p.stat().st_mtime, reverse=True):
        with open(path) as f:
            run_records.append(json.load(f))

    if run_records:
        _upsert_registry(run_records)

    best_id = _find_best_model_id()
    meta    = {}
    if best_id:
        meta_path = MODEL_DIR / f"{best_id}_metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)

    return {
        "status": "success",
        "message": f"Training complete. Best score: {score:.4f}",
        "best_model_id":   best_id,
        "best_model_name": meta.get("model_name"),
        "test_score":      meta.get("test_score"),
        "final_score":     meta.get("final_score"),
        "pickle_download": f"/models/{best_id}/download" if best_id else None,
        "all_models_trained": len(run_records),
    }


# ==============================================================
# ENDPOINTS — HEALTH
# ==============================================================

@app.get("/", tags=["Health"])
def health_check():
    """Health check with links to every endpoint."""
    registry = _load_registry()
    return {
        "status": "ok",
        "service": "ELITE AutoML Inference API v3.0",
        "total_models_in_registry": len(registry),
        "endpoints": {
            "GET  /":                        "This health check",
            "GET  /models":                  "List ALL models from every training run",
            "GET  /models/best":             "Best model details",
            "GET  /models/candidates":       "All available model algorithms",
            "GET  /models/download":         "Download best model .pkl",
            "GET  /models/{id}/download":    "Download specific model .pkl",
            "GET  /metrics":                 "Full metrics for all models",
            "POST /train":                   "Upload CSV + target column → train",
            "POST /train/kaggle":            "Provide Kaggle URL + target → train",
            "POST /predict":                 "JSON rows → predictions",
            "POST /predict/csv":             "Upload CSV → predictions",
            "GET  /predict/example":         "Example predict payload & curl",
        },
    }


# ==============================================================
# ENDPOINTS — MODELS & METRICS
# ==============================================================

@app.get("/models", tags=["Models"])
def list_models(
    task: Optional[str] = Query(None, description="Filter by task: 'classification' or 'regression'"),
    min_test_score: Optional[float] = Query(None, description="Minimum test_score filter"),
    limit: int = Query(50, description="Max number of models to return"),
):
    """
    List **all** models from every training run, sorted by final_score.

    This includes every algorithm tried during every run — not just
    the winner. Use `task` and `min_test_score` query params to filter.
    """
    records = _get_all_models()

    if task:
        records = [r for r in records if r.get("task", "").lower() == task.lower()]
    if min_test_score is not None:
        records = [r for r in records if r.get("test_score", 0) >= min_test_score]

    # Annotate each record with whether its file is downloadable
    for rec in records:
        mid = rec.get("model_id", "")
        rec["downloadable"] = (
            (MODEL_DIR / f"{mid}.pkl").exists() or
            (MODEL_DIR / f"{mid}.joblib").exists()
        )
        rec["download_url"] = f"/models/{mid}/download" if rec["downloadable"] else None

    return {
        "total": len(records),
        "showing": min(len(records), limit),
        "models": records[:limit],
    }


@app.get("/models/candidates", tags=["Models"])
def list_model_candidates(
    task: str = Query("classification", description="'classification' or 'regression'"),
):
    """
    Return the full catalogue of ML algorithms available for training,
    along with their hyperparameter search spaces and tags.
    """
    candidates = (
        CLASSIFICATION_CANDIDATES if task.lower() == "classification"
        else REGRESSION_CANDIDATES
    )
    return {
        "task": task,
        "total_candidates": len(candidates),
        "candidates": {
            name: {
                "sklearn_class": info["class"],
                "hyperparameter_space": info["param_grid"],
                "tags": info["tags"],
            }
            for name, info in candidates.items()
        },
    }


@app.get("/models/best", tags=["Models"])
def best_model():
    """Return full details of the best model across all runs."""
    records = _get_all_models()
    if not records:
        raise HTTPException(404, "No models trained yet.")
    best = records[0]
    best["pickle_download"] = f"/models/{best['model_id']}/download"
    return best


@app.get("/models/download", tags=["Models"])
def download_best_model():
    """Download the best model's .pkl file."""
    model_id = _find_best_model_id()
    if model_id is None:
        raise HTTPException(404, "No models found. Train a model first.")
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


@app.get("/models/{model_id}/download", tags=["Models"])
def download_model_by_id(model_id: str):
    """Download a specific model's .pkl file by its model_id."""
    model_id  = _resolve_model_id(model_id)
    pkl_path  = MODEL_DIR / f"{model_id}.pkl"
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
    Full metrics dashboard for every trained model across all runs.
    Includes CV scores, test scores, latency, final score, and hyperparameters.
    """
    records = _get_all_models()
    if not records:
        raise HTTPException(404, "No models trained yet.")

    best = records[0]
    return {
        "total_models": len(records),
        "best_model": {
            "model_id":        best["model_id"],
            "model_name":      best["model_name"],
            "task":            best["task"],
            "cv_score_mean":   best["cv_score_mean"],
            "cv_score_std":    best["cv_score_std"],
            "test_score":      best["test_score"],
            "final_score":     best["final_score"],
            "latency_ms":      best["latency_ms"],
            "hyperparameters": best["hyperparameters"],
            "pickle_download": f"/models/{best['model_id']}/download",
        },
        "leaderboard": [
            {
                "rank":          i + 1,
                "model_id":      r["model_id"],
                "model_name":    r["model_name"],
                "cv_score_mean": round(r["cv_score_mean"], 4),
                "cv_score_std":  round(r["cv_score_std"], 4),
                "test_score":    round(r["test_score"], 4),
                "final_score":   round(r["final_score"], 4),
                "latency_ms":    round(r["latency_ms"], 4),
                "size_mb":       r.get("size_mb", 0),
                "feature_count": r.get("feature_count", 0),
                "train_samples": r.get("train_samples", 0),
                "hyperparameters": r["hyperparameters"],
                "timestamp":     r.get("timestamp", ""),
                "dataset_hash":  r.get("dataset_hash", ""),
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
    Upload a CSV dataset and run the full AutoML pipeline.

    **All** model candidates (13 classifiers / 9 regressors) will be tried.
    Every trained model is recorded in the global registry at `/models`.
    After training, use the returned `pickle_download` URL to grab the best model.
    """
    csv_path = UPLOAD_DIR / file.filename
    with open(csv_path, "wb") as f:
        f.write(await file.read())
    try:
        result = _run_training(str(csv_path), target_column)
    except Exception:
        raise HTTPException(500, f"Training failed:\n{traceback.format_exc()}")
    return TrainResponse(**result)


@app.post("/train/kaggle", response_model=TrainResponse, tags=["Training"])
def train_from_kaggle(req: TrainRequest):
    """
    Provide a Kaggle dataset URL and target column.
    The server downloads the dataset via the Kaggle CLI and runs AutoML.

    **Prerequisites:**
    ```bash
    pip install kaggle
    export KAGGLE_USERNAME=your_username
    export KAGGLE_KEY=your_api_key
    ```
    """
    csv_path = _download_kaggle_dataset(req.kaggle_url)
    try:
        result = _run_training(csv_path, req.target_column)
    except Exception:
        raise HTTPException(500, f"Training failed:\n{traceback.format_exc()}")
    return TrainResponse(**result)


# ==============================================================
# ENDPOINTS — PREDICTION
# ==============================================================

@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(req: PredictRequest):
    """
    Run predictions for one or more rows.
    Set `model_id` to `"latest"` (default) to auto-select the best model.
    """
    model_id = _resolve_model_id(req.model_id)
    try:
        bundle = _load_bundle(model_id)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    try:
        preds = _preprocess_and_predict(bundle, pd.DataFrame(req.data))
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
    """Upload a CSV file and receive predictions as JSON."""
    model_id = _resolve_model_id(model_id)
    try:
        bundle = _load_bundle(model_id)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    try:
        df = pd.read_csv(io.BytesIO(await file.read()))
    except Exception:
        raise HTTPException(400, "Failed to parse uploaded CSV.")
    try:
        preds = _preprocess_and_predict(bundle, df)
    except Exception as e:
        raise HTTPException(400, f"Prediction failed: {e}")
    record = bundle.get("record", {})
    return {
        "model_id":   model_id,
        "model_name": record.get("model_name", "unknown"),
        "task":       record.get("task", "unknown"),
        "predictions": preds.tolist(),
        "count":      len(preds),
    }


@app.get("/predict/example", tags=["Prediction"])
def predict_example():
    """Example prediction payload + ready-to-paste curl commands."""
    model_id = _find_best_model_id()
    if model_id is None:
        raise HTTPException(404, "No models trained yet. Train a model first via POST /train.")
    try:
        bundle = _load_bundle(model_id)
    except FileNotFoundError:
        raise HTTPException(404, f"Model file not found for {model_id}")

    feature_cols    = bundle.get("feature_cols", [])
    record          = bundle.get("record", {})
    example_row     = {col: 0.0 for col in feature_cols}
    example_payload = {"model_id": "latest", "data": [example_row]}
    payload_json    = json.dumps(example_payload, indent=2)

    return {
        "info":            "Copy the curl command below to test predictions.",
        "model_id":        model_id,
        "model_name":      record.get("model_name", "unknown"),
        "task":            record.get("task", "unknown"),
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
            f"resp = requests.post('http://localhost:8000/predict', json={json.dumps(example_payload)})\n"
            "print(resp.json())"
        ),
    }


# ==============================================================
# MAIN
# ==============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", port=8000, reload=True)