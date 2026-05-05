# ELITE AutoML Platform

A production-grade AutoML system that trains, tunes, and exports the best ML model for any tabular CSV dataset — with a FastAPI inference server included.

---

## ✨ What's New (v2.1)

- **🚀 XGBoost & LightGBM** — Faster gradient boosting for 30% performance boost
- **🎯 Multi-Target Training** — Train separate models for multiple targets simultaneously
- **⚡ Fast Mode** — Top 5 models only for large datasets (5x speedup)
- **🔧 Configuration API** — Toggle features on the fly without code changes

See [ADVANCED_FEATURES.md](ADVANCED_FEATURES.md) for detailed guide.

---

## Features

- **Automated model selection** — trains 16 models (LightGBM, XGBoost, scikit-learn, etc.); picks the best one
- **Multi-target support** — Train separate models for multiple target columns in one run
- **Fast Mode** — Train only top 5 models for large datasets (5x speedup)
- **Hyperparameter tuning** — Optuna (TPE sampler, 50 trials configurable)
- **Stratified K-Fold cross-validation** (5 folds)
- **Automated feature engineering** — interactions, polynomial features
- **Imbalanced data handling** — SMOTE
- **Drift detection** — KS test + PSI
- **Model export** — saves both `.pkl` (pickle) and `.joblib` files
- **Auto-generated summary** — `MODEL_SUMMARY.md` with leaderboard, best model details, and usage instructions
- **FastAPI inference API** — ready-to-deploy prediction server (`app.py`)
- **Configuration API** — toggle FAST_MODE, MULTI_TARGET_MODE, and more

---

## Quick Start (Google Colab)

### 1. Install dependencies

```python
!pip install pandas numpy scikit-learn imbalanced-learn optuna scipy joblib fastapi uvicorn pyngrok tabulate

# Optional (recommended for 30% better performance):
!pip install xgboost lightgbm
```

### 2. Upload or mount your dataset

```python
from google.colab import files
uploaded = files.upload()          # pick your CSV
csv_filename = list(uploaded.keys())[0]
```

### 3. Run the AutoML pipeline

```python
from main import run_automl

score = run_automl(csv_filename, 'YourTargetColumn')
# => trains models, saves .pkl + .joblib, generates MODEL_SUMMARY.md
```

### 4. Download artifacts

```python
import shutil
from google.colab import files

shutil.make_archive('automl_output', 'zip', '.', 'models')
files.download('automl_output.zip')
files.download('MODEL_SUMMARY.md')
files.download('app.py')
```

### 5. Serve predictions inside Colab (optional)

```python
!pip install pyngrok
from pyngrok import ngrok
import subprocess, time

proc = subprocess.Popen(['uvicorn', 'app:app', '--host', '0.0.0.0', '--port', '8000'])
time.sleep(3)
public_url = ngrok.connect(8000)
print(f"API is live at: {public_url}")
```

---

## Quick Start (Local Machine)

```bash
# 1. Clone / copy the project
cd sgp-3

# 2. Create a virtual environment (optional but recommended)
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Install XGBoost and LightGBM for better performance
pip install xgboost lightgbm

# 5. Run the AutoML pipeline
python main.py
#   - Edit CSV_PATH and TARGET_COLUMN at the bottom of main.py first

# 6. Start the inference server
uvicorn app:app --host 0.0.0.0 --port 8000
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/` | Health check + endpoint directory |
| `GET`  | `/config` | Get current AutoML configuration |
| `POST` | `/config` | Update configuration (fast_mode, multi_target_mode, optuna_trials) |
| `GET`  | `/models` | List all models with full metrics |
| `GET`  | `/models/best` | Best model details + download link |
| `GET`  | `/models/download` | Download best model `.pkl` file |
| `GET`  | `/models/{id}/download` | Download a specific model `.pkl` |
| `GET`  | `/metrics` | Full leaderboard with hyperparameters |
| `POST` | `/train` | Upload CSV + target column → train |
| `POST` | `/train/kaggle` | Give Kaggle URL + target → train |
| `POST` | `/train/multi-target` | Upload CSV + multiple targets → train multiple models |
| `POST` | `/predict` | Send JSON rows → get predictions |
| `POST` | `/predict/csv` | Upload CSV → get predictions |
| `GET`  | `/predict/example` | Auto-generated example payload + curl |

### Train from CSV upload

```bash
curl -X POST http://localhost:8000/train \
  -F "file=@Loan_default.csv" \
  -F "target_column=Default"
```

### Train multiple targets (Multi-Target)

```bash
curl -X POST http://localhost:8000/train/multi-target \
  -F "file=@medical_data.csv" \
  -F "target_columns=disease1,disease2,disease3"
```

### Get current configuration

```bash
curl http://localhost:8000/config
```

### Enable Fast Mode (top 5 models only)

```bash
curl -X POST http://localhost:8000/config \
  -H "Content-Type: application/json" \
  -d '{"fast_mode": true}'
```

### Enable Multi-Target Mode

```bash
curl -X POST http://localhost:8000/config \
  -H "Content-Type: application/json" \
  -d '{"multi_target_mode": true}'
```

### Train from Kaggle link

```bash
curl -X POST http://localhost:8000/train/kaggle \
  -H "Content-Type: application/json" \
  -d '{
    "kaggle_url": "https://www.kaggle.com/datasets/nikhil1e9/loan-default",
    "target_column": "Default"
  }'
```

> Requires `pip install kaggle` and Kaggle API credentials (`KAGGLE_USERNAME` + `KAGGLE_KEY`).

### Download best model pickle

```bash
curl -O http://localhost:8000/models/download
```

### View metrics

```bash
curl http://localhost:8000/metrics
```

### Get example prediction payload

```bash
curl http://localhost:8000/predict/example
```

### Predict (JSON)

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "latest",
    "data": [
      {"feature1": 1.0, "feature2": "A"},
      {"feature1": 2.5, "feature2": "B"}
    ]
  }'
```

### Predict (CSV upload)

```bash
curl -X POST http://localhost:8000/predict/csv \
  -F "file=@new_data.csv" \
  -F "model_id=latest"
```

---

## Project Structure

```
sgp-3/
├── main.py              # AutoML pipeline (train + export)
├── app.py               # FastAPI inference server
├── README.md            # This file
├── MODEL_SUMMARY.md     # Auto-generated after a run
├── models/              # Saved model artifacts
│   ├── <model_id>.pkl        ← download via /models/download
│   ├── <model_id>.joblib
│   └── <model_id>_metadata.json
├── uploads/             # Uploaded / Kaggle datasets
└── logs/                # Training logs
```

---

## Requirements

- Python 3.8+
- pandas, numpy, scikit-learn, imbalanced-learn, optuna, scipy, joblib
- fastapi, uvicorn, python-multipart (for the API server)
- tabulate (for MODEL_SUMMARY.md generation)
- kaggle (only needed for `/train/kaggle` endpoint)
- pyngrok (only needed to expose Colab server publicly)

**Optional (recommended):**
- xgboost (30% faster/better gradient boosting)
- lightgbm (fastest gradient boosting)

---

## Model Selection: All vs Fast Mode

| Scenario | Use All Models | Use Fast Mode |
|----------|---|---|
| **Dataset Size** | < 50K rows | > 50K rows |
| **Priority** | Accuracy | Speed |
| **Models** | 16 algorithms | 5 algorithms |
| **Training Time** | 5-30 min | 1-6 min |
| **Best For** | Production | Prototyping |

**All Models (Default):**
- Classification: LogisticRegression, RidgeClassifier, SGDClassifier, DecisionTree, RandomForest, ExtraTrees, GradientBoosting, HistGradientBoosting, AdaBoost, KNN, SVC, GaussianNB, LDA, QDA, XGBoost, LightGBM
- Regression: LinearRegression, Ridge, Lasso, ElasticNet, DecisionTree, RandomForest, ExtraTrees, GradientBoosting, HistGradientBoosting, SVR, XGBoost, LightGBM

**Fast Mode (Top 5):**
- LogisticRegression / LinearRegression (fast baseline)
- DecisionTree (instant, interpretable)
- RandomForest (balanced speed/accuracy)
- GradientBoosting (high accuracy)
- LightGBM (fastest boosting)

---

## For More Information

- **Advanced Features:** See [ADVANCED_FEATURES.md](ADVANCED_FEATURES.md)
- **Model Details:** See [MODEL_SUMMARY.md](MODEL_SUMMARY.md) (auto-generated after first run)

---

## Requirements

- Python 3.8+
- pandas, numpy, scikit-learn, imbalanced-learn, optuna, scipy, joblib
- fastapi, uvicorn, python-multipart (for the API server)
- tabulate (for MODEL_SUMMARY.md generation)
- kaggle (only needed for `/train/kaggle` endpoint)
- pyngrok (only needed to expose Colab server publicly)
