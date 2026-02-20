# AutoML Run Summary

> Generated on 2026-02-20 21:32:01

---

## Dataset

| Property | Value |
|----------|-------|
| File | `uploads\BankNote_Authentication.csv` |
| Rows | 1372 |
| Columns | 5 |
| Task | **CLASSIFICATION** |
| Numeric features | 4 |
| Categorical features | 0 |
| Final feature count (after engineering) | 136 |
| Class imbalanced | No |

---

## Model Leaderboard

| model_name       |   cv_score_mean |   cv_score_std |   test_score |   latency_ms |   final_score |
|:-----------------|----------------:|---------------:|-------------:|-------------:|--------------:|
| GradientBoosting |        0.996359 |     0.00340263 |     1        |    0.0066328 |      0.849999 |
| RandomForest     |        0.999091 |     0.00181818 |     0.996364 |    0.213699  |      0.746342 |

---

## Best Model

| Property | Value |
|----------|-------|
| Model | **GradientBoosting** |
| CV Score (mean +/- std) | 0.9964 +/- 0.0034 |
| Test Score | 1.0000 |
| Final Score (cost-aware) | 0.8500 |
| Latency (ms/sample) | 0.0066 |
| Model ID | `GradientBoosting_b22de4e9f16e_1771603320` |

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| n_estimators | 106 |
| learning_rate | 0.2835559696877989 |
| max_depth | 10 |
| min_samples_split | 3 |
| subsample | 0.6014947911252734 |


---

## Drift Detection

**No significant drift detected.**

---

## Saved Artifacts

| Artifact | Path |
|----------|------|
| Pickle file | `models\GradientBoosting_b22de4e9f16e_1771603320.pkl` |
| Joblib file | `models\GradientBoosting_b22de4e9f16e_1771603320.joblib` |
| Metadata | `models/GradientBoosting_b22de4e9f16e_1771603320_metadata.json` |

---

## How to Load & Predict

```python
import pickle, pandas as pd

# Load pickle
with open('models\GradientBoosting_b22de4e9f16e_1771603320.pkl', 'rb') as f:
    bundle = pickle.load(f)

model = bundle['model']
preprocessor = bundle['preprocessor']
feature_cols = bundle['feature_cols']

# Prepare new data
df = pd.read_csv('new_data.csv')
df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
X = df[feature_cols]
X_processed = preprocessor.transform(X)
predictions = model.predict(X_processed)
```

---

## FastAPI Inference

See `app.py` for a ready-to-use FastAPI server.  
Run with:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```
