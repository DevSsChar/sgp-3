# AutoML Run Summary

> Generated on 2026-02-23 15:53:56

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

| model_name           |   cv_score_mean |   cv_score_std |   test_score |   latency_ms |   final_score |
|:---------------------|----------------:|---------------:|-------------:|-------------:|--------------:|
| LDA                  |        0.989054 |     0.00795721 |     1        |   0.00104427 |      0.975    |
| RidgeClassifier      |        0.992702 |     0.00619148 |     1        |   0.00194073 |      0.975    |
| QDA                  |        0.995442 |     0.00407488 |     0.996364 |   0.0073719  |      0.971363 |
| DecisionTree         |        0.98269  |     0.00442311 |     0.985455 |   0.00127792 |      0.960454 |
| SGDClassifier        |        0.993616 |     0.00364677 |     0.981818 |   0.00500679 |      0.956818 |
| KNN                  |        0.997269 |     0.00223021 |     1        |   0.115685   |      0.949988 |
| SVC                  |        0.998174 |     0.00365297 |     1        |   0.00291109 |      0.9      |
| HistGradientBoosting |        1        |     0          |     1        |   0.0185776  |      0.849998 |
| AdaBoost             |        0.999087 |     0.00182648 |     1        |   0.155673   |      0.849984 |
| GradientBoosting     |        1        |     0          |     1        |   0.00660896 |      0.799999 |
| ExtraTrees           |        1        |     0          |     1        |   0.20067    |      0.64998  |
| GaussianNB           |        0.682868 |     0.0755767  |     0.654545 |   0.00284195 |      0.644545 |
| RandomForest         |        0.999091 |     0.00181818 |     0.996364 |   0.326633   |      0.596331 |

---

## Best Model

| Property | Value |
|----------|-------|
| Model | **LDA** |
| CV Score (mean +/- std) | 0.9891 +/- 0.0080 |
| Test Score | 1.0000 |
| Final Score (cost-aware) | 0.9750 |
| Latency (ms/sample) | 0.0010 |
| Model ID | `LDA_b22de4e9f16e_1771842229` |

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| solver | svd |


---

## Drift Detection

**No significant drift detected.**

---

## Saved Artifacts

| Artifact | Path |
|----------|------|
| Pickle file | `models\LDA_b22de4e9f16e_1771842229.pkl` |
| Joblib file | `models\LDA_b22de4e9f16e_1771842229.joblib` |
| Metadata | `models/LDA_b22de4e9f16e_1771842229_metadata.json` |

---

## How to Load & Predict

```python
import pickle, pandas as pd

# Load pickle
with open('models\LDA_b22de4e9f16e_1771842229.pkl', 'rb') as f:
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
