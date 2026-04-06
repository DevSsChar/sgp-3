# VisualData Project Documentation

## Team Quick Start: How To Access The Endpoint

This section is for your group members so they can start frontend integration immediately.

### Important Hosting Note (Read First)
- Team members can access this API only while the host machine is online and the FastAPI server is running.
- If the host machine is off, sleeping, disconnected from internet/Wi-Fi, or the terminal is closed, the API becomes unavailable.
- This is normal behavior for all self-hosted FastAPI backends.

### 1. Start the API server
From this project folder, run:

```bat
run_api.bat
```

Keep the terminal window open while testing or integrating.

If you close this terminal, the endpoint stops immediately.

### 2. Open API docs (Swagger)
Use this in browser:

```text
http://localhost:8000/docs
```

### 3. Access from same Wi-Fi/LAN (teammates)
1. On host machine, run:

```bat
ipconfig
```

2. Find IPv4 Address (example: 192.168.1.25).
3. Teammates open:

```text
http://192.168.1.25:8000/docs
```

### 4. Access from outside network (optional)
Use a tunnel tool and share generated public URL:

```bat
ngrok http 8000
```

Then teammates can call the HTTPS URL from any network.

### 4.1 Which setup should we use?
- Need quick team testing on same Wi-Fi: use LAN IP (fastest).
- Need access from different networks: use ngrok/public tunnel.
- Need always-on endpoint (even when your laptop is off): deploy to cloud.

### 5. Core endpoints for frontend
- GET /health
- GET /info
- POST /predict
- POST /batch_predict

### 6. First frontend test flow
1. Call GET /health and verify status=healthy.
2. Call GET /info and load model/class metadata in UI.
3. Upload one image to POST /predict.
4. Display predicted_class + confidence + probabilities.

---

## Project Summary

VisualData is an automated image-classification pipeline that trains multiple CNN architectures, compares them across metrics, selects the best model, and serves inference through a FastAPI endpoint for frontend/backend integration.

---

## Final Model Information

Based on saved training outputs:

- Best architecture: CustomResNet
- Accuracy: 0.837 (83.70%)
- F1 score: 0.8358015623538609
- Latency: 3.9108856201171878 ms
- Model memory: 10.604530334472656 MB
- Parameters: 2,779,914
- Overall score: 0.8162256914746457
- Dataset: CIFAR10
- Dataset size: 50,000
- Number of classes: 10
- Class labels: plane, car, bird, cat, deer, dog, frog, horse, ship, truck
- Total end-to-end training time: 188.79 minutes

---

## What The Training Pipeline Did

### 1. Data loading
- Supported CIFAR10 and custom folder datasets.
- Loaded train/test data and class labels.

### 2. Smart dataset analysis
- Detected dataset size category.
- Automatically selected suitable architectures and training settings.
- Adjusted epochs, Optuna trials, and early stopping strategy.

### 3. Data cleaning and preprocessing
- Removed invalid/corrupt image entries.
- Normalized image pixels to [0,1].
- Standardized image size to 32x32.

### 4. Class balancing
- Applied SMOTE for class balancing (with adaptive k-neighbors by dataset size).
- Fallback logic to continue safely if balancing was not applicable.

### 5. Model architecture setup
- Included custom architectures (SimpleCNN, MediumCNN, DeepCNN, CustomResNet, CustomDenseNet).
- Included efficient torchvision options (MobileNetV2, SqueezeNet, ShuffleNetV2, EfficientNetB0 based on selected set).

### 6. Hyperparameter optimization
- Used Optuna trials per architecture.
- Tuned learning rate, batch size, and weight decay.
- Used pruning and early stopping for efficiency.

### 7. Training and evaluation
- Trained each selected architecture.
- Evaluated on test set.
- Compared across multiple metrics:
  - Accuracy
  - F1 score
  - Inference latency
  - Model memory footprint
  - Training time

### 8. Multi-criteria model selection
- Normalized metrics.
- Applied weighted scoring to balance quality and efficiency.
- Selected best model automatically.

### 9. Artifacts generated
- Best model pickle file
- CSV results for all trained architectures
- JSON metrics summary for deployment/API

---

## API Service Implementation Details

### Stack
- FastAPI
- Uvicorn
- PyTorch
- Pillow

### Serving behavior
- Loads model metrics from visualdata_output/best_model_metrics.json
- Loads best model from visualdata_output/CustomResNet_best_model.pkl
- Supports CPU and GPU inference automatically
- Input preprocessing for API inference:
  - RGB conversion
  - Resize to 32x32
  - Normalize to [0,1]
  - Channel-first tensor (C,H,W)

### CORS
- CORS is enabled for all origins to simplify frontend development across teammates.

---

## Endpoint Contract

### GET /
Returns basic API metadata and endpoint list.

### GET /health
Returns service health, device, and model status.

### GET /info
Returns model stats, dataset details, and training summary.

### POST /predict
Accepts one image file and returns:
- success
- filename
- predicted_class
- predicted_index
- confidence
- all_probabilities (per class)
- model_accuracy
- model_info

### POST /batch_predict
Accepts multiple image files and returns:
- total_files
- successful
- failed
- predictions list for each file

---

## Files In This Project (Important)

- run_api.bat: one-click API startup on Windows
- fastapi_endpoint.py: FastAPI inference server
- requirements_endpoint.txt: endpoint dependencies
- FASTAPI_USAGE_GUIDE.md: API setup and usage guide
- visualdata_output/CustomResNet_best_model.pkl: trained best model
- visualdata_output/best_model_metrics.json: model metadata used by API

---

## How To Install Dependencies

```bat
python -m pip install -r requirements_endpoint.txt
```

If needed, install upload dependency explicitly:

```bat
python -m pip install python-multipart
```

---

## Frontend Integration Examples

### JavaScript fetch (single image)

```javascript
async function predictImage(file) {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch('http://localhost:8000/predict', {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    throw new Error('Prediction API failed');
  }

  const data = await res.json();
  return data;
}
```

### Recommended UI fields to display
- Predicted class
- Confidence
- Top 3 probabilities
- Model accuracy from API response

---

## Troubleshooting

### Issue: API not reachable
- Check server terminal is still running.
- Confirm URL and port (8000).
- Check firewall for LAN access.
- Confirm host machine is ON and not in sleep mode.

### Issue: File upload error
- Ensure python-multipart is installed.
- Verify file type is an image.

### Issue: Model file not found
- Confirm visualdata_output folder contains model and metrics files.

### Issue: Teammate cannot access from LAN
- Host must run server with 0.0.0.0 binding (already in run_api.bat).
- Teammate must use host IPv4, not localhost.

---

## Main Project Merge Note

This document is ready to be included in the final main project as a complete technical and integration README for VisualData.
