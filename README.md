# CFAKE — Real Vs AI-Generated Image Detector

A Flask web app that detects whether an image is **REAL or AI-GENERATED** using classical machine learning.

## How It Works

```text
Image
  ↓
Resize to 32×32 RGB
  ↓
Normalize + Flatten (3072 features)
  ↓
StandardScaler
  ↓
PCA (150 components)
  ↓
├── Logistic Regression
├── Decision Tree
└── Random Forest
  ↓
Majority Vote
  ↓
REAL / FAKE / UNCERTAIN
```

## Features

- AI-generated image detection
- 3 ML models with probability-based predictions
- Majority-vote final decision
- Model performance dashboard
- Image upload with drag & drop
- REST API for predictions

## Models

| Model | Configuration |
|---|---|
| Logistic Regression | `max_iter=1000`, balanced classes |
| Decision Tree | `max_depth=15`, balanced classes |
| Random Forest | `200 trees`, `max_depth=20` |

The best model is selected using **F1-score** during training.

## Dataset

Uses the **CIFAKE: Real and AI-Generated Synthetic Images** dataset.

Expected structure:

```text
dataset/
├── train/
│   ├── REAL/
│   └── FAKE/
└── test/
    ├── REAL/
    └── FAKE/
```

## Run Locally

```bash
pip install -r requirements.txt
python train.py
python app.py
```

Then open:

```text
http://localhost:5000
```

> The dataset and trained models are ignored by Git, so `train.py` must be run before starting the app on a fresh clone.

## API

### `POST /api/predict`

Upload an image using the `image` form field.

### `GET /api/results`

Returns model evaluation results.

## Tech Stack

**Python · Flask · Scikit-learn · NumPy · Pillow · Matplotlib · Seaborn**

## Project Structure

```text
Cfake/
├── app.py
├── train.py
├── requirements.txt
├── templates/
├── static/
├── dataset/          # local only
└── models/           # generated locally
```

## Disclaimer

This project is an ML-based classifier for experimentation and educational use. Predictions should not be treated as definitive proof of image authenticity.

## Authors

- **Mithunsurya Kumarasamy**
- **Paari Karthikeyan**
- **Kavin Balakumar**
