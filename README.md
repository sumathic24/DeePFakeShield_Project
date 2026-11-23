# DeepFakeShield 🛡️

**Automated Detection of Deepfakes and Misinformation in Social Media**

> Final Year Major Project — B.Tech CSE (Data Science)
> Aligned with **SDG 16: Peace, Justice and Strong Institutions**

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react)](https://react.dev)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C?logo=pytorch)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Overview

DeepFakeShield is a multimodal detection application with separate text, image, and video workflows:

- **Text** — A configured pretrained detector scores AI-authorship likelihood and SHAP attributes the score to input tokens. A separate pretrained fake-news classifier estimates similarity to its training labels; it does not verify claims against sources.
- **Images** — The configured pretrained classifier distinguishes artificial from real and returns Grad-CAM maps for regions influencing its prediction. It does not identify manipulated images as a separate class.
- **Video** — A configured ConvNeXt detector scores representative frames and returns frame-wise Grad-CAM explanations, scores, and timestamps.
- **Abstention** — Low-confidence text and image scores are returned as uncertain. Video always reports the higher-scoring AI/Real class and flags low-confidence predictions.

The pretrained detectors' confidence scores are not calibrated accuracy estimates. SHAP and Grad-CAM explain model behavior; they do not prove authorship or manipulation.

---

## 🏗️ Project Structure

```
DeepFakeShield/
├── frontend/                    # React + Vite + Tailwind + Framer Motion
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Landing.jsx      # Hero, features, stats, pipeline, CTA
│   │   │   ├── Dashboard.jsx    # Analytics, charts, recent analyses
│   │   │   ├── TextAnalysis.jsx # Input → score and SHAP attribution
│   │   │   ├── ImageAnalysis.jsx# Upload → binary score and Grad-CAM
│   │   │   ├── VideoAnalysis.jsx# Frame scores, timeline, and Grad-CAM
│   │   │   ├── Explainability.jsx# SHAP / Grad-CAM interpretation guide
│   │   │   └── About.jsx        # Models, datasets, tech stack, SDG
│   │   ├── components/
│   │   │   ├── Navbar.jsx
│   │   │   └── Footer.jsx
│   │   └── utils/api.js         # Axios API calls
│   └── package.json
│
├── backend/                     # FastAPI + PyTorch
│   ├── main.py                  # App entry + CORS
│   ├── routers/
│   │   ├── text.py              # /api/text/analyze, /api/text/fake-news
│   │   ├── image.py             # /api/image/analyze
│   │   └── stats.py             # /api/stats
│   ├── services/
│   │   ├── pretrained.py       # Offline-loaded Hugging Face detectors
│   │   ├── explainability.py   # Shared Grad-CAM generation and rendering
│   │   ├── text_service.py     # Text detection, SHAP, optional fake-news model
│   │   ├── image_service.py    # AI-vs-real prediction and Grad-CAM
│   │   └── video_service.py    # Frame sampling, scores, and frame-wise Grad-CAM
│   ├── models/saved/            # Trained model weights (after training)
│   └── requirements.txt
│
├── training/                    # Model training scripts
│   ├── dataset_prep.py          # Download + prepare all datasets
│   ├── train_roberta.py         # Fine-tune RoBERTa for AI text detection
│   ├── train_linguaforge.py     # Train LinguaForge-X (BERT + linguistic features)
│   ├── train_ensemble.py        # Grid search + meta-learner ensemble fusion
│   ├── train_fakenews.py        # BERT fake news classifier
│   └── train_image_classifier.py# EfficientNet / ViT image classifier
│
├── docs/                        # Documentation
├── docker-compose.yml
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- Node.js ≥ 18
- Python ≥ 3.10
- Git

### Windows: Start the app

From the project folder, double-click `start.bat`. It starts the backend and frontend,
waits for the backend and its detectors to finish loading, then opens the app at
**http://127.0.0.1:5173**. Keep the backend and frontend terminal windows open while using
the app. If the virtual environment or frontend dependencies are missing, follow the
manual setup below first.

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/DeepFakeShield.git
cd DeepFakeShield
```

### 2. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at: **http://localhost:5173**

### 3. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

pip install -r requirements.txt

# Start FastAPI server
uvicorn main:app --reload --port 8000
```

Backend API at: **http://localhost:8000**
API Docs at: **http://localhost:8000/docs**

> **Note:** Models are not downloaded at backend startup. Run `python download_models.py`
> from `backend` to fetch all configured models, or use `--text`, `--image`, `--video`, or
> `--misinformation` for selected detectors. The `--text` option downloads both text models.
> The video adapter expects `convnext-best.pth` or
> `checkpoints/convnext/best.pth`; the downloader fetches that ConvNeXt checkpoint only. The repository
> does not declare a model license, so verify your right to use the checkpoint before deployment.
> The defaults can be changed with `TEXT_DETECTOR_MODEL`,
> `IMAGE_DETECTOR_MODEL`, and `VIDEO_DETECTOR_MODEL`; set a detector to `none` to disable it.

---

## 🧠 Model Training

### Step 1 — Prepare Datasets

```bash
cd training
pip install -r ../backend/requirements.txt "datasets==3.6.0" "pyarrow==21.0.0"
pip install pillow

python dataset_prep.py --output_dir ./data
```

This will download and prepare:
- HC3 Dataset (AI vs Human text)
- LIAR Dataset (fake news)
- CIFAKE (AI vs real images)

### Step 2 — Train Text Models

```bash
# Train RoBERTa (AI text detector)
python train_roberta.py \
    --data_path ./data/text/ai_human.jsonl \
    --output_dir ../backend/models/saved/roberta_ai_detector \
    --epochs 5 --batch_size 16 --lr 2e-5

# Train LinguaForge-X (BERT + linguistic features)
python train_linguaforge.py \
    --data_path ./data/text/ai_human.jsonl \
    --output_dir ../backend/models/saved/linguaforge_detector \
    --epochs 5 --batch_size 16

# Train RoBERTa + LinguaForge-X Ensemble
python train_ensemble.py \
    --roberta_dir ../backend/models/saved/roberta_ai_detector \
    --lingua_dir ../backend/models/saved/linguaforge_detector \
    --data_path ./data/text/ai_human.jsonl

# Download the pretrained misinformation classifier
python download_models.py --misinformation
```

Restart the backend after downloading. The classifier estimates patterns learned from its
training data; it does not verify claims against current sources. To fine-tune a local model
instead, prepare LIAR data and use the optional training commands below.

The default is `hamzab/roberta-fake-news-classification`, which is MIT-licensed and trained
on the Kaggle Fake and Real News dataset. Its model card reports 100% accuracy on that dataset;
that figure is not independently verified and should not be treated as expected real-world
accuracy. Set `MISINFO_DETECTOR_MODEL` to another compatible Hugging Face model ID or `none`
to disable the misinformation classifier.

```bash
# Fine-tune a local fake-news classifier
python dataset_prep.py --liar-only --output_dir ./data
python train_fakenews.py \
    --data_path ./data/text/fakenews.jsonl \
    --output_dir ../backend/models/saved/fakenews_classifier \
    --epochs 5
```

`--liar-only` downloads and prepares the LIAR corpus without downloading the other datasets. A locally fine-tuned model in `backend/models/saved/fakenews_classifier` takes precedence over the configured pretrained classifier. This classifier estimates similarity to its training labels; it does not verify claims against current sources.

On Windows, from the project root, download the pretrained model with:

```powershell
cd backend
.\venv\Scripts\python.exe download_models.py --misinformation
```

For optional fine-tuning instead:

```powershell
cd backend
.\venv\Scripts\python.exe -m pip install "datasets==3.6.0" "pyarrow==21.0.0"
cd ..\training
..\backend\venv\Scripts\python.exe dataset_prep.py --liar-only --output_dir .\data
..\backend\venv\Scripts\python.exe train_fakenews.py --data_path .\data\text\fakenews.jsonl --output_dir ..\backend\models\saved\fakenews_classifier
```

These steps require internet access to download LIAR and the BERT base checkpoint. After training completes, close the backend terminal and run `start.bat` again.

### Step 3 — Train Image Models

```bash
# EfficientNet-B4
python train_image_classifier.py \
    --data_dir ./data/images \
    --model efficientnet \
    --epochs 10 --batch_size 32

# Vision Transformer (ViT-B/16)
python train_image_classifier.py \
    --data_dir ./data/images \
    --model vit \
    --epochs 10 --batch_size 32
```

### Evaluation and data integrity

No measured accuracy results are bundled with this repository. Training scripts report accuracy,
precision, recall, F1, a confusion matrix, class labels, and test-set size to JSON files and stdout;
the test set is not used for checkpoint selection. Text records are normalized/deduplicated and
split by source group with a fixed seed. Image training rejects duplicate decoded images that cross
train/validation/test splits and uses validation macro-F1 for early stopping. To evaluate the video
detector, place independently labeled videos in `training/data/videos/test/real/` and
`training/data/videos/test/fake/`, then run from `training`:

```bash
python evaluate_video_detector.py --data_dir ./data/videos/test --output ./video_test_metrics.json
python evaluate_video_detector.py --data_dir ./data/videos/unseen --output ./video_unseen_metrics.json
```

The second directory should contain different source videos from training and the first test set;
report its metrics separately as the unseen-domain result. The video model's fake class targets
face-swapped/deepfake-like content, not arbitrary fully synthetic video.

The checked-in `training/data/text/ai_human.jsonl` is only a placeholder-quality corpus: it has
1,000 rows but just 5 distinct normalized texts. The checked-in `training/data/text/fakenews.jsonl`
is also placeholder-quality: 800 rows but only 4 distinct texts. The training scripts intentionally
reject these files because they cannot support leakage-resistant evaluation. Use
`dataset_prep.py --liar-only` to prepare real LIAR text data before training the fake-news
classifier. Image training data is not included.
Prepare the supported source datasets and use a separately curated, source-disjoint test set before
reporting results. A binary AI-vs-real image classifier cannot determine whether a real-class image
was later edited, and these scripts do not provide a three-way manipulated-media classifier.

---

## 🔌 API Reference

### Text Analysis
```
POST /api/text/analyze
Body: { "text": "Your text here..." }
```

The UI runs AI authorship and misinformation checks separately. `POST /api/text/ai-authorship`
only runs the AI-vs-human detector; `POST /api/text/fake-news` only runs the misinformation
classifier. `/api/text/analyze` remains available for combined analysis.

Response fields include `prediction`, `status`, `confidence`, `ai_confidence`,
`human_confidence`, `fake_news_score` (nullable), `fake_news_prediction` and
`fake_news_status` (nullable until a classifier is downloaded), `model_used`, signed `shap_values`,
`token_importance`, source-preserving `highlighted_text` segments, a SHAP `explanation`
summary, and `processing_time_ms`. SHAP is computed against the active AI-authorship
detector (the configured pretrained model or optional HC3 pipeline).

The fake-news score and prediction are `null` until the configured classifier is downloaded
and loaded. Run `python download_models.py --misinformation` from `backend`, then restart the
API. The dedicated fake-news endpoint returns HTTP 503 when no classifier is available. These
scores estimate similarity to training labels; they are not source-backed fact verification.

### Image Analysis
```
POST /api/image/analyze
Body: multipart/form-data (file: image)

Response fields include `prediction`, `status`, `confidence`, `ai_confidence`,
`real_confidence`, `model_used`, `gradcam_regions`, `explanation`, and `gradcam`
heatmap/overlay data URIs, plus `processing_time_ms`. The original upload remains available
in the browser preview for comparison.
```

`confidence` is the larger of the two model scores. By default an image result is uncertain below
0.65; adjust this abstention threshold with `IMAGE_MIN_CONFIDENCE`. `IMAGE_AI_THRESHOLD` controls
the binary decision boundary. These are operating thresholds, not evidence of calibration.

### Video Analysis

`POST /api/video/analyze` accepts MP4, MOV, AVI, MKV, and WEBM uploads (up to 200 MB and five
minutes). The adapter samples up to 16 evenly spaced frames and applies the configured ConvNeXt
frame classifier to those same frames. It returns the overall score, per-frame scores and
timestamps, Grad-CAM heatmaps/overlays, and suspicious-frame timestamps. Run
`python download_models.py --video` and restart the API if the checkpoint is not present.

### Dashboard Stats
```
GET /api/stats
```

---

## 🐳 Docker Deployment

```bash
# Build and run entire stack
docker-compose up --build

# Frontend: http://localhost:5173
# Backend:  http://localhost:8000
```

---

## 📊 Datasets

| Dataset | Type | Samples | Source |
|---|---|---|---|
| HC3 | AI vs Human Text | Downloaded at preparation time; deduplicated and grouped by question | Hugging Face |
| LIAR | News veracity statements | Downloaded at preparation time; grouped by speaker | Hugging Face |
| CIFAKE | AI vs Real Images | Up to 5,000 per class; deterministic image-hash split | Hugging Face |

---

## 🛠️ Tech Stack

**Frontend:** React 18 · Vite · Tailwind CSS · Framer Motion · Recharts · Axios

**Backend:** FastAPI · Python 3.11 · PyTorch 2.2 · Transformers 4.48 · SHAP 0.45

**Trainable model families:** RoBERTa · LinguaForge-X · BERT · EfficientNet-B4 · ViT-B/16

**Model explanations:** SHAP for active text detectors; gradient-based Grad-CAM for the active SwinV2 image detector and ConvNeXt video frame classifier.

**Database:** SQLite analysis history

---

## 🌏 SDG Alignment

This project aligns with **UN Sustainable Development Goal 16** —
_Peace, Justice and Strong Institutions_ — by providing tools to detect and
counter AI-generated misinformation that undermines democratic processes,
public trust, and social cohesion.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) file.

---

*Built with ❤️ as a Final Year Major Project — CSE (Data Science)*
