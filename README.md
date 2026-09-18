# ChestAI — NIH ChestX-ray14 Multi-label Classification

### ResNet-18 · FAISS RAG · Groq (openai/gpt-oss-120b) · React + Node.js Gateway + FastAPI

**Note:** The previous Railway deployment has expired. Currently runnable locally — see Local Setup below.

---

## Overview

AI-powered chest X-ray analysis system that classifies 14 thoracic diseases using deep learning, explains predictions via Grad-CAM, and provides RAG-grounded LLM interpretation through a ChatGPT-style interface.

Trained on NIH ChestX-ray14 (112,120 images) using three CNN architectures with patient-wise splits and class-weighted loss to handle severe label imbalance.

A Node.js/Express gateway sits in front of the FastAPI ML backend, handling authentication, upload validation, and rate limiting before any request reaches the model-serving code.

---

## Results

| Model | Mean AUROC | Mean PR-AUC | Micro F1 |
|---|---|---|---|
| **ResNet-18** | **0.8179** | **0.2272** | **0.2373** |
| VGG-19 | 0.7543 | 0.1397 | 0.1822 |
| Custom CNN | 0.7143 | 0.1056 | 0.1500 |

### Per-class AUROC (ResNet-18)

| Disease | AUROC | Disease | AUROC |
|---|---|---|---|
| Hernia | 0.914 | Emphysema | 0.903 |
| Cardiomegaly | 0.897 | Edema | 0.875 |
| Effusion | 0.867 | Pneumothorax | 0.857 |
| Atelectasis | 0.787 | Consolidation | 0.804 |
| Mass | 0.807 | Fibrosis | 0.801 |
| Pleural Thickening | 0.796 | Nodule | 0.726 |
| Pneumonia | 0.719 | Infiltration | 0.697 |

---

## Architecture

```
User uploads X-ray
       ↓
Node.js/Express Gateway → JWT auth, upload validation, rate limiting
       ↓
ResNet-18 inference → 14 disease probability scores
       ↓
Grad-CAM heatmap (top predicted class)
       ↓
FAISS vector search → top-3 knowledge base chunks
       ↓
Groq (openai/gpt-oss-120b) → structured clinical interpretation (streaming)
       ↓
React UI → ChatGPT-style conversational follow-up (requires login)
```

---

## Tech Stack

**Frontend**
- React 18 + Vite
- ChatGPT-style dark UI with drag-and-drop X-ray upload
- Login/register screen — required since the gateway now sits between the UI and the model
- Server-Sent Events for streaming LLM responses

**Gateway**
- Node.js + Express
- JWT authentication (`/auth/register`, `/auth/login`) with role-based permission middleware
- Upload validation — MIME type + size + magic-byte checks (rejects renamed/spoofed files before they reach FastAPI)
- Rate limiting (tighter on `/predict`, since each call triggers full model inference)
- Helmet + locked-down CORS
- Proxies `/predict` and streams `/chat/stream` through to FastAPI unchanged
- Known limitations: user store is in-memory (not a database), no refresh-token flow — a solid demo of the pattern, not production-hardened

**Backend**
- FastAPI + Uvicorn
- PyTorch ResNet-18 inference
- Grad-CAM explainability (layer4 activations)
- FAISS + sentence-transformers (all-MiniLM-L6-v2) RAG
- Groq `openai/gpt-oss-120b` streaming responses

**Model Storage**
- ResNet-18 checkpoint hosted on HuggingFace Hub
- Auto-downloaded at server startup

---

## Project Structure

```
NIH-ChestXray14-MultiLabel-CNN-RAG/
├── backend/
│   ├── main.py              # FastAPI — /predict, /chat/stream, /health
│   └── requirements.txt
├── gateway/
│   ├── src/
│   │   ├── routes/          # auth.js, predict.js, chat.js
│   │   ├── middleware/      # auth.js, validateUpload.js, errorHandler.js
│   │   ├── utils/           # logger.js, userStore.js
│   │   ├── config.js
│   │   └── server.js
│   ├── .env.example
│   └── package.json
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # Full React UI (login, home, chat pages)
│   │   └── main.jsx
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
├── checkpoints/
│   └── ResNet18_best.pth
├── notebooks/
│   └── nih-chestxray14-multilabel-cnn-rag.ipynb
├── rag/
│   ├── knowledge_base.txt
│   ├── rag_pipeline.py
│   └── sample_rag_output.txt
├── report/
│   ├── NIH_ChestXray14_Report.pdf
│   └── gradcam_results.png
├── Dockerfile
├── railway.toml
└── README.md
```

---

## Features

- **14-label multi-label classification** — simultaneous detection of all conditions
- **Grad-CAM heatmap** — visual explanation of model focus regions
- **RAG pipeline** — FAISS retrieval from curated medical knowledge base
- **Streaming LLM responses** — word-by-word like ChatGPT
- **Conversational follow-up** — ask questions about findings
- **Confidence-aware output** — High / Moderate / Low confidence pills
- **Patient-wise splits** — zero data leakage between train/val/test
- **Class-weighted BCE loss** — handles extreme imbalance (Hernia: 523x weight)
- **JWT-authenticated gateway** — auth, upload validation (magic-byte checks), and rate limiting sit in front of the model, not inside it

---

## Local Setup

Run all three services locally, each in its own terminal, in this order:

### 1. Backend (FastAPI)
```bash
cd backend
pip install -r requirements.txt

# Create .env file
echo "GROQ_API_KEY=your_key_here" > .env

uvicorn main:app --reload --port 8000
```

### 2. Gateway (Node.js)
```bash
cd gateway
npm install
cp .env.example .env
# Open .env and set a real JWT_SECRET, e.g.:
# node -e "console.log(require('crypto').randomBytes(32).toString('hex'))"
npm start
# Runs on http://localhost:4000
```

### 3. Frontend (React)
```bash
cd frontend
npm install
npm run dev
# Open http://localhost:3000
```

The frontend calls the gateway (`http://localhost:4000`), not FastAPI directly. You'll need to register/log in through the UI before uploading an X-ray — every `/predict` and `/chat/stream` call requires a valid JWT.

### API Endpoints

All routes below are served through the gateway (`http://localhost:4000`), which forwards validated, authenticated requests to FastAPI internally.

| Method | Endpoint | Auth required | Description |
|--------|----------|:---:|-------------|
| POST | `/auth/register` | – | Create an account, returns a JWT |
| POST | `/auth/login` | – | Log in, returns a JWT |
| POST | `/predict` | ✓ | Upload X-ray → predictions + Grad-CAM + LLM summary |
| POST | `/chat/stream` | ✓ | Streaming follow-up chat (SSE) |
| GET | `/health` | – | Backend health check |

---

## Dataset

NIH ChestX-ray14 — 112,120 frontal chest X-rays, 14 disease labels, NLP-mined from radiology reports.

[Download on Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data)

**Note:** Labels have an estimated 10-20% error rate due to NLP extraction. All model outputs should be interpreted with this limitation in mind.

---

## Disclaimer

This system is strictly assistive. **NOT a diagnostic tool.** All predictions must be reviewed by a qualified radiologist before any clinical decision is made.

---