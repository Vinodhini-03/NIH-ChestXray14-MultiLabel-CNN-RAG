# ChestAI — NIH ChestX-ray14 Multi-label Classification

### ResNet-18 · FAISS RAG · Groq LLaMA-3.3-70B · React + FastAPI

**Live Demo → [chestai.vercel.app](https://chestai.vercel.app)**

---

## Overview

AI-powered chest X-ray analysis system that classifies 14 thoracic diseases using deep learning, explains predictions via Grad-CAM, and provides RAG-grounded LLM interpretation through a ChatGPT-style interface.

Trained on NIH ChestX-ray14 (112,120 images) using three CNN architectures with patient-wise splits and class-weighted loss to handle severe label imbalance.

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
ResNet-18 inference → 14 disease probability scores
       ↓
Grad-CAM heatmap (top predicted class)
       ↓
FAISS vector search → top-3 knowledge base chunks
       ↓
Groq LLaMA-3.3-70B → structured clinical interpretation (streaming)
       ↓
React UI → ChatGPT-style conversational follow-up
```

---

## Tech Stack

**Frontend**
- React 18 + Vite
- ChatGPT-style dark UI with drag-and-drop X-ray upload
- Server-Sent Events for streaming LLM responses
- Deployed on **Vercel**

**Backend**
- FastAPI + Uvicorn
- PyTorch ResNet-18 inference
- Grad-CAM explainability (layer4 activations)
- FAISS + sentence-transformers (all-MiniLM-L6-v2) RAG
- Groq LLaMA-3.3-70B streaming responses
- Deployed on **Railway** (Docker)

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
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # Full React UI (home + chat pages)
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

---

## Local Setup

### Backend
```bash
cd backend
pip install -r requirements.txt

# Create .env file
echo "GROQ_API_KEY=your_key_here" > .env

uvicorn main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
# Open http://localhost:3000
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | Upload X-ray → predictions + Grad-CAM + LLM summary |
| POST | `/chat/stream` | Streaming follow-up chat (SSE) |
| GET | `/health` | Backend health check |

---

## Dataset

NIH ChestX-ray14 — 112,120 frontal chest X-rays, 14 disease labels, NLP-mined from radiology reports.

[Download on Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data)

**Note:** Labels have an estimated 10-20% error rate due to NLP extraction. All model outputs should be interpreted with this limitation in mind.

---

## Disclaimer

This system is strictly assistive. **NOT a diagnostic tool.** All predictions must be reviewed by a qualified radiologist before any clinical decision is made.

---
