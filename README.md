# NIH ChestX-ray14: Multi-label Thoracic Disease Classification
### CNNs + RAG-grounded LLM Interpretation Layer

## Overview
Multi-label chest X-ray classifier trained on NIH ChestX-ray14 (112,120 images, 14 disease labels) using three CNN architectures with a RAG pipeline for grounded LLM interpretation.

## Results
| Model | Mean AUROC | Mean PR-AUC | Micro F1 |
|---|---|---|---|
| ResNet-18 | 0.8179 | 0.2272 | 0.2373 |
| VGG-19 | 0.7543 | 0.1397 | 0.1822 |
| Custom CNN | 0.7143 | 0.1056 | 0.1500 |

## Project Structure
```
├── notebooks/        # Kaggle training notebook (15 cells)
├── rag/              # RAG pipeline + knowledge base
├── report/           # Final PDF report + images
├── checkpoints/      # Model checkpoints (.pth files)
├── requirements.txt  # Dependencies
└── README.md
```

## Models
- **ResNet-18** — Transfer learning, all layers fine-tuned, best performance
- **VGG-19** — Transfer learning, feature layers frozen
- **Custom CNN** — 4-block CNN trained from scratch

## RAG Pipeline
- Knowledge base: 15 disease chunks
- Embeddings: sentence-transformers (all-MiniLM-L6-v2)
- Vector search: FAISS IndexFlatL2
- Output: Citation-backed, uncertainty-aware summaries

## Key Features
- Patient-wise splits (zero data leakage)
- Class-weighted BCE loss (Hernia weight: 523x)
- Grad-CAM explainability
- Non-diagnostic disclaimer enforced

## Dataset
NIH ChestX-ray14 — [Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data)

## Disclaimer
This system is strictly assistive and is NOT a diagnostic tool.