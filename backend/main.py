"""
ChestAI — FastAPI Backend
Endpoints: /predict, /chat, /gradcam
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
import os, io, base64, json
import cv2
from dotenv import load_dotenv

app = FastAPI(title="ChestAI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Constants ──────────────────────────────────────────────────────────────────
LABELS = [
    'Atelectasis','Cardiomegaly','Effusion','Infiltration',
    'Mass','Nodule','Pneumonia','Pneumothorax',
    'Consolidation','Edema','Emphysema','Fibrosis',
    'Pleural_Thickening','Hernia'
]
load_dotenv() 
GROQ_KEY   = os.getenv("GROQ_API_KEY", "")
CHECKPOINT = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'ResNet18_best.pth')

KNOWLEDGE = [
    "DISEASE: Atelectasis\nCollapse or incomplete expansion of lung tissue. Increased opacity, volume loss, mediastinal shift toward affected side. Label noise present in NIH dataset.",
    "DISEASE: Cardiomegaly\nEnlarged heart, cardiothoracic ratio > 0.5. AP views magnify cardiac size artificially. Associated with heart failure, hypertension.",
    "DISEASE: Effusion\nAbnormal fluid accumulation in pleural space. Blunting of costophrenic angles, meniscus sign. Small effusions may be missed on supine films.",
    "DISEASE: Infiltration\nDense material in lung parenchyma. Ill-defined opacities. High label noise — overlaps significantly with consolidation and edema.",
    "DISEASE: Mass\nPulmonary opacity > 3cm. Associated with malignancy — requires urgent CT confirmation and clinical follow-up.",
    "DISEASE: Nodule\nRounded opacity < 3cm. Difficult to detect if < 1cm on plain X-ray. CT needed for full characterisation and malignancy risk stratification.",
    "DISEASE: Pneumonia\nInfection causing air sac inflammation. Lobar consolidation, air bronchograms. Significant label noise in NIH dataset — many cases mislabelled as consolidation.",
    "DISEASE: Pneumothorax\nAir in pleural space causing lung collapse. Visible pleural line with absent lung markings. Tension pneumothorax is a life-threatening emergency.",
    "DISEASE: Consolidation\nAlveolar air replaced by fluid or cells. Homogeneous opacity with air bronchograms. Overlaps heavily with pneumonia predictions.",
    "DISEASE: Edema\nFluid in lung interstitium and alveoli. Bilateral perihilar opacities, Kerley B lines. Co-occurs frequently with cardiomegaly and effusion.",
    "DISEASE: Emphysema\nPermanent airspace enlargement with alveolar wall destruction. Hyperinflation, flattened diaphragms, bullae. Plain X-ray underestimates severity vs CT.",
    "DISEASE: Fibrosis\nLung scarring causing progressive function loss. Reticular opacities, honeycombing, traction bronchiectasis. CT is gold standard for diagnosis.",
    "DISEASE: Pleural_Thickening\nPleural scarring often due to prior inflammation or asbestos exposure. Irregular pleural surface, blunted costophrenic angles.",
    "DISEASE: Hernia\nAbdominal contents through diaphragm. Rarest class — only 227 training cases. Class weight 523x applied. Predictions have lowest reliability.",
    "DATASET LIMITATIONS: Labels NLP-mined from radiology reports with 10-20% estimated error rate. Not radiologist-verified. Model is strictly assistive — NOT a diagnostic tool.",
]

# ── Model Loading ──────────────────────────────────────────────────────────────
_model = None
_embed_model = None
_faiss_index = None
_gradcam_features = {}
_gradcam_grads = {}

def get_model():
    global _model
    if _model is None:
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, 14)
        if os.path.exists(CHECKPOINT):
            m.load_state_dict(torch.load(CHECKPOINT, map_location='cpu'))
        m.eval()
        _model = m
    return _model

def get_rag():
    global _embed_model, _faiss_index
    if _embed_model is None:
        _embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        emb = _embed_model.encode(KNOWLEDGE).astype('float32')
        _faiss_index = faiss.IndexFlatL2(emb.shape[1])
        _faiss_index.add(emb)
    return _embed_model, _faiss_index

# ── Transforms ────────────────────────────────────────────────────────────────
TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def preprocess(img: Image.Image) -> torch.Tensor:
    return TF(img.convert('RGB')).unsqueeze(0)

# ── Grad-CAM ──────────────────────────────────────────────────────────────────
def compute_gradcam(model, tensor, class_idx):
    features, grads = {}, {}

    def fwd_hook(m, i, o):
        features['x'] = o.detach()

    def bwd_hook(m, gi, go):
        grads['x'] = go[0].detach()

    handle_f = model.layer4.register_forward_hook(fwd_hook)
    handle_b = model.layer4.register_full_backward_hook(bwd_hook)

    model.zero_grad()
    out = model(tensor)
    out[0, class_idx].backward()

    handle_f.remove()
    handle_b.remove()

    weights = grads['x'].mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * features['x']).sum(dim=1, keepdim=True))
    cam = cam.squeeze().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam = cv2.resize(cam, (224, 224))

    heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    orig = np.array(tensor.squeeze().permute(1,2,0).numpy())
    orig = (orig - orig.min()) / (orig.max() - orig.min() + 1e-8)
    orig = (orig * 255).astype(np.uint8)

    overlay = cv2.addWeighted(orig, 0.55, heatmap, 0.45, 0)

    _, buf = cv2.imencode('.png', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf).decode()

# ── RAG Retrieve ──────────────────────────────────────────────────────────────
def retrieve(query: str, k: int = 3) -> list[str]:
    em, idx = get_rag()
    _, I = idx.search(em.encode([query]).astype('float32'), k)
    return [KNOWLEDGE[i] for i in I[0]]

# ── Groq Prompt ───────────────────────────────────────────────────────────────
def build_prompt(probs, chunks, question=None):
    detected = sorted(
        [(LABELS[i], float(probs[i])) for i in range(14) if probs[i] >= 0.5],
        key=lambda x: x[1], reverse=True
    )
    findings = "\n".join(
        f"- {l}: {p:.1%} ({'High confidence' if p>=0.8 else 'Moderate confidence' if p>=0.65 else 'Low confidence'})"
        for l, p in detected
    ) if detected else "No abnormalities detected above 0.5 threshold."
    ctx = "\n\n".join(chunks)

    if question:
        return f"""You are ChestAI, an expert AI assistant for chest X-ray interpretation.

X-ray findings from ResNet-18:
{findings}

Medical knowledge base:
{ctx}

User question: {question}

Answer clearly and helpfully in 2-4 sentences. Be conversational but medically accurate. Always end with: ⚠️ AI-assisted only — consult a qualified radiologist."""
    else:
        return f"""You are ChestAI, an expert AI assistant for chest X-ray interpretation.

ResNet-18 detected:
{findings}

Medical knowledge base (ground your response in this):
{ctx}

Write a structured clinical interpretation:

**Overall Assessment**
One concise sentence summarising the findings.

**Detected Findings**
For each detected condition, explain what it means clinically using the knowledge base. Be specific.

**Confidence Notes**
Briefly comment on prediction confidence levels and any caveats.

**Limitations**
One brief paragraph on label noise, model limitations, and dataset caveats.

End with:
⚠️ **Disclaimer:** This is AI-assisted analysis only. NOT a diagnostic tool. All findings must be reviewed by a qualified radiologist before any clinical decision."""

# ── Endpoints ─────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str
    probs: list[float]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        model = get_model()

        tensor = preprocess(img)
        with torch.no_grad():
            probs = torch.sigmoid(model(tensor))[0].numpy().tolist()

        # Top detected
        detected = [LABELS[i] for i in range(14) if probs[i] >= 0.5]
        query = "Information about " + ", ".join(detected) if detected else "normal chest xray no findings"
        chunks = retrieve(query)

        # Grad-CAM on top predicted class
        top_idx = int(np.argmax(probs))
        tensor_grad = preprocess(img).requires_grad_(False)
        gradcam_b64 = compute_gradcam(get_model(), preprocess(img), top_idx)

        # Groq streaming response — collect full for /predict
        client = Groq(api_key=os.getenv("GROQ_API_KEY", ""))
        prompt = build_prompt(probs, chunks)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.3,
        )
        summary = response.choices[0].message.content

        return {
            "probs": probs,
            "labels": LABELS,
            "summary": summary,
            "gradcam": gradcam_b64,
            "top_label": LABELS[top_idx],
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    chunks = retrieve(req.question)
    client = Groq(api_key=GROQ_KEY)
    prompt = build_prompt(req.probs, chunks, question=req.question)

    def generate():
        stream = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.3,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                yield f"data: {json.dumps({'text': delta})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/health")
def health():
    return {"status": "ok", "model": os.path.exists(CHECKPOINT)}