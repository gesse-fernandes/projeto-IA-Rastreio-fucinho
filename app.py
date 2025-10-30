# app.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import cv2, numpy as np, os, json, faiss
from PIL import Image
from embedder import ResnetEmbedder
from detector import NoseDetector

app = FastAPI(title="Bovine Nose ID")
INDEX_DIR="index"
ENROLL_DIR="data/enroll"
os.makedirs(ENROLL_DIR, exist_ok=True)

# Carrega em memória
emb = ResnetEmbedder()
det = NoseDetector(weights_path=None)
index = faiss.read_index(os.path.join(INDEX_DIR, "nose.index")) if os.path.exists(os.path.join(INDEX_DIR, "nose.index")) else None
labels_path = os.path.join(INDEX_DIR, "labels.json")
labels = json.load(open(labels_path, "r", encoding="utf-8")) if os.path.exists(labels_path) else []

def _embed_from_bytes(bts):
    arr = np.frombuffer(bts, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    crop = det.crop_nose(img)
    pil = Image.fromarray(crop[:, :, ::-1])
    v = emb.embed_pil(pil)
    return v

@app.post("/identify")
async def identify(file: UploadFile = File(...), threshold: float = 0.65, k: int = 3):
    if index is None or len(labels) == 0:
        return JSONResponse({"error": "Índice vazio. Cadastre animais primeiro."}, status_code=400)
    v = _embed_from_bytes(await file.read())[None, :]
    sims, ids = index.search(v, k)
    sims, ids = sims[0], ids[0]
    best_sim, best_idx = float(sims[0]), int(ids[0])
    candidate = labels[best_idx] if best_idx >= 0 else None

    # 🔹 carrega rastreabilidade estática
    trace_data = {}
    if os.path.exists("data/trace.json"):
        trace_data = json.load(open("data/trace.json", "r", encoding="utf-8"))

    rastreio = trace_data.get(candidate, {}) if candidate else {}

    return {
        "candidate": candidate if best_sim >= threshold else None,
        "confidence": best_sim,
        "topk": [{"id": labels[i], "score": float(s)} for i, s in zip(ids, sims)],
        "rastreio": rastreio   # 🔹 anexa o histórico
    }


@app.post("/enroll")
async def enroll(animal_id: str = Form(...), file: UploadFile = File(...)):
    global index, labels
    raw = await file.read()
    v = _embed_from_bytes(raw)[None, :]

    # 🔹 Verifica se esse embedding já existe no índice
    if index is not None and len(labels) > 0:
        sims, ids = index.search(v, 1)  # busca vizinho mais próximo
        if float(sims[0][0]) > 0.95:    # limiar de "mesma imagem"
            return {"status": "skip", "msg": "Imagem já cadastrada", "animal_id": labels[ids[0][0]]}

    # Se não existe, adiciona normalmente
    if index is None:
        index = faiss.IndexFlatIP(v.shape[1])
        labels = []
    index.add(v)
    labels.append(animal_id)
    faiss.write_index(index, os.path.join(INDEX_DIR, "nose.index"))
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)
    return {"status": "ok", "animal_id": animal_id}


# Rodar: uvicorn app:app --reload --port 8000
