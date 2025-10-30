# infer.py
import json, faiss, cv2, numpy as np
from PIL import Image
from embedder import ResnetEmbedder
from detector import NoseDetector

class NoseRecognizer:
    def __init__(self, index_path="index/nose.index", labels_path="index/labels.json", det_weights=None):
        self.index = faiss.read_index(index_path)
        self.labels = json.load(open(labels_path, "r", encoding="utf-8"))
        self.emb = ResnetEmbedder()
        self.det = NoseDetector(det_weights)

    def identify(self, img_bgr, k=3, threshold=0.65):
        crop = self.det.crop_nose(img_bgr)
        pil = Image.fromarray(crop[:, :, ::-1])  # BGR->RGB
        q = self.emb.embed_pil(pil)[None, :]     # (1,512)
        sims, ids = self.index.search(q, k)      # inner product ~ cos
        sims, ids = sims[0], ids[0]
        best_sim, best_idx = float(sims[0]), int(ids[0])
        candidate = self.labels[best_idx] if best_idx >=0 else None
        is_confident = best_sim >= threshold
        return {
            "candidate": candidate if is_confident else None,
            "confidence": best_sim,
            "topk": [{"id": self.labels[i], "score": float(s)} for i, s in zip(ids, sims)]
        }

# Exemplo de uso:
if __name__ == "__main__":
    img = cv2.imread("teste.jpg")
    rec = NoseRecognizer()
    print(rec.identify(img))
