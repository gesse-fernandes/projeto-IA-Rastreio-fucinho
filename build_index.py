# build_index.py
import os, glob, json
import faiss, cv2
from PIL import Image
from embedder import ResnetEmbedder
from detector import NoseDetector

ENROLL_DIR = "data/enroll"
INDEX_DIR  = "index"
os.makedirs(INDEX_DIR, exist_ok=True)

def load_pairs():
    # retorna [(path, animal_id), ...]
    pairs = []
    for aid in sorted(os.listdir(ENROLL_DIR)):
        d = os.path.join(ENROLL_DIR, aid)
        if not os.path.isdir(d): continue
        for p in glob.glob(os.path.join(d, "*.jpg")) + glob.glob(os.path.join(d, "*.png")):
            pairs.append((p, aid))
    return pairs

def main():
    emb = ResnetEmbedder()
    det = NoseDetector(weights_path=None)  # coloque seu .pt se tiver
    vecs, labels = [], []
    for path, aid in load_pairs():
        img = cv2.imread(path)[:, :, ::-1]  # BGR->RGB
        crop = det.crop_nose(img)
        pil = Image.fromarray(crop)
        v = emb.embed_pil(pil)
        vecs.append(v)
        labels.append(aid)

    xb = np.stack(vecs)  # (N,512)
    index = faiss.IndexFlatIP(xb.shape[1])      # IP com vetores L2-normalizados = cos similarity
    index.add(xb)

    faiss.write_index(index, os.path.join(INDEX_DIR, "nose.index"))
    with open(os.path.join(INDEX_DIR, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)
    print(f"Index salvo com {len(labels)} embeddings.")

if __name__ == "__main__":
    import numpy as np
    main()
