"""Utility module for performing bovine nose identification against the database."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image

from database import get_session, init_db
from detector import NoseDetector
from embedder import ResnetEmbedder
from models import Animal, NoseEmbedding


@dataclass
class RecognitionResult:
    candidate: Optional[Animal]
    confidence: float
    topk: List[tuple[Animal, float]]


class NoseRecognizer:
    """Identify bovine noses using embeddings stored in the database."""

    def __init__(self, detector_weights: str | None = None) -> None:
        init_db()
        self.embedder = ResnetEmbedder()
        self.detector = NoseDetector(detector_weights)

    def _compute_embedding(self, img_bgr: np.ndarray) -> np.ndarray:
        crop = self.detector.crop_nose(img_bgr)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        return self.embedder.embed_pil(pil)

    def identify(
        self, img_bgr: np.ndarray, k: int = 3, threshold: float = 0.65
    ) -> RecognitionResult:
        query = self._compute_embedding(img_bgr)

        session = get_session()
        try:
            embeddings = session.query(NoseEmbedding).join(Animal).all()
            if not embeddings:
                return RecognitionResult(candidate=None, confidence=0.0, topk=[])

            vectors = np.stack(
                [np.frombuffer(embedding.vector, dtype=np.float32) for embedding in embeddings]
            )
            scores = vectors @ query
            order = np.argsort(scores)[::-1]

            topk_results: List[tuple[Animal, float]] = []
            for idx in order[:k]:
                embedding = embeddings[idx]
                topk_results.append((embedding.animal, float(scores[idx])))

            best_animal, best_score = topk_results[0]
            candidate = best_animal if best_score >= threshold else None
            return RecognitionResult(
                candidate=candidate,
                confidence=float(best_score),
                topk=topk_results,
            )
        finally:
            session.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Identificação de focinhos bovinos")
    parser.add_argument("image", help="Caminho da imagem a ser identificada")
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    frame = cv2.imread(args.image)
    if frame is None:
        raise SystemExit(f"Não foi possível carregar {args.image}")

    recognizer = NoseRecognizer()
    result = recognizer.identify(frame, k=args.top_k, threshold=args.threshold)

    print("Melhor candidato:", result.candidate.external_id if result.candidate else None)
    print("Confiança:", result.confidence)
    print("Top-k:")
    for animal, score in result.topk:
        print(f" - {animal.external_id}: {score:.4f}")
