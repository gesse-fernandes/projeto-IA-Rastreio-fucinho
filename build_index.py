"""Utility script to batch-enroll bovine nose images into the MySQL database."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, Tuple

import cv2
from PIL import Image

from database import get_session, init_db
from detector import NoseDetector
from embedder import ResnetEmbedder
from models import Animal, NoseEmbedding

ENROLL_DIR = Path("data/enroll")
ENROLL_DIR.mkdir(parents=True, exist_ok=True)


def iter_images() -> Iterable[Tuple[Path, str]]:
    """Yield (path, animal_id) pairs for every file inside data/enroll."""
    for animal_dir in sorted(ENROLL_DIR.iterdir()):
        if not animal_dir.is_dir():
            continue
        animal_id = animal_dir.name
        for pattern in ("*.jpg", "*.jpeg", "*.png"):
            for file_path in sorted(animal_dir.glob(pattern)):
                yield file_path, animal_id


def parse_trace_file() -> Dict[str, Dict[str, str]]:
    trace_path = Path("data/trace.json")
    if not trace_path.exists():
        return {}
    with trace_path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def main() -> None:
    init_db()
    session = get_session()
    embedder = ResnetEmbedder()
    detector = NoseDetector(weights_path=os.getenv("NOSE_DETECTOR_WEIGHTS"))
    traces = parse_trace_file()

    added = 0
    for file_path, animal_external_id in iter_images():
        img = cv2.imread(str(file_path))
        if img is None:
            print(f"[WARN] Não foi possível ler {file_path}")
            continue
        crop = detector.crop_nose(img)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        vector = embedder.embed_pil(pil)

        trace_info = traces.get(animal_external_id)
        trace_json = json.dumps(trace_info, ensure_ascii=False) if trace_info else None

        animal = (
            session.query(Animal).filter(Animal.external_id == animal_external_id).one_or_none()
        )
        if animal is None:
            animal = Animal(external_id=animal_external_id, trace_json=trace_json)
            session.add(animal)
            session.flush()
        elif trace_json:
            animal.trace_json = trace_json

        embedding = NoseEmbedding(
            animal_id=animal.id,
            vector=vector.tobytes(),
            image_filename=str(file_path),
        )
        session.add(embedding)
        added += 1

    session.commit()
    session.close()
    print(f"Adicionados {added} embeddings ao banco de dados.")


if __name__ == "__main__":
    main()
