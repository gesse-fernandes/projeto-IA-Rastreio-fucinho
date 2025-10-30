"""Flask application providing CRUD APIs for bovine nose traceability."""
from __future__ import annotations

import json
import os
from typing import Any, Tuple
from uuid import uuid4

import cv2
import numpy as np
from flask import Flask, jsonify, request, g
from sqlalchemy.exc import IntegrityError
from werkzeug.utils import secure_filename
from PIL import Image

from database import get_session, init_db
from embedder import ResnetEmbedder
from detector import NoseDetector
from models import Animal, NoseEmbedding

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "data/uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


def create_app() -> Flask:
    app = Flask(__name__)
    init_db()

    embedder = ResnetEmbedder()
    detector = NoseDetector(weights_path=os.getenv("NOSE_DETECTOR_WEIGHTS"))

    def get_db():
        if "db" not in g:
            g.db = get_session()
        return g.db

    @app.teardown_appcontext
    def shutdown_session(exception: Exception | None = None) -> None:  # pragma: no cover
        db = g.pop("db", None)
        if db is not None:
            db.close()

    def extract_trace_json(raw_value: Any) -> str | None:
        if raw_value is None:
            return None
        if isinstance(raw_value, (dict, list)):
            return json.dumps(raw_value, ensure_ascii=False)
        if isinstance(raw_value, str) and raw_value.strip():
            try:
                parsed = json.loads(raw_value)
                return json.dumps(parsed, ensure_ascii=False)
            except json.JSONDecodeError:
                return json.dumps({"notes": raw_value}, ensure_ascii=False)
        return None

    def load_image_from_upload(
        file_storage, persist: bool = True
    ) -> Tuple[np.ndarray, str | None]:
        if file_storage is None or file_storage.filename == "":
            raise ValueError("Nenhuma imagem foi enviada.")
        data = file_storage.read()
        if not data:
            raise ValueError("Arquivo de imagem vazio.")
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Não foi possível decodificar a imagem enviada.")
        path: str | None = None
        if persist:
            filename = secure_filename(file_storage.filename) or f"{uuid4().hex}.jpg"
            path = os.path.join(UPLOAD_DIR, filename)
            with open(path, "wb") as fp:
                fp.write(data)
        return img, path

    def compute_embedding(img: np.ndarray) -> np.ndarray:
        crop = detector.crop_nose(img)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        return embedder.embed_pil(pil)

    @app.route("/health", methods=["GET"])
    def healthcheck():
        return jsonify({"status": "ok"})

    @app.route("/animals", methods=["GET"])
    def list_animals():
        session = get_db()
        animals = session.query(Animal).order_by(Animal.created_at.desc()).all()
        return jsonify([animal.to_dict() for animal in animals])

    @app.route("/animals", methods=["POST"])
    def create_animal():
        session = get_db()
        data = request.form if request.form else request.json or {}
        external_id = data.get("external_id")
        if not external_id:
            return jsonify({"error": "external_id é obrigatório"}), 400

        trace_json = extract_trace_json(data.get("trace_info"))
        animal = Animal(
            external_id=external_id,
            name=data.get("name"),
            breed=data.get("breed"),
            farm=data.get("farm"),
            trace_json=trace_json,
        )
        session.add(animal)
        try:
            session.flush()
        except IntegrityError:
            session.rollback()
            return jsonify({"error": "Já existe um animal com esse external_id."}), 409

        file_storage = request.files.get("image") if request.files else None
        if file_storage is not None:
            try:
                img, saved_path = load_image_from_upload(file_storage)
                vector = compute_embedding(img)
                embedding = NoseEmbedding(
                    animal_id=animal.id,
                    vector=vector.tobytes(),
                    image_filename=saved_path,
                )
                session.add(embedding)
            except ValueError as err:
                session.rollback()
                return jsonify({"error": str(err)}), 400

        session.commit()
        session.refresh(animal)
        return jsonify(animal.to_dict(include_embeddings=True)), 201

    @app.route("/animals/<int:animal_id>", methods=["GET"])
    def get_animal(animal_id: int):
        session = get_db()
        animal = session.get(Animal, animal_id)
        if animal is None:
            return jsonify({"error": "Animal não encontrado."}), 404
        return jsonify(animal.to_dict(include_embeddings=True))

    @app.route("/animals/<int:animal_id>", methods=["PUT"])
    def update_animal(animal_id: int):
        session = get_db()
        animal = session.get(Animal, animal_id)
        if animal is None:
            return jsonify({"error": "Animal não encontrado."}), 404

        payload = request.json or {}
        for field in ("name", "breed", "farm"):
            if field in payload:
                setattr(animal, field, payload[field])
        if "trace_info" in payload:
            animal.trace_json = extract_trace_json(payload.get("trace_info"))

        session.commit()
        session.refresh(animal)
        return jsonify(animal.to_dict(include_embeddings=True))

    @app.route("/animals/<int:animal_id>", methods=["DELETE"])
    def delete_animal(animal_id: int):
        session = get_db()
        animal = session.get(Animal, animal_id)
        if animal is None:
            return jsonify({"error": "Animal não encontrado."}), 404
        session.delete(animal)
        session.commit()
        return ("", 204)

    @app.route("/animals/<int:animal_id>/embeddings", methods=["POST"])
    def add_embedding(animal_id: int):
        session = get_db()
        animal = session.get(Animal, animal_id)
        if animal is None:
            return jsonify({"error": "Animal não encontrado."}), 404
        file_storage = request.files.get("image") if request.files else None
        if file_storage is None:
            return jsonify({"error": "Envie uma imagem no campo 'image'."}), 400
        try:
            img, saved_path = load_image_from_upload(file_storage)
            vector = compute_embedding(img)
        except ValueError as err:
            return jsonify({"error": str(err)}), 400
        embedding = NoseEmbedding(
            animal_id=animal.id,
            vector=vector.tobytes(),
            image_filename=saved_path,
        )
        session.add(embedding)
        session.commit()
        session.refresh(animal)
        return jsonify(animal.to_dict(include_embeddings=True)), 201

    @app.route(
        "/animals/<int:animal_id>/embeddings/<int:embedding_id>", methods=["DELETE"]
    )
    def delete_embedding(animal_id: int, embedding_id: int):
        session = get_db()
        embedding = (
            session.query(NoseEmbedding)
            .filter(
                NoseEmbedding.id == embedding_id,
                NoseEmbedding.animal_id == animal_id,
            )
            .one_or_none()
        )
        if embedding is None:
            return jsonify({"error": "Registro de focinho não encontrado."}), 404
        session.delete(embedding)
        session.commit()
        return ("", 204)

    @app.route("/identify", methods=["POST"])
    def identify_animal():
        session = get_db()
        file_storage = request.files.get("image") if request.files else None
        if file_storage is None:
            return jsonify({"error": "Envie uma imagem no campo 'image'."}), 400

        try:
            img, _ = load_image_from_upload(file_storage, persist=False)
            query_vec = compute_embedding(img)
        except ValueError as err:
            return jsonify({"error": str(err)}), 400

        embeddings = (
            session.query(NoseEmbedding).join(Animal).all()
        )
        if not embeddings:
            return jsonify({"error": "Nenhum focinho cadastrado."}), 400

        vectors = np.stack([np.frombuffer(e.vector, dtype=np.float32) for e in embeddings])
        scores = vectors @ query_vec
        threshold = float(request.form.get("threshold", request.args.get("threshold", 0.65)))
        top_k = int(request.form.get("top_k", request.args.get("top_k", 3)))
        order = np.argsort(scores)[::-1]
        results = []
        for idx in order[:top_k]:
            embedding = embeddings[idx]
            results.append(
                {
                    "animal": embedding.animal.to_dict(),
                    "score": float(scores[idx]),
                }
            )
        best = results[0]
        candidate = best["animal"] if best["score"] >= threshold else None
        return jsonify(
            {
                "candidate": candidate,
                "confidence": float(best["score"]),
                "topk": results,
            }
        )

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)
