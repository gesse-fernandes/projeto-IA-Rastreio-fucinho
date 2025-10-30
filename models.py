"""SQLAlchemy models for bovine nose traceability."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import Column, DateTime, ForeignKey, Integer, LargeBinary, String, Text
from sqlalchemy.orm import relationship

from database import Base


class Animal(Base):
    __tablename__ = "animals"

    id = Column(Integer, primary_key=True)
    external_id = Column(String(64), nullable=False, unique=True, index=True)
    name = Column(String(120))
    breed = Column(String(120))
    farm = Column(String(120))
    trace_json = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    embeddings = relationship(
        "NoseEmbedding",
        back_populates="animal",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def to_dict(self, include_embeddings: bool = False) -> Dict[str, Any]:
        trace_info: Optional[Dict[str, Any]] = None
        if self.trace_json:
            try:
                import json

                trace_info = json.loads(self.trace_json)
            except json.JSONDecodeError:
                trace_info = None
        data: Dict[str, Any] = {
            "id": self.id,
            "external_id": self.external_id,
            "name": self.name,
            "breed": self.breed,
            "farm": self.farm,
            "trace_info": trace_info,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "embeddings_count": len(self.embeddings),
        }
        if include_embeddings:
            data["embeddings"] = [emb.to_dict() for emb in self.embeddings]
        return data


class NoseEmbedding(Base):
    __tablename__ = "nose_embeddings"

    id = Column(Integer, primary_key=True)
    animal_id = Column(
        Integer, ForeignKey("animals.id", ondelete="CASCADE"), nullable=False, index=True
    )
    vector = Column(LargeBinary, nullable=False)
    image_filename = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    animal = relationship("Animal", back_populates="embeddings")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "animal_id": self.animal_id,
            "image_filename": self.image_filename,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
