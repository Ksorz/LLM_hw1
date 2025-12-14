"""Database models."""

from sqlalchemy import Column, Integer, String, DateTime, Text, Float, JSON
from sqlalchemy.sql import func

from .database import Base


class RequestLog(Base):
    """Log of API requests and responses."""
    __tablename__ = "request_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    input_text = Column(Text, nullable=False)
    output_text = Column(Text, nullable=True)
    processing_time_ms = Column(Float, nullable=True)
    model_name = Column(String, nullable=True)
    device = Column(String, nullable=True)


class DatasetSample(Base):
    """User-provided samples for retraining."""

    __tablename__ = "dataset_samples"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    text = Column(Text, nullable=False)
    label = Column(Text, nullable=True)


class Experiment(Base):
    """Experiments for retrain/deploy flow."""

    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True, index=True)
    status = Column(String, nullable=False, default="pending")
    checkpoint_path = Column(String, nullable=True)
    onnx_path = Column(String, nullable=True)
    metrics = Column(JSON, nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    finished_at = Column(DateTime(timezone=True), nullable=True)
