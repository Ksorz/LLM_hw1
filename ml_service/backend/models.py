"""Database models."""
from sqlalchemy import Column, Integer, String, DateTime, Text, Float
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
