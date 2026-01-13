from sqlalchemy import Column, String, Float, Integer, JSON, TIMESTAMP, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import declarative_base
import uuid
from datetime import datetime

Base = declarative_base()

class AudioDiagnosticRun(Base):
    __tablename__ = "audio_diagnostic_run"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String, nullable=False)
    duration = Column(Float)
    sample_rate = Column(Integer)
    windows = Column(Integer)
    run_metadata = Column("metadata", JSON)
    created_at = Column(TIMESTAMP, default=datetime.utcnow)


class AudioWindowDetection(Base):
    __tablename__ = "audio_window_detection"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id = Column(UUID(as_uuid=True), ForeignKey("audio_diagnostic_run.id"))
    window_index = Column(Integer)
    start_time = Column(Float)
    end_time = Column(Float)


class AudioWindowLabel(Base):
    __tablename__ = "audio_window_label"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    window_id = Column(UUID(as_uuid=True), ForeignKey("audio_window_detection.id"))
    label = Column(String)
    confidence = Column(Float)


class AudioLabelSummary(Base):
    __tablename__ = "audio_label_summary"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id = Column(UUID(as_uuid=True), ForeignKey("audio_diagnostic_run.id"))
    label = Column(String)
    confidence = Column(Float)
    occurrences = Column(Integer)


