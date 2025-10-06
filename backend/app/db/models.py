# backend/app/db/models.py
import enum
import uuid
from sqlalchemy import Column, String, DateTime, Enum as SQLAlchemyEnum, ForeignKey, CheckConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func

Base = declarative_base()

class RecognitionStatusEnum(str, enum.Enum):
    pending = 'pending'
    in_progress = 'in_progress'
    success = 'success'
    error = 'error'

class AudioFile(Base):
    __tablename__ = 'audio_files'
    
    file_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    file_name = Column(String(255), nullable=False)
    file_path = Column(String, nullable=False)
    file_extension = Column(String(10), CheckConstraint("file_extension IN ('mp3', 'wav')"), nullable=False)
    upload_date = Column(DateTime, server_default=func.current_timestamp(), nullable=False)
    
    recognition = relationship("AudioEmotionRecognition", back_populates="audiofile", uselist=False, cascade="all, delete-orphan")

class AudioEmotionRecognition(Base):
    __tablename__ = 'audio_emotion_recognition'
    
    recognition_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    file_id = Column(UUID(as_uuid=True), ForeignKey('audio_files.file_id', ondelete='CASCADE'), nullable=False)
    recognition_path = Column(String, nullable=False)
    recognition_date = Column(DateTime, server_default=func.current_timestamp(), nullable=False)
    recognition_status = Column(SQLAlchemyEnum(RecognitionStatusEnum, name='recognition_status_enum'), nullable=False)
    
    audiofile = relationship("AudioFile", back_populates="recognition")
