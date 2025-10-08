import uuid
from datetime import datetime
from pydantic import BaseModel
from typing import Optional, Dict, List
from app.db.models import RecognitionStatusEnum

class AudioFileBase(BaseModel):
    file_name: str
    file_extension: str

class AudioFileCreate(AudioFileBase):
    pass

class AudioFileInDB(AudioFileBase):
    file_id: uuid.UUID
    upload_date: datetime
    file_path: str

    class Config:
        orm_mode = True

class RecognitionResultSchema(BaseModel):
    recognition_id: uuid.UUID
    file_id: uuid.UUID
    recognition_date: datetime
    recognition_status: RecognitionStatusEnum
    
    class Config:
        orm_mode = True

class AudioFileWithStatus(AudioFileInDB):
    recognition_status: Optional[RecognitionStatusEnum]
    recognition_date: Optional[datetime]

class ProbabilitySegment(BaseModel):
    start_ms: int
    end_ms: int
    probabilities: Dict[str, float]

class FullRecognitionResult(BaseModel):
    segments: List[ProbabilitySegment]
    average_mood: str

class AudioFileWithRecognition(AudioFileInDB):
    recognition: Optional[RecognitionResultSchema] = None 