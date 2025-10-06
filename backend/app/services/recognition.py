import os
import json
import uuid
from datetime import datetime
from sqlalchemy.orm import Session
from app.db import models
import time
import random

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'results'))
os.makedirs(RESULTS_DIR, exist_ok=True)

class MockEmotionModel:
    def __init__(self):
        self.emotions = ["angry", "sad", "neutral", "positive", "other"]

    def predict(self, audio_path: str) -> dict:
        time.sleep(random.uniform(2, 5))
        
        if random.random() < 0.1:
            raise Exception("Mock model processing failed: unsupported audio format.")
            
        probabilities = {emotion: random.random() for emotion in self.emotions}
        total = sum(probabilities.values())
        normalized_probabilities = {k: v / total for k, v in probabilities.items()}
        
        return normalized_probabilities

model = MockEmotionModel()

def create_initial_recognition_record(db: Session, file_id: uuid.UUID):
    existing = db.query(models.AudioEmotionRecognition).filter_by(file_id=file_id).first()
    if not existing:
        placeholder_path = os.path.join(RESULTS_DIR, f"{file_id}.json")
        
        db_rec = models.AudioEmotionRecognition(
            file_id=file_id, 
            recognition_status=models.RecognitionStatusEnum.pending,
            recognition_path=placeholder_path
        )
        db.add(db_rec)
        db.commit()

def run_recognition_pipeline(db: Session, file_id: uuid.UUID):
    recognition_record = db.query(models.AudioEmotionRecognition).filter_by(file_id=file_id).first()
    audio_file = db.query(models.AudioFile).filter_by(file_id=file_id).first()

    if not recognition_record or not audio_file:
        return

    recognition_record.recognition_status = models.RecognitionStatusEnum.in_progress
    db.commit()

    try:
        overall_probabilities = model.predict(audio_file.file_path)
        
        result_json = {
            "segments": [
                {
                    "startms": 0,
                    "endms": 10000,
                    "probabilities": overall_probabilities
                }
            ],
            "average_mood": max(overall_probabilities, key=overall_probabilities.get)
        }
        
        result_filename = f"{file_id}.json"
        result_filepath = os.path.join(RESULTS_DIR, result_filename)
        with open(result_filepath, 'w') as f:
            json.dump(result_json, f, indent=2)

        recognition_record.recognition_status = models.RecognitionStatusEnum.success
        recognition_record.recognition_path = result_filepath
        recognition_record.recognition_date = datetime.utcnow()

    except Exception as e:
        recognition_record.recognition_status = models.RecognitionStatusEnum.error
        recognition_record.recognition_date = datetime.utcnow()
    
    finally:
        db.commit()

def get_recognition_result_json(db: Session, file_id: uuid.UUID):
    recognition = db.query(models.AudioEmotionRecognition).filter_by(
        file_id=file_id, 
        recognition_status=models.RecognitionStatusEnum.success
    ).first()
    
    if not recognition or not recognition.recognition_path:
        return None
    
    try:
        with open(recognition.recognition_path, 'r') as f:
            data = json.load(f)
        return data
    except (FileNotFoundError, json.JSONDecodeError):
        return None
