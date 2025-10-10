import os
import json
import torch
import torchaudio
import numpy as np
from datetime import datetime

from app.db import models
from app.db.session import SessionLocal
from app.services import file_handler

from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# CONFIG: Put your model directory here (should contain model.safetensors, config.json, preprocessor_config.json)
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'emotion_recognition', 'fine-tuned-emotion-model'))

EMOTION_LABELS = ['angry', 'sad', 'neutral', 'positive', 'other']
NUM_CLASSES = len(EMOTION_LABELS)

N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512
SAMPLING_RATE = 16000  # must match your training config

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'results'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model & feature extractor ---
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_DIR)
model = AutoModelForAudioClassification.from_pretrained(MODEL_DIR)
model = model.to(device)
model.eval()

def create_initial_recognition_record(db, file_id):
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

def update_recognition_status(db, file_id, status):
    """Helper function to update the status of a recognition record."""
    recognition_record = db.query(models.AudioEmotionRecognition).filter_by(file_id=file_id).first()
    if recognition_record:
        recognition_record.recognition_status = status
        db.commit()

def get_recognition_result_json(db, file_id):
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

def run_recognition_pipeline(db, file_id):
    recognition_record = db.query(models.AudioEmotionRecognition).filter_by(file_id=file_id).first()
    audio_file = db.query(models.AudioFile).filter_by(file_id=file_id).first()
    if not recognition_record or not audio_file:
        return

    try:
        file_path = audio_file.file_path
        waveform, sr = torchaudio.load(file_path)

        # Resample if necessary
        if sr != SAMPLING_RATE:
            waveform = torchaudio.transforms.Resample(sr, SAMPLING_RATE)(waveform)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        SEGMENT_SEC = 5  # segment length in seconds
        segment_samples = int(SAMPLING_RATE * SEGMENT_SEC)
        total_samples = waveform.shape[1]
        num_segments = int(np.ceil(total_samples / segment_samples))

        segments = []

        for i in range(num_segments):
            start = i * segment_samples
            end = min((i + 1) * segment_samples, total_samples)
            segment_waveform = waveform[:, start:end]

            if segment_waveform.shape[1] < segment_samples:
                segment_waveform = torch.nn.functional.pad(segment_waveform, (0, segment_samples - segment_waveform.shape[1]))

            audio_input = segment_waveform.squeeze().cpu().numpy()
            processed = feature_extractor(
                audio_input,
                sampling_rate=SAMPLING_RATE,
                max_length=segment_samples,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )

            with torch.no_grad():
                logits = model(**processed.to(device)).logits
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            segments.append({
                "start_ms": int(start * 1000 / SAMPLING_RATE),
                "end_ms": int(end * 1000 / SAMPLING_RATE),
                "probabilities": {emo: float(probs[j]) for j, emo in enumerate(EMOTION_LABELS)}
            })

        # Average mood
        avg_probs = np.mean([np.array(list(seg["probabilities"].values())) for seg in segments], axis=0)
        avg_mood = EMOTION_LABELS[int(np.argmax(avg_probs))]

        result_json = {
            "segments": segments,
            "average_mood": avg_mood
        }

        result_filename = f"{file_id}.json"
        result_filepath = os.path.join(RESULTS_DIR, result_filename)
        with open(result_filepath, 'w') as f:
            json.dump(result_json, f, indent=2)

        recognition_record.recognition_status = models.RecognitionStatusEnum.success
        recognition_record.recognition_path = result_filepath
        recognition_record.recognition_date = datetime.utcnow()

    except Exception as e:
        print(f"Error during recognition for file_id {file_id}: {e}")
        recognition_record.recognition_status = models.RecognitionStatusEnum.error
        recognition_record.recognition_date = datetime.utcnow()
    finally:
        db.commit()
