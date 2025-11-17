import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List
from fastapi.responses import FileResponse
import os

from app.db.session import get_db
from app.db import models
from app import services
from . import schemas

router = APIRouter()

@router.post("/files", response_model=schemas.AudioFileInDB)
async def upload_audio_file(background_tasks: BackgroundTasks, file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename.endswith(('.mp3', '.wav')):
        raise HTTPException(status_code=400, detail="Invalid file type. Only .mp3 and .wav are accepted.")
    
    audio_file = await services.file_handler.save_uploaded_file(db=db, file=file)
    
    background_tasks.add_task(services.recognition.create_initial_recognition_record, db=db, file_id=audio_file.file_id)
    
    return audio_file

@router.get("/files", response_model=List[schemas.AudioFileWithRecognition])
def list_audio_files(db: Session = Depends(get_db)):
    return services.file_handler.get_all_audio_files_with_status(db)

@router.post("/files/{file_id}/recognize", status_code=202)
def request_recognition(file_id: uuid.UUID, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    audio_file = services.file_handler.get_audio_file_by_id(db=db, file_id=file_id)
    if not audio_file:
        raise HTTPException(status_code=404, detail="Audio file not found.")
    
    # Update status to "in_progress" immediately
    services.recognition.update_recognition_status(db=db, file_id=file_id, status="in_progress")
    
    background_tasks.add_task(services.recognition.run_recognition_pipeline, db=db, file_id=file_id)
    
    return {"message": "Recognition process started."}

@router.get("/files/{file_id}/recognition", response_model=schemas.FullRecognitionResult)
def get_recognition_result(file_id: uuid.UUID, db: Session = Depends(get_db)):
    result = services.recognition.get_recognition_result_json(db=db, file_id=file_id)
    if not result:
        raise HTTPException(status_code=404, detail="Recognition result not found or not complete.")
    return result

@router.get("/files/{file_id}/progress", response_model=schemas.RecognitionResultSchema)
def get_recognition_progress(file_id: uuid.UUID, db: Session = Depends(get_db)):
    recognition = db.query(models.RecognitionResult).filter_by(file_id=file_id).first()
    if not recognition:
        raise HTTPException(status_code=404, detail="Recognition record not found.")
    return recognition

@router.get("/files/{file_id}/audio")
def get_audio_file_data(file_id: uuid.UUID, db: Session = Depends(get_db)):
    audio_file = services.file_handler.get_audio_file_by_id(db=db, file_id=file_id)
    if not audio_file or not os.path.exists(audio_file.file_path):
        raise HTTPException(status_code=404, detail="Audio file not found.")
    return FileResponse(audio_file.file_path, media_type=f"audio/{audio_file.file_extension}")
