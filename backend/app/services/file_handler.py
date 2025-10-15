import shutil
import os
import uuid
from fastapi import UploadFile
from sqlalchemy.orm import Session, joinedload
from app.db import models

UPLOAD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'uploads'))
os.makedirs(UPLOAD_DIR, exist_ok=True)

async def save_uploaded_file(db: Session, file: UploadFile) -> models.AudioFile:
    file_id = uuid.uuid4()
    extension = os.path.splitext(file.filename)[1].lower()
    server_filename = f"{file_id}{extension}"
    file_path = os.path.join(UPLOAD_DIR, server_filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    db_file = models.AudioFile(
        file_id=file_id,
        file_name=file.filename,
        file_path=file_path,
        file_extension=extension.strip('.'),
    )
    db.add(db_file)
    db.commit()
    db.refresh(db_file)
    return db_file

def get_audio_file_by_id(db: Session, file_id: uuid.UUID):
    return db.query(models.AudioFile).filter(models.AudioFile.file_id == file_id).first()

def get_all_audio_files_with_status(db: Session):
    return db.query(models.AudioFile).options(joinedload(models.AudioFile.recognition)).order_by(models.AudioFile.upload_date.desc()).all()
