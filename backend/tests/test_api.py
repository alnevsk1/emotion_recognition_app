import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.main import app
from app.db.session import get_db
from app.db.models import Base
import os

DATABASE_URL="postgresql://emotion_recognition_admin:pudge@localhost:5432/emotion_recognition_db_test"

engine = create_engine(DATABASE_URL)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(bind=engine)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)

@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown_database():
    # Setup: create tables
    Base.metadata.create_all(bind=engine)
    yield
    # Teardown: drop all tables to clean up the database
    Base.metadata.drop_all(bind=engine)

def test_upload_audio_file():
    # Create a dummy file to upload
    with open("test.wav", "w") as f:
        f.write("dummy audio data")

    with open("test.wav", "rb") as f:
        response = client.post("/api/v1/files", files={"file": ("test.wav", f, "audio/wav")})
    
    os.remove("test.wav")

    assert response.status_code == 200
    data = response.json()
    assert "file_id" in data
    assert data["file_name"] == "test.wav"
    return data["file_id"]

def test_list_audio_files():
    response = client.get("/api/v1/files")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_request_recognition():
    file_id = test_upload_audio_file()
    response = client.post(f"/api/v1/files/{file_id}/recognize")
    assert response.status_code == 202
    assert response.json() == {"message": "Recognition process started."}

def test_get_recognition_result_not_found():
    # Use a new UUID that doesn't exist
    import uuid
    random_uuid = uuid.uuid4()
    response = client.get(f"/api/v1/files/{random_uuid}/recognition")
    assert response.status_code == 404
