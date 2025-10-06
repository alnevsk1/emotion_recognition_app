from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api.v1 import endpoints
from .db.session import engine
from .db import models

# This line creates the database tables based on your models
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Speech Emotion Recognition API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # The default Vite dev server address
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(endpoints.router, prefix="/api/v1")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Speech Emotion Recognition API"}