# Speech Emotion Recognition App

This repository contains a simple prototype application that performs speech emotion recognition on uploaded audio files. It includes a FastAPI backend that accepts audio uploads, runs a (mock) emotion recognition pipeline, stores results, and a React + Vite frontend for uploading files and visualizing emotion probabilities.

## Repository layout

- `backend/` - FastAPI backend

  - `app/main.py` - FastAPI application entrypoint
  - `app/api/v1/endpoints.py` - API endpoints (upload, list files, start recognition, fetch result)
  - `app/db` - SQLAlchemy models and DB session
  - `app/services` - File handling and recognition pipeline (mock model implementation)
  - `data/uploads` - stored uploaded audio files
  - `data/results` - recognition JSON results
  - `requirements.txt` - Python dependencies
  - `models/` - training & inference code and a saved model (not wired into the prototype backend)
  - `scripts`/ - SQL script for generating DB at postgresql
- `frontend/` - React (Vite) single-page app

  - `src/` - React source code
  - `src/services/api.js` - API wrappers used by the UI
  - `package.json` - dev/start scripts

## Features

- Upload audio (.mp3, .wav) via the frontend or direct API.
- Background recognition job that writes a JSON file with per-segment emotion probabilities and an average mood.
- File history UI and chart visualization of recognition results.

> Note: The backend currently uses a MockEmotionModel (see `backend/app/services/recognition.py`) that returns randomized probabilities and simulates processing time. There are model-training/inference scripts under `models/emotion_recognition/` for future integration.

## Requirements

- Python 3.10+ (for backend)
- Node 18+ / npm or yarn (for frontend)
- PostgreSQL for production; the project expects a `DATABASE_URL` env var. For quick local testing you can point `DATABASE_URL` to a local Postgres instance or adjust to an SQLite URL (see notes).

## Backend — Setup & Run

1. Create a virtual environment and install dependencies

```powershell
cd backend
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Configure environment

Create a `.env` file in `backend/` with at least:

```
DATABASE_URL=postgresql://user:password@localhost:5432/emotion_db
```

If you don't have Postgres handy for quick testing, you can use SQLite by setting `DATABASE_URL=sqlite:///./test.db`. Note that the code imports `psycopg2-binary` and uses PostgreSQL UUID types in models (for full compatibility prefer Postgres). If using SQLite, you may need to adjust model UUID column types.

3. Run the FastAPI app (development)

```powershell
# from backend/
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Once running, the API root is available at `http://localhost:8000/` and the API router is mounted at `/api/v1`.

### Backend endpoints

- POST /api/v1/files

  - Uploads an audio file (multipart/form-data `file`). Accepts `.mp3` and `.wav`.
  - Returns the stored audio file DB record (including `file_id`).
- GET /api/v1/files

  - Returns list of uploaded files and their recognition status.
- POST /api/v1/files/{file_id}/recognize

  - Triggers a background recognition job for the given file id. Returns 202.
- GET /api/v1/files/{file_id}/recognition

  - Returns the recognition JSON result (when processing finished).

Recognition results are saved to `backend/data/results/{file_id}.json`.

## Frontend — Setup & Run

1. Install dependencies

```powershell
cd frontend
npm install
```

2. Start dev server

```powershell
npm run dev
```

The frontend expects the backend base URL to be `http://localhost:8000/api/v1` (configured in `src/services/api.js`). If your backend runs on a different host/port, update `baseURL` accordingly.

## Models

The repository includes a `models/emotion_recognition/` folder with:

- `model_training.py` — training utilities (experimental)
- `model_inference.py` — example inference wrapper
- `my_model.pth` and a `fine-tuned-emotion-model/` snapshot — provided artifacts.

These are not currently integrated into the FastAPI service. The backend uses `MockEmotionModel` for deterministic, fast local testing. To integrate a real model:

1. Replace `backend/app/services/recognition.py`'s `MockEmotionModel` with your model loader/predictor.
2. Ensure the prediction returns a dict of emotion -> probability and that `run_recognition_pipeline` writes the JSON with the same structure.

## Tests

There is a basic API test at `backend/tests/test_api.py`. To run tests, use pytest (install it in the backend venv):

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
pip install pytest
pytest -q
```

## Notes & Next steps

- Replace mock model with real inference code (possible starting point: `models/emotion_recognition/model_inference.py`).
- Add authentication/authorization for file uploads and result access.
- Harden DB portability (SQLite compatibility) or provide docker-compose for Postgres + app bootstrapping.
- Add proper background worker (Celery/RQ) for robust job handling.

## Troubleshooting

- CORS: The backend allows `http://localhost:5173` by default (Vite dev server). If your frontend runs on a different port, update `app/main.py` or set proper origins.
- Database: If you see `Connected to DB!` printed from `session.py` but tables are not created, ensure your database URL is reachable and that the DB user has CREATE privileges.

## License

This project is licensed under the MIT License. See the `LICENSE` file in the repository root for the full text and copyright information.

## Contact

If you want help integrating a real model or turning this into a production-ready service, open an issue or reach out in the repo.
