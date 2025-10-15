# Speech Emotion Recognition App

This repository contains a comprehensive application for speech emotion recognition. It features a FastAPI backend for audio processing and a React frontend for user interaction and data visualization.

## Table of Contents

- [Features](#features)
- [Repository Layout](#repository-layout)
- [Requirements](#requirements)
- [Backend Setup](#backend--setup--run)
- [Frontend Setup](#frontend--setup--run)
- [API Endpoints](#backend-endpoints)
- [Model Integration](#models)
- [Testing](#tests)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Contact](#contact)

## Features

- **Audio Upload**: Supports `.mp3` and `.wav` file formats.
- **Background Processing**: Emotion recognition runs as a background task to prevent UI blocking.
- **File History**: Displays a list of uploaded files with their recognition status.
- **Data Visualization**: Renders a plot of emotion probabilities over time for successfully processed files.
- **Localization**: UI elements and labels are translated for a better user experience.

## Repository Layout

- `backend/`: The FastAPI backend application.
  - `app/main.py`: FastAPI application entrypoint.
  - `app/api/v1/endpoints.py`: API endpoints for file handling and recognition.
  - `app/db/`: SQLAlchemy models and database session management.
  - `app/services/`: Business logic for file handling and the recognition pipeline.
  - `data/uploads/`: Directory for storing uploaded audio files.
  - `data/results/`: Directory for storing recognition JSON results.
  - `models/`: Contains the machine learning model, training, and inference code.
  - `requirements.txt`: A list of Python dependencies for the backend.
- `frontend/`: The React single-page application.
  - `src/`: The main source code for the React application.
  - `src/services/api.js`: API wrappers for communicating with the backend.
  - `package.json`: Contains scripts and dependencies for the frontend.

## Requirements

- **Python 3.10+**: For the backend.
- **Node.js 18+**: For the frontend.
- **PostgreSQL**: The recommended database for production.

## Backend — Setup & Run

1. **Install Dependencies**:

   ```bash
   cd backend
   python -m venv .venv
   source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
   pip install -r requirements.txt
   ```
2. **Configure Environment**:
   Create a `.env` file in the `backend/` directory with your database URL:

   ```
   DATABASE_URL=postgresql://user:password@localhost:5432/emotion_db
   ```
3. **Run the Application**:

   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

   The API will be available at `http://localhost:8000/api/v1`.

## Frontend — Setup & Run

1. **Install Dependencies**:

   ```bash
   cd frontend
   npm install
   ```
2. **Start the Development Server**:

   ```bash
   npm run dev
   ```

   The frontend will be accessible at `http://localhost:5173`.

## Backend Endpoints

- `POST /api/v1/files`: Uploads an audio file.
- `GET /api/v1/files`: Retrieves a list of all uploaded files and their recognition status.
- `POST /api/v1/files/{file_id}/recognize`: Starts the emotion recognition process for a given file.
- `GET /api/v1/files/{file_id}/recognition`: Fetches the recognition results for a completed analysis.

## Models

The `models/emotion_recognition/` directory includes:

- `model_training.py`: Scripts for training the emotion recognition model.
- `fine-tuned-emotion-model`: The trained model saved in `.safetensors` format.

To integrate a different model, replace the existing model files and update the prediction logic in `backend/app/services/recognition.py`.

## Tests

To run the backend tests, use `pytest`:

```bash
cd backend
pytest
```

## Troubleshooting

- **CORS Issues**: If the frontend and backend are on different ports, update the CORS settings in `app/main.py`.
- **Database Connection**: Ensure your database URL is correct and the database server is running.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## TODO

- Upgrade fuzzy logic;
- Configure normal .env file;
- Add docker file;
