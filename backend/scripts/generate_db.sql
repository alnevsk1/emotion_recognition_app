CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE audio_files (

    file_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    file_name VARCHAR(255) NOT NULL,

    file_path TEXT NOT NULL,

    file_extension VARCHAR(10) CHECK (file_extension IN ('mp3', 'wav')) NOT NULL,

    upload_date TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL
);

CREATE TYPE recognition_status_enum AS ENUM (
    'pending',
    'in_progress',
    'success',
    'error'
);

CREATE TABLE audio_emotion_recognition (
    recognition_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    file_id UUID NOT NULL REFERENCES audio_files(file_id) ON DELETE CASCADE,

    recognition_path TEXT NOT NULL,

    recognition_date TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    
    recognition_status recognition_status_enum NOT NULL
);
