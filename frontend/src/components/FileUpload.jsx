import React, { useState } from 'react';
import { uploadFile } from '../services/api';

const FileUpload = ({ onUploadSuccess }) => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      alert('Сначала выберите файл.');
      return;
    }
    setIsUploading(true);
    try {
      await uploadFile(selectedFile);
      onUploadSuccess(); 
      setSelectedFile(null);
    } catch (error) {
      console.error('Error uploading file:', error);
      alert('Неудалось загрузить файл: ' + error.message); 
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div>
      <h3>Загрузка аудио файла: .mp3 или .wav</h3>
      <input type="file" accept=".mp3,.wav" onChange={handleFileChange} />
      <button onClick={handleUpload} disabled={isUploading || !selectedFile}>
        {isUploading ? 'Загрузка...' : 'Загрузить'}
      </button>
    </div>
  );
};

export default FileUpload;
