// src/components/FileUpload.jsx
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
      onUploadSuccess(); // Notify parent component to refresh file list
      setSelectedFile(null); // Reset the input
    } catch (error) {
      console.error('Error uploading file:', error);
      alert('Неудалось загрузить файл.');
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div>
      <h3>Загрузка аудио файла</h3>
      <input type="file" accept=".mp3,.wav" onChange={handleFileChange} />
      <button onClick={handleUpload} disabled={isUploading || !selectedFile}>
        {isUploading ? 'Загрузка...' : 'Загрузить'}
      </button>
    </div>
  );
};

export default FileUpload;
