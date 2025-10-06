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
      alert('Please select a file first!');
      return;
    }
    setIsUploading(true);
    try {
      await uploadFile(selectedFile);
      onUploadSuccess(); // Notify parent component to refresh file list
      setSelectedFile(null); // Reset the input
    } catch (error) {
      console.error('Error uploading file:', error);
      alert('File upload failed.');
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div>
      <h2>Upload Audio File</h2>
      <input type="file" accept=".mp3,.wav" onChange={handleFileChange} />
      <button onClick={handleUpload} disabled={isUploading || !selectedFile}>
        {isUploading ? 'Uploading...' : 'Upload'}
      </button>
    </div>
  );
};

export default FileUpload;
