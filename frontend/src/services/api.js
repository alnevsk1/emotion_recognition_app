// src/services/api.js
import axios from 'axios';

const apiClient = axios.create({
  baseURL: 'http://localhost:8000/api/v1', // Backend URL
  headers: {
    'Content-Type': 'application/json',
  },
});

export const uploadFile = (file) => {
  const formData = new FormData();
  formData.append('file', file);
  return apiClient.post('/files', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
};

export const getFiles = () => {
  return apiClient.get('/files');
};

export const startRecognition = (fileId) => {
  return apiClient.post(`/files/${fileId}/recognize`);
};

export const getRecognitionResult = (fileId) => {
  return apiClient.get(`/files/${fileId}/recognition`);
};

export const getAudioUrl = (fileId) => {
  return `${apiClient.defaults.baseURL}/files/${fileId}/audio`;
};