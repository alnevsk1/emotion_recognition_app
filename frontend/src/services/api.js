// src/services/api.js
import axios from 'axios';

// Determine base URL based on environment
const getBaseURL = () => {
  // Check if running in Docker (nginx proxy) or locally
  if (window.location.hostname === 'localhost' && window.location.port === '5173') {
    // Local development - direct connection to backend
    return 'http://localhost:8000/api/v1';
  }
  // Docker/production - use relative path for nginx proxy
  return '/api/v1';
};

const apiClient = axios.create({
  baseURL: getBaseURL(),
  headers: {
    'Content-Type': 'application/json',
  },
});

apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response && error.response.data) {
      const customError = new Error(error.response.data.detail || 'Ошибка сервера');
      customError.status = error.response.status;
      customError.detail = error.response.data.detail;
      return Promise.reject(customError);
    }
    return Promise.reject(error);
  }
);

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

export const getRecognitionProgress = (fileId) => {
  return apiClient.get(`/files/${fileId}/progress`);
};