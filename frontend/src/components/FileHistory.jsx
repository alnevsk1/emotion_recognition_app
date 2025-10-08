// src/components/FileHistory.jsx
import React, { useState } from 'react';
import { startRecognition } from '../services/api';

const STATUS_TRANSLATIONS = {
  success: 'Успешно',
  in_progress: 'В обработке',
  error: 'Ошибка',
  pending: 'В ожидании',
};

const FileHistory = ({ files, onRecognizeStart, onSelectFile, onRefresh, isLoading }) => {
  const [selectedFileId, setSelectedFileId] = useState(null);

  const handleRecognize = async (e, fileId) => {
    e.stopPropagation(); 
    try {
      await startRecognition(fileId);
      onRecognizeStart();
    } catch (error) {
      console.error('Error starting recognition:', error);
      alert('Неудалось начать распознавание.');
    }
  };

  const handleSelect = (file) => {
    setSelectedFileId(file.file_id);
    onSelectFile(file);
  };

  const getStatusClass = (status) => {
    switch (status) {
      case 'success': return 'status-success';
      case 'in_progress': return 'status-processing';
      case 'error': return 'status-failed';
      case 'pending': return 'status-pending';
      default: return 'status-pending';
    }
  };

  return (
    <>
      <div className="file-history-header">
        <h3>Файлы</h3>
        <button onClick={onRefresh} disabled={isLoading} className="refresh-btn">
          {isLoading ? 'Обновление...' : 'Обновить'}
        </button>
      </div>
      <ul className="file-history-list">
        {files.map((file) => (
          <li
            key={file.file_id}
            className={`file-history-item ${selectedFileId === file.file_id ? 'selected' : ''}`}
            onClick={() => handleSelect(file)}
          >
            <div className="file-info">
              <span>{file.file_name}</span>
              <small>{new Date(file.upload_date).toLocaleString()}</small>
            </div>
            <div className="file-actions">
              <span className={`file-status ${getStatusClass(file.recognition.recognition_status)}`}>
                {STATUS_TRANSLATIONS[file.recognition.recognition_status] || file.recognition.recognition_status}
              </span>
              <button
                onClick={(e) => handleRecognize(e, file.file_id)}
                disabled={file.recognition.recognition_status === 'in_progress' || file.recognition.recognition_status === 'success'}
              >
                Распознать
              </button>
              {file.recognition.recognition_status === 'success' && (
                <button
                  className="view-plot-btn"
                  onClick={(e) => {
                    e.stopPropagation();
                    onSelectFile(file);
                  }}
                >
                  Посмотреть график
                </button>
              )}
            </div>
          </li>
        ))}
      </ul>
    </>
  );
};

export default FileHistory;
