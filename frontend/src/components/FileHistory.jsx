import React, { useState, useEffect } from 'react';
import { startRecognition, getRecognitionProgress } from '../services/api';

const STATUS_TRANSLATIONS = {
  success: 'Успешно',
  in_progress: 'В обработке',
  error: 'Ошибка',
  pending: 'В ожидании',
};

const FileHistory = ({ files = [], onRecognizeStart, onSelectFile, onRefresh, isLoading }) => {
  const [selectedFileId, setSelectedFileId] = useState(null);
  const [progressData, setProgressData] = useState({});

  const handleRecognize = async (e, fileId) => {
    e.stopPropagation(); 
    try {
      await startRecognition(fileId);
      onRecognizeStart();
    } catch (error) {
      console.error('Error starting recognition:', error);
      alert('Неудалось начать распознавание: ' + error.message);
    }
  };

  const handleSelect = (file) => {
    setSelectedFileId(file.file_id);
    onSelectFile(file);
  };

  // Track progress for files in progress
  useEffect(() => {
    const arr = Array.isArray(files) ? files : [];
    const inProgressFiles = arr.filter(
      file => file.recognition && file.recognition.recognition_status === 'in_progress'
    );
    if (inProgressFiles.length === 0) return;
  
    const progressInterval = setInterval(async () => {
      for (const file of inProgressFiles) {
        try {
          const response = await getRecognitionProgress(file.file_id);
          const progress = response.data.progress;
          setProgressData(prev => ({ ...prev, [file.file_id]: progress }));
          if (progress >= 100) {
            setTimeout(() => { onRefresh(); }, 1000);
          }
        } catch (error) {
          console.error('Error fetching progress:', error);
        }
      }
    }, 2000);
  
    return () => clearInterval(progressInterval);
  }, [files, onRefresh]);

  const getStatusClass = (status) => {
    switch (status) {
      case 'success': return 'status-success';
      case 'in_progress': return 'status-processing';
      case 'error': return 'status-failed';
      case 'pending': return 'status-pending';
      default: return 'status-pending';
    }
  };
  const list = Array.isArray(files) ? files : [];
  return (
    <>
      <div className="file-history-header">
        <h3>Загруженные файлы</h3>
        <button onClick={onRefresh} disabled={isLoading} className="refresh-btn">
          {isLoading ? 'Обновление...' : 'Обновить'}
        </button>
      </div>
      <ul className="file-history-list">
        {list.map((file) => (
          <li
            key={file.file_id}
            className={`file-history-item ${selectedFileId === file.file_id ? 'selected' : ''}`}
            onClick={() => handleSelect(file)}
          >
            <div className="file-info">
              <span>{file.file_name} </span>
              <small>{new Date(file.upload_date).toLocaleString()}</small>
            </div>
            <div className="file-actions">
              <span className={`file-status ${getStatusClass(file.recognition.recognition_status)}`}>
                {STATUS_TRANSLATIONS[file.recognition.recognition_status] || file.recognition.recognition_status}
              </span>
              {file.recognition.recognition_status === 'in_progress' && (
                <div className="progress-container">
                  <div className="progress-bar">
                    <div 
                      className="progress-fill" 
                      style={{ width: `${progressData[file.file_id] || 0}%` }}
                    ></div>
                  </div>
                  <span className="progress-text">
                    {progressData[file.file_id] || 0}%
                  </span>
                </div>
              )}
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
                  Посмотреть результат
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
