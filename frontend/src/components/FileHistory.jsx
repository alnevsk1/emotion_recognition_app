// src/components/FileHistory.jsx
import React from 'react';
import { startRecognition } from '../services/api';

const FileHistory = ({ files, onRecognizeStart, onSelectFile }) => {
  const handleRecognize = async (fileId) => {
    try {
      await startRecognition(fileId);
      onRecognizeStart();
    } catch (error) {
      console.error('Error starting recognition:', error);
      alert('Failed to start recognition.');
    }
  };

  const getStatusClass = (status) => {
    switch (status) {
      case 'success': return 'success';
      case 'inprogress': return 'inprogress';
      case 'error': return 'error';
      case 'pending': return 'pending';
      default: return 'default';
    }
  };

  return (
    <div>
      {/* Table is now styled via styles.css */}
      <table>
        <thead>
          <tr>
            <th>Filename</th>
            <th>Upload Date</th>
            <th>Status</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {files.map((file) => (
            <tr key={file.fileid}>
              <td data-label="Filename">{file.filename}</td>
              <td data-label="Upload Date">{new Date(file.uploaddate).toLocaleString()}</td>
              <td data-label="Status">
                <span className={`status ${getStatusClass(file.recognitionstatus)}`}>
                  {file.recognitionstatus || 'Not Started'}
                </span>
              </td>
              <td data-label="Actions">
                <button onClick={() => handleRecognize(file.fileid)} disabled={file.recognitionstatus === 'inprogress'}>
                  Recognize
                </button>
                <button onClick={() => onSelectFile(file)} disabled={file.recognitionstatus !== 'success'} style={{marginLeft: '10px'}}>
                  View Plot
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default FileHistory;
