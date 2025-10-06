// src/App.jsx
import React, { useState, useEffect } from 'react';
import FileUpload from './components/FileUpload';
import FileHistory from './components/FileHistory';
import EmotionPlot from './components/EmotionPlot';
import { getFiles, getRecognitionResult } from './services/api';
import './assets/styles.css';

const App = () => {
  const [files, setFiles] = useState([]);
  const [selectedResult, setSelectedResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const fetchFiles = async () => {
    setIsLoading(true);
    try {
      const response = await getFiles();
      setFiles(response.data);
    } catch (error) {
      console.error("Failed to fetch files:", error);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchFiles();
    const interval = setInterval(fetchFiles, 5000);
    return () => clearInterval(interval);
  }, []);

  const handleSelectFileForResult = async (file) => {
    if (file.recognitionstatus === 'success') {
      try {
        const resultResponse = await getRecognitionResult(file.fileid);
        setSelectedResult(resultResponse.data);
      } catch (error) {
        console.error("Failed to fetch recognition result:", error);
      }
    }
  };

  return (
    <div className="App">
      <header className="app-header">
        <h1>Speech Emotion Recognition</h1>
      </header>

      <aside className="sidebar">
        <div className="section-container">
          <h2>1. Upload Audio</h2>
          <FileUpload onUploadSuccess={fetchFiles} />
        </div>
        <div className="section-container" style={{ flexGrow: 1 }}>
          <h2>2. File History</h2>
          {isLoading && <p>Refreshing file list...</p>}
          <FileHistory
            files={files}
            onRecognizeStart={fetchFiles}
            onSelectFile={handleSelectFileForResult}
          />
        </div>
      </aside>

      <main className="main-content">
        <div className="section-container plot-container">
          <h2>3. Emotion Analysis Plot</h2>
          <EmotionPlot recognitionData={selectedResult} />
        </div>
      </main>
    </div>
  );
};

export default App;
