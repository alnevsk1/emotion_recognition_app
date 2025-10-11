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
  const [selectedFileId, setSelectedFileId] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const fetchFiles = async () => {
    setIsLoading(true);
    try {
      const response = await getFiles();
      setFiles(response.data);
    } catch (error) {
      console.error("Неудалось обработать файлы:", error);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchFiles();
  }, []);

  const handleSelectFileForResult = async (file) => {
    if (file.recognition.recognition_status === 'success') {
      try {
        const resultResponse = await getRecognitionResult(file.file_id);
        setSelectedResult(resultResponse.data);
        setSelectedFileId(file.file_id);
      } catch (error) {
        console.error("Неудалось обработать результат распознавания:", error);
        setSelectedResult(null);
        setSelectedFileId(null);
      }
    } else {
      setSelectedResult(null);
      setSelectedFileId(null);
    }
  };

  return (
    <div className="App">
      <header className="app-header">
        <h1>Распознавание эмоций в речи</h1>
      </header>

      <aside className="sidebar">
        <div className="section-container">
          <FileUpload onUploadSuccess={fetchFiles} />
        </div>
        <div className="section-container">
          <FileHistory
            files={files}
            onRecognizeStart={fetchFiles}
            onSelectFile={handleSelectFileForResult}
            onRefresh={fetchFiles}
            isLoading={isLoading}
          />
        </div>
      </aside>

      <main className="main-content">
        <div className="plot-container">
          <h2>Анализ графика эмоций</h2>
          <EmotionPlot recognitionData={selectedResult} fileId={selectedFileId} />
        </div>
      </main>
    </div>
  );
};

export default App;
