// src/components/EmotionPlot.jsx
import React from 'react';
import { Line } from 'react-chartjs-2';
import { getAudioUrl } from '../services/api';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend } from 'chart.js';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

const EMOTION_COLORS = {
  angry: 'rgba(255, 99, 132, 1)',    // Red
  sad: 'rgba(54, 162, 235, 1)',      // Blue
  neutral: 'rgba(201, 203, 207, 1)', // Grey
  positive: 'rgba(75, 192, 192, 1)',  // Green
  other: 'rgba(153, 102, 255, 1)',   // Purple
};

const EMOTION_TRANSLATIONS = {
  angry: 'Злость',
  sad: 'Грусть',
  neutral: 'Нейтральность',
  positive: 'Позитив',
  other: 'Другое',
};

const EmotionPlot = ({ recognitionData, fileId }) => {
  if (!recognitionData || !recognitionData.segments || recognitionData.segments.length === 0) {
    return <p style={{ textAlign: 'center', marginTop: '40px', color: '#667' }}>Выберите успешно распознанный файл для просмотра графика.</p>;
  }


  const audioUrl = fileId ? getAudioUrl(fileId) : null;
  const averageMood = recognitionData.average_mood ? EMOTION_TRANSLATIONS[recognitionData.average_mood] || recognitionData.average_mood : 'Не определено';


  const labels = recognitionData.segments.map(segment => `${segment.start_ms / 1000}s`);
  const datasets = Object.keys(recognitionData.segments[0].probabilities).map(emotion => {
    const color = EMOTION_COLORS[emotion] || `rgba(${Math.random() * 255},${Math.random() * 255},${Math.random() * 255},1)`;
    return {
      label: EMOTION_TRANSLATIONS[emotion] || emotion,
      data: recognitionData.segments.map(s => s.probabilities[emotion]),
      fill: false,
      borderColor: color,
      backgroundColor: color,
    };
  });

  const data = { labels, datasets };

  const options = {
    responsive: true,
    maintainAspectRatio: false, 
    plugins: {
      legend: { position: 'top' },
      title: { display: true, text: 'Вероятность эмоций с течением времени', font: { size: 16 } },
    },
    scales: {
      y: {
        min: 0,
        max: 1,
        title: { display: true, text: 'Вероятность' }
      },
      x: {
        title: { display: true, text: 'Время (секунды)' }
      }
    }
  };

  return (
    <div className="chart-container" style={{ height: '500px', position: 'relative' }}>
      {/* Audio and average mood block here */}
      <div style={{ marginBottom: '20px' }}>
        <h3>Среднее настроение: <span style={{ color: EMOTION_COLORS[recognitionData.average_mood] || '#000' }}>{averageMood}</span></h3>
        {audioUrl && (
          <audio controls src={audioUrl} style={{ width: '100%' }}>
            Your browser does not support the audio element.
          </audio>
        )}
      </div>
      <Line options={options} data={data} />
    </div>
  );
};

export default EmotionPlot;
