// src/components/EmotionPlot.jsx
import React from 'react';
import { Line } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend } from 'chart.js';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

const EmotionPlot = ({ recognitionData }) => {
  if (!recognitionData) {
    return <p style={{ textAlign: 'center', marginTop: '40px', color: '#666' }}>Select a successful recognition result to view the plot.</p>;
  }

  // ... (keep the existing data processing logic)
  const labels = recognitionData.segments.map(segment => `${segment.startms / 1000}s`);
  const datasets = Object.keys(recognitionData.segments[0].probabilities).map(emotion => {
    return {
      label: emotion,
      data: recognitionData.segments.map(s => s.probabilities[emotion]),
      fill: false,
      // You can add specific colors for each emotion here
    };
  });

  const data = { labels, datasets };

  const options = {
    responsive: true,
    maintainAspectRatio: false, // This is the key change
    plugins: {
      legend: { position: 'top' },
      title: { display: true, text: 'Emotion Probabilities Over Time', font: { size: 16 } },
    },
    scales: {
      y: {
        min: 0,
        max: 1,
        title: { display: true, text: 'Probability' }
      },
      x: {
        title: { display: true, text: 'Time (seconds)' }
      }
    }
  };

  return <Line options={options} data={data} />;
};

export default EmotionPlot;
