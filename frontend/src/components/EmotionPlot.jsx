import React, { useState, useMemo } from 'react';
import { Line } from 'react-chartjs-2';
import { getAudioUrl } from '../services/api';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend
} from 'chart.js';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

const EMOTION_COLORS = {
    angry: 'rgba(255, 99, 132, 1)',
    sad: 'rgba(54, 162, 235, 1)',
    neutral: 'rgba(201, 203, 207, 1)',
    positive: 'rgba(75, 192, 75, 1)',
    other: 'rgba(153, 102, 255, 1)',
};

const EMOTION_TRANSLATIONS = {
    angry: 'Злость',
    sad: 'Грусть',
    neutral: 'Нейтральность',
    positive: 'Позитив',
    other: 'Другое',
};

const EmotionPlot = ({ recognitionData, fileId }) => {
    const [hoveredDatasetIndex, setHoveredDatasetIndex] = useState(null);

    const handleExportJson = () => {
        if (!recognitionData) return;
        const jsonString = JSON.stringify(recognitionData, null, 2);
        const blob = new Blob([jsonString], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `recognition_result_${fileId}.json`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    };

    const datasets = useMemo(() => {
        if (!recognitionData || !recognitionData.segments || recognitionData.segments.length === 0) {
            return [];
    }
        const emotions = Object.keys(recognitionData.segments[0].probabilities);
        return emotions.map((emotion, index) => {
            const originalColor = EMOTION_COLORS[emotion] || 'rgba(0,0,0,1)';
            const isHovered = index === hoveredDatasetIndex;
            const isAnotherHovered = hoveredDatasetIndex !== null && !isHovered;
            
            let lineBorderColor = originalColor;
            if (isAnotherHovered) {
                const rgb = originalColor.match(/\d+/g).slice(0, 3).join(', ');
                lineBorderColor = `rgba(${rgb}, 0.2)`; // Dim the color for the line
            }

            return {
                label: EMOTION_TRANSLATIONS[emotion] || emotion,
                data: recognitionData.segments.map(s => s.probabilities[emotion]),
                fill: false,
                borderColor: lineBorderColor, 
                originalColor: originalColor, 
                borderWidth: isHovered ? 4 : 2,
                pointRadius: 0,
            };
        });
    }, [recognitionData, hoveredDatasetIndex]);


    if (!recognitionData || !recognitionData.segments || recognitionData.segments.length === 0) {
        return <p style={{ textAlign: 'center', marginTop: '40px', color: '#667' }}>Выберите успешно распознанный файл для просмотра графика.</p>;
    }

    const audioUrl = fileId ? getAudioUrl(fileId) : null;
    const averageMood = recognitionData.average_mood ? EMOTION_TRANSLATIONS[recognitionData.average_mood] : recognitionData.average_mood;
    const labels = recognitionData.segments.map(segment => (segment.start_ms / 1000).toFixed(1) + 's');
    const data = { labels, datasets };

    const options = {
        responsive: true,
        maintainAspectRatio: false,
        onHover: (event, chartElement, chart) => {
            const hoveredElements = chart.getElementsAtEventForMode(event, 'dataset', { intersect: false }, true);
            if (hoveredElements.length > 0) {
                setHoveredDatasetIndex(hoveredElements[0].datasetIndex);
            }
        },
        plugins: {
            legend: {
                position: 'top',
                labels: {
                    generateLabels: function(chart) {
                        const originalLabels = ChartJS.defaults.plugins.legend.labels.generateLabels(chart);
                        originalLabels.forEach(label => {
                            const dataset = chart.data.datasets[label.datasetIndex];
                            if (dataset && dataset.originalColor) {
                                label.fillStyle = dataset.originalColor; 
                            }
                        });
                        return originalLabels;
                    }
                }
            },
            tooltip: {
                enabled: true,
                usePointStyle: true,
                itemSort: (a, b) => b.parsed.y - a.parsed.y,
                callbacks: {
                    label: (context) => `${context.dataset.label}: ${context.parsed.y.toFixed(3)}`,
                    labelColor: function(context) {
                        const originalColor = context.chart.data.datasets[context.datasetIndex].originalColor;
                        return {
                            borderColor: originalColor,
                            backgroundColor: originalColor,
                        };
                    },
                }
            }
        },
        interaction: {
            mode: 'index',
            intersect: false,
        },
        scales: {
            y: { min: 0, max: 1, title: { display: true, text: 'Вероятность' } },
            x: { title: { display: true, text: 'Время (секунды)' } }
        }
    };

    return (
        <div className="chart-container" style={{ height: '500px', position: 'relative' }}>
            <div style={{ marginBottom: '20px' }}>
                <h3>
                    Среднее настроение: <span style={{ color: EMOTION_COLORS[recognitionData.average_mood] || '#000' }}>{averageMood}</span>
                </h3>
                {audioUrl && <audio controls src={audioUrl} style={{ width: '100%' }} />}
                <button onClick={handleExportJson} style={{ marginTop: '10px', padding: '8px 12px', cursor: 'pointer' }}>
                    Экспорт в JSON
                </button>
            </div>
            <Line 
                options={options} 
                data={data} 
                onMouseLeave={() => {
                    setHoveredDatasetIndex(null);
                }}
            />
        </div>
    );
};

export default EmotionPlot;

