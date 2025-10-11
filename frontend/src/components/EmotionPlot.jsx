// src/components/EmotionPlot.jsx
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

// Emotion colors and translations (no changes)
const EMOTION_COLORS = {
    angry: 'rgba(255, 99, 132, 1)',
    sad: 'rgba(54, 162, 235, 1)',
    neutral: 'rgba(201, 203, 207, 1)',
    positive: 'rgba(75, 192, 192, 1)',
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

    if (!recognitionData || !recognitionData.segments || recognitionData.segments.length === 0) {
        return <p style={{ textAlign: 'center', marginTop: '40px', color: '#667' }}>Выберите успешно распознанный файл для просмотра графика.</p>;
    }
    
    const audioUrl = fileId ? getAudioUrl(fileId) : null;
    const averageMood = recognitionData.average_mood ? EMOTION_TRANSLATIONS[recognitionData.average_mood] : recognitionData.average_mood;

    const datasets = useMemo(() => {
        const emotions = Object.keys(recognitionData.segments[0].probabilities);
        return emotions.map((emotion, index) => {
            // The original, full-opacity color
            const originalColor = EMOTION_COLORS[emotion] || 'rgba(0,0,0,1)';
            const isHovered = index === hoveredDatasetIndex;
            const isAnotherHovered = hoveredDatasetIndex !== null && !isHovered;
            
            // This color will be used for the line on the chart
            let lineBorderColor = originalColor;
            if (isAnotherHovered) {
                const rgb = originalColor.match(/\d+/g).slice(0, 3).join(', ');
                lineBorderColor = `rgba(${rgb}, 0.2)`; // Dim the color for the line
            }

            return {
                label: EMOTION_TRANSLATIONS[emotion] || emotion,
                data: recognitionData.segments.map(s => s.probabilities[emotion]),
                fill: false,
                // 1. Set line color and custom property
                borderColor: lineBorderColor, // Use the dynamic color for the line itself
                originalColor: originalColor, // Store the permanent color for legend/tooltips
                borderWidth: isHovered ? 4 : 2,
                pointRadius: 0,
            };
        });
    }, [recognitionData, hoveredDatasetIndex]);

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
            // 2. Configure the legend to use the original color
            legend: {
                position: 'top',
                labels: {
                    generateLabels: function(chart) {
                        const originalLabels = ChartJS.defaults.plugins.legend.labels.generateLabels(chart);
                        originalLabels.forEach(label => {
                            const dataset = chart.data.datasets[label.datasetIndex];
                            if (dataset && dataset.originalColor) {
                                label.fillStyle = dataset.originalColor; // Use original color for the legend box
                            }
                        });
                        return originalLabels;
                    }
                }
            },
            // 3. Configure the tooltip to use the original color
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

