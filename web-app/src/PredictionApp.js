import React, { useState, useEffect } from 'react';

function PredictionApp() {
  const [predictions, setPredictions] = useState([]);
  const [currentPrediction, setCurrentPrediction] = useState('');
  const [index, setIndex] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const [intervalId, setIntervalId] = useState(null);

  useEffect(() => {
    if (isRunning && predictions.length > 0) {
      const id = setInterval(() => {
        setCurrentPrediction(predictions[index]);
        setIndex((prevIndex) => (prevIndex + 1) % predictions.length);
      }, 1000);
      setIntervalId(id);

      // Clear the interval when the effect or component unmounts
      return () => clearInterval(id);
    }
  }, [isRunning, predictions, index]);

  const fetchPredictions = async () => {
    try {
      const response = await fetch('http://localhost:5000/predictions');
      const data = await response.json();
      setPredictions(data);
      setIsRunning(true); // Start displaying predictions immediately after fetching
    } catch (error) {
      console.error('Error fetching predictions:', error);
    }
  };

  const startPrediction = () => {
    if (predictions.length === 0) {
      fetchPredictions();
    } else {
      setIsRunning(true);
    }
  };

  const pausePrediction = () => {
    setIsRunning(false);
  };

  const resetPrediction = () => {
    setIsRunning(false);
    setPredictions([]);
    setIndex(0);
    setCurrentPrediction('');
    clearInterval(intervalId); // Clear the interval on reset
  };

  // Function to determine the class based on prediction
  const getPredictionClass = (prediction) => {
    if (prediction === 'POSITIVE') return 'positive';
    if (prediction === 'NEUTRAL') return 'neutral';
    if (prediction === 'NEGATIVE') return 'negative';
    return ''; // Default class
  };

  return (
    <div className="container my-5">
      <div className="card p-4 shadow">
        <h1 className="text-center mb-4">Emotion Prediction</h1>
        <div className="d-flex justify-content-center mb-4">
          <button 
            className="btn btn-primary mx-2" 
            onClick={startPrediction} 
            disabled={isRunning}
          >
            Start
          </button>
          <button 
            className="btn btn-warning mx-2" 
            onClick={pausePrediction} 
            disabled={!isRunning}
          >
            Pause
          </button>
          <button 
            className="btn btn-danger mx-2" 
            onClick={resetPrediction}
          >
            Reset
          </button>
        </div>
        <h2 className="text-center">Predictions:</h2>
        <h3 className={`text-center ${getPredictionClass(currentPrediction)}`}>
          {currentPrediction}
        </h3>
      </div>
    </div>
  );
}

export default PredictionApp;
