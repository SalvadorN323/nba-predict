/**
 * Custom hook for NBA prediction functionality.
 * 
 * This hook manages the prediction state and API calls,
 * providing a clean interface for prediction operations.
 */

import { useState, useCallback } from 'react';
import ApiService from '../services/api';
import { toast } from 'react-toastify';

export const usePrediction = () => {
  const [prediction, setPrediction] = useState('');
  const [winProbability, setWinProbability] = useState(null);
  const [lossProbability, setLossProbability] = useState(null);
  const [graphData, setGraphData] = useState(null);
  const [analysis, setAnalysis] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  /**
   * Make a prediction for a matchup.
   * 
   * @param {string} teamA - Name of the first team
   * @param {string} teamB - Name of the second team
   */
  const makePrediction = useCallback(async (teamA, teamB) => {
    if (!teamA || !teamB) {
      setError('Please select both teams');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const data = await ApiService.predictMatchup(teamA, teamB);
      
      setPrediction(data.prediction);
      setWinProbability(data.win_probability);
      setLossProbability(data.loss_probability);
      setGraphData(data.graph_data);
      setAnalysis(data.analysis);
      
      toast.success('Prediction completed successfully!');
    } catch (error) {
      setError(error.message);
      toast.error(`Prediction failed: ${error.message}`);
    } finally {
      setLoading(false);
    }
  }, []);

  /**
   * Clear the current prediction results.
   */
  const clearPrediction = useCallback(() => {
    setPrediction('');
    setWinProbability(null);
    setLossProbability(null);
    setGraphData(null);
    setAnalysis('');
    setError(null);
  }, []);

  /**
   * Check the health status of the API.
   * 
   * @returns {Promise<Object>} Health status
   */
  const checkApiHealth = useCallback(async () => {
    try {
      return await ApiService.checkHealth();
    } catch (error) {
      console.error('Health check failed:', error);
      throw error;
    }
  }, []);

  return {
    // State
    prediction,
    winProbability,
    lossProbability,
    graphData,
    analysis,
    loading,
    error,
    
    // Actions
    makePrediction,
    clearPrediction,
    checkApiHealth,
  };
}; 