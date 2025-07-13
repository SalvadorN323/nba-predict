/**
 * API service for NBA Predictor frontend.
 * 
 * This service handles all API calls to the backend,
 * providing a centralized interface for data communication.
 */

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://127.0.0.1:8080/api/v1';

class ApiService {
  /**
   * Make a prediction for a matchup between two teams.
   * 
   * @param {string} teamA - Name of the first team
   * @param {string} teamB - Name of the second team
   * @returns {Promise<Object>} Prediction results
   */
  static async predictMatchup(teamA, teamB) {
    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ team_a: teamA, team_b: teamB }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Prediction request failed');
      }

      return await response.json();
    } catch (error) {
      console.error('Prediction API error:', error);
      throw error;
    }
  }

  /**
   * Get list of available NBA teams.
   * 
   * @returns {Promise<Array>} List of team names
   */
  static async getTeams() {
    try {
      const response = await fetch(`${API_BASE_URL}/teams`);

      if (!response.ok) {
        throw new Error('Failed to fetch teams');
      }

      const data = await response.json();
      return data.teams || [];
    } catch (error) {
      console.error('Teams API error:', error);
      throw error;
    }
  }

  /**
   * Check the health status of the API.
   * 
   * @returns {Promise<Object>} Health status
   */
  static async checkHealth() {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);

      if (!response.ok) {
        throw new Error('Health check failed');
      }

      return await response.json();
    } catch (error) {
      console.error('Health check error:', error);
      throw error;
    }
  }

  /**
   * Get detailed health status including component checks.
   * 
   * @returns {Promise<Object>} Detailed health status
   */
  static async getDetailedHealth() {
    try {
      const response = await fetch(`${API_BASE_URL}/health/detailed`);

      if (!response.ok) {
        throw new Error('Detailed health check failed');
      }

      return await response.json();
    } catch (error) {
      console.error('Detailed health check error:', error);
      throw error;
    }
  }
}

export default ApiService; 