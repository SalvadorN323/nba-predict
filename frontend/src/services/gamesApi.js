/**
 * BallDon'tLie API service for NBA Predictor frontend.
 * 
 * This service handles all API calls to the BallDon'tLie API
 * for fetching upcoming games and team data.
 */

import { ballDontLieRateLimiter, RateLimiter } from '../utils/rateLimiter';

const BALLDONTLIE_API_URL = 'https://api.balldontlie.io/v1';
const API_KEY = 'd1710c2b-4c0f-4f21-ac88-487e55085ea7';

class GamesApiService {
  /**
   * Fetch games for specific dates.
   * 
   * @param {string|Array} dates - Date(s) in YYYY-MM-DD format. Can be a single date string or array of dates
   * @returns {Promise<Object>} Games data
   */
  static async getGames(dates) {
    try {
      // Check rate limit before making request
      if (!ballDontLieRateLimiter.canMakeRequest()) {
        const delay = ballDontLieRateLimiter.getDelay();
        throw new Error(`Rate limit exceeded. Please wait ${Math.ceil(delay / 1000)} seconds before trying again.`);
      }

      // Convert single date to array if needed
      const dateArray = Array.isArray(dates) ? dates : [dates];
      
      // Build the query string with dates[] parameters
      const dateParams = dateArray.map(date => `dates[]=${date}`).join('&');
      
      // Record the request
      ballDontLieRateLimiter.recordRequest();
      
      const response = await fetch(
        `${BALLDONTLIE_API_URL}/games?${dateParams}`,
        {
          headers: {
            'Authorization': API_KEY
          }
        }
      );

      if (response.status === 429) {
        // Handle rate limit error
        const retryAfter = response.headers.get('Retry-After');
        const retrySeconds = retryAfter ? parseInt(retryAfter) : 60;
        ballDontLieRateLimiter.handleRateLimit(retrySeconds);
        throw new Error(`Rate limit exceeded. Please try again in ${retrySeconds} seconds.`);
      }

      if (!response.ok) {
        throw new Error(`Failed to fetch games: ${response.status} ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Games API error:', error);
      throw error;
    }
  }

  /**
   * Fetch games for testing (June 22, 2025 - NBA Finals game).
   * 
   * @returns {Promise<Array>} Array of games
   */
  static async getTomorrowGames() {
    // Use June 22, 2025 as test date - NBA Finals game
    const tomorrowDate = '2025-06-22';

    try {
      const data = await this.getGames(tomorrowDate);
      return data.data || [];
    } catch (error) {
      console.error('Error fetching tomorrow games:', error);
      throw error;
    }
  }

  /**
   * Fetch team information.
   * 
   * @param {number} teamId - Team ID
   * @returns {Promise<Object>} Team data
   */
  static async getTeam(teamId) {
    try {
      // Check rate limit before making request
      if (!ballDontLieRateLimiter.canMakeRequest()) {
        const delay = ballDontLieRateLimiter.getDelay();
        throw new Error(`Rate limit exceeded. Please wait ${Math.ceil(delay / 1000)} seconds before trying again.`);
      }

      // Record the request
      ballDontLieRateLimiter.recordRequest();

      const response = await fetch(
        `${BALLDONTLIE_API_URL}/teams/${teamId}`,
        {
          headers: {
            'Authorization': API_KEY
          }
        }
      );

      if (response.status === 429) {
        // Handle rate limit error
        const retryAfter = response.headers.get('Retry-After');
        const retrySeconds = retryAfter ? parseInt(retryAfter) : 60;
        ballDontLieRateLimiter.handleRateLimit(retrySeconds);
        throw new Error(`Rate limit exceeded. Please try again in ${retrySeconds} seconds.`);
      }

      if (!response.ok) {
        throw new Error(`Failed to fetch team data: ${response.status} ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Team API error:', error);
      throw error;
    }
  }

  /**
   * Search for teams by name.
   * 
   * @param {string} searchTerm - Search term for team name
   * @returns {Promise<Array>} Array of matching teams
   */
  static async searchTeams(searchTerm) {
    try {
      // Check rate limit before making request
      if (!ballDontLieRateLimiter.canMakeRequest()) {
        const delay = ballDontLieRateLimiter.getDelay();
        throw new Error(`Rate limit exceeded. Please wait ${Math.ceil(delay / 1000)} seconds before trying again.`);
      }

      // Record the request
      ballDontLieRateLimiter.recordRequest();

      const response = await fetch(
        `${BALLDONTLIE_API_URL}/teams?search=${encodeURIComponent(searchTerm)}`,
        {
          headers: {
            'Authorization': API_KEY
          }
        }
      );

      if (response.status === 429) {
        // Handle rate limit error
        const retryAfter = response.headers.get('Retry-After');
        const retrySeconds = retryAfter ? parseInt(retryAfter) : 60;
        ballDontLieRateLimiter.handleRateLimit(retrySeconds);
        throw new Error(`Rate limit exceeded. Please try again in ${retrySeconds} seconds.`);
      }

      if (!response.ok) {
        throw new Error(`Failed to search teams: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      return data.data || [];
    } catch (error) {
      console.error('Team search error:', error);
      throw error;
    }
  }
}

export default GamesApiService; 