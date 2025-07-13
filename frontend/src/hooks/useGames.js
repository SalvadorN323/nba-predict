/**
 * Custom hook for NBA games functionality.
 * 
 * This hook manages the games state and API calls,
 * providing a clean interface for games operations.
 */

import { useState, useEffect, useCallback } from 'react';
import GamesApiService from '../services/gamesApi';
import { toast } from 'react-toastify';

export const useGames = () => {
  const [games, setGames] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  /**
   * Fetch test games (June 22, 2025 - NBA Finals game).
   */
  const fetchTomorrowGames = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const gamesData = await GamesApiService.getTomorrowGames();
      setGames(gamesData);
      
      if (gamesData.length === 0) {
        toast.info('No games scheduled for June 22, 2025 (NBA Finals test date).');
      }
    } catch (error) {
      setError(error.message);
      
      // Show different messages based on error type
      if (error.message.includes('Rate limit exceeded')) {
        toast.warning(`Rate limit exceeded. Please wait before trying again.`);
      } else {
        toast.error(`Failed to fetch games: ${error.message}`);
      }
    } finally {
      setLoading(false);
    }
  }, []);

  /**
   * Fetch games for specific dates.
   * 
   * @param {string|Array} dates - Date(s) in YYYY-MM-DD format. Can be a single date string or array of dates
   */
  const fetchGamesByDateRange = useCallback(async (dates) => {
    setLoading(true);
    setError(null);

    try {
      const data = await GamesApiService.getGames(dates);
      setGames(data.data || []);
    } catch (error) {
      setError(error.message);
      toast.error(`Failed to fetch games: ${error.message}`);
    } finally {
      setLoading(false);
    }
  }, []);

  /**
   * Search for teams.
   * 
   * @param {string} searchTerm - Search term for team name
   * @returns {Promise<Array>} Array of matching teams
   */
  const searchTeams = useCallback(async (searchTerm) => {
    try {
      return await GamesApiService.searchTeams(searchTerm);
    } catch (error) {
      console.error('Team search failed:', error);
      throw error;
    }
  }, []);

  /**
   * Get team information.
   * 
   * @param {number} teamId - Team ID
   * @returns {Promise<Object>} Team data
   */
  const getTeam = useCallback(async (teamId) => {
    try {
      return await GamesApiService.getTeam(teamId);
    } catch (error) {
      console.error('Team fetch failed:', error);
      throw error;
    }
  }, []);

  // Fetch test games (NBA Finals) on component mount
  useEffect(() => {
    fetchTomorrowGames();
  }, [fetchTomorrowGames]);

  return {
    // State
    games,
    loading,
    error,
    
    // Actions
    fetchTomorrowGames,
    fetchGamesByDateRange,
    searchTeams,
    getTeam,
    
    // Utility
    retry: fetchTomorrowGames, // Allow retrying the fetch
  };
}; 