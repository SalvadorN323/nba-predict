/**
 * NBA Matchup Prediction Form Component
 * 
 * A modern betting-style interface for NBA game predictions
 * with machine learning-powered insights and real-time odds.
 * 
 * Features:
 * - Fetches and displays upcoming NBA games
 * - Allows team selection for predictions
 * - Displays ML predictions with probabilities
 * - Shows statistical comparisons with charts
 * - Provides AI-powered analysis
 * 
 * @author NBA Predictor Team
 * @version 2.0.0
 */

import React, { useState } from 'react';
import { ClipLoader } from 'react-spinners';
import { toast } from 'react-toastify';
import { Bar } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend } from 'chart.js';
import { usePrediction } from '../../hooks/usePrediction';
import { useGames } from '../../hooks/useGames';
import { TEAM_LOGOS, CHART_CONFIG, SPINNER_CONFIG, getTeamAbbreviation } from '../../utils/constants';
import RetryButton from '../common/RetryButton';
import RateLimitInfo from '../common/RateLimitInfo';
import BettingCalculator from '../common/BettingCalculator';
import './MatchupForm.css';

// Register necessary Chart.js components for bar chart functionality
ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

/**
 * Main MatchupForm component for NBA game predictions
 * 
 * @returns {JSX.Element} The rendered matchup form component
 */
const MatchupForm = () => {
  // State for selected teams
  const [teamA, setTeamA] = useState('');
  const [teamB, setTeamB] = useState('');

  // Custom hooks for prediction and games functionality
  const {
    prediction,
    winProbability,
    lossProbability,
    graphData,
    analysis,
    loading: predictionLoading,
    error: predictionError,
    makePrediction,
    clearPrediction,
  } = usePrediction();

  const {
    games,
    loading: gamesLoading,
    error: gamesError,
    fetchTomorrowGames,
    retry: retryGames,
  } = useGames();

  // The test date for games being fetched
  const gamesDate = 'June 22, 2025';

  /**
   * Handle form submission for prediction requests
   * 
   * @param {Event} e - The form submission event
   */
  const handleSubmit = async (e) => {
    e.preventDefault();
    await makePrediction(teamA, teamB);
  };

  /**
   * Handle team selection from game list
   * 
   * @param {string} homeTeam - Home team name
   * @param {string} visitorTeam - Visitor team name
   */
  const handleTeamSelection = (homeTeam, visitorTeam) => {
    console.log('Selecting teams:', { homeTeam, visitorTeam });
    console.log('Team A abbreviation:', getTeamAbbreviation(homeTeam));
    console.log('Team B abbreviation:', getTeamAbbreviation(visitorTeam));
    console.log('Team A logo path:', TEAM_LOGOS[getTeamAbbreviation(homeTeam)]);
    console.log('Team B logo path:', TEAM_LOGOS[getTeamAbbreviation(visitorTeam)]);
    
    setTeamA(homeTeam);
    setTeamB(visitorTeam);
    clearPrediction(); // Clear previous prediction when selecting new teams
  };

  /**
   * Handle refresh of games data
   */
  const handleRefreshGames = () => {
    fetchTomorrowGames();
  };

  return (
    <div className="matchup-form-container">
      {/* Rate limit information */}
      <RateLimitInfo visible={gamesError && gamesError.includes('Rate limit exceeded')} />

      {/* Games Section */}
      <section className="games-section">
        <div className="games-header">
          <h2 className="games-title">Games for {gamesDate}</h2>
        </div>

        {/* Loading spinner while fetching data */}
        {gamesLoading && (
          <div className="loading-container">
            <div className="loading-spinner"></div>
            <p className="loading-text">Loading games...</p>
          </div>
        )}
        
        {/* Error message display */}
        {gamesError && !gamesLoading && (
          <div className="error-container">
            <RetryButton 
              onRetry={retryGames}
              loading={gamesLoading}
              error={gamesError}
            >
              Refresh Games
            </RetryButton>
          </div>
        )}

        {/* Display upcoming games for NBA Finals test date */}
        {!gamesLoading && !gamesError && (
          <div>
            {games.length > 0 ? (
              <ul className="games-list">
                {games.map((game) => (
                  <li key={game.id} className={`game-item ${teamA === game.home_team.full_name && teamB === game.visitor_team.full_name ? 'selected' : ''}`}>
                    <button 
                      onClick={() => handleTeamSelection(game.home_team.full_name, game.visitor_team.full_name)}
                      className="game-button"
                    >
                      <div className="game-teams">
                        <div className="team-info">
                          <img 
                            src={TEAM_LOGOS[game.home_team.abbreviation]} 
                            alt={game.home_team.full_name} 
                            className="team-logo"
                            onError={(e) => {
                              e.target.style.display = 'none';
                            }}
                          />
                          <span className="team-name">{game.home_team.full_name}</span>
                        </div>
                        <span className="vs-text">vs</span>
                        <div className="team-info">
                          <img 
                            src={TEAM_LOGOS[game.visitor_team.abbreviation]} 
                            alt={game.visitor_team.full_name} 
                            className="team-logo"
                            onError={(e) => {
                              e.target.style.display = 'none';
                            }}
                          />
                          <span className="team-name">{game.visitor_team.full_name}</span>
                        </div>
                      </div>

                    </button>
                  </li>
                ))}
              </ul>
                          ) : (
                <p className="no-games">No games scheduled for June 22, 2025 (NBA Finals test date).</p>
              )}
          </div>
        )}
      </section>

      {/* Prediction Section */}
      <section className="prediction-section">
        <div className="prediction-header">
          <h2 className="prediction-title"> Matchup Prediction</h2>
        </div>

        {/* Team Selection */}
        <div className="team-selection">
          <div className="teams-display">
            <div className={`team-card ${teamA ? 'selected' : ''}`}>
              {teamA ? (
                <>
                  <img 
                    src={TEAM_LOGOS[getTeamAbbreviation(teamA)] || '/logos/default_team.png'} 
                    alt={teamA} 
                    className="team-card-logo"
                    onError={(e) => {
                      e.target.style.display = 'none';
                    }}
                  />
                  <div className="team-card-name">{teamA}</div>
                </>
              ) : (
                <div className="team-card-name">Select Team A</div>
              )}
            </div>
            
            <div className="vs-divider">VS</div>
            
            <div className={`team-card ${teamB ? 'selected' : ''}`}>
              {teamB ? (
                <>
                  <img 
                    src={TEAM_LOGOS[getTeamAbbreviation(teamB)] || '/logos/default_team.png'} 
                    alt={teamB} 
                    className="team-card-logo"
                    onError={(e) => {
                      e.target.style.display = 'none';
                    }}
                  />
                  <div className="team-card-name">{teamB}</div>
                </>
              ) : (
                <div className="team-card-name">Select Team B</div>
              )}
            </div>
          </div>

          {/* Prediction Button */}
          <button 
            onClick={handleSubmit}
            className="predict-button"
            disabled={!teamA || !teamB || predictionLoading}
          >
            {predictionLoading ? 'Analyzing Matchup...' : 'Predict Winner'}
          </button>
        </div>

        {/* Prediction Results */}
        {prediction && !predictionLoading && (
          <div className="prediction-results">
            <div className="result-card">
              <h3 className="prediction-result">{prediction}</h3>

              {/* Display win/loss probabilities */}
              {winProbability !== null && lossProbability !== null && (
                <div className="odds-display">
                  <div className="odds-card">
                    <div className="odds-value">{(winProbability * 100).toFixed(1)}%</div>
                    <div className="odds-label">{teamA} Win</div>
                  </div>
                  <div className="odds-card">
                    <div className="odds-value">{(lossProbability * 100).toFixed(1)}%</div>
                    <div className="odds-label">{teamB} Win</div>
                  </div>
                </div>
              )}

              {/* Betting Calculators */}
              <div className="betting-calculators">
                <BettingCalculator 
                  winProbability={winProbability} 
                  teamName={teamA}
                />
                <BettingCalculator 
                  winProbability={lossProbability} 
                  teamName={teamB}
                />
              </div>

              {/* Display statistical comparison chart */}
              {graphData && (
                <div className="chart-container">
                  <h4 className="chart-title">Statistical Comparison</h4>
                  <Bar
                    data={{
                      labels: graphData.labels,
                      datasets: [
                        {
                          label: teamA,
                          data: graphData.team_a,
                          backgroundColor: CHART_CONFIG.colors.teamA,
                        },
                        {
                          label: teamB,
                          data: graphData.team_b,
                          backgroundColor: CHART_CONFIG.colors.teamB,
                        },
                      ],
                    }}
                    options={CHART_CONFIG.options}
                  />
                </div>
              )}

              {/* Display AI-generated analysis */}
              {analysis && (
                <div className="analysis-container">
                  <h4 className="analysis-title">AI Analysis</h4>
                  <div className="analysis-content">{analysis}</div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Loading indicator during prediction */}
        {predictionLoading && (
          <div className="loading-container">
            <div className="loading-spinner"></div>
            <p className="loading-text">Analyzing matchup...</p>
          </div>
        )}

        {/* Prediction error display */}
        {predictionError && (
          <div className="error-container">
            <p className="error-message">{predictionError}</p>
          </div>
        )}
      </section>
    </div>
  );
};

export default MatchupForm; 