import React, { useState, useEffect } from 'react';
import { ClipLoader } from 'react-spinners';
import { toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import { Bar } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend } from 'chart.js';
import './MatchupForm.css';

// Register necessary Chart.js components
ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);
 
// Logos for teams (You can use URLs to your logos here)
const teamLogos = {
  'ATL': '/logos/atlanta_hawks.png',
  'BOS': '/logos/boston_celtics.png',
  'BKN': '/logos/brooklyn_nets.png',
  'CHA': '/logos/charlotte_hornets.png',
  'CHI': '/logos/chicago_bulls.png',
  'CLE': '/logos/cleveland_cavaliers.png',
  'DAL': '/logos/dallas_mavericks.png',
  'DEN': '/logos/denver_nuggets.png',
  'DET': '/logos/detroit_pistons.png',
  'GSW': '/logos/golden_state_warriors.png',
  'HOU': '/logos/houston_rockets.png',
  'IND': '/logos/indiana_pacers.png',
  'LAC': '/logos/los_angeles_clippers.png',
  'LAL': '/logos/los_angeles_lakers.png',
  'MEM': '/logos/memphis_grizzlies.png',
  'MIA': '/logos/miami_heat.png',
  'MIL': '/logos/milwaukee_bucks.png',
  'MIN': '/logos/minnesota_timberwolves.png',
  'NOP': '/logos/new_orleans_pelicans.png',
  'NYK': '/logos/new_york_knicks.png',
  'OKC': '/logos/oklahoma_city_thunder.png',
  'ORL': '/logos/orlando_magic.png',
  'PHI': '/logos/philadelphia_76ers.png',
  'PHX': '/logos/phoenix_suns.png',
  'POR': '/logos/portland_trail_blazers.png',
  'SAC': '/logos/sacramento_kings.png',
  'SAS': '/logos/san_antonio_spurs.png',
  'TOR': '/logos/toronto_raptors.png',
  'UTA': '/logos/utah_jazz.png',
  'WAS': '/logos/washington_wizards.png',
};

const MatchupForm = () => {
  const [games, setGames] = useState([]);  // Store games data
  const [teamA, setTeamA] = useState('');
  const [teamB, setTeamB] = useState('');
  const [prediction, setPrediction] = useState('');
  const [winProbability, setWinProbability] = useState(null);
  const [lossProbability, setLossProbability] = useState(null);
  const [graphData, setGraphData] = useState(null);
  const [analysis, setAnalysis] = useState('');  // New state for analysis
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const currDay = new Date(); // Current date
  currDay.setDate(currDay.getDate() + 1); // No change needed here
  const newDay = currDay.toLocaleDateString('en-CA'); // YYYY-MM-DD format

  // Fetch games for tomorrow (2024-11-28) from BallDon'tLie API
  useEffect(() => {
    const startDate = newDay;
    const endDate = newDay;

    // Fetch games for the next day (from startDate to endDate)
    const getGames = async () => {
      try {
        const response = await fetch(
          `https://api.balldontlie.io/v1/games?start_date=${startDate}&end_date=${endDate}`,
          {
            headers: {
              'Authorization': 'd1710c2b-4c0f-4f21-ac88-487e55085ea7'  // API Key
            }
          }
        );
 
        if (!response.ok) {
          throw new Error('Failed to fetch games');
        }

        const data = await response.json();
        setGames(data.data);  // Set games data to state
      } catch (error) {
        setError(error.message);
        toast.error('Failed to fetch games. Please try again.');
      } finally {
        setLoading(false);
      }
    };

    getGames();
  }, []);  // This effect runs only once on component mount

  // Handle prediction submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!teamA || !teamB) {
      setPrediction('Please select both teams');
      return;
    }

    setLoading(true);
    try {
      const response = await fetch('https://nba-predict.onrender.com/predict', {
        method: 'POST',
        credentials: 'include',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ team_a: teamA, team_b: teamB }),
      });

      const data = await response.json();
      setPrediction(data.prediction);  // Store the prediction (win/loss)
      setWinProbability(data.win_probability);  // Store the win probability
      setLossProbability(data.loss_probability);  // Store the loss probability
      setGraphData(data.graph_data);  // Store the graph data
      setAnalysis(data.analysis);  // Store the OpenAI analysis
    } catch (error) {
      setPrediction('Error in prediction. Please try again.');
      toast.error('Error in prediction. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="matchup-form-container">
      <h1>NBA Games for Tomorrow {newDay}</h1>

      {/* Side-by-side layout: Games on the left, Predictions on the right */}
      <div className="games-container">
        {loading && <ClipLoader color="#007bff" size={50} />}
        {error && <p className="error">{error}</p>}

        {/* Display games for tomorrow */}
        {games.length > 0 ? (
          <div>
            <h2>Select a Game (H vs A)</h2>
            <ul>
              {games.map((game) => (
                <li key={game.id}>
                  <button onClick={() => { setTeamA(game.home_team.full_name); setTeamB(game.visitor_team.full_name); }}>
                    <img src={teamLogos[game.home_team.abbreviation]} alt={game.home_team.full_name} width="30" />
                    {game.home_team.full_name} vs. 
                    <img src={teamLogos[game.visitor_team.abbreviation]} alt={game.visitor_team.full_name} width="30" />
                    {game.visitor_team.full_name}
                  </button>
                </li>
              ))}
            </ul>
          </div>
        ) : (
          <p>No games scheduled for tomorrow.</p>
        )}
      </div>

      <div className="prediction-container">
        <h2>Matchup Prediction</h2>

        <form onSubmit={handleSubmit}>
          <div className="input-container">
            <label>Team A: {teamA}</label>
            <br />
            <label>Team B: {teamB}</label>
          </div>

          <button type="submit" className="submit-button">Predict Winner</button>
        </form>

        {loading && <div className="loading">Loading prediction...</div>}

        {prediction && (
          <div className="result">
            <p>{prediction}</p>

            {/* Display win/loss probabilities */}
            {winProbability !== null && lossProbability !== null && (
              <div className="probabilities">
                <p>Win Probability for {teamA}: {winProbability.toFixed(2) * 100}%</p>
                <p>Win Probability for {teamB}: {lossProbability.toFixed(2) * 100}%</p>
              </div>
            )}

            {graphData && (
              <div className="graph">
                <Bar
                  data={{
                    labels: graphData.labels,
                    datasets: [
                      {
                        label: teamA,
                        data: graphData.team_a,
                        backgroundColor: 'rgba(75, 192, 192, 0.6)',
                      },
                      {
                        label: teamB,
                        data: graphData.team_b,
                        backgroundColor: 'rgba(153, 102, 255, 0.6)',
                      },
                    ],
                  }}
                  options={{
                    responsive: true,
                    plugins: {
                      legend: {
                        position: 'top',
                      },
                      title: {
                        display: true,
                        text: 'Matchup Comparison',
                      },
                    },
                  }}
                />
              </div>
            )}

            {/* Display OpenAI's analysis */}
            {analysis && (
              <div className="analysis">
                <h3>Matchup Analysis</h3>
                <p>{analysis}</p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default MatchupForm;