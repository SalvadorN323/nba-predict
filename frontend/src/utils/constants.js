/**
 * Constants for the NBA Predictor frontend.
 * 
 * This file contains all constant values used throughout the application,
 * including team logos, API endpoints, and configuration values.
 */

// Team logo mapping
export const TEAM_LOGOS = {
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

// Team name to abbreviation mapping
export const TEAM_NAME_TO_ABBR = {
  'Atlanta Hawks': 'ATL',
  'Boston Celtics': 'BOS',
  'Brooklyn Nets': 'BKN',
  'Charlotte Hornets': 'CHA',
  'Chicago Bulls': 'CHI',
  'Cleveland Cavaliers': 'CLE',
  'Dallas Mavericks': 'DAL',
  'Denver Nuggets': 'DEN',
  'Detroit Pistons': 'DET',
  'Golden State Warriors': 'GSW',
  'Houston Rockets': 'HOU',
  'Indiana Pacers': 'IND',
  'Los Angeles Clippers': 'LAC',
  'Los Angeles Lakers': 'LAL',
  'Memphis Grizzlies': 'MEM',
  'Miami Heat': 'MIA',
  'Milwaukee Bucks': 'MIL',
  'Minnesota Timberwolves': 'MIN',
  'New Orleans Pelicans': 'NOP',
  'New York Knicks': 'NYK',
  'Oklahoma City Thunder': 'OKC',
  'Orlando Magic': 'ORL',
  'Philadelphia 76ers': 'PHI',
  'Phoenix Suns': 'PHX',
  'Portland Trail Blazers': 'POR',
  'Sacramento Kings': 'SAC',
  'San Antonio Spurs': 'SAS',
  'Toronto Raptors': 'TOR',
  'Utah Jazz': 'UTA',
  'Washington Wizards': 'WAS',
};

/**
 * Get team abbreviation from full team name
 * @param {string} teamName - Full team name
 * @returns {string} Team abbreviation or null if not found
 */
export const getTeamAbbreviation = (teamName) => {
  return TEAM_NAME_TO_ABBR[teamName] || null;
};

// Chart configuration
export const CHART_CONFIG = {
  colors: {
    teamA: 'rgba(75, 192, 192, 0.6)',
    teamB: 'rgba(153, 102, 255, 0.6)',
  },
  options: {
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
  },
};

// API configuration
export const API_CONFIG = {
  baseUrl: process.env.REACT_APP_API_URL || 'http://127.0.0.1:8080/api/v1',
  timeout: 10000, // 10 seconds
};

// Application configuration
export const APP_CONFIG = {
  name: 'NBA Game Predictor',
  version: '1.0.0',
  description: 'Machine learning-powered NBA game predictions',
};

// Toast configuration
export const TOAST_CONFIG = {
  position: 'top-right',
  autoClose: 5000,
  hideProgressBar: false,
  closeOnClick: true,
  pauseOnHover: true,
  draggable: true,
};

// Loading spinner configuration
export const SPINNER_CONFIG = {
  color: '#007bff',
  size: 50,
  loadingText: 'Loading...',
}; 