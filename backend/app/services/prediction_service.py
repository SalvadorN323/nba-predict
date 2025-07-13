"""
Prediction service for NBA game predictions.

This service handles all prediction-related operations including
model loading, data processing, and prediction generation.
"""

import pandas as pd
import joblib
import os
from typing import Dict, List, Any, Optional
from app.utils.logger import get_logger
from config.settings import get_config

logger = get_logger(__name__)

class PredictionService:
    """Service for handling NBA game predictions."""
    
    def __init__(self):
        """Initialize the prediction service."""
        self.config = get_config()
        self.model = None
        self.data = None
        self._load_model()
        self._load_data()
    
    def _load_model(self):
        """Load the trained machine learning model."""
        try:
            model_path = self.config.MODEL_PATH
            if not os.path.exists(model_path):
                # Try relative path from current directory
                model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'model.pkl')
            
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                logger.info("ML model loaded successfully")
            else:
                logger.error(f"Model file not found at {model_path}")
                raise FileNotFoundError(f"Model file not found at {model_path}")
                
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def _load_data(self):
        """Load the NBA games dataset."""
        try:
            data_path = self.config.DATA_PATH
            if not os.path.exists(data_path):
                # Try relative path from current directory
                data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'nba_games.csv')
            
            if os.path.exists(data_path):
                self.data = pd.read_csv(data_path)
                self.data['GAME_DATE'] = pd.to_datetime(self.data['GAME_DATE'])
                logger.info("NBA games data loaded successfully")
            else:
                logger.error(f"Data file not found at {data_path}")
                raise FileNotFoundError(f"Data file not found at {data_path}")
                
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def get_available_teams(self) -> List[str]:
        """
        Get list of available NBA teams.
        
        Returns:
            List of team names
        """
        if self.data is None:
            return []
        
        return sorted(self.data['TEAM_NAME'].unique().tolist())
    
    def check_model_status(self) -> Dict[str, Any]:
        """
        Check the status of the ML model.
        
        Returns:
            Dictionary with model status information
        """
        try:
            if self.model is None:
                return {'status': 'error', 'message': 'Model not loaded'}
            
            return {
                'status': 'healthy',
                'model_type': type(self.model).__name__,
                'features': self._get_model_features()
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def _get_model_features(self) -> List[str]:
        """Get the features used by the model."""
        # These are the features the model was trained on
        return ['PTS', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 
                'FTM', 'FTA', 'REB', 'AST', 'TOV', 'STL', 'BLK', 'OREB', 'DREB']
    
    def predict_matchup(self, team_a: str, team_b: str) -> Dict[str, Any]:
        """
        Predict the winner of a matchup between two teams.
        
        Args:
            team_a: Name of the first team
            team_b: Name of the second team
            
        Returns:
            Dictionary containing prediction results
        """
        try:
            # Validate teams exist in data
            if team_a not in self.data['TEAM_NAME'].values:
                return {
                    'success': False,
                    'error': f'Team not found: {team_a}'
                }
            
            if team_b not in self.data['TEAM_NAME'].values:
                return {
                    'success': False,
                    'error': f'Team not found: {team_b}'
                }
            
            # Get recent data for both teams
            team_a_data = self.data[self.data['TEAM_NAME'] == team_a].sort_values(by='GAME_DATE', ascending=False).head(1)
            team_b_data = self.data[self.data['TEAM_NAME'] == team_b].sort_values(by='GAME_DATE', ascending=False).head(1)
            
            if team_a_data.empty or team_b_data.empty:
                return {
                    'success': False,
                    'error': f'No recent data found for {team_a} or {team_b}'
                }
            
            # Extract features for prediction
            features = self._get_model_features()
            team_a_stats = team_a_data[features].iloc[0]
            team_b_stats = team_b_data[features].iloc[0]
            
            # Prepare data for model (using Team A's stats)
            matchup_data = {feature: float(team_a_stats[feature]) for feature in features}
            game_data = pd.DataFrame([matchup_data])
            
            # Make prediction
            prediction = self.model.predict(game_data)[0]
            prediction_probabilities = self.model.predict_proba(game_data)[0]
            
            # Extract probabilities
            win_probability = float(prediction_probabilities[1])  # Team A wins
            loss_probability = float(prediction_probabilities[0])  # Team B wins
            
            # Convert prediction to human-readable format
            result = "Team A wins!" if prediction == 1 else "Team B wins!"
            
            # Prepare graph data for visualization
            graph_features = ['PTS', 'FG_PCT', 'REB', 'AST', 'TOV', 'STL', 'BLK', 'OREB', 'DREB']
            graph_data = {
                "labels": graph_features,
                "team_a": [float(team_a_stats[feature]) for feature in graph_features],
                "team_b": [float(team_b_stats[feature]) for feature in graph_features]
            }
            
            logger.info(f"Prediction completed: {result} (Team A: {win_probability:.2f}, Team B: {loss_probability:.2f})")
            
            return {
                'success': True,
                'data': {
                    'prediction': result,
                    'win_probability': win_probability,
                    'loss_probability': loss_probability,
                    'graph_data': graph_data,
                    'matchup_data': matchup_data
                }
            }
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            return {
                'success': False,
                'error': f'Prediction failed: {str(e)}'
            } 