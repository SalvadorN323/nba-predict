"""
Prediction routes for the NBA Predictor API.

This module contains all routes related to game predictions,
including the main prediction endpoint and related functionality.
"""

from flask import Blueprint, request, jsonify, current_app
from app.services.prediction_service import PredictionService
from app.services.openai_service import OpenAIService
from app.utils.validators import validate_prediction_request
from app.utils.logger import get_logger

# Create blueprint
predictions_bp = Blueprint('predictions', __name__)
logger = get_logger(__name__)

@predictions_bp.route('/predict', methods=['POST'])
def predict():
    """
    Predict the winner of an NBA matchup between two teams.
    
    This endpoint:
    1. Validates the incoming request
    2. Retrieves recent performance data for both teams
    3. Uses the trained ML model to predict the winner
    4. Generates AI-powered analysis using OpenAI
    5. Returns comprehensive prediction results
    
    Returns:
        JSON response containing prediction results and analysis
    """
    try:
        # Validate request
        data = request.get_json()
        validation_result = validate_prediction_request(data)
        
        if not validation_result['valid']:
            return jsonify({'error': validation_result['message']}), 400
        
        team_a = data.get('team_a')
        team_b = data.get('team_b')
        
        logger.info(f"Processing prediction request for {team_a} vs {team_b}")
        
        # Initialize services
        prediction_service = PredictionService()
        openai_service = OpenAIService()
        
        # Get prediction
        prediction_result = prediction_service.predict_matchup(team_a, team_b)
        
        if not prediction_result['success']:
            return jsonify({'error': prediction_result['error']}), 400
        
        # Generate AI analysis
        analysis_result = openai_service.generate_analysis(
            team_a=team_a,
            team_b=team_b,
            prediction=prediction_result['data']['prediction'],
            win_probability=prediction_result['data']['win_probability'],
            loss_probability=prediction_result['data']['loss_probability'],
            stats=prediction_result['data']['matchup_data']
        )
        
        # Prepare response
        response_data = {
            'prediction': prediction_result['data']['prediction'],
            'win_probability': prediction_result['data']['win_probability'],
            'loss_probability': prediction_result['data']['loss_probability'],
            'graph_data': prediction_result['data']['graph_data'],
            'analysis': analysis_result.get('analysis', 'Analysis not available')
        }
        
        logger.info(f"Prediction completed successfully for {team_a} vs {team_b}")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in prediction endpoint: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred during prediction'}), 500

@predictions_bp.route('/teams', methods=['GET'])
def get_teams():
    """
    Get list of available NBA teams.
    
    Returns:
        JSON response containing list of team names
    """
    try:
        prediction_service = PredictionService()
        teams = prediction_service.get_available_teams()
        
        return jsonify({
            'teams': teams,
            'count': len(teams)
        })
        
    except Exception as e:
        logger.error(f"Error getting teams: {str(e)}")
        return jsonify({'error': 'Failed to retrieve teams'}), 500 