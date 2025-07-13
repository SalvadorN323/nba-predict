"""
Health check routes for the NBA Predictor API.

This module contains health check endpoints for monitoring
the application status and external service connectivity.
"""

from flask import Blueprint, jsonify, current_app
from app.utils.logger import get_logger

# Create blueprint
health_bp = Blueprint('health', __name__)
logger = get_logger(__name__)

@health_bp.route('/health', methods=['GET'])
def health_check():
    """
    Basic health check endpoint.
    
    Returns:
        JSON response indicating service status
    """
    return jsonify({
        'status': 'healthy',
        'service': 'NBA Predictor API',
        'version': '1.0.0'
    })

@health_bp.route('/health/detailed', methods=['GET'])
def detailed_health_check():
    """
    Detailed health check including external service status.
    
    Returns:
        JSON response with detailed service status
    """
    try:
        # Check model availability
        from app.services.prediction_service import PredictionService
        prediction_service = PredictionService()
        model_status = prediction_service.check_model_status()
        
        # Check OpenAI service
        from app.services.openai_service import OpenAIService
        openai_service = OpenAIService()
        openai_status = openai_service.check_service_status()
        
        return jsonify({
            'status': 'healthy',
            'service': 'NBA Predictor API',
            'version': '1.0.0',
            'components': {
                'model': model_status,
                'openai': openai_status
            }
        })
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'service': 'NBA Predictor API',
            'version': '1.0.0',
            'error': str(e)
        }), 500 