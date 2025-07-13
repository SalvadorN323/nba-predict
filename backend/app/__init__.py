"""
Flask application factory for the NBA Predictor backend.

This module creates and configures the Flask application with all necessary
extensions, blueprints, and error handlers.
"""

from flask import Flask
from flask_cors import CORS
from config.settings import get_config

def create_app(config_name=None):
    """
    Create and configure the Flask application.
    
    Args:
        config_name: Configuration name to use (development, production, testing)
    
    Returns:
        Flask: Configured Flask application
    """
    app = Flask(__name__)
    
    # Load configuration
    if config_name:
        app.config.from_object(get_config())
    else:
        app.config.from_object(get_config())
    
    # Initialize extensions
    CORS(app, resources={r'/*': {'origins': app.config['CORS_ORIGINS']}})
    
    # Register blueprints
    from app.routes.predictions import predictions_bp
    from app.routes.health import health_bp
    
    app.register_blueprint(predictions_bp, url_prefix='/api/v1')
    app.register_blueprint(health_bp, url_prefix='/api/v1')
    
    # Register error handlers
    register_error_handlers(app)
    
    return app

def register_error_handlers(app):
    """Register error handlers for the application."""
    
    @app.errorhandler(400)
    def bad_request(error):
        return {'error': 'Bad request', 'message': str(error)}, 400
    
    @app.errorhandler(404)
    def not_found(error):
        return {'error': 'Not found', 'message': 'The requested resource was not found'}, 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return {'error': 'Internal server error', 'message': 'An unexpected error occurred'}, 500 