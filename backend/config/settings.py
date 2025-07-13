"""
Configuration settings for the NBA Predictor backend.

This module contains all configuration settings for different environments
(development, production, testing) and external API configurations.
"""

import os
from typing import List

# Load environment variables
# load_dotenv() # This line is removed as per the new_code, as the dotenv import is removed.

class Config:
    """Base configuration class."""
    
    # Flask settings
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = False
    TESTING = False
    
    # API settings
    API_TITLE = 'NBA Game Predictor API'
    API_VERSION = 'v1'
    
    # Model settings
    MODEL_PATH = os.getenv('MODEL_PATH', 'models/model.pkl')
    DATA_PATH = os.getenv('DATA_PATH', 'data/nba_games.csv')
    
    # OpenAI settings
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
    
    # CORS settings
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', 'https://nba-predict-frontend.onrender.com,http://localhost:3000').split(',')
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'

class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    LOG_LEVEL = 'INFO'

class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    DEBUG = True

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

def get_config():
    """Get configuration based on environment."""
    env = os.getenv('FLASK_ENV', 'development')
    return config.get(env, config['default']) 