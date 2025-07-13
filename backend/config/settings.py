"""
Configuration settings for the NBA Predictor backend.

This module contains all configuration settings for different environments
(development, production, testing) and external API configurations.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Base configuration class."""
    
    # Flask settings
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = False
    TESTING = False
    
    # API settings
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    BALLDONTLIE_API_KEY = os.getenv('BALLDONTLIE_API_KEY', 'd1710c2b-4c0f-4f21-ac88-487e55085ea7')
    
    # Model settings
    MODEL_PATH = os.getenv('MODEL_PATH', 'models/model.pkl')
    DATA_PATH = os.getenv('DATA_PATH', 'data/nba_games.csv')
    
    # CORS settings
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', 'https://nba-predict-1.onrender.com').split(',')
    
    # OpenAI settings
    OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
    OPENAI_MAX_TOKENS = int(os.getenv('OPENAI_MAX_TOKENS', '500'))
    
    # Rate limiting
    RATE_LIMIT_PER_MINUTE = int(os.getenv('RATE_LIMIT_PER_MINUTE', '60'))

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    CORS_ORIGINS = ['http://localhost:3000', 'http://127.0.0.1:3000']

class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False

class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    DEBUG = True

# Configuration mapping
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