"""
Main entry point for the NBA Predictor backend application.

This script starts the Flask application with proper configuration
and logging setup.
"""

import os
from app import create_app
from app.utils.logger import setup_logging

# Create Flask application instance for gunicorn
app = create_app()

def main():
    """Main application entry point."""
    # Setup logging
    setup_logging(level=os.getenv('LOG_LEVEL', 'INFO'))
    
    # Get configuration
    port = int(os.environ.get('PORT', 8080))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    # Start the application
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug
    )

if __name__ == '__main__':
    main() 