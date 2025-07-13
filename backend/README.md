# NBA Predictor Backend

This is the backend API for the NBA Game Predictor application, built with Flask and Python.

## Features

- **RESTful API**: Clean API endpoints for predictions and health checks
- **Machine Learning**: XGBoost model integration for game predictions
- **AI Analysis**: OpenAI GPT-4 integration for expert insights
- **Error Handling**: Comprehensive error handling and validation
- **Logging**: Structured logging for monitoring and debugging
- **Health Checks**: API health monitoring endpoints

## Project Structure

```
backend/
├── app/                    # Application package
│   ├── routes/            # API route blueprints
│   ├── services/          # Business logic services
│   └── utils/             # Utility functions
├── config/                # Configuration settings
├── data/                  # Data files
├── models/                # ML model files
├── requirements.txt       # Python dependencies
├── run.py                 # Application entry point
└── README.md             # This file
```

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd backend
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp env.example .env
   # Edit .env with your configuration
   ```

5. **Run the application**
   ```bash
   python run.py
   ```

## API Endpoints

### Predictions
- `POST /api/v1/predict` - Make a game prediction
- `GET /api/v1/teams` - Get available teams

### Health Checks
- `GET /api/v1/health` - Basic health check
- `GET /api/v1/health/detailed` - Detailed health check

## Configuration

The application uses environment variables for configuration. See `env.example` for all available options.

### Required Environment Variables
- `OPENAI_API_KEY` - OpenAI API key for AI analysis
- `SECRET_KEY` - Flask secret key

### Optional Environment Variables
- `FLASK_ENV` - Environment (development/production)
- `PORT` - Server port (default: 8080)
- `LOG_LEVEL` - Logging level (default: INFO)

## Development

### Running in Development Mode
```bash
export FLASK_ENV=development
python run.py
```

### Running Tests
```bash
pytest
```

### Code Formatting
```bash
black .
```

### Linting
```bash
flake8
```

## Deployment

### Render Deployment
1. Connect your GitHub repository to Render
2. Set environment variables in Render dashboard
3. Deploy as a Python web service

### Local Production
```bash
gunicorn run:main --bind 0.0.0.0:8080
```

## Monitoring

The application includes health check endpoints for monitoring:
- Basic health: `/api/v1/health`
- Detailed health: `/api/v1/health/detailed`

## Logging

Logs are structured and include:
- Request/response logging
- Error tracking
- Performance metrics
- Model prediction logs

## Security

- Input validation on all endpoints
- CORS configuration
- Environment variable protection
- Rate limiting (configurable) 