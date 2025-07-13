# NBA Game Predictor

A machine learning-powered web application that predicts NBA game outcomes using historical data and provides detailed analysis with AI-powered insights.

## 🏀 Features

- **Game Prediction**: Predicts winners of NBA matchups using XGBoost machine learning model with **85% accuracy**
- **Real-time Data**: Fetches upcoming games from the BallDon'tLie API
- **Statistical Analysis**: Compares team statistics with interactive charts
- **AI Analysis**: Provides detailed matchup analysis using OpenAI GPT-4
- **Team Logos**: Visual representation with all 30 NBA team logos
- **Probability Scores**: Shows win/loss probabilities for each team
- **Modern UI**: Betting-style interface with glassmorphism design
- **Rate Limiting**: Built-in protection against API rate limits

## 🚀 Live Demo

Visit the application: [NBA Predictor](https://nba-predict-1.onrender.com)

## 🏗️ Architecture

This is a full-stack application with a modern, scalable architecture:

### Backend (Flask API)
- **Framework**: Flask with application factory pattern
- **Structure**: Modular design with blueprints, services, and utilities
- **ML Model**: XGBoost classifier trained on historical NBA data (85% accuracy)
- **APIs**: BallDon'tLie API for game data, OpenAI API for analysis
- **Deployment**: Render (Python web service)

### Frontend (React.js)
- **Framework**: React 18 with functional components and hooks
- **Architecture**: Service layer pattern with custom hooks
- **State Management**: React hooks for local state
- **UI**: Responsive design with Chart.js visualizations and modern betting-style interface
- **Deployment**: Render (Static site)

## 📁 Project Structure

```
nba/
├── backend/                    # Flask API backend
│   ├── app/                   # Application package
│   │   ├── routes/           # API route blueprints
│   │   │   ├── predictions.py # Prediction endpoints
│   │   │   └── health.py     # Health check endpoints
│   │   ├── services/         # Business logic services
│   │   │   ├── prediction_service.py # ML prediction service
│   │   │   └── openai_service.py    # OpenAI integration
│   │   └── utils/            # Utility functions
│   │       ├── logger.py     # Logging utilities
│   │       └── validators.py # Input validation
│   ├── config/               # Configuration settings
│   │   └── settings.py       # Environment-based config
│   ├── data/                 # Data files
│   │   └── nba_games.csv     # Historical NBA data
│   ├── models/               # ML model files
│   │   └── model.pkl         # Trained XGBoost model
│   ├── requirements.txt      # Python dependencies
│   ├── run.py               # Application entry point
│   └── README.md            # Backend documentation
├── frontend/                  # React.js frontend
│   ├── public/              # Static assets
│   │   └── logos/           # NBA team logos
│   ├── src/                 # Source code
│   │   ├── components/      # React components
│   │   │   ├── MatchupForm/ # Main prediction component
│   │   │   └── common/      # Reusable UI components
│   │   ├── hooks/           # Custom React hooks
│   │   │   ├── usePrediction.js # Prediction hook
│   │   │   └── useGames.js      # Games data hook
│   │   ├── services/        # API services
│   │   │   ├── api.js       # Backend API service
│   │   │   └── gamesApi.js  # BallDon'tLie API service
│   │   ├── utils/           # Utility functions
│   │   │   ├── constants.js # Application constants
│   │   │   └── rateLimiter.js # Rate limiting utilities
│   │   └── App.js           # Main React component
│   ├── package.json         # Node.js dependencies
│   └── README.md           # Frontend documentation
├── DOCUMENTATION.md          # Technical documentation
└── README.md                # This file
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- Node.js 14+
- npm or yarn

### Backend Setup

1. **Navigate to backend directory**
   ```bash
   cd backend
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp env.example .env
   # Edit .env with your configuration
   ```

5. **Run the Flask backend**
   ```bash
   python run.py
   ```
   The API will be available at `http://localhost:8080`

### Frontend Setup

1. **Navigate to frontend directory**
   ```bash
   cd frontend
   ```

2. **Install Node.js dependencies**
   ```bash
   npm install
   ```

3. **Set up environment variables**
   ```bash
   cp env.example .env
   # Edit .env with your configuration
   ```

4. **Start the development server**
   ```bash
   npm start
   ```
   The React app will open at `http://localhost:3000`

## 🤖 Machine Learning Model

### Model Details
- **Algorithm**: XGBoost Classifier
- **Training Data**: NBA regular season games (2010-2024)
- **Features**: 16 statistical features including:
  - Points (PTS)
  - Field Goal Percentage (FG_PCT)
  - Rebounds (REB)
  - Assists (AST)
  - Turnovers (TOV)
  - Steals (STL)
  - Blocks (BLK)
  - And more...

### Model Performance
- **Accuracy**: **85%** on test data
- **Features**: Uses recent team performance data
- **Prediction**: Provides win/loss probabilities

## 📊 API Endpoints

### Backend API (Flask)

#### Predictions
- `POST /api/v1/predict` - Make a game prediction
- `GET /api/v1/teams` - Get available teams

#### Health Checks
- `GET /api/v1/health` - Basic health check
- `GET /api/v1/health/detailed` - Detailed health check

### External APIs
- **BallDon'tLie API**: For fetching NBA game data
- **OpenAI API**: For generating matchup analysis

## 🎯 Usage

1. **Select Teams**: Choose from upcoming games or manually select teams
2. **Get Prediction**: Click "Predict Winner" to get the ML prediction
3. **View Analysis**: See detailed statistics comparison and AI analysis
4. **Explore Data**: Interactive charts show team performance metrics

## 🔧 Configuration

### Backend Environment Variables
- `OPENAI_API_KEY`: Required for AI analysis
- `SECRET_KEY`: Flask secret key
- `PORT`: Backend port (default: 8080)
- `FLASK_ENV`: Environment (development/production)

### Frontend Environment Variables
- `REACT_APP_API_URL`: Backend API URL (default: http://127.0.0.1:8080/api/v1)

## 🚀 Deployment

### Backend (Render)
1. Connect your GitHub repository to Render
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `python run.py`
4. Configure environment variables

### Frontend (Render)
1. Connect your GitHub repository to Render
2. Set build command: `npm run build`
3. Set publish directory: `build`
4. Configure environment variables

## 📈 Data Sources

- **Historical Data**: NBA regular season statistics (2010-2024)
- **Live Games**: BallDon'tLie API for upcoming games
- **Team Logos**: Local assets for all 30 NBA teams

## 🧪 Development

### Backend Development
```bash
cd backend
export FLASK_ENV=development
python run.py
```

### Frontend Development
```bash
cd frontend
npm start
```

### Testing
```bash
# Backend tests
cd backend
pytest

# Frontend tests
cd frontend
npm test
```

## 🔒 Security Features

- Input validation on all endpoints
- CORS configuration
- Environment variable protection
- Rate limiting (configurable)
- Secure API key handling

## 📝 Code Quality

### Backend
- Modular Flask application structure
- Comprehensive error handling
- Structured logging
- Type hints and documentation
- Service layer pattern

### Frontend
- Modern React with hooks
- Custom hooks for reusable logic
- Service layer for API calls
- Comprehensive error handling
- JSDoc documentation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new features
5. Update documentation
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **BallDon'tLie API** for providing NBA game data
- **OpenAI** for AI-powered analysis
- **Chart.js** for data visualization
- **React** and **Flask** communities for excellent documentation
