# NBA Game Predictor - Technical Documentation

## Table of Contents
1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Data Pipeline](#data-pipeline)
4. [Machine Learning Model](#machine-learning-model)
5. [API Documentation](#api-documentation)
6. [Frontend Architecture](#frontend-architecture)
7. [Deployment Guide](#deployment-guide)
8. [Testing Strategy](#testing-strategy)
9. [Performance Considerations](#performance-considerations)
10. [Security Considerations](#security-considerations)
11. [Troubleshooting](#troubleshooting)

## Overview

The NBA Game Predictor is a full-stack web application that uses machine learning to predict NBA game outcomes. The system combines historical data analysis, real-time game information, and AI-powered insights to provide comprehensive game predictions.

### Key Features
- **ML-Powered Predictions**: XGBoost ensemble model for win/loss prediction
- **Real-time Data**: Integration with BallDon'tLie API for live game data
- **AI Analysis**: OpenAI GPT-4 integration for expert-level insights
- **Interactive Visualizations**: Chart.js integration for statistical comparisons
- **Responsive Design**: Modern React frontend with mobile-friendly interface

## System Architecture

### High-Level Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React Frontend│    │   Flask Backend │    │   ML Model      │
│                 │    │                 │    │                 │
│ - MatchupForm   │◄──►│ - /predict API  │◄──►│ - XGBoost Model │
│ - Charts        │    │ - Data Processing│    │ - Feature Eng.  │
│ - Team Logos    │    │ - OpenAI Int.   │    │ - Model.pkl     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   BallDon'tLie  │    │   OpenAI API    │    │   Historical    │
│   API           │    │                 │    │   NBA Data      │
│                 │    │                 │    │                 │
│ - Upcoming Games│    │ - GPT-4 Analysis│    │ - 2010-2024     │
│ - Team Data     │    │ - Expert Insights│   │ - Game Stats    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Technology Stack
- **Frontend**: React.js 18.3.1, Chart.js 4.4.6, React-Chartjs-2 5.2.0
- **Backend**: Flask 3.0.3, Python 3.8+
- **ML**: XGBoost, Scikit-learn, Pandas
- **APIs**: BallDon'tLie API, OpenAI API
- **Deployment**: Render (Backend), Render (Frontend)

## Data Pipeline

### Data Sources
1. **Historical Data**: NBA regular season statistics (2010-2024)
   - Source: `nba_games.csv`
   - Features: 16 statistical metrics per game
   - Records: ~35,000+ games

2. **Real-time Data**: BallDon'tLie API
   - Endpoint: `https://api.balldontlie.io/v1/games`
   - Purpose: Fetch upcoming games
   - Authentication: API key required

3. **AI Analysis**: OpenAI GPT-4
   - Model: `gpt-4o-mini`
   - Purpose: Generate expert analysis
   - Input: Team stats, predictions, probabilities

### Data Flow
```
Historical Data → Feature Engineering → Model Training → Model.pkl
     ↓
Real-time Games → Team Selection → Prediction Request
     ↓
ML Model → Prediction + Probabilities → OpenAI Analysis
     ↓
Frontend Display ← JSON Response ← Flask API
```

## Machine Learning Model

### Model Architecture
The system uses an ensemble approach combining multiple algorithms:

1. **Classification Model (Win/Loss Prediction)**
   - Algorithm: Voting Classifier with soft voting
   - Base Models: Random Forest, Gradient Boosting, XGBoost
   - Output: Win probability (0-1)

2. **Regression Models (Score Prediction)**
   - Algorithm: Random Forest Regressor
   - Separate models for Team A and Team B scores
   - Output: Predicted final scores

### Feature Engineering
The model uses 10 engineered features:

**Difference Features (Team A - Team B):**
- `PTS_DIFF`: Point difference
- `FG_PCT_DIFF`: Field goal percentage difference
- `REB_DIFF`: Rebound difference
- `AST_DIFF`: Assist difference
- `STL_DIFF`: Steal difference
- `BLK_DIFF`: Block difference
- `FT_PCT_DIFF`: Free throw percentage difference

**Rolling Average Features (5-game window):**
- `PTS_ROLLING_AVG`: Average points over last 5 games
- `REB_ROLLING_AVG`: Average rebounds over last 5 games
- `AST_ROLLING_AVG`: Average assists over last 5 games

### Model Performance
- **Classification Accuracy**: **85%**
- **Cross-validation Score**: ~85%
- **Mean Squared Error (Regression)**: ~150-200 points²

### Training Process
```python
# 1. Data Loading and Preprocessing
df = pd.read_csv('regular_season_totals_2010_2024.csv')

# 2. Feature Engineering
df['PTS_ROLLING_AVG'] = df['PTS'].rolling(window=5).mean().shift(-5)
df['PTS_DIFF'] = df['PTS'] - df['PTS'].shift(-1)
# ... additional features

# 3. Model Training
voting_clf = VotingClassifier([
    ('rf', RandomForestClassifier()),
    ('gb', GradientBoostingClassifier()),
    ('xgb', XGBClassifier())
], voting='soft')

# 4. Model Evaluation and Saving
joblib.dump(voting_clf, 'model.pkl')
```

### Current Test Configuration
The application is currently configured to test against NBA Finals games from June 22, 2025, providing a consistent test environment with known historical outcomes.

## API Documentation

### Backend API Endpoints

#### POST `/predict`
Predicts the winner of an NBA matchup.

**Request:**
```json
{
  "team_a": "Los Angeles Lakers",
  "team_b": "Boston Celtics"
}
```

**Response:**
```json
{
  "prediction": "Team A wins!",
  "win_probability": 0.65,
  "loss_probability": 0.35,
  "graph_data": {
    "labels": ["PTS", "FG_PCT", "REB", "AST", "TOV", "STL", "BLK", "OREB", "DREB"],
    "team_a": [110.5, 0.48, 45.2, 25.1, 12.3, 8.5, 4.2, 10.1, 35.1],
    "team_b": [108.3, 0.46, 42.1, 23.8, 13.1, 7.9, 3.8, 9.8, 32.3]
  },
  "analysis": "Detailed AI analysis of the matchup..."
}
```

**Error Responses:**
```json
{
  "error": "Both teams must be selected"
}
```

### External API Integrations

#### BallDon'tLie API
- **Base URL**: `https://api.balldontlie.io/v1`
- **Authentication**: API key in headers
- **Rate Limit**: 1000 requests per day
- **Endpoint**: `/games?start_date={date}&end_date={date}`

#### OpenAI API
- **Model**: `gpt-4o-mini`
- **Authentication**: API key from environment
- **Rate Limit**: Based on OpenAI plan
- **Usage**: Analysis generation for predictions

## Frontend Architecture

### Component Structure
```
App.js
└── MatchupForm.js
    ├── Game Selection
    ├── Prediction Form
    ├── Results Display
    ├── Chart Visualization
    └── AI Analysis
```

### State Management
The application uses React hooks for state management:

```javascript
const [games, setGames] = useState([]);           // Upcoming games
const [teamA, setTeamA] = useState('');           // Selected Team A
const [teamB, setTeamB] = useState('');           // Selected Team B
const [prediction, setPrediction] = useState(''); // ML prediction
const [winProbability, setWinProbability] = useState(null);
const [lossProbability, setLossProbability] = useState(null);
const [graphData, setGraphData] = useState(null); // Chart data
const [analysis, setAnalysis] = useState('');     // AI analysis
const [loading, setLoading] = useState(true);     // Loading state
const [error, setError] = useState(null);         // Error state
```

### Key Features
1. **Game Selection**: Interactive buttons with team logos
2. **Loading States**: Spinners and progress indicators
3. **Error Handling**: Toast notifications for user feedback
4. **Responsive Design**: Mobile-friendly layout
5. **Data Visualization**: Interactive bar charts

## Deployment Guide

### Backend Deployment (Render)

1. **Repository Setup**
   ```bash
   git clone <repository-url>
   cd nba
   ```

2. **Environment Variables**
   ```env
   OPENAI_API_KEY=your_openai_api_key
   PORT=8080
   ```

3. **Requirements**
   ```txt
   Flask==3.0.3
   joblib==1.4.2
   openai==0.28.0
   pandas==2.2.3
   python-dotenv==1.0.1
   ```

4. **Deploy to Render**
   - Connect GitHub repository
   - Set build command: `pip install -r requirements.txt`
   - Set start command: `python app.py`
   - Configure environment variables

### Frontend Deployment (Netlify)

1. **Build the Application**
   ```bash
   npm run build
   ```

2. **Deploy to Netlify**
   - Connect GitHub repository
   - Set build command: `npm run build`
   - Set publish directory: `build`
   - Configure environment variables if needed

### Environment Configuration
```bash
# Backend (.env)
OPENAI_API_KEY=sk-...
PORT=8080

# Frontend (if needed)
REACT_APP_API_URL=https://your-backend-url.onrender.com
```

## Testing Strategy

### Backend Testing
```python
# Unit tests for API endpoints
def test_predict_endpoint():
    response = client.post('/predict', json={
        'team_a': 'Los Angeles Lakers',
        'team_b': 'Boston Celtics'
    })
    assert response.status_code == 200
    assert 'prediction' in response.json
```

### Frontend Testing
```javascript
// Component testing with React Testing Library
test('renders game selection', () => {
  render(<MatchupForm />);
  expect(screen.getByText(/NBA Games for Tomorrow/)).toBeInTheDocument();
});
```

### Model Testing
```python
# Model validation
def test_model_accuracy():
    X_test, y_test = load_test_data()
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    assert accuracy > 0.6  # Minimum acceptable accuracy
```

## Performance Considerations

### Backend Optimization
1. **Model Loading**: Pre-load model at startup
2. **Data Caching**: Cache frequently accessed data
3. **API Rate Limiting**: Implement request throttling
4. **Database Optimization**: Use efficient data structures

### Frontend Optimization
1. **Code Splitting**: Lazy load components
2. **Image Optimization**: Compress team logos
3. **Caching**: Cache API responses
4. **Bundle Optimization**: Minimize bundle size

### API Performance
- **Response Time**: < 2 seconds for predictions
- **Concurrent Requests**: Handle multiple users
- **Error Handling**: Graceful degradation
- **Monitoring**: Track API usage and performance

## Security Considerations

### API Security
1. **CORS Configuration**: Restrict origins
2. **Input Validation**: Sanitize user inputs
3. **Rate Limiting**: Prevent abuse
4. **API Key Protection**: Secure environment variables

### Data Security
1. **Sensitive Data**: No PII stored
2. **API Keys**: Environment variables only
3. **HTTPS**: All communications encrypted
4. **Access Control**: No admin interfaces

### Model Security
1. **Input Validation**: Validate team names
2. **Output Sanitization**: Clean prediction results
3. **Model Protection**: Prevent model extraction
4. **Audit Logging**: Track prediction requests

## Troubleshooting

### Common Issues

#### Backend Issues
1. **Model Loading Error**
   ```bash
   # Check if model.pkl exists
   ls -la model.pkl
   
   # Re-train model if needed
   python model/model.py
   ```

2. **API Key Issues**
   ```bash
   # Verify environment variables
   echo $OPENAI_API_KEY
   
   # Check .env file
   cat .env
   ```

3. **CORS Errors**
   ```python
   # Update CORS configuration in app.py
   CORS(app, resources={r'/*': {'origins': 'your-frontend-url'}})
   ```

#### Frontend Issues
1. **API Connection Errors**
   ```javascript
   // Check API URL in MatchupForm.js
   const response = await fetch('https://your-backend-url/predict', {
     // ... configuration
   });
   ```

2. **Chart Rendering Issues**
   ```javascript
   // Ensure Chart.js is properly registered
   ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);
   ```

3. **Team Logo Issues**
   ```bash
   # Verify logo files exist
   ls -la public/logos/
   ```

### Debugging Tools
1. **Backend Logs**: Check Render logs
2. **Frontend Console**: Browser developer tools
3. **Network Tab**: Monitor API requests
4. **Environment Variables**: Verify configuration

### Performance Monitoring
1. **Response Times**: Monitor API performance
2. **Error Rates**: Track failed requests
3. **User Analytics**: Monitor usage patterns
4. **Model Accuracy**: Regular model evaluation

---

**Last Updated**: December 2024
**Version**: 1.0.0
**Maintainer**: NBA Predictor Team 