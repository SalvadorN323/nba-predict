# NBA Predictor Frontend

This is the frontend application for the NBA Game Predictor, built with React.js and modern web technologies.

## Features

- **Modern React**: Built with React 18 and functional components
- **Custom Hooks**: Reusable hooks for API calls and state management
- **Service Layer**: Clean separation of API calls and business logic
- **Responsive Design**: Mobile-friendly interface
- **Interactive Charts**: Chart.js integration for data visualization
- **Error Handling**: Comprehensive error handling with user feedback
- **Loading States**: Smooth loading experiences

## Project Structure

```
frontend/
├── public/                 # Static assets
│   ├── logos/             # Team logos
│   └── index.html         # HTML template
├── src/                   # Source code
│   ├── components/        # React components
│   │   └── MatchupForm/   # Main prediction component
│   ├── hooks/             # Custom React hooks
│   ├── services/          # API services
│   ├── utils/             # Utility functions
│   ├── assets/            # Static assets
│   └── styles/            # CSS styles
├── package.json           # Dependencies
└── README.md             # This file
```

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd frontend
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Set up environment variables**
   ```bash
   cp env.example .env
   # Edit .env with your configuration
   ```

4. **Start development server**
   ```bash
   npm start
   ```

## Available Scripts

- `npm start` - Start development server
- `npm build` - Build for production
- `npm test` - Run tests
- `npm eject` - Eject from Create React App

## Configuration

### Environment Variables
- `REACT_APP_API_URL` - Backend API URL (default: https://nba-predict.onrender.com/api/v1)

## Development

### Project Structure

#### Components
- `MatchupForm/` - Main prediction interface component
- `common/` - Reusable UI components
- `layout/` - Layout components

#### Hooks
- `usePrediction.js` - Prediction functionality hook
- `useGames.js` - Games data management hook

#### Services
- `api.js` - Backend API service
- `gamesApi.js` - BallDon'tLie API service

#### Utils
- `constants.js` - Application constants
- Additional utility functions

### Code Style

The project follows modern React best practices:
- Functional components with hooks
- Custom hooks for reusable logic
- Service layer for API calls
- Proper error handling
- TypeScript-like JSDoc comments

## Deployment

### Build for Production
```bash
npm run build
```

### Deploy to Render
1. Connect your GitHub repository to Render
2. Set build command: `npm run build`
3. Set publish directory: `build`
4. Configure environment variables

### Deploy to Netlify
1. Connect your GitHub repository to Netlify
2. Set build command: `npm run build`
3. Set publish directory: `build`

## API Integration

The frontend integrates with two APIs:

### Backend API
- Prediction endpoints
- Health check endpoints
- Team data endpoints

### BallDon'tLie API
- Upcoming games
- Team information
- Game schedules

## Performance

- Code splitting with React.lazy()
- Optimized bundle size
- Efficient re-renders
- Image optimization
- Caching strategies

## Browser Support

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

## Testing

```bash
# Run tests
npm test

# Run tests with coverage
npm test -- --coverage
```

## Contributing

1. Follow the existing code style
2. Add tests for new features
3. Update documentation
4. Use conventional commits 