/**
 * NBA Predictor - Main Application Component
 * 
 * A modern betting-style interface for NBA game predictions
 * with machine learning-powered insights and real-time odds.
 */

import React from 'react';
import { ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import MatchupForm from './components/MatchupForm/MatchupForm';
import './App.css';

const App = () => {
  return (
    <div className="App">
      {/* Header */}
      <header className="app-header">
        <div className="header-content">
          <a href="/" className="logo">
            🏀 NBA Predictor Pro
          </a>
          <div className="header-stats">
            <span className="stat-item">
              <span className="stat-label">Accuracy</span>
              <span className="stat-value">85%</span>
            </span>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="main-content">
      <MatchupForm />
      </main>

      {/* Toast Notifications */}
      <ToastContainer
        position="top-right"
        autoClose={5000}
        hideProgressBar={false}
        newestOnTop={false}
        closeOnClick
        rtl={false}
        pauseOnFocusLoss
        draggable
        pauseOnHover
        theme="dark"
        toastStyle={{
          background: 'rgba(255, 255, 255, 0.1)',
          backdropFilter: 'blur(20px)',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          color: '#ffffff',
        }}
      />
    </div>
  );
};
 
export default App;
