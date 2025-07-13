/**
 * Betting Calculator Component
 * 
 * A betting-style calculator that shows potential winnings,
 * odds conversion, and betting recommendations.
 */

import React, { useState } from 'react';
import './BettingCalculator.css';

const BettingCalculator = ({ winProbability, teamName, onBetAmountChange }) => {
  const [betAmount, setBetAmount] = useState(10);
  const [betType, setBetType] = useState('moneyline');

  // Calculate odds and potential winnings
  const calculateOdds = (probability) => {
    if (probability >= 0.5) {
      // Favorite (negative odds)
      const odds = Math.round((probability / (1 - probability)) * -100);
      return odds;
    } else {
      // Underdog (positive odds)
      const odds = Math.round(((1 - probability) / probability) * 100);
      return odds;
    }
  };

  const calculatePayout = (amount, odds) => {
    if (odds > 0) {
      // Positive odds (underdog)
      return amount + (amount * odds / 100);
    } else {
      // Negative odds (favorite)
      return amount + (amount * 100 / Math.abs(odds));
    }
  };

  const americanOdds = calculateOdds(winProbability);
  const decimalOdds = (1 / winProbability).toFixed(2);
  const fractionalOdds = winProbability >= 0.5 
    ? `${Math.round((1 - winProbability) / winProbability * 100)}/100`
    : `${Math.round((1 - winProbability) / winProbability)}/1`;
  
  const potentialPayout = calculatePayout(betAmount, americanOdds);
  const profit = potentialPayout - betAmount;

  const handleBetAmountChange = (e) => {
    const amount = parseFloat(e.target.value) || 0;
    setBetAmount(amount);
    if (onBetAmountChange) {
      onBetAmountChange(amount);
    }
  };

  const quickAmounts = [5, 10, 25, 50, 100, 250];

  return (
    <div className="betting-calculator">
      <div className="calculator-header">
        <h4>Betting Calculator</h4>
        <span className="team-name">{teamName}</span>
      </div>

      {/* Odds Display */}
      <div className="odds-display-grid">
        <div className="odds-card">
          <div className="odds-label">American</div>
          <div className={`odds-value ${americanOdds > 0 ? 'positive' : 'negative'}`}>
            {americanOdds > 0 ? '+' : ''}{americanOdds}
          </div>
        </div>
        <div className="odds-card">
          <div className="odds-label">Decimal</div>
          <div className="odds-value">{decimalOdds}</div>
        </div>
        <div className="odds-card">
          <div className="odds-label">Fractional</div>
          <div className="odds-value">{fractionalOdds}</div>
        </div>
        <div className="odds-card">
          <div className="odds-label">Implied</div>
          <div className="odds-value">{(winProbability * 100).toFixed(1)}%</div>
        </div>
      </div>

      {/* Bet Amount Input */}
      <div className="bet-amount-section">
        <label className="bet-amount-label">Bet Amount ($)</label>
        <div className="bet-amount-input-container">
          <span className="currency-symbol">$</span>
          <input
            type="number"
            value={betAmount}
            onChange={handleBetAmountChange}
            className="bet-amount-input"
            min="1"
            step="1"
            placeholder="Enter amount"
          />
        </div>
        
        {/* Quick Amount Buttons */}
        <div className="quick-amounts">
          {quickAmounts.map(amount => (
            <button
              key={amount}
              onClick={() => {
                setBetAmount(amount);
                if (onBetAmountChange) onBetAmountChange(amount);
              }}
              className={`quick-amount-btn ${betAmount === amount ? 'active' : ''}`}
            >
              ${amount}
            </button>
          ))}
        </div>
      </div>

      {/* Payout Display */}
      <div className="payout-section">
        <div className="payout-card">
          <div className="payout-label">Potential Payout</div>
          <div className="payout-amount">${potentialPayout.toFixed(2)}</div>
        </div>
        <div className="payout-card">
          <div className="payout-label">Profit</div>
          <div className={`payout-profit ${profit > 0 ? 'positive' : 'negative'}`}>
            ${profit.toFixed(2)}
          </div>
        </div>
      </div>

      {/* Betting Recommendation */}
      <div className="betting-recommendation">
        <div className="recommendation-header">
          <span className="recommendation-icon">💡</span>
          <span className="recommendation-title">Betting Tip</span>
        </div>
        <div className="recommendation-content">
          {winProbability > 0.6 ? (
            <p>Strong favorite - consider a larger bet for consistent returns</p>
          ) : winProbability > 0.45 ? (
            <p>Close matchup - moderate bet size recommended</p>
          ) : (
            <p>Underdog opportunity - smaller bet with higher potential return</p>
          )}
        </div>
      </div>
    </div>
  );
};

export default BettingCalculator; 