/**
 * Rate limit information component.
 * 
 * This component displays helpful information about API rate limits
 * and what users can do when they encounter rate limiting.
 */

import React from 'react';
import './RateLimitInfo.css';

const RateLimitInfo = ({ visible = false }) => {
  if (!visible) return null;

  return (
    <div className="rate-limit-info">
      <div className="info-icon">ℹ️</div>
      <div className="info-content">
        <h4>Rate Limit Information</h4>
        <p>
          The BallDon'tLie API has a rate limit of 60 requests per minute. 
          If you see a rate limit error, please wait a moment before trying again.
        </p>
        <ul>
          <li>Free tier: 60 requests per minute</li>
          <li>Wait 60 seconds if you hit the limit</li>
          <li>Use the retry button to try again</li>
        </ul>
      </div>
    </div>
  );
};

export default RateLimitInfo; 