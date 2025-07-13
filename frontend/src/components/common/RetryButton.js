/**
 * Retry button component for failed API calls.
 * 
 * This component provides a user-friendly way to retry failed operations
 * with proper loading states and error handling.
 */

import React from 'react';
import './RetryButton.css';

const RetryButton = ({ onRetry, loading, error, children = 'Retry' }) => {
  return (
    <div className="retry-container">
      {error && (
        <div className="error-message">
          {error}
        </div>
      )}
      <button 
        className="retry-button"
        onClick={onRetry}
        disabled={loading}
      >
        {loading ? 'Retrying...' : children}
      </button>
    </div>
  );
};

export default RetryButton; 