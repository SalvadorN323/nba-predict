/**
 * Rate limiter utility for API calls.
 * 
 * This utility helps manage API rate limits by implementing
 * exponential backoff and request queuing.
 */

class RateLimiter {
  constructor(maxRequests = 60, timeWindow = 60000) { // 60 requests per minute
    this.maxRequests = maxRequests;
    this.timeWindow = timeWindow;
    this.requests = [];
    this.isBlocked = false;
    this.blockUntil = 0;
  }

  /**
   * Check if we can make a request.
   * 
   * @returns {boolean} True if request can be made
   */
  canMakeRequest() {
    const now = Date.now();
    
    // Check if we're currently blocked
    if (this.isBlocked && now < this.blockUntil) {
      return false;
    }
    
    // Clear expired requests
    this.requests = this.requests.filter(time => now - time < this.timeWindow);
    
    // Check if we're under the limit
    return this.requests.length < this.maxRequests;
  }

  /**
   * Record a request.
   */
  recordRequest() {
    this.requests.push(Date.now());
  }

  /**
   * Handle rate limit error and set block period.
   * 
   * @param {number} retryAfter - Retry after time in seconds
   */
  handleRateLimit(retryAfter = 60) {
    this.isBlocked = true;
    this.blockUntil = Date.now() + (retryAfter * 1000);
    console.warn(`Rate limited. Retry after ${retryAfter} seconds.`);
  }

  /**
   * Get delay before next request can be made.
   * 
   * @returns {number} Delay in milliseconds
   */
  getDelay() {
    const now = Date.now();
    
    if (this.isBlocked && now < this.blockUntil) {
      return this.blockUntil - now;
    }
    
    if (this.requests.length >= this.maxRequests) {
      const oldestRequest = this.requests[0];
      return this.timeWindow - (now - oldestRequest);
    }
    
    return 0;
  }

  /**
   * Wait for the specified delay.
   * 
   * @param {number} delay - Delay in milliseconds
   * @returns {Promise} Promise that resolves after delay
   */
  static async wait(delay) {
    return new Promise(resolve => setTimeout(resolve, delay));
  }
}

// Create a singleton instance for the BallDon'tLie API
export const ballDontLieRateLimiter = new RateLimiter(60, 60000); // 60 requests per minute

export default RateLimiter; 