# Rate Limiting Solution

This document explains how the NBA Predictor frontend handles API rate limiting for the BallDon'tLie API.

## Problem

The BallDon'tLie API has a rate limit of 60 requests per minute for the free tier. When this limit is exceeded, the API returns a 429 (Too Many Requests) status code.

## Solution

### 1. Rate Limiter Utility (`src/utils/rateLimiter.js`)

A client-side rate limiter that:
- Tracks API requests within a time window
- Prevents requests when the limit is reached
- Handles 429 responses with exponential backoff
- Provides user-friendly error messages

### 2. Enhanced API Service (`src/services/gamesApi.js`)

Updated to:
- Check rate limits before making requests
- Handle 429 responses gracefully
- Provide detailed error messages
- Record all API calls for tracking

### 3. User Interface Components

#### RetryButton Component (`src/components/common/RetryButton.js`)
- Provides a user-friendly way to retry failed requests
- Shows loading states during retry attempts
- Displays error messages clearly

#### RateLimitInfo Component (`src/components/common/RateLimitInfo.js`)
- Explains rate limiting to users
- Provides helpful information about limits
- Shows when rate limiting is active

### 4. Enhanced Error Handling

The `useGames` hook now:
- Distinguishes between rate limit errors and other errors
- Shows appropriate toast notifications
- Provides retry functionality

## Usage

### For Users
1. If you see a "Rate limit exceeded" error, wait 60 seconds
2. Use the "Retry" button to try again
3. The app will automatically handle rate limiting

### For Developers
1. The rate limiter is automatically applied to all BallDon'tLie API calls
2. Rate limit errors are handled gracefully with user-friendly messages
3. The system tracks requests and prevents exceeding limits

## Configuration

The rate limiter is configured for:
- **60 requests per minute** (BallDon'tLie free tier limit)
- **60-second retry delay** when rate limited
- **Automatic request tracking** across the application

## Error Messages

- **Rate limit exceeded**: Wait 60 seconds before retrying
- **API error**: General API failure, try again later
- **Network error**: Check your internet connection

## Best Practices

1. **Don't spam the refresh button** - it won't help with rate limits
2. **Wait for the retry delay** - the app will tell you when to try again
3. **Use the retry button** - it's designed to handle rate limiting properly
4. **Check the rate limit info** - it explains what's happening

## Technical Details

### Rate Limiter Implementation
```javascript
// Check if request can be made
if (!ballDontLieRateLimiter.canMakeRequest()) {
  const delay = ballDontLieRateLimiter.getDelay();
  throw new Error(`Rate limit exceeded. Please wait ${Math.ceil(delay / 1000)} seconds.`);
}

// Record the request
ballDontLieRateLimiter.recordRequest();
```

### Error Handling
```javascript
if (response.status === 429) {
  const retryAfter = response.headers.get('Retry-After');
  const retrySeconds = retryAfter ? parseInt(retryAfter) : 60;
  ballDontLieRateLimiter.handleRateLimit(retrySeconds);
  throw new Error(`Rate limit exceeded. Please try again in ${retrySeconds} seconds.`);
}
```

## Future Improvements

1. **Caching**: Cache API responses to reduce API calls
2. **Queue System**: Queue requests when rate limited
3. **Progressive Backoff**: Implement exponential backoff for retries
4. **User Preferences**: Allow users to set their own rate limit preferences 