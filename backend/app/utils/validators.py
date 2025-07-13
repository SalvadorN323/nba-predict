"""
Validation utilities for the NBA Predictor backend.

This module provides validation functions for API requests
and data validation.
"""

from typing import Dict, Any, List

def validate_prediction_request(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate a prediction request.
    
    Args:
        data: Request data dictionary
        
    Returns:
        Dictionary with validation result
    """
    if not data:
        return {
            'valid': False,
            'message': 'Request data is required'
        }
    
    # Check required fields
    required_fields = ['team_a', 'team_b']
    for field in required_fields:
        if field not in data:
            return {
                'valid': False,
                'message': f'Missing required field: {field}'
            }
        
        if not data[field] or not isinstance(data[field], str):
            return {
                'valid': False,
                'message': f'Field {field} must be a non-empty string'
            }
    
    # Check if teams are different
    if data['team_a'] == data['team_b']:
        return {
            'valid': False,
            'message': 'Team A and Team B must be different teams'
        }
    
    return {'valid': True}

def validate_team_name(team_name: str) -> bool:
    """
    Validate a team name.
    
    Args:
        team_name: Team name to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not team_name or not isinstance(team_name, str):
        return False
    
    # Basic validation - team name should be reasonable length
    if len(team_name.strip()) < 3 or len(team_name.strip()) > 50:
        return False
    
    return True

def sanitize_team_name(team_name: str) -> str:
    """
    Sanitize a team name for safe processing.
    
    Args:
        team_name: Raw team name
        
    Returns:
        Sanitized team name
    """
    if not team_name:
        return ""
    
    # Remove extra whitespace and normalize
    sanitized = team_name.strip()
    
    # Basic sanitization - remove potentially dangerous characters
    # Keep alphanumeric, spaces, and common punctuation
    import re
    sanitized = re.sub(r'[^\w\s\-\.]', '', sanitized)
    
    return sanitized 