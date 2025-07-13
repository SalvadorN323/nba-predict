"""
OpenAI service for generating AI-powered analysis.

This service handles all interactions with the OpenAI API
for generating expert analysis of NBA matchups.
"""

import openai
from typing import Dict, Any, Optional
from app.utils.logger import get_logger
from config.settings import get_config

logger = get_logger(__name__)

class OpenAIService:
    """Service for OpenAI API interactions."""
    
    def __init__(self):
        """Initialize the OpenAI service."""
        self.config = get_config()
        openai.api_key = self.config.OPENAI_API_KEY
        self.model = self.config.OPENAI_MODEL
        self.max_tokens = self.config.OPENAI_MAX_TOKENS
    
    def check_service_status(self) -> Dict[str, Any]:
        """
        Check the status of the OpenAI service.
        
        Returns:
            Dictionary with service status information
        """
        try:
            if not self.config.OPENAI_API_KEY:
                return {'status': 'error', 'message': 'OpenAI API key not configured'}
            
            # Try a simple API call to test connectivity
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            
            return {
                'status': 'healthy',
                'model': self.model,
                'max_tokens': self.max_tokens
            }
            
        except Exception as e:
            logger.error(f"OpenAI service check failed: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def generate_analysis(self, team_a: str, team_b: str, prediction: str, 
                         win_probability: float, loss_probability: float, 
                         stats: Dict[str, float]) -> Dict[str, Any]:
        """
        Generate AI-powered analysis for a matchup.
        
        Args:
            team_a: Name of the first team
            team_b: Name of the second team
            prediction: Model prediction result
            win_probability: Probability of Team A winning
            loss_probability: Probability of Team B winning
            stats: Team statistics for analysis
            
        Returns:
            Dictionary containing the generated analysis
        """
        try:
            if not self.config.OPENAI_API_KEY:
                logger.warning("OpenAI API key not configured, skipping analysis")
                return {'analysis': 'AI analysis not available - API key not configured'}
            
            # Create the analysis prompt
            prompt = self._create_analysis_prompt(
                team_a=team_a,
                team_b=team_b,
                prediction=prediction,
                win_probability=win_probability,
                loss_probability=loss_probability,
                stats=stats
            )
            
            # Generate analysis using OpenAI
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                temperature=0.7
            )
            
            analysis = response['choices'][0]['message']['content'].strip()
            
            logger.info(f"AI analysis generated successfully for {team_a} vs {team_b}")
            
            return {'analysis': analysis}
            
        except Exception as e:
            logger.error(f"Error generating AI analysis: {str(e)}")
            return {'analysis': f'AI analysis failed: {str(e)}'}
    
    def _create_analysis_prompt(self, team_a: str, team_b: str, prediction: str,
                               win_probability: float, loss_probability: float,
                               stats: Dict[str, float]) -> str:
        """
        Create the analysis prompt for OpenAI.
        
        Args:
            team_a: Name of the first team
            team_b: Name of the second team
            prediction: Model prediction result
            win_probability: Probability of Team A winning
            loss_probability: Probability of Team B winning
            stats: Team statistics for analysis
            
        Returns:
            Formatted prompt string
        """
        # Format key statistics for the prompt
        key_stats = {
            'Points': stats.get('PTS', 0),
            'Field Goal %': f"{stats.get('FG_PCT', 0):.3f}",
            'Rebounds': stats.get('REB', 0),
            'Assists': stats.get('AST', 0),
            'Turnovers': stats.get('TOV', 0),
            'Steals': stats.get('STL', 0),
            'Blocks': stats.get('BLK', 0)
        }
        
        prompt = f"""
        Analyze the following NBA game prediction as an expert basketball analyst:

        MATCHUP: {team_a} vs {team_b}
        
        MODEL PREDICTION: {prediction}
        - {team_a} Win Probability: {win_probability:.1%}
        - {team_b} Win Probability: {loss_probability:.1%}
        
        TEAM A ({team_a}) RECENT PERFORMANCE:
        {chr(10).join([f"- {stat}: {value}" for stat, value in key_stats.items()])}
        
        Please provide a concise, expert analysis that includes:
        1. Key factors influencing this prediction
        2. Statistical insights about the matchup
        3. Potential game-changing elements
        4. Why the model prediction makes sense based on the data
        
        Keep the analysis professional, informative, and under 200 words.
        Focus on basketball insights rather than gambling advice.
        """
        
        return prompt.strip() 