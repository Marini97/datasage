"""
Enhanced LLM generator with multiple model support and fallback strategies.

This module provides a robust generator that can try multiple approaches
to generate high-quality data analysis reports.
"""

import os
import logging
from typing import Dict, Any, List, Optional, Union
from .model import LocalLLMGenerator

logger = logging.getLogger(__name__)


class EnhancedLLMGenerator:
    """Enhanced generator that tries multiple approaches for best quality."""
    
    def __init__(self, preferred_model: Optional[str] = None):
        """Initialize the enhanced generator.
        
        Args:
            preferred_model: Preferred model to try first
        """
        self.preferred_model = preferred_model
        self.fallback_models = [
            "google/flan-t5-large",    # Larger model if available
            "google/flan-t5-base",     # Reliable fallback
        ]
        self.current_generator = None
        
    def _try_openai_compatible(self, prompt: str) -> Optional[str]:
        """Try OpenAI-compatible API if available.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated text or None if not available
        """
        try:
            # Check for OpenAI API key or compatible endpoint
            api_key = os.getenv('OPENAI_API_KEY')
            api_base = os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1')
            
            if not api_key:
                return None
                
            import openai
            
            client = openai.OpenAI(
                api_key=api_key,
                base_url=api_base
            )
            
            # Try a small, efficient model for data analysis
            models_to_try = ['gpt-3.5-turbo', 'gpt-4o-mini', 'gpt-4']
            
            for model in models_to_try:
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {
                                "role": "system", 
                                "content": "You are an expert data analyst specializing in data quality assessment and reporting. Provide detailed, actionable insights in professional markdown format."
                            },
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=1500,
                        temperature=0.3
                    )
                    
                    return response.choices[0].message.content.strip()
                    
                except Exception as e:
                    logger.debug(f"Failed to use {model}: {e}")
                    continue
            
            return None
            
        except ImportError:
            logger.debug("OpenAI library not available")
            return None
        except Exception as e:
            logger.debug(f"OpenAI API failed: {e}")
            return None
    
    def _try_local_models(self, prompt: str) -> Optional[str]:
        """Try local models in order of preference.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated text or None if all fail
        """
        models_to_try = []
        
        if self.preferred_model:
            models_to_try.append(self.preferred_model)
        
        models_to_try.extend(self.fallback_models)
        
        for model_name in models_to_try:
            try:
                generator = LocalLLMGenerator(model_name=model_name)
                result = generator.generate_markdown(prompt)
                
                # Basic quality check
                if result and len(result.strip()) > 100 and "Error generating response" not in result:
                    self.current_generator = generator
                    return result
                    
            except Exception as e:
                logger.debug(f"Failed to use local model {model_name}: {e}")
                continue
        
        return None
    
    def generate_markdown(self, prompt: str) -> str:
        """Generate markdown using the best available method.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated markdown text
        """
        # Try OpenAI-compatible API first (usually higher quality)
        result = self._try_openai_compatible(prompt)
        if result:
            logger.info("Using OpenAI-compatible API for generation")
            return result
        
        # Try local models as fallback
        result = self._try_local_models(prompt)
        if result:
            logger.info(f"Using local model for generation")
            return result
        
        # If everything fails, return an error message
        logger.error("All LLM generation methods failed")
        return "**Unable to generate AI analysis.** All available language models failed to produce output. Please check your configuration and try again."
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model.
        
        Returns:
            Model information dictionary
        """
        if self.current_generator:
            return self.current_generator.get_model_info()
        
        return {
            "status": "No model loaded",
            "attempted_models": self.fallback_models,
            "preferred_model": self.preferred_model
        }
