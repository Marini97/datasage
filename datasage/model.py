"""
Local LLM model generation for DataSage.

This module provides utilities for loading and running local Hugging Face models
for text generation with deterministic settings.
"""

import os
from typing import Dict, Any, List, Optional
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class LocalLLMGenerator:
    """Local language model generator using Hugging Face transformers."""
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: str = "cpu",
        **generation_kwargs
    ):
        """Initialize the local LLM generator.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run the model on ('cpu' or 'cuda')
            **generation_kwargs: Additional generation parameters
        """
        self.model_name = model_name or os.getenv("DATASAGE_MODEL", "google/flan-t5-base")
        
        # Default generation parameters for deterministic output
        # Configure generation parameters for more reliable output
        self.generation_config = {
            "max_new_tokens": 300,     # Reduced for more focused output
            "temperature": 0.7,        # More randomness to break repetition
            "do_sample": True,         # Enable sampling
            "top_p": 0.9,              # Nucleus sampling
            "repetition_penalty": 2.0, # Strong penalty to prevent loops
            "no_repeat_ngram_size": 3, # Prevent 3-gram repetition
        }
        
        self.pipeline = None
        self.tokenizer = None
        self._model_loaded = False
    
    def _load_model(self) -> None:
        """Load the model and tokenizer lazily."""
        if self._model_loaded:
            return
        
        try:
            from transformers import pipeline, AutoTokenizer
            import torch
            
            logger.info(f"Loading model: {self.model_name}")
            
            # Load tokenizer first to get pad_token_id
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Set pad_token_id if not set
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            self.generation_config["pad_token_id"] = self.tokenizer.pad_token_id
            
            # Auto-detect best available device
            if torch.cuda.is_available():
                device = 0  # Use first GPU
                device_name = torch.cuda.get_device_name(0)
                torch_dtype = torch.float16  # Use half precision on GPU for speed
                logger.info(f"🚀 GPU detected: {device_name}. Using GPU for faster generation!")
                print(f"Device set to use GPU ({device_name}) - this should be faster!")
            else:
                device = -1  # Use CPU
                torch_dtype = torch.float32
                logger.info("No GPU found, using CPU (still works, just a bit slower)")
                print("Device set to use CPU")
            
            # Load the pipeline with optimal settings
            self.pipeline = pipeline(
                "text2text-generation",
                model=self.model_name,
                tokenizer=self.tokenizer,
                device=device,
                torch_dtype=torch_dtype,
                model_kwargs={"low_cpu_mem_usage": True} if device == -1 else {}
            )
            
            self._model_loaded = True
            logger.info("Model loaded successfully")
            
        except ImportError as e:
            logger.error(f"Failed to import transformers: {e}")
            raise RuntimeError(
                "Please install transformers: pip install transformers torch"
            ) from e
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e
    
    def generate_markdown(self, prompt: str) -> str:
        """Generate markdown text from a prompt.
        
        Args:
            prompt: Input prompt for the model
            
        Returns:
            Generated markdown text
        """
        self._load_model()
        
        if not self.pipeline:
            raise RuntimeError("Model pipeline not loaded")
        
        try:
            # Generate text
            response = self.pipeline(
                prompt,
                max_new_tokens=self.generation_config.get("max_new_tokens", 300),
                temperature=self.generation_config.get("temperature", 0.0),
                top_p=self.generation_config.get("top_p", 1.0),
                do_sample=self.generation_config.get("temperature", 0.0) > 0,
                pad_token_id=self.generation_config.get("pad_token_id"),
            )
            
            if isinstance(response, list) and len(response) > 0:
                generated_text = response[0].get("generated_text", "")
            else:
                generated_text = str(response)
            
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"**Error generating response**: {str(e)}"
    
    def generate_batch(self, prompts: List[str]) -> List[str]:
        """Generate markdown text for multiple prompts.
        
        Args:
            prompts: List of input prompts
            
        Returns:
            List of generated markdown texts
        """
        results = []
        for prompt in prompts:
            result = self.generate_markdown(prompt)
            results.append(result)
        
        return results
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for a given text.
        
        Args:
            text: Input text to count tokens for
            
        Returns:
            Estimated token count
        """
        # Simple estimation: roughly 4 characters per token
        return len(text) // 4
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "generation_config": self.generation_config,
            "loaded": self._model_loaded
        }
