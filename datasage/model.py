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
    """Enhanced local LLM generator with multiple model support and better prompting."""
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        **generation_kwargs
    ):
        """Initialize the LLM generator with intelligent model selection.
        
        Args:
            model_name: HuggingFace model identifier
            **generation_kwargs: Additional generation parameters
        """
        # Intelligent model selection - prefer larger, more capable models
        if model_name is None:
            # Try to use better models if available, fall back to FLAN-T5
            available_models = [
                "microsoft/DialoGPT-medium",   # More reasonable size
                "google/flan-t5-large",        # Larger FLAN-T5 if available
                "google/flan-t5-base"          # Reliable fallback
            ]
            self.model_name = self._select_best_available_model(available_models)
        else:
            self.model_name = model_name
        
        # Enhanced generation parameters for better quality
        self.generation_config = {
            "max_new_tokens": 800,  # Increased for more detailed output
            "temperature": 0.3,     # Lower for more focused responses
            "do_sample": True,
            "top_p": 0.85,          # Slightly more focused
            "top_k": 50,            # Add top-k for better control
            "repetition_penalty": 1.2,  # Stronger penalty
            "no_repeat_ngram_size": 3,   # Prevent longer repeats
            # Removed early_stopping as it's not supported by all models
        }
        
        self.model = None
        self.tokenizer = None
        self.device = None
        self._model_loaded = False
    
    def _select_best_available_model(self, model_list: List[str]) -> str:
        """Select the best available model from a list.
        
        Args:
            model_list: List of model names in order of preference
            
        Returns:
            The best available model name
        """
        # For now, just return the last (most reliable) option
        # In the future, we could check model availability
        return model_list[-1]
    
    def _is_generative_model(self) -> bool:
        """Check if the current model is a generative (GPT-style) model."""
        return "gpt" in self.model_name.lower() or "dialog" in self.model_name.lower()
    
    def _load_model(self) -> None:
        """Load the model and tokenizer with enhanced error handling and model support."""
        if self._model_loaded:
            return
        
        try:
            import torch
            
            logger.info(f"Loading model: {self.model_name}")
            
            # Handle different model types
            if self._is_generative_model():
                # GPT-style models
                from transformers import AutoTokenizer, AutoModelForCausalLM
                
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                # Load model with GPU support
                if torch.cuda.is_available():
                    device_name = torch.cuda.get_device_name(0)
                    logger.info(f"🚀 GPU detected: {device_name}. Loading model on GPU!")
                    print(f"🚀 GPU acceleration enabled: {device_name}")
                    
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        device_map="auto",
                        torch_dtype=torch.float16
                    )
                    self.device = "cuda"
                else:
                    logger.info("No GPU found, using CPU")
                    print("Using CPU (consider using GPU for faster generation)")
                    self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
                    self.device = "cpu"
            
            else:
                # T5-style models (encoder-decoder)
                from transformers import T5Tokenizer, T5ForConditionalGeneration
                
                self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
                
                # Load model with GPU support
                if torch.cuda.is_available():
                    device_name = torch.cuda.get_device_name(0)
                    logger.info(f"🚀 GPU detected: {device_name}. Loading model on GPU!")
                    print(f"🚀 GPU acceleration enabled: {device_name}")
                    
                    self.model = T5ForConditionalGeneration.from_pretrained(
                        self.model_name, 
                        device_map="auto",
                        torch_dtype=torch.float16
                    )
                    self.device = "cuda"
                else:
                    logger.info("No GPU found, using CPU")
                    print("Using CPU (consider using GPU for faster generation)")
                    self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
                    self.device = "cpu"
            
            self._model_loaded = True
            logger.info(f"Model {self.model_name} loaded successfully")
            
        except ImportError as e:
            logger.error(f"Failed to import transformers: {e}")
            raise RuntimeError(
                "Please install transformers: pip install transformers torch"
            ) from e
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e
    
    def generate_markdown(self, prompt: str) -> str:
        """Generate markdown text from a prompt with enhanced generation.
        
        Args:
            prompt: Input prompt for the model
            
        Returns:
            Generated markdown text
        """
        self._load_model()
        
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded")
        
        try:
            import torch
            
            # Enhanced prompt preprocessing
            processed_prompt = self._preprocess_prompt(prompt)
            
            # Tokenize input and move to appropriate device
            inputs = self.tokenizer(
                processed_prompt, 
                return_tensors="pt", 
                padding=True,
                truncation=True,
                max_length=1024  # Ensure we don't exceed context
            )
            input_ids = inputs.input_ids
            attention_mask = inputs.attention_mask
            
            if self.device == "cuda":
                input_ids = input_ids.to("cuda")
                attention_mask = attention_mask.to("cuda")
            
            # Generate with enhanced parameters
            with torch.no_grad():
                if self._is_generative_model():
                    # For GPT-style models
                    outputs = self.model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=self.generation_config.get("max_new_tokens", 800),
                        temperature=self.generation_config.get("temperature", 0.3),
                        top_p=self.generation_config.get("top_p", 0.85),
                        top_k=self.generation_config.get("top_k", 50),
                        do_sample=self.generation_config.get("do_sample", True),
                        repetition_penalty=self.generation_config.get("repetition_penalty", 1.2),
                        no_repeat_ngram_size=self.generation_config.get("no_repeat_ngram_size", 3),
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                    # For generative models, skip the input when decoding
                    input_length = input_ids.shape[1]
                    generated_text = self.tokenizer.decode(
                        outputs[0][input_length:], 
                        skip_special_tokens=True
                    )
                else:
                    # For T5-style models
                    outputs = self.model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=self.generation_config.get("max_new_tokens", 800),
                        temperature=self.generation_config.get("temperature", 0.3),
                        top_p=self.generation_config.get("top_p", 0.85),
                        top_k=self.generation_config.get("top_k", 50),
                        do_sample=self.generation_config.get("do_sample", True),
                        repetition_penalty=self.generation_config.get("repetition_penalty", 1.2),
                        no_repeat_ngram_size=self.generation_config.get("no_repeat_ngram_size", 3),
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                    generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Post-process the output
            return self._postprocess_output(generated_text.strip())
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"**Error generating response**: {str(e)}"
    
    def _preprocess_prompt(self, prompt: str) -> str:
        """Preprocess the prompt to improve generation quality.
        
        Args:
            prompt: Original prompt
            
        Returns:
            Enhanced prompt
        """
        # For T5 models, ensure the prompt is properly formatted
        if not self._is_generative_model():
            return prompt
        
        # For GPT-style models, add context and formatting instructions
        enhanced_prompt = f"""You are an expert data analyst creating professional data quality reports. 

{prompt}

Please provide a detailed, well-structured markdown report following the requested format. Use clear headings, bullet points, and professional language. Focus on actionable insights and practical recommendations.

Report:"""
        
        return enhanced_prompt
    
    def _postprocess_output(self, text: str) -> str:
        """Post-process the generated text to improve quality.
        
        Args:
            text: Raw generated text
            
        Returns:
            Cleaned and enhanced text
        """
        # Remove common repetitions and artifacts
        lines = text.split('\n')
        clean_lines = []
        prev_line = ""
        
        for line in lines:
            line = line.strip()
            # Skip empty lines and repetitions
            if line and line != prev_line:
                clean_lines.append(line)
                prev_line = line
        
        # Join and clean up
        cleaned_text = '\n'.join(clean_lines)
        
        # Remove any remaining artifacts
        cleaned_text = cleaned_text.replace('**Error generating response**:', '')
        cleaned_text = cleaned_text.replace('Response:', '')
        
        # Ensure minimum content quality
        if len(cleaned_text.strip()) < 100:
            return "**Unable to generate detailed analysis.** The model output was too brief or incomplete."
        
        return cleaned_text
    
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
            "device": self.device,
            "generation_config": self.generation_config,
            "loaded": self._model_loaded
        }
