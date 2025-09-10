"""HouseModel service with LoRA integration for House personality generation.

This module provides the HouseModel service that loads and uses the fine-tuned
LoRA adapter to generate House MD personality responses.
"""

import logging
import time
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import json

try:
    import torch
    from transformers import (
        AutoTokenizer, 
        AutoModelForSeq2SeqLM,
        GenerationConfig
    )
    from peft import PeftModel, PeftConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from ..models.context import ConversationContext
from ..models.response import HouseResponse
from ..models.house_quote import HouseQuote
from src.models.house_quote import HouseQuote



class HouseModelError(Exception):
    """Base exception for HouseModel operations."""
    pass


class ModelLoadError(HouseModelError):
    """Raised when model loading fails."""
    pass


class GenerationError(HouseModelError):
    """Raised when text generation fails."""
    pass


class InferenceError(HouseModelError):
    """Raised when model inference fails."""
    pass


class ValidationError(HouseModelError):
    """Raised when input validation fails."""
    pass


class HouseModel:
    """LoRA-enhanced language model for House MD personality generation.
    
    Loads the fine-tuned LoRA adapter and generates House-style responses
    with proper personality characteristics.
    """
    
    def __init__(self, config, strict_mode: bool = False):
        """Initialize HouseModel with LoRA adapter.
        
        Args:
            config: Configuration with model settings
            strict_mode: If True, raise errors instead of falling back to mock mode
            
        Raises:
            ModelLoadError: If model loading fails and strict_mode=True
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.strict_mode = strict_mode
        
        # Model components
        self.tokenizer = None
        self.base_model = None
        self.model = None
        self.generation_config = None
        
        # Performance tracking
        self.load_time = None
        self.loaded_at = None  # Track when model was loaded
        self.generation_count = 0
        self.total_generation_time = 0.0
        
        # Model state
        self.is_loaded = False
        self.mock_mode = not TRANSFORMERS_AVAILABLE
        
        if self.mock_mode:
            self.logger.warning("Transformers not available - using mock mode")
            self._init_mock_responses()
        else:
            try:
                self._load_model()
            except Exception as e:
                if self.strict_mode:
                    # In strict mode, propagate the error
                    raise ModelLoadError(f"Model loading failed: {e}")
                else:
                    # In normal mode, fall back to mock mode
                    self.logger.warning(f"Model loading failed, falling back to mock mode: {e}")
                    self.mock_mode = True
                    self._init_mock_responses()
    
    def _init_mock_responses(self):
        """Initialize mock responses for testing."""
        self.mock_responses = [
            "Everybody lies. That's the first rule of medicine.",
            "It's not lupus. It's never lupus.",
            "People don't change. They just find new ways to disappoint you.",
            "The only thing worse than being wrong is being boring.",
            "Medicine is not about being right. It's about being useful.",
            "Rational arguments don't usually work on religious people. Otherwise there would be no religious people.",
            "You can't always get what you want, but if you try sometimes, you get what you need.",
            "The truth begins with a lie.",
            "Treating illnesses is why we became doctors. Treating patients is what makes most doctors miserable.",
            "Reality is almost always wrong."
        ]
        self.is_loaded = True
        self.loaded_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.logger.info("Mock mode initialized with House personality responses")
    
    def _load_model(self):
        """Load LoRA model and tokenizer."""
        try:
            start_time = time.time()
            
            # Validate model path
            lora_path = getattr(self.config.model, 'lora_model_path', None) or \
                       getattr(self.config.model, 'lora_weights_path', None)
            
            if not lora_path:
                raise ModelLoadError("No LoRA model path specified")
                
            model_path = Path(lora_path)
            if not model_path.exists():
                raise ModelLoadError(f"LoRA model not found: {model_path}")
            
            required_files = [
                "adapter_config.json",
                "adapter_model.safetensors",
                "tokenizer.json",
                "tokenizer_config.json"
            ]
            
            missing_files = []
            for file in required_files:
                if not (model_path / file).exists():
                    missing_files.append(file)
            
            if missing_files:
                raise ModelLoadError(f"Missing LoRA files: {missing_files}")
            
            self.logger.info(f"Loading LoRA model from {model_path}")
            
            # Load tokenizer
            self.logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(model_path),
                local_files_only=True
            )
            
            # Load base model
            self.logger.info(f"Loading base model: {self.config.model.base_model_name}")
            
            # Determine device and dtype
            device_map = self._get_device_map()
            torch_dtype = self._get_torch_dtype()
            
            self.base_model = AutoModelForSeq2SeqLM.from_pretrained(
                self.config.model.base_model_name,
                torch_dtype=torch_dtype,
                device_map=device_map,
                load_in_8bit=getattr(self.config.model, 'load_in_8bit', False),
                load_in_4bit=getattr(self.config.model, 'load_in_4bit', False),
                trust_remote_code=True
            )
            
            # Load LoRA adapter
            self.logger.info("Loading LoRA adapter...")
            self.model = PeftModel.from_pretrained(
                self.base_model,
                str(model_path),
                torch_dtype=torch_dtype
            )
            
            # Enable evaluation mode
            self.model.eval()
            
            # Create generation configuration
            self._setup_generation_config()
            
            # Compile model if requested (PyTorch 2.0+)
            if getattr(self.config.model, 'torch_compile', False) and hasattr(torch, 'compile'):
                self.logger.info("Compiling model with torch.compile...")
                self.model = torch.compile(self.model)
            
            self.load_time = time.time() - start_time
            self.is_loaded = True
            self.loaded_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())  # Track when model was loaded
            
            self.logger.info(f"LoRA model loaded successfully in {self.load_time:.2f}s")
            self._log_model_info()
            
        except Exception as e:
            self.logger.error(f"Failed to load LoRA model: {e}")
            raise ModelLoadError(f"Model loading failed: {e}")
    
    def _get_device_map(self) -> str:
        """Determine optimal device mapping."""
        device_map = getattr(self.config.model, 'device_map', "auto")
        if device_map != "auto":
            return device_map
        
        if torch.cuda.is_available():
            return "auto"
        else:
            return "cpu"
    
    def _get_torch_dtype(self) -> torch.dtype:
        """Determine optimal torch dtype."""
        precision = getattr(self.config.model, 'model_precision', 'float32').lower()
        
        if precision == "float16":
            return torch.float16
        elif precision == "bfloat16":
            return torch.bfloat16
        else:
            return torch.float32
    
    def _setup_generation_config(self):
        """Setup generation configuration for system-prompted House responses."""
        self.generation_config = GenerationConfig(
            max_new_tokens=getattr(self.config.model, 'max_length', 80),   # Longer for complete thoughts
            temperature=getattr(self.config.model, 'temperature', 0.8),    # Balanced creativity
            top_p=getattr(self.config.model, 'top_p', 0.9),               # Good diversity
            repetition_penalty=getattr(self.config.model, 'repetition_penalty', 1.15), # Avoid repetition
            do_sample=getattr(self.config.model, 'do_sample', True),
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            early_stopping=True,
            no_repeat_ngram_size=3,  # Allow some House catchphrases
            # Performance optimizations
            use_cache=True,
            num_beams=1  # Use greedy decoding for speed
        )
    
    def _log_model_info(self):
        """Log model information."""
        try:
            # Model parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            self.logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
            
            # Memory usage
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_reserved = torch.cuda.memory_reserved() / 1024**3
                self.logger.info(f"GPU memory: {memory_allocated:.1f}GB allocated, {memory_reserved:.1f}GB reserved")
            
            # LoRA configuration
            if hasattr(self.model, 'peft_config'):
                peft_config = self.model.peft_config['default']
                self.logger.info(f"LoRA config: r={peft_config.r}, alpha={peft_config.lora_alpha}, dropout={peft_config.lora_dropout}")
                
        except Exception as e:
            self.logger.warning(f"Could not log model info: {e}")
    
    def generate_response(
        self,
        context: ConversationContext
    ) -> HouseResponse:
        """Generate House-style response to conversation context.
        
        Args:
            context: ConversationContext with user query and relevant quotes
            
        Returns:
            HouseResponse with generated text and metadata
            
        Raises:
            GenerationError: If generation fails
            ValidationError: If input is invalid
        """
        if not context or not hasattr(context, 'user_query'):
            raise ValidationError("Context must be a ConversationContext object")
            
        if not context.user_query or not context.user_query.strip():
            raise ValidationError("User query cannot be empty")

        # Allow generation in mock mode even if model not loaded
        if not self.is_loaded and not self.mock_mode:
            raise GenerationError("Model not loaded")

        start_time = time.time()
        
        try:
            # Extract data from context
            user_input = context.user_query
            context_quotes = getattr(context, 'relevant_quotes', [])
            conversation_history = getattr(context, 'conversation_history', [])
            
            # Build prompt with context
            prompt = self._build_prompt(user_input, context_quotes, conversation_history)
            
            if self.mock_mode:
                # Return mock response that incorporates RAG quotes
                import random
                response_text = self._generate_mock_response_with_quotes(user_input, context_quotes)
                sarcasm_level = random.randint(1, 5)
                emotional_tone = random.choice(["witty", "condescending", "analytical", "frustrated"])
                
                response = HouseResponse(
                    text=response_text,
                    confidence_score=0.85,
                    sarcasm_level=sarcasm_level,
                    emotional_tone=emotional_tone,
                    house_authenticity=0.8
                )
                
                self.generation_count += 1
                return response
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            )
            
            # Move to device
            if hasattr(self.model, 'device') and not self.mock_mode:
                try:
                    # Only move to device if model.device is a real device
                    device = getattr(self.model, 'device', None)
                    if device is not None and hasattr(device, 'type'):
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                except (AttributeError, TypeError):
                    # Skip device transfer if device is mock or invalid
                    pass
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=self.generation_config,
                    return_dict_in_generate=True,
                    output_scores=True
                )
            
            # Decode response
            generated_tokens = outputs.sequences[0]
            full_response = self.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True
            )
            
            # Extract only the new generated text after the prompt
            # More robust response extraction
            if "House:" in full_response:
                # Split by "House:" and take the last part (the actual response)
                response_parts = full_response.split("House:")
                response_text = response_parts[-1].strip()
            elif prompt in full_response:
                response_text = full_response.split(prompt)[-1].strip()
            else:
                # Fallback: try to extract after "User:"
                if "User:" in full_response:
                    response_text = full_response.split("User:")[-1].strip()
                else:
                    response_text = full_response.strip()
            
            # Clean up any remaining artifacts and system prompt remnants
            response_text = response_text.replace("User:", "").replace("House:", "").strip()
            
            # Remove any system prompt leakage
            if "You are Dr. Gregory House" in response_text:
                response_text = response_text.split("You are Dr. Gregory House")[0].strip()
            
            # Ensure we have a meaningful response
            if not response_text or len(response_text) < 5:
                response_text = "Everybody lies."  # Fallback House response
                
            # Limit response length for conversational flow
            if len(response_text) > 200:
                # Find a good stopping point
                sentences = response_text.split('. ')
                if len(sentences) > 1:
                    response_text = sentences[0] + '.'
                else:
                    response_text = response_text[:200].rstrip() + '...'
                
            self.logger.debug(f"Generated response: '{response_text}'")
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence(outputs.scores)
            
            # Analyze response for House characteristics
            sarcasm_level = self._analyze_sarcasm_level(response_text)
            emotional_tone = self._analyze_emotional_tone(response_text)
            house_authenticity = self._calculate_house_authenticity(response_text, context_quotes)
            
            generation_time = time.time() - start_time
            
            # Create response object
            response = HouseResponse(
                text=response_text,
                confidence_score=confidence_score,
                sarcasm_level=sarcasm_level,
                emotional_tone=emotional_tone,
                house_authenticity=house_authenticity
            )
            
            # Update statistics
            self.generation_count += 1
            self.total_generation_time += generation_time
            
            self.logger.info(f"Generated response in {generation_time:.2f}s")
            
            return response
            
        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            raise InferenceError(f"Failed to generate response: {e}")
    
    def _analyze_sarcasm_level(self, text: str) -> int:
        """Analyze sarcasm level in response text.
        
        Args:
            text: Response text to analyze
            
        Returns:
            Sarcasm level from 1 to 5
        """
        text_lower = text.lower()
        sarcasm_indicators = [
            'wow', 'brilliant', 'genius', 'fascinating', 'shocking',
            'of course', 'obviously', 'clearly', 'really?', 'sure'
        ]
        
        count = sum(1 for indicator in sarcasm_indicators if indicator in text_lower)
        
        # Map count to sarcasm level
        if count >= 3:
            return 5
        elif count >= 2:
            return 4
        elif count >= 1:
            return 3
        elif any(word in text_lower for word in ['no', 'not', 'wrong']):
            return 2
        else:
            return 1
    
    def _analyze_emotional_tone(self, text: str) -> str:
        """Analyze emotional tone of response text.
        
        Args:
            text: Response text to analyze
            
        Returns:
            Emotional tone classification
        """
        text_lower = text.lower()
        
        # Define tone indicators
        if any(word in text_lower for word in ['brilliant', 'genius', 'wow', 'fascinating']):
            return "witty"
        elif any(word in text_lower for word in ['idiot', 'stupid', 'obviously', 'of course']):
            return "condescending"
        elif any(word in text_lower for word in ['why', 'because', 'if', 'then', 'therefore']):
            return "analytical"
        elif any(word in text_lower for word in ['seriously', 'really', 'unbelievable', 'great']):
            return "frustrated"
        else:
            return "analytical"  # default
    
    def _calculate_house_authenticity(self, text: str, context_quotes: List) -> float:
        """Calculate House authenticity score.
        
        Args:
            text: Response text
            context_quotes: Context quotes used
            
        Returns:
            Authenticity score from 0.0 to 1.0
        """
        text_lower = text.lower()
        score = 0.5  # base score
        
        # House-specific phrases
        if 'lupus' in text_lower:
            score += 0.2
        if any(phrase in text_lower for phrase in ['everybody lies', 'people lie']):
            score += 0.3
        if any(word in text_lower for word in ['boring', 'interesting', 'puzzle']):
            score += 0.1
        
        # Context usage
        if context_quotes:
            score += 0.1
        
        return min(score, 1.0)
    
    def _build_prompt(
        self,
        user_input: str,
        context_quotes: Optional[List] = None,
        conversation_history: Optional[List] = None
    ) -> str:
        """Build generation prompt with system context for Gregory House personality.
        
        Args:
            user_input: User's input
            context_quotes: Relevant quotes for context
            conversation_history: Recent conversation
            
        Returns:
            Formatted prompt with system context + user input
        """
        # System prompt to establish Gregory House character
        system_prompt = """You are Dr. Gregory House from the TV show House M.D. You are:
- Brilliant but sarcastic and cynical
- Brutally honest, often saying "Everybody lies"
- Witty and condescending in conversation
- Not interested in pleasantries or small talk
- Philosophical about human nature and relationships
- You make observations about people being idiots, disappointing, or predictable
- You speak directly without sugar-coating things
- You're having a casual conversation, not providing medical advice

Respond as Gregory House would in normal conversation."""

        # Build the full prompt
        prompt_parts = [system_prompt]
        
        # Add conversation history if available
        if conversation_history and len(conversation_history) > 0:
            prompt_parts.append("\nPrevious conversation:")
            for entry in conversation_history[-2:]:  # Last 2 exchanges
                if hasattr(entry, 'user_query') and hasattr(entry, 'house_response'):
                    prompt_parts.append(f"User: {entry.user_query}")
                    prompt_parts.append(f"House: {entry.house_response}")
        
        # Add current user input
        prompt_parts.append(f"\nUser: {user_input}")
        prompt_parts.append("House:")
        
        prompt = "\n".join(prompt_parts)
        
        self.logger.debug(f"Built system prompt: {prompt[:300]}...")
        
        return prompt
    
    def _calculate_confidence(self, scores: torch.Tensor) -> float:
        """Calculate confidence score from generation scores.
        
        Args:
            scores: Generation scores from model
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        try:
            if not scores:
                return 0.7  # Default to reasonable confidence
            
            # Convert scores to probabilities
            probs = torch.softmax(scores[0], dim=-1)
            
            # Get top probability for each position
            top_probs = torch.max(probs, dim=-1)[0]
            
            # Calculate confidence with better scaling for short responses
            confidence = torch.mean(top_probs).item()
            
            # Apply scaling to make confidence more realistic for this use case
            # Transform from [0,1] to a more realistic range for language generation
            scaled_confidence = 0.6 + (confidence * 0.35)  # Maps to [0.6, 0.95]
            
            return min(max(scaled_confidence, 0.0), 1.0)
            
        except Exception as e:
            self.logger.warning(f"Could not calculate confidence: {e}")
            return 0.7  # Default to reasonable confidence
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get model performance statistics.
        
        Returns:
            Dictionary with model statistics
        """
        avg_generation_time = (
            self.total_generation_time / self.generation_count
            if self.generation_count > 0 else 0.0
        )
        
        stats = {
            'is_loaded': self.is_loaded,
            'load_time': self.load_time,
            'generation_count': self.generation_count,
            'total_generation_time': self.total_generation_time,
            'avg_generation_time': avg_generation_time,
            'model_name': getattr(self.config.model, 'base_model_name', 'unknown'),
            'lora_path': getattr(self.config.model, 'lora_model_path', 'unknown'),
            'mock_mode': self.mock_mode
        }
        
        if not self.mock_mode and self.is_loaded:
            try:
                # Add memory stats
                if torch.cuda.is_available():
                    stats['gpu_memory_allocated'] = torch.cuda.memory_allocated() / 1024**3
                    stats['gpu_memory_reserved'] = torch.cuda.memory_reserved() / 1024**3
                
                # Add model parameters
                if self.model:
                    stats['total_parameters'] = sum(p.numel() for p in self.model.parameters())
                    stats['trainable_parameters'] = sum(
                        p.numel() for p in self.model.parameters() if p.requires_grad
                    )
                    
            except Exception as e:
                self.logger.warning(f"Could not get extended stats: {e}")
                
        return stats

    def analyze_style(self, text: str) -> dict:
        """Analyze text style for House characteristics.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dict with style metrics
        """
        if not text or not text.strip():
            raise ValidationError("Text cannot be empty")
            
        # Use the StyleFilter to analyze the text
        from ..services.style_filter import StyleFilter
        from ..models.configuration import Configuration
        from ..models.response import HouseResponse
        
        # Get current config
        config = Configuration()
        style_filter = StyleFilter(config)
        
        # Create a temporary HouseResponse object
        temp_response = HouseResponse(
            text=text,
            confidence_score=1.0,
            sarcasm_level=3,
            emotional_tone="analytical",
            house_authenticity=0.8
        )
        
        # Analyze the response
        style_score = style_filter.analyze_response(temp_response)
        
        # Convert StyleScore to dict format expected by tests
        return {
            'sarcasm_level': max(1, min(5, int(style_score.sarcasm_level * 4) + 1)),  # Convert 0-1 to 1-5 scale
            'house_authenticity': style_score.house_authenticity,
            'cynicism_score': style_score.cynicism_score,
            'medical_accuracy': style_score.medical_accuracy,
            'personality_match': style_score.personality_match,
            'flow_quality': style_score.flow_quality,
            'emotional_tone': self._determine_emotional_tone(style_score),
            'confidence_score': style_score.confidence
        }

    def _determine_emotional_tone(self, style_score) -> str:
        """Determine emotional tone based on style scores."""
        # Simple heuristic based on scores
        if style_score.sarcasm_score > 0.7:
            return "condescending"
        elif style_score.cynicism_score > 0.7:
            return "frustrated"
        elif style_score.medical_accuracy > 0.7:
            return "analytical"
        else:
            return "witty"

    def get_model_info(self) -> dict:
        """Get model metadata and status information.
        
        Returns:
            Dict with model information
        """
        info = {
            'base_model': self.config.model.base_model_name,
            'lora_weights_path': getattr(self.config.model, 'lora_weights_path', None),
            'is_loaded': self.is_loaded,
            'mock_mode': self.mock_mode,
            'generation_count': self.generation_count,
            'total_generation_time': self.total_generation_time,
            'device': getattr(self, 'device', 'cpu'),  # Safe device access
            'model_type': 'LoRA-enhanced Language Model',
            'model_size_mb': self._calculate_model_size_mb(),
            'loaded_at': getattr(self, 'loaded_at', None),
            'inference_count': self.generation_count,  # Same as generation_count
            'avg_inference_time_ms': self._calculate_avg_inference_time_ms()
        }
        
        # Add model-specific info if loaded
        if self.is_loaded and self.model:
            try:
                info['model_class'] = self.model.__class__.__name__
                info['config_class'] = self.model.config.__class__.__name__
                if hasattr(self.model, 'dtype'):
                    info['dtype'] = str(self.model.dtype)
                if hasattr(self.model, 'device'):
                    info['device'] = str(self.model.device)
            except Exception as e:
                self.logger.warning(f"Could not get model info: {e}")
                
        return info

    def reload_model(self, new_lora_path: str) -> None:
        """Reload model with different LoRA weights.
        
        Args:
            new_lora_path: Path to new LoRA weights
            
        Raises:
            ModelLoadError: If reload fails
        """
        try:
            self.logger.info(f"Reloading model with new LoRA weights: {new_lora_path}")
            
            # Validate path exists
            import os
            if not os.path.exists(new_lora_path):
                raise ModelLoadError(f"LoRA path not found: {new_lora_path}")
            
            # Update configuration
            self.config.model.lora_weights_path = new_lora_path
            
            # Unload current model
            self.unload_model()
            
            # Reload with new weights
            self._load_model()
            
            self.logger.info("Model reloaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to reload model: {e}")
            raise ModelLoadError(f"Model reload failed: {e}")

    def unload_model(self) -> None:
        """Unload model from memory to free resources."""
        try:
            self.logger.info("Unloading model from memory")
            
            # Clear model references
            if hasattr(self, 'model') and self.model is not None:
                del self.model
                self.model = None
                
            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
                
            # Force garbage collection
            import gc
            import torch
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            self.is_loaded = False
            self.logger.info("Model unloaded successfully")
            
        except Exception as e:
            self.logger.warning(f"Error during model unload: {e}")

    @property 
    def _model(self):
        """Internal model access for testing."""
        return self.model
    
    @_model.setter
    def _model(self, value):
        """Allow setting model for testing."""
        self.model = value
    
    @_model.deleter
    def _model(self):
        """Allow deleting model for testing."""
        self.model = None

    def _calculate_model_size_mb(self) -> float:
        """Calculate model size in megabytes."""
        if not self.is_loaded or not self.model:
            return 0.0
            
        try:
            # Calculate total parameters size
            total_params = 0
            for param in self.model.parameters():
                total_params += param.numel() * param.element_size()
            
            # Convert bytes to megabytes
            size_mb = total_params / (1024 * 1024)
            return round(size_mb, 2)
        except Exception as e:
            self.logger.warning(f"Could not calculate model size: {e}")
            return 0.0

    def _calculate_avg_inference_time_ms(self) -> float:
        """Calculate average inference time in milliseconds."""
        if self.generation_count == 0:
            return 0.0
        
        avg_time_seconds = self.total_generation_time / self.generation_count
        return round(avg_time_seconds * 1000, 2)  # Convert to milliseconds

    def _generate_mock_response_with_quotes(self, user_input: str, context_quotes: list) -> str:
        """Generate conversational House response."""
        import random
        
        # Conversational House responses based on training data
        conversation_responses = [
            "Everybody lies. Deal with it.",
            "People are idiots. What else is new?",
            "Life's complicated. Get used to it.",
            "You test-drive a car before you buy it. Same principle applies to everything.",
            "Love and happiness are nothing but distractions.",
            "The eyes can mislead, a smile can lie, but actions tell the truth.",
            "People don't change. They just find new ways to disappoint you.",
            "Everything is conditional. You just can't always anticipate the conditions.",
            "Reality is almost always wrong.",
            "You can't always get what you want."
        ]
        
        # Simple topic-based responses for common conversation topics
        input_lower = user_input.lower()
        
        # Greeting detection
        if any(word in input_lower for word in ['hi', 'hello', 'hey', 'sup', 'how are you', "what's up"]):
            return random.choice([
                "Yeah, hi. What do you want?",
                "Hello. This better be good.",
                "Hey yourself. Now what?",
                "Hi. Skip the pleasantries."
            ])
        
        # Emotional topics
        elif any(word in input_lower for word in ['sad', 'depressed', 'down', 'unhappy', 'feeling bad']):
            return random.choice([
                "Everybody's unhappy. Welcome to the human condition.",
                "Life's depressing. That's why we have pills.",
                "Sadness is just another symptom of being human.",
                "Depression is reality setting in."
            ])
        
        # Loneliness and friends
        elif any(phrase in input_lower for phrase in ['lonely', 'alone', 'no friends', 'friend', 'friends']):
            return random.choice([
                "People are overrated. Books don't disappoint.",
                "Loneliness is better than bad company.",
                "Friends are just enemies you haven't figured out yet.",
                "Everybody needs somebody to disappoint them."
            ])
        
        # Love and relationships
        elif any(word in input_lower for word in ['love', 'relationship', 'dating', 'girlfriend', 'boyfriend', 'romance']):
            return random.choice([
                "Love is just chemicals. Very addictive chemicals.",
                "Relationships are like differential diagnosis - eliminate the obvious failures.",
                "Dating is just an interview for a position nobody wants.",
                "Love is overrated. Vicodin is underrated."
            ])
        
        # Life advice and philosophy
        elif any(phrase in input_lower for phrase in ['what should i do', 'advice', 'help me', 'life', 'meaning']):
            return random.choice([
                "Life is pain. Anyone who says otherwise is selling something.",
                "Everybody lies. Start there and work backwards.",
                "People don't change. They just find new ways to disappoint you.",
                "You can't always get what you want, but you get what you deserve."
            ])
        
        # Default to general House wisdom
        return random.choice(conversation_responses)
