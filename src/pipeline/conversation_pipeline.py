"""End-to-end conversation pipeline integration for HouseGPT.

Orchestrates the complete conversation flow from user input to House response
with comprehensive error handling, fallback mechanisms, and performance monitoring.
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import traceback

from ..models.user_input import UserInput
from ..models.house_response import HouseResponse
from ..models.context import ConversationContext
from ..models.conversation_entry import ConversationEntry
from ..models.configuration import HouseGPTConfig
from ..services.house_model import HouseModel
from ..services.style_filter import StyleFilter
from ..services.voice_output import VoiceOutput
from ..services.wake_word import WakeWordDetector
from ..services.rag_store import RAGStore
from ..services.conversation_logger import ConversationLogger
from ..lib.text_utils import TextCleaner, TextValidator
from ..lib.audio_utils import AudioProcessor


class PipelineStage(Enum):
    """Pipeline processing stages."""
    INPUT_VALIDATION = "input_validation"
    TEXT_PREPROCESSING = "text_preprocessing"
    RAG_RETRIEVAL = "rag_retrieval"
    CONTEXT_BUILDING = "context_building"
    RESPONSE_GENERATION = "response_generation"
    STYLE_FILTERING = "style_filtering"
    VOICE_SYNTHESIS = "voice_synthesis"
    LOGGING = "logging"
    COMPLETE = "complete"
    ERROR = "error"


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"          # Non-critical, continue processing
    MEDIUM = "medium"    # Warning, use fallback
    HIGH = "high"        # Critical, abort processing
    FATAL = "fatal"      # System-level error


@dataclass
class PipelineError:
    """Pipeline error information."""
    stage: PipelineStage
    severity: ErrorSeverity
    message: str
    exception: Optional[Exception] = None
    timestamp: float = field(default_factory=time.time)
    details: Optional[Dict[str, Any]] = None


@dataclass
class PipelineMetrics:
    """Pipeline performance metrics."""
    conversation_id: str
    start_time: float
    end_time: Optional[float] = None
    total_duration_ms: float = 0.0
    
    # Stage timings
    input_validation_ms: float = 0.0
    text_preprocessing_ms: float = 0.0
    rag_retrieval_ms: float = 0.0
    context_building_ms: float = 0.0
    response_generation_ms: float = 0.0
    style_filtering_ms: float = 0.0
    voice_synthesis_ms: float = 0.0
    logging_ms: float = 0.0
    
    # Quality metrics
    input_word_count: int = 0
    response_word_count: int = 0
    style_score: float = 0.0
    rag_relevance_score: float = 0.0
    
    # Error tracking
    errors: List[PipelineError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    fallbacks_used: List[str] = field(default_factory=list)


@dataclass
class PipelineResult:
    """Result of pipeline processing."""
    success: bool
    conversation_id: str
    user_input: UserInput
    house_response: Optional[HouseResponse] = None
    voice_audio: Optional[bytes] = None
    metrics: Optional[PipelineMetrics] = None
    errors: List[PipelineError] = field(default_factory=list)
    fallback_used: bool = False
    fallback_reason: Optional[str] = None


class ConversationPipeline:
    """End-to-end conversation processing pipeline."""
    
    def __init__(self, config: HouseGPTConfig):
        """Initialize conversation pipeline.
        
        Args:
            config: HouseGPT configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Services (initialized externally)
        self.house_model: Optional[HouseModel] = None
        self.style_filter: Optional[StyleFilter] = None
        self.voice_output: Optional[VoiceOutput] = None
        self.wake_word_detector: Optional[WakeWordDetector] = None
        self.rag_store: Optional[RAGStore] = None
        self.conversation_logger: Optional[ConversationLogger] = None
        
        # Utilities
        self.text_cleaner = TextCleaner()
        self.text_validator = TextValidator()
        self.audio_processor: Optional[AudioProcessor] = None
        
        # Performance tracking
        self.total_conversations = 0
        self.successful_conversations = 0
        self.failed_conversations = 0
        self.average_response_time = 0.0
        
        # Error handling
        self.max_retries = 3
        self.timeout_seconds = 30.0
        self.fallback_responses = [
            "I'm having trouble processing that right now. Try again.",
            "Something's not working properly. Can you rephrase?",
            "Technical difficulties. Ask me something else.",
            "My brain isn't cooperating. What else can I help with?"
        ]
        
        # Context management
        self.active_contexts: Dict[str, ConversationContext] = {}
        self.max_context_age = 3600  # 1 hour
    
    def initialize_services(self, services: Dict[str, Any]) -> None:
        """Initialize pipeline services.
        
        Args:
            services: Dictionary of service instances
        """
        self.house_model = services.get('house_model')
        self.style_filter = services.get('style_filter')
        self.voice_output = services.get('voice_output')
        self.wake_word_detector = services.get('wake_word_detector')
        self.rag_store = services.get('rag_store')
        self.conversation_logger = services.get('conversation_logger')
        
        if self.config.audio:
            self.audio_processor = AudioProcessor(
                sample_rate=self.config.audio.sample_rate,
                vad_enabled=self.config.audio.vad_enabled
            )
        
        self.logger.info("Pipeline services initialized")
    
    async def process_conversation(
        self,
        user_input_text: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        audio_data: Optional[bytes] = None,
        context: Optional[ConversationContext] = None
    ) -> PipelineResult:
        """Process complete conversation from input to response.
        
        Args:
            user_input_text: User's input text
            session_id: Optional session identifier
            user_id: Optional user identifier  
            audio_data: Optional audio data for voice input
            context: Optional conversation context
            
        Returns:
            PipelineResult with response and metrics
        """
        conversation_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Initialize metrics
        metrics = PipelineMetrics(
            conversation_id=conversation_id,
            start_time=start_time
        )
        
        # Initialize result
        result = PipelineResult(
            success=False,
            conversation_id=conversation_id,
            user_input=UserInput(
                text=user_input_text,
                session_id=session_id,
                user_id=user_id,
                timestamp=start_time
            ),
            metrics=metrics
        )
        
        try:
            self.logger.info(f"🔄 Processing conversation {conversation_id[:8]}")
            
            # Stage 1: Input Validation
            stage_start = time.time()
            if not await self._validate_input(result):
                return result
            metrics.input_validation_ms = (time.time() - stage_start) * 1000
            
            # Stage 2: Text Preprocessing
            stage_start = time.time()
            if not await self._preprocess_text(result):
                return result
            metrics.text_preprocessing_ms = (time.time() - stage_start) * 1000
            
            # Stage 3: RAG Retrieval (if enabled)
            stage_start = time.time()
            rag_quotes = await self._retrieve_rag_context(result)
            metrics.rag_retrieval_ms = (time.time() - stage_start) * 1000
            
            # Stage 4: Context Building
            stage_start = time.time()
            conversation_context = await self._build_context(result, context, rag_quotes)
            metrics.context_building_ms = (time.time() - stage_start) * 1000
            
            # Stage 5: Response Generation
            stage_start = time.time()
            if not await self._generate_response(result, conversation_context):
                return await self._handle_generation_fallback(result)
            metrics.response_generation_ms = (time.time() - stage_start) * 1000
            
            # Stage 6: Style Filtering
            stage_start = time.time()
            if not await self._apply_style_filtering(result):
                return await self._handle_style_fallback(result)
            metrics.style_filtering_ms = (time.time() - stage_start) * 1000
            
            # Stage 7: Voice Synthesis (if enabled)
            stage_start = time.time()
            await self._synthesize_voice(result)
            metrics.voice_synthesis_ms = (time.time() - stage_start) * 1000
            
            # Stage 8: Logging
            stage_start = time.time()
            await self._log_conversation(result, conversation_context)
            metrics.logging_ms = (time.time() - stage_start) * 1000
            
            # Complete metrics
            metrics.end_time = time.time()
            metrics.total_duration_ms = (metrics.end_time - metrics.start_time) * 1000
            
            if result.house_response:
                metrics.response_word_count = len(result.house_response.text.split())
                metrics.style_score = result.house_response.style_score.overall_score
            
            result.success = True
            self.successful_conversations += 1
            
            self.logger.info(f"✅ Conversation completed in {metrics.total_duration_ms:.0f}ms")
            return result
            
        except Exception as e:
            self._handle_pipeline_error(result, PipelineStage.ERROR, ErrorSeverity.FATAL, str(e), e)
            self.failed_conversations += 1
            return result
        
        finally:
            self.total_conversations += 1
            self._update_performance_metrics(metrics)
    
    async def _validate_input(self, result: PipelineResult) -> bool:
        """Validate user input.
        
        Args:
            result: Pipeline result to update
            
        Returns:
            True if validation passed
        """
        try:
            user_input = result.user_input
            
            # Basic validation
            if not user_input.text or not user_input.text.strip():
                self._handle_pipeline_error(
                    result, PipelineStage.INPUT_VALIDATION, 
                    ErrorSeverity.HIGH, "Empty input text"
                )
                return False
            
            # Length validation
            if len(user_input.text) > self.config.security.max_input_length:
                self._handle_pipeline_error(
                    result, PipelineStage.INPUT_VALIDATION,
                    ErrorSeverity.HIGH, f"Input too long: {len(user_input.text)} chars"
                )
                return False
            
            # Content validation using text validator
            is_valid, errors = self.text_validator.validate_user_input(user_input.text)
            if not is_valid:
                warning_msg = f"Input validation warnings: {', '.join(errors)}"
                result.metrics.warnings.append(warning_msg)
                self.logger.warning(warning_msg)
            
            # Update metrics
            result.metrics.input_word_count = len(user_input.text.split())
            
            return True
            
        except Exception as e:
            self._handle_pipeline_error(
                result, PipelineStage.INPUT_VALIDATION,
                ErrorSeverity.HIGH, f"Input validation failed: {e}", e
            )
            return False
    
    async def _preprocess_text(self, result: PipelineResult) -> bool:
        """Preprocess user input text.
        
        Args:
            result: Pipeline result to update
            
        Returns:
            True if preprocessing succeeded
        """
        try:
            # Clean and normalize text
            cleaned_text = self.text_cleaner.clean_user_input(result.user_input.text)
            
            # Update user input with cleaned text
            result.user_input.text = cleaned_text
            result.user_input.original_text = result.user_input.text
            
            return True
            
        except Exception as e:
            self._handle_pipeline_error(
                result, PipelineStage.TEXT_PREPROCESSING,
                ErrorSeverity.MEDIUM, f"Text preprocessing failed: {e}", e
            )
            return False
    
    async def _retrieve_rag_context(self, result: PipelineResult) -> List[Dict[str, Any]]:
        """Retrieve relevant context using RAG.
        
        Args:
            result: Pipeline result to update
            
        Returns:
            List of relevant quotes/context
        """
        try:
            if not self.rag_store or not self.config.rag.enabled:
                return []
            
            # Search for relevant quotes
            quotes = await self.rag_store.search_quotes(
                result.user_input.text,
                max_results=self.config.rag.max_retrieved_quotes
            )
            
            if quotes:
                # Calculate average relevance score
                relevance_scores = [quote.get('similarity_score', 0.0) for quote in quotes]
                result.metrics.rag_relevance_score = sum(relevance_scores) / len(relevance_scores)
                
                self.logger.debug(f"Retrieved {len(quotes)} relevant quotes")
            
            return quotes
            
        except Exception as e:
            self._handle_pipeline_error(
                result, PipelineStage.RAG_RETRIEVAL,
                ErrorSeverity.LOW, f"RAG retrieval failed: {e}", e
            )
            return []
    
    async def _build_context(
        self,
        result: PipelineResult,
        existing_context: Optional[ConversationContext],
        rag_quotes: List[Dict[str, Any]]
    ) -> ConversationContext:
        """Build conversation context.
        
        Args:
            result: Pipeline result
            existing_context: Existing conversation context
            rag_quotes: Retrieved RAG quotes
            
        Returns:
            Updated conversation context
        """
        try:
            # Use existing context or create new one
            if existing_context:
                context = existing_context
            else:
                session_id = result.user_input.session_id or result.conversation_id
                context = ConversationContext(session_id=session_id)
            
            # Add user input to context
            context.add_user_input(result.user_input)
            
            # Add RAG quotes to context
            if rag_quotes:
                context.add_rag_context(rag_quotes)
            
            # Store context for session continuity
            if result.user_input.session_id:
                self.active_contexts[result.user_input.session_id] = context
            
            return context
            
        except Exception as e:
            # Create minimal context on error
            session_id = result.user_input.session_id or result.conversation_id
            context = ConversationContext(session_id=session_id)
            context.add_user_input(result.user_input)
            
            self._handle_pipeline_error(
                result, PipelineStage.CONTEXT_BUILDING,
                ErrorSeverity.LOW, f"Context building failed: {e}", e
            )
            
            return context
    
    async def _generate_response(self, result: PipelineResult, context: ConversationContext) -> bool:
        """Generate House response.
        
        Args:
            result: Pipeline result to update
            context: Conversation context
            
        Returns:
            True if generation succeeded
        """
        try:
            if not self.house_model:
                self._handle_pipeline_error(
                    result, PipelineStage.RESPONSE_GENERATION,
                    ErrorSeverity.FATAL, "House model not available"
                )
                return False
            
            # Generate response with timeout
            response_task = self.house_model.generate_response(
                result.user_input.text,
                context=context
            )
            
            response = await asyncio.wait_for(response_task, timeout=self.timeout_seconds)
            
            if not response or not response.text:
                self._handle_pipeline_error(
                    result, PipelineStage.RESPONSE_GENERATION,
                    ErrorSeverity.HIGH, "Empty response generated"
                )
                return False
            
            result.house_response = response
            return True
            
        except asyncio.TimeoutError:
            self._handle_pipeline_error(
                result, PipelineStage.RESPONSE_GENERATION,
                ErrorSeverity.HIGH, f"Response generation timeout ({self.timeout_seconds}s)"
            )
            return False
        
        except Exception as e:
            self._handle_pipeline_error(
                result, PipelineStage.RESPONSE_GENERATION,
                ErrorSeverity.HIGH, f"Response generation failed: {e}", e
            )
            return False
    
    async def _apply_style_filtering(self, result: PipelineResult) -> bool:
        """Apply style filtering to response.
        
        Args:
            result: Pipeline result to update
            
        Returns:
            True if filtering succeeded
        """
        try:
            if not self.style_filter or not result.house_response:
                return True
            
            # Apply style filtering
            filtered_response = self.style_filter.apply_style(result.house_response)
            
            if filtered_response:
                result.house_response = filtered_response
                return True
            else:
                self._handle_pipeline_error(
                    result, PipelineStage.STYLE_FILTERING,
                    ErrorSeverity.MEDIUM, "Style filtering failed"
                )
                return False
            
        except Exception as e:
            self._handle_pipeline_error(
                result, PipelineStage.STYLE_FILTERING,
                ErrorSeverity.MEDIUM, f"Style filtering error: {e}", e
            )
            return False
    
    async def _synthesize_voice(self, result: PipelineResult) -> None:
        """Synthesize voice audio for response.
        
        Args:
            result: Pipeline result to update
        """
        try:
            if not self.voice_output or not result.house_response:
                return
            
            # Synthesize voice
            audio_data = self.voice_output.synthesize(result.house_response.text)
            
            if audio_data:
                result.voice_audio = audio_data
            else:
                result.metrics.warnings.append("Voice synthesis failed")
                
        except Exception as e:
            self._handle_pipeline_error(
                result, PipelineStage.VOICE_SYNTHESIS,
                ErrorSeverity.LOW, f"Voice synthesis error: {e}", e
            )
    
    async def _log_conversation(self, result: PipelineResult, context: ConversationContext) -> None:
        """Log conversation entry.
        
        Args:
            result: Pipeline result
            context: Conversation context
        """
        try:
            if not self.conversation_logger or not result.house_response:
                return
            
            # Create conversation entry
            entry = ConversationEntry(
                conversation_id=result.conversation_id,
                user_input=result.user_input,
                house_response=result.house_response,
                context=context,
                timestamp=time.time()
            )
            
            # Log conversation
            self.conversation_logger.log_conversation_entry(entry)
            
        except Exception as e:
            self._handle_pipeline_error(
                result, PipelineStage.LOGGING,
                ErrorSeverity.LOW, f"Conversation logging error: {e}", e
            )
    
    async def _handle_generation_fallback(self, result: PipelineResult) -> PipelineResult:
        """Handle response generation fallback.
        
        Args:
            result: Pipeline result
            
        Returns:
            Updated pipeline result with fallback response
        """
        import random
        
        fallback_text = random.choice(self.fallback_responses)
        
        # Create fallback response
        from ..models.house_response import HouseResponse
        from ..models.style_score import StyleScore
        
        fallback_response = HouseResponse(
            text=fallback_text,
            confidence=0.5,
            processing_time=0.1,
            style_score=StyleScore()
        )
        
        result.house_response = fallback_response
        result.fallback_used = True
        result.fallback_reason = "Response generation failed"
        result.success = True
        
        result.metrics.fallbacks_used.append("response_generation")
        
        self.logger.warning(f"Using fallback response: {fallback_text}")
        return result
    
    async def _handle_style_fallback(self, result: PipelineResult) -> PipelineResult:
        """Handle style filtering fallback.
        
        Args:
            result: Pipeline result
            
        Returns:
            Updated pipeline result
        """
        # Use original response without style filtering
        result.fallback_used = True
        result.fallback_reason = "Style filtering failed"
        result.success = True
        
        result.metrics.fallbacks_used.append("style_filtering")
        
        self.logger.warning("Using response without style filtering")
        return result
    
    def _handle_pipeline_error(
        self,
        result: PipelineResult,
        stage: PipelineStage,
        severity: ErrorSeverity,
        message: str,
        exception: Optional[Exception] = None
    ) -> None:
        """Handle pipeline error.
        
        Args:
            result: Pipeline result to update
            stage: Pipeline stage where error occurred
            severity: Error severity
            message: Error message
            exception: Optional exception object
        """
        error = PipelineError(
            stage=stage,
            severity=severity,
            message=message,
            exception=exception,
            details={"conversation_id": result.conversation_id}
        )
        
        result.errors.append(error)
        result.metrics.errors.append(error)
        
        # Log based on severity
        if severity == ErrorSeverity.FATAL:
            self.logger.error(f"FATAL {stage.value}: {message}")
            if exception:
                self.logger.error(traceback.format_exc())
        elif severity == ErrorSeverity.HIGH:
            self.logger.error(f"ERROR {stage.value}: {message}")
        elif severity == ErrorSeverity.MEDIUM:
            self.logger.warning(f"WARNING {stage.value}: {message}")
        else:
            self.logger.debug(f"INFO {stage.value}: {message}")
    
    def _update_performance_metrics(self, metrics: PipelineMetrics) -> None:
        """Update overall performance metrics.
        
        Args:
            metrics: Conversation metrics
        """
        if metrics.total_duration_ms > 0:
            # Update running average response time
            if self.successful_conversations == 1:
                self.average_response_time = metrics.total_duration_ms
            else:
                total_time = self.average_response_time * (self.successful_conversations - 1)
                self.average_response_time = (total_time + metrics.total_duration_ms) / self.successful_conversations
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline performance statistics.
        
        Returns:
            Dictionary with pipeline statistics
        """
        success_rate = (self.successful_conversations / self.total_conversations * 100 
                       if self.total_conversations > 0 else 0)
        
        return {
            "total_conversations": self.total_conversations,
            "successful_conversations": self.successful_conversations,
            "failed_conversations": self.failed_conversations,
            "success_rate_percent": success_rate,
            "average_response_time_ms": self.average_response_time,
            "active_contexts": len(self.active_contexts),
            "services_initialized": {
                "house_model": self.house_model is not None,
                "style_filter": self.style_filter is not None,
                "voice_output": self.voice_output is not None,
                "rag_store": self.rag_store is not None,
                "conversation_logger": self.conversation_logger is not None
            }
        }
    
    def cleanup_old_contexts(self) -> int:
        """Clean up old conversation contexts.
        
        Returns:
            Number of contexts cleaned up
        """
        current_time = time.time()
        old_sessions = []
        
        for session_id, context in self.active_contexts.items():
            if current_time - context.created_at > self.max_context_age:
                old_sessions.append(session_id)
        
        for session_id in old_sessions:
            del self.active_contexts[session_id]
        
        if old_sessions:
            self.logger.info(f"Cleaned up {len(old_sessions)} old conversation contexts")
        
        return len(old_sessions)
