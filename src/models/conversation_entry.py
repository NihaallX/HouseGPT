"""Conversation log data model with JSON serialization.

This module provides the ConversationEntry model for logging and
analyzing conversation interactions with comprehensive metadata.
"""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
from datetime import datetime
import json
import uuid


@dataclass
class ConversationEntry:
    """Data model for conversation log entries with comprehensive metadata.
    
    Stores complete conversation interactions including user input,
    AI response, processing metadata, and performance metrics.
    """
    
    timestamp: datetime
    """When the conversation occurred."""
    
    user_query: str
    """The user's input query."""
    
    house_response: str
    """The AI's response text."""
    
    entry_id: Optional[str] = None
    """Unique identifier for this conversation entry."""
    
    session_id: Optional[str] = None
    """Session this conversation belongs to."""
    
    user_id: Optional[str] = None
    """User identifier (if available)."""
    
    wake_word_detected: bool = False
    """Whether this conversation started with wake word detection."""
    
    input_mode: str = "text"
    """Input mode: text, voice, wake_word."""
    
    confidence_score: Optional[float] = None
    """Speech-to-text confidence for voice input."""
    
    rag_quotes_found: int = 0
    """Number of relevant RAG quotes found."""
    
    rag_quotes_used: Optional[List[Dict[str, Any]]] = None
    """Details of RAG quotes that influenced the response."""
    
    response_confidence: float = 0.0
    """AI model confidence in the response."""
    
    sarcasm_level: int = 3
    """Sarcasm level of the response (0-5)."""
    
    emotional_tone: str = "analytical"
    """Emotional tone of the response."""
    
    house_authenticity: float = 0.0
    """How House-like the response was (0.0-1.0)."""
    
    processing_time_ms: Optional[float] = None
    """Total processing time in milliseconds."""
    
    rag_search_time_ms: Optional[float] = None
    """Time spent on RAG search."""
    
    model_generation_time_ms: Optional[float] = None
    """Time spent on model generation."""
    
    style_filter_time_ms: Optional[float] = None
    """Time spent on style filtering."""
    
    voice_synthesis_time_ms: Optional[float] = None
    """Time spent on voice synthesis."""
    
    user_satisfaction: Optional[int] = None
    """User satisfaction rating (1-5) if provided."""
    
    follow_up_questions: int = 0
    """Number of follow-up questions in this session."""
    
    conversation_length: int = 1
    """Position in conversation sequence."""
    
    error_occurred: bool = False
    """Whether any errors occurred during processing."""
    
    error_details: Optional[Dict[str, Any]] = None
    """Details of any errors that occurred."""
    
    model_version: Optional[str] = None
    """Version of AI model used."""
    
    filter_violations: Optional[List[str]] = None
    """Style filter violations found."""
    
    transformations_applied: Optional[List[str]] = None
    """Transformations applied to improve response."""
    
    quality_score: Optional[float] = None
    """Overall response quality score (0.0-1.0)."""
    
    urgency_level: str = "low"
    """Assessed urgency level: low, medium, high."""
    
    medical_topic: Optional[str] = None
    """Medical topic category if applicable."""
    
    def __post_init__(self):
        """Initialize entry with defaults and validation."""
        if self.entry_id is None:
            self.entry_id = str(uuid.uuid4())
        
        if self.rag_quotes_used is None:
            self.rag_quotes_used = []
        
        if self.filter_violations is None:
            self.filter_violations = []
        
        if self.transformations_applied is None:
            self.transformations_applied = []
        
        if self.error_details is None and self.error_occurred:
            self.error_details = {}
        
        self._validate_entry()
        self._calculate_derived_metrics()
    
    def _validate_entry(self):
        """Validate conversation entry fields."""
        # Required fields validation
        if not self.user_query or not self.user_query.strip():
            raise ValueError("User query cannot be empty")
        
        if not self.house_response or not self.house_response.strip():
            raise ValueError("House response cannot be empty")
        
        # Numeric field validation
        if self.response_confidence < 0.0 or self.response_confidence > 1.0:
            raise ValueError("Response confidence must be between 0.0 and 1.0")
        
        if not 0 <= self.sarcasm_level <= 5:
            raise ValueError("Sarcasm level must be between 0 and 5")
        
        if self.house_authenticity < 0.0 or self.house_authenticity > 1.0:
            raise ValueError("House authenticity must be between 0.0 and 1.0")
        
        if self.user_satisfaction is not None and not 1 <= self.user_satisfaction <= 5:
            raise ValueError("User satisfaction must be between 1 and 5")
        
        # Input mode validation
        valid_modes = ["text", "voice", "wake_word"]
        if self.input_mode not in valid_modes:
            raise ValueError(f"Invalid input mode. Must be one of: {valid_modes}")
        
        # Urgency level validation
        valid_urgency = ["low", "medium", "high"]
        if self.urgency_level not in valid_urgency:
            raise ValueError(f"Invalid urgency level. Must be one of: {valid_urgency}")
    
    def _calculate_derived_metrics(self):
        """Calculate derived metrics from available data."""
        # Calculate total processing time if components are available
        if self.processing_time_ms is None:
            component_times = [
                self.rag_search_time_ms or 0,
                self.model_generation_time_ms or 0,
                self.style_filter_time_ms or 0,
                self.voice_synthesis_time_ms or 0
            ]
            if any(time > 0 for time in component_times):
                self.processing_time_ms = sum(component_times)
        
        # Calculate quality score if not provided
        if self.quality_score is None:
            self.quality_score = self._calculate_quality_score()
    
    def _calculate_quality_score(self) -> float:
        """Calculate overall quality score from available metrics.
        
        Returns:
            Quality score between 0.0 and 1.0
        """
        score = 0.0
        factors = 0
        
        # Response confidence (weight: 0.3)
        if self.response_confidence > 0:
            score += self.response_confidence * 0.3
            factors += 0.3
        
        # House authenticity (weight: 0.4)
        if self.house_authenticity > 0:
            score += self.house_authenticity * 0.4
            factors += 0.4
        
        # Sarcasm appropriateness (weight: 0.2)
        sarcasm_score = min(self.sarcasm_level / 5.0, 1.0)
        if sarcasm_score >= 0.4:  # Prefer some sarcasm
            score += sarcasm_score * 0.2
            factors += 0.2
        
        # Error penalty (weight: 0.1)
        if not self.error_occurred:
            score += 0.1
            factors += 0.1
        
        # Normalize score
        return score / factors if factors > 0 else 0.0
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Get processing duration in seconds."""
        if self.processing_time_ms is None:
            return None
        return self.processing_time_ms / 1000.0
    
    @property
    def was_voice_input(self) -> bool:
        """Check if input was voice-based."""
        return self.input_mode in ["voice", "wake_word"]
    
    @property
    def had_rag_context(self) -> bool:
        """Check if RAG context was available."""
        return self.rag_quotes_found > 0
    
    @property
    def was_high_quality(self) -> bool:
        """Check if response was high quality."""
        return (self.quality_score or 0) >= 0.7
    
    @property
    def was_fast_response(self) -> bool:
        """Check if response was generated quickly (< 3 seconds)."""
        return (self.processing_time_ms or 0) < 3000
    
    @property
    def required_transformations(self) -> bool:
        """Check if style transformations were needed."""
        return len(self.transformations_applied) > 0
    
    def add_rag_quote(self, quote: Dict[str, Any]):
        """Add a RAG quote that influenced this conversation.
        
        Args:
            quote: Quote data including text, relevance, etc.
        """
        self.rag_quotes_used.append(quote)
        self.rag_quotes_found = len(self.rag_quotes_used)
    
    def add_error(self, error_type: str, error_message: str, 
                  component: Optional[str] = None):
        """Add error information to the entry.
        
        Args:
            error_type: Type of error (e.g., "connection_error")
            error_message: Detailed error message
            component: Component where error occurred
        """
        self.error_occurred = True
        
        if self.error_details is None:
            self.error_details = {}
        
        error_info = {
            "type": error_type,
            "message": error_message,
            "timestamp": datetime.now().isoformat(),
            "component": component
        }
        
        if "errors" not in self.error_details:
            self.error_details["errors"] = []
        
        self.error_details["errors"].append(error_info)
    
    def add_transformation(self, transformation: str):
        """Add a style transformation that was applied.
        
        Args:
            transformation: Description of transformation
        """
        if transformation not in self.transformations_applied:
            self.transformations_applied.append(transformation)
    
    def set_performance_metrics(self, 
                               rag_time: Optional[float] = None,
                               model_time: Optional[float] = None,
                               filter_time: Optional[float] = None,
                               voice_time: Optional[float] = None):
        """Set performance timing metrics.
        
        Args:
            rag_time: RAG search time in milliseconds
            model_time: Model generation time in milliseconds
            filter_time: Style filter time in milliseconds
            voice_time: Voice synthesis time in milliseconds
        """
        if rag_time is not None:
            self.rag_search_time_ms = rag_time
        if model_time is not None:
            self.model_generation_time_ms = model_time
        if filter_time is not None:
            self.style_filter_time_ms = filter_time
        if voice_time is not None:
            self.voice_synthesis_time_ms = voice_time
        
        # Recalculate total time
        self._calculate_derived_metrics()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert conversation entry to dictionary for serialization.
        
        Returns:
            Dictionary representation with proper datetime handling
        """
        data = asdict(self)
        
        # Convert datetime to ISO string
        if isinstance(data['timestamp'], datetime):
            data['timestamp'] = data['timestamp'].isoformat()
        
        return data
    
    def to_json(self, indent: Optional[int] = None) -> str:
        """Convert conversation entry to JSON string.
        
        Args:
            indent: JSON indentation level
            
        Returns:
            JSON representation
        """
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationEntry':
        """Create ConversationEntry from dictionary.
        
        Args:
            data: Dictionary containing entry data
            
        Returns:
            ConversationEntry instance
        """
        # Handle datetime parsing
        if isinstance(data.get('timestamp'), str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        elif data.get('timestamp') is None:
            data['timestamp'] = datetime.now()
        
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ConversationEntry':
        """Create ConversationEntry from JSON string.
        
        Args:
            json_str: JSON string containing entry data
            
        Returns:
            ConversationEntry instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    @classmethod
    def create_simple(cls, user_query: str, house_response: str,
                     session_id: Optional[str] = None) -> 'ConversationEntry':
        """Create a simple conversation entry with minimal data.
        
        Args:
            user_query: User's input
            house_response: AI response
            session_id: Optional session identifier
            
        Returns:
            ConversationEntry instance
        """
        return cls(
            timestamp=datetime.now(),
            user_query=user_query,
            house_response=house_response,
            session_id=session_id
        )
