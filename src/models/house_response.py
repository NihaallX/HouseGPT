"""House response data model with processing stages.

This module provides the HouseResponse model for AI-generated responses
with tracking of processing stages and transformations.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class ProcessingStage(Enum):
    """Enumeration of response processing stages."""
    INITIAL_GENERATION = "initial_generation"
    RAG_ENHANCED = "rag_enhanced" 
    STYLE_FILTERED = "style_filtered"
    FINAL_OUTPUT = "final_output"


class ResponseQuality(Enum):
    """Enumeration of response quality levels."""
    EXCELLENT = "excellent"  # High confidence, authentic, relevant
    GOOD = "good"           # Good quality with minor issues
    ACCEPTABLE = "acceptable"  # Meets minimum requirements
    POOR = "poor"           # Below quality threshold
    REJECTED = "rejected"   # Failed quality checks


@dataclass
class ProcessingMetadata:
    """Metadata for a processing stage."""
    
    stage: ProcessingStage
    timestamp: datetime
    processing_time_ms: float
    transformations_applied: List[str]
    quality_scores: Dict[str, float]
    notes: Optional[str] = None


@dataclass
class HouseResponse:
    """Data model for House-style AI responses with processing tracking.
    
    Tracks the complete lifecycle of response generation from initial
    AI output through style filtering to final delivery.
    """
    
    text: str
    """The response text (current version)."""
    
    confidence_score: float
    """AI model confidence in the response (0.0-1.0)."""
    
    sarcasm_level: int
    """Sarcasm intensity from 0 (none) to 5 (maximum)."""
    
    emotional_tone: str
    """Emotional tone: sarcastic, condescending, analytical, witty, cynical."""
    
    house_authenticity: float
    """How House-like the response is (0.0-1.0)."""
    
    response_id: Optional[str] = None
    """Unique identifier for tracking."""
    
    session_id: Optional[str] = None
    """Session this response belongs to."""
    
    user_query: Optional[str] = None
    """Original user query that triggered this response."""
    
    rag_quotes_used: Optional[List[Dict[str, Any]]] = None
    """RAG quotes that influenced this response."""
    
    processing_history: Optional[List[ProcessingMetadata]] = None
    """History of processing stages and transformations."""
    
    quality_assessment: Optional[ResponseQuality] = None
    """Overall quality assessment."""
    
    generation_timestamp: Optional[datetime] = None
    """When the response was first generated."""
    
    final_timestamp: Optional[datetime] = None
    """When the response was finalized."""
    
    model_version: Optional[str] = None
    """Version of the AI model used."""
    
    tokens_used: Optional[int] = None
    """Number of tokens consumed in generation."""
    
    original_text: Optional[str] = None
    """Original text before any transformations."""
    
    def __post_init__(self):
        """Initialize response with validation and defaults."""
        if self.generation_timestamp is None:
            self.generation_timestamp = datetime.now()
        
        if self.processing_history is None:
            self.processing_history = []
        
        if self.rag_quotes_used is None:
            self.rag_quotes_used = []
        
        if self.original_text is None:
            self.original_text = self.text
        
        self._validate_response()
        self._assess_quality()
    
    def _validate_response(self):
        """Validate response fields."""
        # Text validation
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("Response text must be a non-empty string")
        
        if len(self.text) > 1000:
            raise ValueError("Response text exceeds maximum length of 1000 characters")
        
        # Confidence score validation
        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError("Confidence score must be between 0.0 and 1.0")
        
        # Sarcasm level validation
        if not 0 <= self.sarcasm_level <= 5:
            raise ValueError("Sarcasm level must be between 0 and 5")
        
        # House authenticity validation
        if not 0.0 <= self.house_authenticity <= 1.0:
            raise ValueError("House authenticity must be between 0.0 and 1.0")
        
        # Emotional tone validation
        valid_tones = [
            "sarcastic", "condescending", "analytical", "witty", "cynical", 
            "matter-of-fact", "frustrated", "clinical", "philosophical", 
            "ironic", "cheerful", "reassuring", "sympathetic", "self-deprecating",
            "enthusiastic", "romantic", "childlike"
        ]
        if self.emotional_tone not in valid_tones:
            raise ValueError(f"Invalid emotional tone. Must be one of: {valid_tones}")
    
    def _assess_quality(self):
        """Assess overall response quality."""
        if self.quality_assessment is not None:
            return  # Already assessed
        
        # Quality scoring based on multiple factors
        quality_score = 0.0
        
        # Confidence score weight (25%)
        quality_score += self.confidence_score * 0.25
        
        # House authenticity weight (35%)
        quality_score += self.house_authenticity * 0.35
        
        # Sarcasm appropriateness weight (20%)
        sarcasm_score = min(self.sarcasm_level / 5.0, 1.0)  # Normalize to 0-1
        if sarcasm_score >= 0.4:  # Prefer some sarcasm
            quality_score += sarcasm_score * 0.20
        else:
            quality_score += (sarcasm_score * 0.5) * 0.20  # Penalty for low sarcasm
        
        # Text quality weight (20%)
        text_score = self._assess_text_quality()
        quality_score += text_score * 0.20
        
        # Determine quality level
        if quality_score >= 0.85:
            self.quality_assessment = ResponseQuality.EXCELLENT
        elif quality_score >= 0.70:
            self.quality_assessment = ResponseQuality.GOOD
        elif quality_score >= 0.55:
            self.quality_assessment = ResponseQuality.ACCEPTABLE
        elif quality_score >= 0.40:
            self.quality_assessment = ResponseQuality.POOR
        else:
            self.quality_assessment = ResponseQuality.REJECTED
    
    def _assess_text_quality(self) -> float:
        """Assess the quality of the response text.
        
        Returns:
            Text quality score (0.0-1.0)
        """
        text_lower = self.text.lower()
        score = 0.5  # Base score
        
        # Positive indicators
        house_indicators = ["patient", "diagnosis", "symptom", "lie", "lies", "differential"]
        house_score = sum(1 for indicator in house_indicators if indicator in text_lower)
        score += min(house_score * 0.1, 0.3)  # Up to 0.3 bonus
        
        # Negative indicators
        negative_indicators = ["wonderful", "amazing", "fantastic", "love", "perfect"]
        negative_score = sum(1 for indicator in negative_indicators if indicator in text_lower)
        score -= min(negative_score * 0.15, 0.4)  # Up to 0.4 penalty
        
        # Length appropriateness
        if 50 <= len(self.text) <= 300:  # Good length range
            score += 0.1
        elif len(self.text) < 20:  # Too short
            score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    @property
    def is_high_quality(self) -> bool:
        """Check if response meets high quality standards."""
        return self.quality_assessment in [ResponseQuality.EXCELLENT, ResponseQuality.GOOD]
    
    @property
    def is_acceptable(self) -> bool:
        """Check if response meets minimum acceptable standards."""
        return self.quality_assessment != ResponseQuality.REJECTED
    
    @property
    def was_transformed(self) -> bool:
        """Check if response was transformed from original."""
        return self.text != self.original_text
    
    @property
    def processing_time_total(self) -> float:
        """Get total processing time across all stages."""
        if not self.processing_history:
            return 0.0
        return sum(stage.processing_time_ms for stage in self.processing_history)
    
    @property
    def rag_quote_count(self) -> int:
        """Get number of RAG quotes used."""
        return len(self.rag_quotes_used) if self.rag_quotes_used else 0
    
    def add_processing_stage(self, stage: ProcessingStage, 
                           processing_time_ms: float,
                           transformations: List[str],
                           quality_scores: Dict[str, float],
                           notes: Optional[str] = None):
        """Add a processing stage to the history.
        
        Args:
            stage: Processing stage enum
            processing_time_ms: Time taken for this stage
            transformations: List of transformations applied
            quality_scores: Quality metrics for this stage
            notes: Optional notes about the processing
        """
        metadata = ProcessingMetadata(
            stage=stage,
            timestamp=datetime.now(),
            processing_time_ms=processing_time_ms,
            transformations_applied=transformations,
            quality_scores=quality_scores,
            notes=notes
        )
        
        if self.processing_history is None:
            self.processing_history = []
        
        self.processing_history.append(metadata)
        
        # Update final timestamp
        self.final_timestamp = datetime.now()
    
    def update_text(self, new_text: str, transformation_note: str):
        """Update response text and track the change.
        
        Args:
            new_text: New response text
            transformation_note: Description of the transformation
        """
        old_text = self.text
        self.text = new_text
        self._validate_response()  # Re-validate
        self._assess_quality()     # Re-assess quality
        
        # Log the transformation
        if hasattr(self, '_transformations'):
            self._transformations.append({
                'from': old_text,
                'to': new_text,
                'note': transformation_note,
                'timestamp': datetime.now()
            })
        else:
            self._transformations = [{
                'from': old_text,
                'to': new_text,
                'note': transformation_note,
                'timestamp': datetime.now()
            }]
    
    def add_rag_quote(self, quote_data: Dict[str, Any]):
        """Add a RAG quote that influenced this response.
        
        Args:
            quote_data: Quote metadata and content
        """
        if self.rag_quotes_used is None:
            self.rag_quotes_used = []
        
        self.rag_quotes_used.append(quote_data)
    
    def get_stage_by_type(self, stage_type: ProcessingStage) -> Optional[ProcessingMetadata]:
        """Get processing metadata for a specific stage.
        
        Args:
            stage_type: Stage to find
            
        Returns:
            Processing metadata or None
        """
        if not self.processing_history:
            return None
        
        for stage in self.processing_history:
            if stage.stage == stage_type:
                return stage
        
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary for serialization.
        
        Returns:
            Dictionary representation of the response
        """
        def serialize_datetime(dt):
            return dt.isoformat() if dt else None
        
        def serialize_enum(enum_val):
            return enum_val.value if enum_val else None
        
        return {
            "response_id": self.response_id,
            "session_id": self.session_id,
            "text": self.text,
            "original_text": self.original_text,
            "confidence_score": self.confidence_score,
            "sarcasm_level": self.sarcasm_level,
            "emotional_tone": self.emotional_tone,
            "house_authenticity": self.house_authenticity,
            "user_query": self.user_query,
            "rag_quotes_used": self.rag_quotes_used,
            "quality_assessment": serialize_enum(self.quality_assessment),
            "generation_timestamp": serialize_datetime(self.generation_timestamp),
            "final_timestamp": serialize_datetime(self.final_timestamp),
            "model_version": self.model_version,
            "tokens_used": self.tokens_used,
            "processing_time_total": self.processing_time_total,
            "rag_quote_count": self.rag_quote_count,
            "was_transformed": self.was_transformed,
            "is_high_quality": self.is_high_quality,
            "is_acceptable": self.is_acceptable,
            "processing_history": [
                {
                    "stage": serialize_enum(stage.stage),
                    "timestamp": serialize_datetime(stage.timestamp),
                    "processing_time_ms": stage.processing_time_ms,
                    "transformations_applied": stage.transformations_applied,
                    "quality_scores": stage.quality_scores,
                    "notes": stage.notes
                }
                for stage in (self.processing_history or [])
            ]
        }
    
    @classmethod
    def create_initial(cls, text: str, confidence_score: float,
                      user_query: Optional[str] = None,
                      session_id: Optional[str] = None) -> 'HouseResponse':
        """Create initial response before processing.
        
        Args:
            text: Initial response text
            confidence_score: AI confidence score
            user_query: Original user query
            session_id: Session identifier
            
        Returns:
            HouseResponse instance
        """
        return cls(
            text=text,
            confidence_score=confidence_score,
            sarcasm_level=2,  # Default low sarcasm for initial generation
            emotional_tone="analytical",  # Default neutral tone
            house_authenticity=0.5,  # Default moderate authenticity
            user_query=user_query,
            session_id=session_id
        )
    
    @classmethod
    def create_house_style(cls, text: str, confidence_score: float,
                          sarcasm_level: int = 4,
                          emotional_tone: str = "sarcastic",
                          house_authenticity: float = 0.9) -> 'HouseResponse':
        """Create response with House-style characteristics.
        
        Args:
            text: Response text
            confidence_score: AI confidence
            sarcasm_level: Sarcasm level (default high)
            emotional_tone: Emotional tone (default sarcastic)
            house_authenticity: Authenticity score (default high)
            
        Returns:
            HouseResponse instance
        """
        return cls(
            text=text,
            confidence_score=confidence_score,
            sarcasm_level=sarcasm_level,
            emotional_tone=emotional_tone,
            house_authenticity=house_authenticity
        )
