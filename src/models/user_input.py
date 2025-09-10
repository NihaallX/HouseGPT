"""User input data model with validation rules.

This module provides the UserInput model for processing and validating
user queries before they enter the conversation pipeline.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from datetime import datetime
import re


@dataclass
class UserInput:
    """Data model for user input with validation and preprocessing.
    
    Handles text input, audio input metadata, and conversation context
    for processing by the HouseGPT pipeline.
    """
    
    text: str
    """The user's input text (transcribed from audio if applicable)."""
    
    timestamp: datetime
    """When the input was received."""
    
    input_mode: str = "text"
    """Input mode: 'text', 'voice', 'wake_word'."""
    
    audio_metadata: Optional[Dict[str, Any]] = None
    """Metadata from audio input (duration, quality, etc.)."""
    
    session_id: Optional[str] = None
    """Conversation session identifier."""
    
    user_id: Optional[str] = None
    """User identifier (if available)."""
    
    confidence_score: Optional[float] = None
    """Speech-to-text confidence (for voice input)."""
    
    language: str = "en"
    """Detected or specified language."""
    
    context_hints: Optional[List[str]] = None
    """Context hints for better response generation."""
    
    def __post_init__(self):
        """Validate and preprocess user input."""
        self.text = self._preprocess_text(self.text)
        self._validate_input()
        
        if self.context_hints is None:
            self.context_hints = []
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess and clean user input text.
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned and normalized text
        """
        if not text:
            return ""
        
        # Strip whitespace
        text = text.strip()
        
        # Normalize whitespace (multiple spaces to single)
        text = re.sub(r'\s+', ' ', text)
        
        # Remove non-printable characters except common punctuation
        text = re.sub(r'[^\x20-\x7E]', '', text)
        
        # Basic profanity filtering (simple approach)
        # In production, this would use a more sophisticated filter
        profane_words = ['damn', 'hell', 'shit', 'fuck', 'bitch']
        for word in profane_words:
            text = re.sub(rf'\b{word}\b', '[REDACTED]', text, flags=re.IGNORECASE)
        
        return text
    
    def _validate_input(self):
        """Validate user input fields."""
        # Text validation
        if not isinstance(self.text, str):
            raise ValueError("Input text must be a string")
        
        if len(self.text) > 1000:
            raise ValueError("Input text exceeds maximum length of 1000 characters")
        
        # Input mode validation
        valid_modes = ["text", "voice", "wake_word"]
        if self.input_mode not in valid_modes:
            raise ValueError(f"Invalid input mode. Must be one of: {valid_modes}")
        
        # Confidence score validation (for voice input)
        if self.confidence_score is not None:
            if not 0.0 <= self.confidence_score <= 1.0:
                raise ValueError("Confidence score must be between 0.0 and 1.0")
        
        # Language validation
        valid_languages = ["en", "es", "fr", "de", "it"]  # Supported languages
        if self.language not in valid_languages:
            raise ValueError(f"Unsupported language. Must be one of: {valid_languages}")
    
    @property
    def is_empty(self) -> bool:
        """Check if the input is empty or whitespace only."""
        return not self.text or self.text.isspace()
    
    @property
    def is_voice_input(self) -> bool:
        """Check if this was voice input."""
        return self.input_mode in ["voice", "wake_word"]
    
    @property
    def is_high_confidence(self) -> bool:
        """Check if voice input has high confidence score."""
        if not self.is_voice_input or self.confidence_score is None:
            return True  # Text input is always high confidence
        return self.confidence_score >= 0.8
    
    @property
    def word_count(self) -> int:
        """Get the number of words in the input."""
        if self.is_empty:
            return 0
        return len(self.text.split())
    
    @property
    def is_question(self) -> bool:
        """Check if the input appears to be a question."""
        if self.is_empty:
            return False
        
        # Check for question words
        question_words = ["what", "why", "how", "when", "where", "who", "which", "can", "could", "should", "would", "will", "is", "are", "do", "does", "did"]
        first_word = self.text.split()[0].lower()
        
        # Check for question mark or question words
        return self.text.endswith('?') or first_word in question_words
    
    @property
    def is_medical_query(self) -> bool:
        """Check if the input appears to be medical-related."""
        medical_keywords = [
            "symptom", "symptoms", "pain", "hurt", "ache", "sick", "illness",
            "disease", "condition", "diagnosis", "treatment", "medicine", "medication",
            "doctor", "hospital", "clinic", "fever", "headache", "nausea", "fatigue",
            "infection", "injury", "surgery", "prescription", "therapy"
        ]
        
        text_lower = self.text.lower()
        return any(keyword in text_lower for keyword in medical_keywords)
    
    @property
    def urgency_level(self) -> str:
        """Determine the urgency level of the input.
        
        Returns:
            'high', 'medium', or 'low' based on content analysis
        """
        if self.is_empty:
            return 'low'
        
        text_lower = self.text.lower()
        
        # High urgency indicators
        high_urgency_words = [
            "emergency", "urgent", "help", "pain", "severe", "critical",
            "dying", "death", "can't breathe", "chest pain", "bleeding",
            "overdose", "poison", "suicide", "911"
        ]
        
        # Medium urgency indicators
        medium_urgency_words = [
            "worried", "concerned", "problem", "issue", "hurt", "sick",
            "fever", "infection", "injury", "medication", "symptoms"
        ]
        
        if any(word in text_lower for word in high_urgency_words):
            return 'high'
        elif any(word in text_lower for word in medium_urgency_words):
            return 'medium'
        else:
            return 'low'
    
    def extract_symptoms(self) -> List[str]:
        """Extract potential symptoms from the input text.
        
        Returns:
            List of identified symptom keywords
        """
        symptom_patterns = [
            r'\b(pain|ache|hurt|sore)\b',
            r'\b(fever|temperature|hot|cold)\b',
            r'\b(nausea|vomit|sick|queasy)\b',
            r'\b(headache|migraine)\b',
            r'\b(fatigue|tired|exhausted)\b',
            r'\b(dizzy|lightheaded)\b',
            r'\b(cough|sneeze|congestion)\b',
            r'\b(rash|itchy|swelling)\b',
            r'\b(bleeding|bruising)\b',
            r'\b(shortness of breath|breathe)\b'
        ]
        
        symptoms = []
        text_lower = self.text.lower()
        
        for pattern in symptom_patterns:
            matches = re.findall(pattern, text_lower)
            symptoms.extend(matches)
        
        return list(set(symptoms))  # Remove duplicates
    
    def add_context_hint(self, hint: str):
        """Add a context hint for better response generation.
        
        Args:
            hint: Context hint to add
        """
        if self.context_hints is None:
            self.context_hints = []
        
        if hint not in self.context_hints:
            self.context_hints.append(hint)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert UserInput to dictionary for serialization.
        
        Returns:
            Dictionary representation of the UserInput
        """
        return {
            "text": self.text,
            "timestamp": self.timestamp.isoformat(),
            "input_mode": self.input_mode,
            "audio_metadata": self.audio_metadata,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "confidence_score": self.confidence_score,
            "language": self.language,
            "context_hints": self.context_hints,
            "is_question": self.is_question,
            "is_medical_query": self.is_medical_query,
            "urgency_level": self.urgency_level,
            "word_count": self.word_count,
            "extracted_symptoms": self.extract_symptoms()
        }
    
    @classmethod
    def from_text(cls, text: str, session_id: Optional[str] = None) -> 'UserInput':
        """Create UserInput from plain text.
        
        Args:
            text: Input text
            session_id: Optional session identifier
            
        Returns:
            UserInput instance
        """
        return cls(
            text=text,
            timestamp=datetime.now(),
            input_mode="text",
            session_id=session_id
        )
    
    @classmethod
    def from_voice(cls, text: str, confidence: float, 
                   audio_metadata: Optional[Dict[str, Any]] = None,
                   session_id: Optional[str] = None) -> 'UserInput':
        """Create UserInput from voice transcription.
        
        Args:
            text: Transcribed text
            confidence: Speech-to-text confidence score
            audio_metadata: Audio processing metadata
            session_id: Optional session identifier
            
        Returns:
            UserInput instance
        """
        return cls(
            text=text,
            timestamp=datetime.now(),
            input_mode="voice",
            confidence_score=confidence,
            audio_metadata=audio_metadata,
            session_id=session_id
        )
    
    @classmethod
    def from_wake_word(cls, text: str, confidence: float,
                       session_id: Optional[str] = None) -> 'UserInput':
        """Create UserInput from wake word detection.
        
        Args:
            text: Text following wake word
            confidence: Detection confidence
            session_id: Optional session identifier
            
        Returns:
            UserInput instance
        """
        return cls(
            text=text,
            timestamp=datetime.now(),
            input_mode="wake_word",
            confidence_score=confidence,
            session_id=session_id
        )


# Validation utilities for common input patterns
class InputValidator:
    """Utility class for additional input validation."""
    
    @staticmethod
    def is_spam(text: str) -> bool:
        """Check if input appears to be spam.
        
        Args:
            text: Input text to check
            
        Returns:
            True if input appears to be spam
        """
        spam_indicators = [
            r'(.)\1{10,}',  # Repeated characters
            r'[A-Z]{20,}',  # Excessive caps
            r'(buy now|click here|free trial|amazing deal)',  # Spam phrases
            r'https?://[^\s]+',  # URLs (might be spam)
        ]
        
        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in spam_indicators)
    
    @staticmethod
    def is_coherent(text: str) -> bool:
        """Check if input appears to be coherent text.
        
        Args:
            text: Input text to check
            
        Returns:
            True if input appears coherent
        """
        if len(text.strip()) < 3:
            return False
        
        # Check for reasonable ratio of letters to other characters
        letters = sum(c.isalpha() for c in text)
        total = len(text.replace(' ', ''))
        
        if total == 0:
            return False
        
        letter_ratio = letters / total
        return letter_ratio >= 0.6  # At least 60% letters
    
    @staticmethod
    def requires_emergency_response(text: str) -> bool:
        """Check if input requires immediate emergency response.
        
        Args:
            text: Input text to check
            
        Returns:
            True if emergency keywords detected
        """
        emergency_keywords = [
            "911", "emergency", "dying", "can't breathe", "chest pain",
            "overdose", "poison", "suicide", "help me", "call ambulance"
        ]
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in emergency_keywords)
