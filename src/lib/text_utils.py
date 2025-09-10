"""Text utilities for HouseGPT preprocessing and validation.

Provides text cleaning, validation, and preprocessing utilities for
user input and model responses.
"""

import logging
import re
import string
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
from enum import Enum


class TextType(Enum):
    """Types of text content."""
    USER_INPUT = "user_input"
    MODEL_RESPONSE = "model_response"
    HOUSE_QUOTE = "house_quote"
    SYSTEM_MESSAGE = "system_message"


@dataclass
class TextMetrics:
    """Metrics for analyzed text."""
    char_count: int
    word_count: int
    sentence_count: int
    avg_word_length: float
    readability_score: float
    contains_profanity: bool
    language_detected: str


class TextCleaner:
    """Text cleaning and normalization utilities."""
    
    def __init__(self):
        """Initialize text cleaner."""
        self.logger = logging.getLogger(__name__)
        
        # Common contractions for expansion
        self.contractions = {
            "don't": "do not",
            "won't": "will not", 
            "can't": "cannot",
            "n't": " not",
            "'re": " are",
            "'ve": " have",
            "'ll": " will",
            "'d": " would",
            "'m": " am",
            "i'm": "i am",
            "you're": "you are",
            "it's": "it is",
            "that's": "that is",
            "what's": "what is",
            "here's": "here is",
            "there's": "there is"
        }
        
        # Profanity filter (basic - can be expanded)
        self.profanity_words = {
            "gandu"
        }
        
        # House MD specific terms that should be preserved
        self.house_terms = {
            "vicodin", "lupus", "sarcoidosis", "differential", 
            "foreman", "cuddy", "wilson", "clinic", "diagnostics"
        }
    
    def clean_user_input(self, text: str) -> str:
        """Clean and normalize user input text.
        
        Args:
            text: Raw user input
            
        Returns:
            Cleaned and normalized text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\?\!\,\;\:\-\'\"]', '', text)
        
        # Normalize case (but preserve House terms)
        words = text.split()
        normalized_words = []
        
        for word in words:
            lower_word = word.lower()
            if lower_word in self.house_terms:
                normalized_words.append(lower_word)
            else:
                normalized_words.append(word)
        
        text = ' '.join(normalized_words)
        
        # Expand contractions
        for contraction, expansion in self.contractions.items():
            text = re.sub(r'\b' + re.escape(contraction) + r'\b', expansion, text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def clean_model_response(self, text: str) -> str:
        """Clean model response text for output.
        
        Args:
            text: Raw model response
            
        Returns:
            Cleaned response text
        """
        if not text:
            return ""
        
        # Remove model artifacts and special tokens
        text = re.sub(r'<[^>]+>', '', text)  # Remove XML-like tags
        text = re.sub(r'\[.*?\]', '', text)  # Remove bracketed content
        text = re.sub(r'\{.*?\}', '', text)  # Remove braced content
        
        # Clean up spacing and punctuation
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'\.{2,}', '.', text)  # Multiple periods to single
        text = re.sub(r'\?{2,}', '?', text)  # Multiple question marks
        text = re.sub(r'!{2,}', '!', text)   # Multiple exclamations
        
        # Ensure proper sentence ending
        if text and not text[-1] in '.!?':
            text += '.'
        
        return text
    
    def extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        if not text:
            return []
        
        # Simple sentence splitting on periods, exclamations, questions
        sentences = re.split(r'[.!?]+', text)
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 3:  # Minimum sentence length
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def remove_profanity(self, text: str, replacement: str = "****") -> str:
        """Remove or replace profanity in text.
        
        Args:
            text: Input text
            replacement: Replacement string for profanity
            
        Returns:
            Text with profanity replaced
        """
        if not text:
            return ""
        
        words = text.split()
        cleaned_words = []
        
        for word in words:
            # Check if word (without punctuation) is profanity
            clean_word = word.lower().strip(string.punctuation)
            if clean_word in self.profanity_words:
                cleaned_words.append(replacement)
            else:
                cleaned_words.append(word)
        
        return ' '.join(cleaned_words)


class TextValidator:
    """Text validation utilities."""
    
    def __init__(self):
        """Initialize text validator."""
        self.logger = logging.getLogger(__name__)
        
        # Validation rules
        self.max_input_length = 1000
        self.min_input_length = 3
        self.max_response_length = 500
        self.min_response_length = 5
    
    def validate_user_input(self, text: str) -> Tuple[bool, List[str]]:
        """Validate user input text.
        
        Args:
            text: User input to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        if not text or not text.strip():
            errors.append("Input cannot be empty")
            return False, errors
        
        # Length checks
        if len(text) < self.min_input_length:
            errors.append(f"Input too short (minimum {self.min_input_length} characters)")
        
        if len(text) > self.max_input_length:
            errors.append(f"Input too long (maximum {self.max_input_length} characters)")
        
        # Content checks
        if text.strip() == text.strip().lower() and len(text) > 20:
            # Warn about all lowercase for longer inputs
            errors.append("Consider using proper capitalization")
        
        # Check for repeated characters (spam detection)
        if re.search(r'(.)\1{5,}', text):
            errors.append("Input contains excessive repeated characters")
        
        # Check for valid characters
        if not re.match(r'^[a-zA-Z0-9\s\.\?\!\,\;\:\-\'\"]+$', text):
            errors.append("Input contains invalid characters")
        
        return len(errors) == 0, errors
    
    def validate_model_response(self, text: str) -> Tuple[bool, List[str]]:
        """Validate model response text.
        
        Args:
            text: Model response to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        if not text or not text.strip():
            errors.append("Response cannot be empty")
            return False, errors
        
        # Length checks
        if len(text) < self.min_response_length:
            errors.append(f"Response too short (minimum {self.min_response_length} characters)")
        
        if len(text) > self.max_response_length:
            errors.append(f"Response too long (maximum {self.max_response_length} characters)")
        
        # Quality checks
        if text.count('.') == 0 and text.count('!') == 0 and text.count('?') == 0:
            errors.append("Response should end with proper punctuation")
        
        # Check for model artifacts
        if re.search(r'<[^>]+>|\[.*?\]|\{.*?\}', text):
            errors.append("Response contains formatting artifacts")
        
        return len(errors) == 0, errors
    
    def is_house_appropriate(self, text: str) -> Tuple[bool, str]:
        """Check if text is appropriate for House MD personality.
        
        Args:
            text: Text to check
            
        Returns:
            Tuple of (is_appropriate, reason)
        """
        if not text:
            return False, "Empty text"
        
        text_lower = text.lower()
        
        # Check for House-like characteristics
        house_indicators = [
            "everybody lies", "idiot", "moron", "people are", 
            "obviously", "brilliant", "differential", "interesting"
        ]
        
        # Anti-House indicators (too positive/nice)
        anti_house = [
            "wonderful", "amazing", "fantastic", "lovely", 
            "please", "thank you", "sorry", "apologize"
        ]
        
        has_house_style = any(indicator in text_lower for indicator in house_indicators)
        has_anti_house = any(indicator in text_lower for indicator in anti_house)
        
        if has_anti_house and not has_house_style:
            return False, "Too polite/positive for House character"
        
        if len(text) > 100 and not has_house_style:
            return False, "Lacks House personality markers"
        
        return True, "Appropriate for House character"


class TextAnalyzer:
    """Text analysis and metrics utilities."""
    
    def __init__(self):
        """Initialize text analyzer."""
        self.logger = logging.getLogger(__name__)
    
    def analyze_text(self, text: str) -> TextMetrics:
        """Analyze text and return metrics.
        
        Args:
            text: Text to analyze
            
        Returns:
            TextMetrics object with analysis results
        """
        if not text:
            return TextMetrics(0, 0, 0, 0.0, 0.0, False, "unknown")
        
        # Basic counts
        char_count = len(text)
        words = text.split()
        word_count = len(words)
        sentences = self._split_sentences(text)
        sentence_count = len(sentences)
        
        # Average word length
        avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0.0
        
        # Simple readability score (based on word and sentence length)
        readability_score = self._calculate_readability(words, sentences)
        
        # Profanity check
        contains_profanity = self._contains_profanity(text)
        
        # Language detection (simple heuristic)
        language_detected = self._detect_language(text)
        
        return TextMetrics(
            char_count=char_count,
            word_count=word_count,
            sentence_count=sentence_count,
            avg_word_length=avg_word_length,
            readability_score=readability_score,
            contains_profanity=contains_profanity,
            language_detected=language_detected
        )
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _calculate_readability(self, words: List[str], sentences: List[str]) -> float:
        """Calculate simple readability score (0-100, higher = easier)."""
        if not words or not sentences:
            return 0.0
        
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Simple formula: shorter sentences and words = higher readability
        score = 100 - (avg_sentence_length * 2) - (avg_word_length * 5)
        return max(0, min(100, score))  # Clamp to 0-100 range
    
    def _contains_profanity(self, text: str) -> bool:
        """Check if text contains profanity."""
        profanity_words = {
            "damn", "hell", "crap", "shit", "fuck", "bitch", 
            "asshole", "bastard", "idiot", "moron", "stupid"
        }
        
        words = re.findall(r'\b\w+\b', text.lower())
        return any(word in profanity_words for word in words)
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection (English vs other)."""
        # Very basic heuristic - count common English words
        common_english = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "is", "are", "was", "were", "be", "been", "have",
            "has", "had", "do", "does", "did", "will", "would", "could", "should"
        }
        
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return "unknown"
        
        english_count = sum(1 for word in words if word in common_english)
        english_ratio = english_count / len(words)
        
        return "english" if english_ratio > 0.3 else "other"


def preprocess_for_model(text: str, max_length: int = 512) -> str:
    """Preprocess text for model input.
    
    Args:
        text: Raw text input
        max_length: Maximum length for model
        
    Returns:
        Preprocessed text ready for model
    """
    cleaner = TextCleaner()
    
    # Clean the text
    cleaned = cleaner.clean_user_input(text)
    
    # Truncate if too long
    if len(cleaned) > max_length:
        # Try to truncate at sentence boundary
        sentences = cleaner.extract_sentences(cleaned)
        truncated = ""
        for sentence in sentences:
            if len(truncated + sentence) <= max_length:
                truncated += sentence + ". "
            else:
                break
        cleaned = truncated.strip()
    
    return cleaned


def postprocess_model_output(text: str) -> str:
    """Postprocess model output for display.
    
    Args:
        text: Raw model output
        
    Returns:
        Cleaned text ready for display
    """
    cleaner = TextCleaner()
    return cleaner.clean_model_response(text)


if __name__ == "__main__":
    # Test text utilities
    analyzer = TextAnalyzer()
    validator = TextValidator()
    
    test_input = "Hey House, what do you think about love?"
    metrics = analyzer.analyze_text(test_input)
    is_valid, errors = validator.validate_user_input(test_input)
    
    print(f"Text: {test_input}")
    print(f"Words: {metrics.word_count}, Readability: {metrics.readability_score:.1f}")
    print(f"Valid: {is_valid}, Errors: {errors}")
    
    test_response = "Love is just chemicals. Very addictive chemicals."
    is_appropriate, reason = validator.is_house_appropriate(test_response)
    print(f"House appropriate: {is_appropriate} - {reason}")
