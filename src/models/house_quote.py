"""House quote data model with embedding vector support.

This module provides the HouseQuote model for storing and retrieving
House MD quotes with semantic search capabilities.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import numpy as np
import json


@dataclass
class HouseQuote:
    """Data model for House MD quotes with embedding vectors.
    
    Stores quote text, metadata, and vector embeddings for semantic search
    and retrieval-augmented generation.
    """
    
    text: str
    """The actual quote text from House MD."""
    
    episode_info: Optional[str] = None
    """Episode information (e.g., 'Season 1, Episode 5')."""
    
    sarcasm_level: int = 3
    """Sarcasm intensity from 0 (none) to 5 (maximum)."""
    
    emotional_tone: str = "analytical"
    """Emotional tone: analytical, sarcastic, condescending, witty, cynical."""
    
    similarity_score: Optional[float] = None
    """Similarity score when retrieved from search (0.0-1.0)."""
    
    context: Optional[str] = None
    """Additional context about when/why the quote was said."""
    
    character_target: Optional[str] = None
    """Who the quote was directed at (patient, Wilson, Cuddy, etc.)."""
    
    medical_topic: Optional[str] = None
    """Medical topic if applicable (diagnosis, treatment, etc.)."""
    
    embedding_vector: Optional[List[float]] = None
    """Semantic embedding vector for similarity search."""
    
    quote_id: Optional[int] = None
    """Unique identifier in the database."""
    
    def __post_init__(self):
        """Validate quote fields after initialization."""
        self._validate_quote()
    
    def _validate_quote(self):
        """Validate quote fields."""
        # Text validation
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("Quote text must be a non-empty string")
        
        if len(self.text) > 500:
            raise ValueError("Quote text exceeds maximum length of 500 characters")
        
        # Sarcasm level validation
        if not 0 <= self.sarcasm_level <= 5:
            raise ValueError("Sarcasm level must be between 0 and 5")
        
        # Emotional tone validation
        valid_tones = [
            "sarcastic", "condescending", "analytical", "witty", "cynical", 
            "matter-of-fact", "frustrated", "clinical", "philosophical", 
            "ironic", "cheerful", "reassuring", "sympathetic", "self-deprecating",
            "enthusiastic", "romantic", "childlike"
        ]
        if self.emotional_tone not in valid_tones:
            raise ValueError(f"Invalid emotional tone. Must be one of: {valid_tones}")
        
        # Similarity score validation
        if self.similarity_score is not None:
            if not 0.0 <= self.similarity_score <= 1.0:
                raise ValueError("Similarity score must be between 0.0 and 1.0")
        
        # Embedding vector validation
        if self.embedding_vector is not None:
            if not isinstance(self.embedding_vector, list):
                raise ValueError("Embedding vector must be a list of floats")
            
            if len(self.embedding_vector) not in [384, 512, 768, 1024]:
                raise ValueError("Embedding vector must be standard size (384, 512, 768, or 1024)")
            
            if not all(isinstance(x, (int, float)) for x in self.embedding_vector):
                raise ValueError("Embedding vector must contain only numeric values")
    
    @property
    def is_high_sarcasm(self) -> bool:
        """Check if quote has high sarcasm level (4-5)."""
        return self.sarcasm_level >= 4
    
    @property
    def is_medical_related(self) -> bool:
        """Check if quote is related to medical topics."""
        if self.medical_topic:
            return True
        
        medical_keywords = [
            "patient", "diagnosis", "symptom", "treatment", "medicine", "disease",
            "condition", "hospital", "clinic", "doctor", "surgery", "prescription",
            "infection", "therapy", "medical", "health", "illness", "pain"
        ]
        
        text_lower = self.text.lower()
        return any(keyword in text_lower for keyword in medical_keywords)
    
    @property
    def has_embedding(self) -> bool:
        """Check if quote has an embedding vector."""
        return self.embedding_vector is not None and len(self.embedding_vector) > 0
    
    @property
    def is_highly_relevant(self) -> bool:
        """Check if quote is highly relevant (similarity > 0.8)."""
        return self.similarity_score is not None and self.similarity_score > 0.8
    
    def get_embedding_array(self) -> Optional[np.ndarray]:
        """Get embedding as numpy array.
        
        Returns:
            Numpy array of embedding vector or None
        """
        if not self.has_embedding:
            return None
        return np.array(self.embedding_vector, dtype=np.float32)
    
    def set_embedding_from_array(self, embedding: np.ndarray):
        """Set embedding from numpy array.
        
        Args:
            embedding: Numpy array of embedding vector
        """
        if not isinstance(embedding, np.ndarray):
            raise ValueError("Embedding must be a numpy array")
        
        if embedding.dtype not in [np.float32, np.float64]:
            embedding = embedding.astype(np.float32)
        
        self.embedding_vector = embedding.tolist()
        self._validate_quote()  # Re-validate with new embedding
    
    def calculate_similarity(self, other_embedding: List[float]) -> float:
        """Calculate cosine similarity with another embedding.
        
        Args:
            other_embedding: Another embedding vector
            
        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        if not self.has_embedding:
            raise ValueError("Quote has no embedding vector")
        
        if len(other_embedding) != len(self.embedding_vector):
            raise ValueError("Embedding vectors must have same dimensions")
        
        # Convert to numpy arrays
        vec1 = np.array(self.embedding_vector, dtype=np.float32)
        vec2 = np.array(other_embedding, dtype=np.float32)
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        
        if norm_product == 0:
            return 0.0
        
        similarity = dot_product / norm_product
        return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert quote to dictionary for serialization.
        
        Returns:
            Dictionary representation of the quote
        """
        return {
            "quote_id": self.quote_id,
            "text": self.text,
            "episode_info": self.episode_info,
            "sarcasm_level": self.sarcasm_level,
            "emotional_tone": self.emotional_tone,
            "similarity_score": self.similarity_score,
            "context": self.context,
            "character_target": self.character_target,
            "medical_topic": self.medical_topic,
            "embedding_vector": self.embedding_vector,
            "is_high_sarcasm": self.is_high_sarcasm,
            "is_medical_related": self.is_medical_related,
            "has_embedding": self.has_embedding
        }
    
    def to_json(self) -> str:
        """Convert quote to JSON string.
        
        Returns:
            JSON representation of the quote
        """
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HouseQuote':
        """Create HouseQuote from dictionary.
        
        Args:
            data: Dictionary containing quote data
            
        Returns:
            HouseQuote instance
        """
        # Extract only the fields that exist in the dataclass
        valid_fields = {
            "text", "episode_info", "sarcasm_level", "emotional_tone",
            "similarity_score", "context", "character_target", "medical_topic",
            "embedding_vector", "quote_id"
        }
        
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'HouseQuote':
        """Create HouseQuote from JSON string.
        
        Args:
            json_str: JSON string containing quote data
            
        Returns:
            HouseQuote instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    @classmethod
    def create_simple(cls, text: str, sarcasm_level: int = 3,
                      emotional_tone: str = "analytical") -> 'HouseQuote':
        """Create a simple quote with minimal metadata.
        
        Args:
            text: Quote text
            sarcasm_level: Sarcasm level (0-5)
            emotional_tone: Emotional tone
            
        Returns:
            HouseQuote instance
        """
        return cls(
            text=text,
            sarcasm_level=sarcasm_level,
            emotional_tone=emotional_tone
        )
