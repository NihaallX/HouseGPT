"""House response data model with style metadata.

This module provides the HouseResponse model for House personality model output.
"""

from typing import Optional


class HouseResponse:
    """Data model for House personality model responses.
    
    This is a stub implementation that will be replaced during implementation phase.
    """
    
    def __init__(self,
                 text: str,
                 confidence_score: float,
                 sarcasm_level: int,
                 emotional_tone: str,
                 house_authenticity: Optional[float] = None):
        """Initialize House response.
        
        Args:
            text: Generated response text
            confidence_score: Model confidence (0.0-1.0)
            sarcasm_level: Sarcasm level (1-5)
            emotional_tone: Emotional tone classification
            house_authenticity: House authenticity score (0.0-1.0)
        """
        self.text = text
        self.confidence_score = confidence_score
        self.sarcasm_level = sarcasm_level
        self.emotional_tone = emotional_tone
        self.house_authenticity = house_authenticity
