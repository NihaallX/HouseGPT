"""Style score data model for House MD personality analysis.

This module provides the StyleScore model for style analysis results.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class StyleScore:
    """Data model for style analysis scoring results.
    
    Captures various metrics for House MD personality authenticity.
    """
    
    overall_score: float  # 0.0 to 1.0
    sarcasm_score: float  # 0.0 to 1.0 (replaces sarcasm_level)
    cynicism_score: float  # 0.0 to 1.0
    medical_accuracy: float  # 0.0 to 1.0
    personality_match: float  # 0.0 to 1.0
    conversational_flow: float  # 0.0 to 1.0 (replaces flow_quality)
    passes_filter: bool
    reasoning: str
    suggestions: List[str]
    analysis_time: float
    confidence: float
    
    # Properties for backward compatibility with the expected API
    @property
    def sarcasm_level(self) -> float:
        """Backward compatibility property."""
        return self.sarcasm_score
    
    @property
    def house_authenticity(self) -> float:
        """Backward compatibility property."""
        return self.overall_score
    
    @property
    def flow_quality(self) -> float:
        """Backward compatibility property."""
        return self.conversational_flow
    
    def is_house_like(self, threshold: float = 0.7) -> bool:
        """Check if response meets House authenticity threshold."""
        return self.overall_score >= threshold
