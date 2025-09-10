"""Filter result data model for style validation.

This module provides the FilterResult model for style filter output.
"""

from typing import List, Dict, Any, Optional


class FilterResult:
    """Data model for style filter validation results.
    
    This is a stub implementation that will be replaced during implementation phase.
    """
    
    def __init__(self,
                 is_valid: bool,
                 violations: List[Dict[str, Any]],
                 score: float,
                 filtered_response,
                 applied_transformations: Optional[List[Dict[str, Any]]] = None):
        """Initialize filter result.
        
        Args:
            is_valid: Whether response passes all rules
            violations: List of rule violations
            score: Overall filter score
            filtered_response: Corrected response (if applicable)
            applied_transformations: List of transformations applied
        """
        self.is_valid = is_valid
        self.violations = violations
        self.score = score
        self.filtered_response = filtered_response
        self.applied_transformations = applied_transformations or []
