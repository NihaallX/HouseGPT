"""Conversation context data model for House model input.

This module provides the ConversationContext model for passing context
to the House personality model.
"""

from typing import List, Dict, Any, Optional


class ConversationContext:
    """Data model for conversation context input to House model.
    
    This is a stub implementation that will be replaced during implementation phase.
    """
    
    def __init__(self,
                 user_query: str,
                 relevant_quotes: List = None,
                 conversation_history: List[Dict[str, str]] = None):
        """Initialize conversation context.
        
        Args:
            user_query: User's input query
            relevant_quotes: List of relevant House quotes from RAG
            conversation_history: Previous conversation turns
        """
        self.user_query = user_query
        self.relevant_quotes = relevant_quotes or []
        self.conversation_history = conversation_history or []
