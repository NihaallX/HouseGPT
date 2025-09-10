"""RAG store service for House quote retrieval.

This module provides the RAGStore service for semantic similarity search
of House M.D. quotes using local vector storage with FAISS.
"""

# Import everything from the local implementation
from src.services.rag_store_local import (
    RAGStore,
    RAGStoreError,
    VectorStoreError,
    ValidationError,
    DuplicateQuoteError
)

# Make the classes available at module level
__all__ = [
    'RAGStore',
    'RAGStoreError', 
    'VectorStoreError',
    'ValidationError',
    'DuplicateQuoteError'
]
