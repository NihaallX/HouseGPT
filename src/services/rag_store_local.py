"""RAG store service using local vector storage for House quote retrieval.

This module provides the RAGStore service for semantic similarity search
of House M.D. quotes using vector embeddings with local file storage.
"""

import json
import logging
import uuid
import pickle
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import numpy as np

try:
    import faiss
    import sentence_transformers
    VECTOR_AVAILABLE = True
except ImportError:
    VECTOR_AVAILABLE = False

from src.models.house_quote import HouseQuote


class RAGStoreError(Exception):
    """Base exception for RAG store operations."""
    pass


class VectorStoreError(RAGStoreError):
    """Raised when vector store operations fail."""
    pass


class ValidationError(RAGStoreError):
    """Raised when input validation fails."""
    pass


class DuplicateQuoteError(RAGStoreError):
    """Raised when attempting to add duplicate quote."""
    pass


class RAGStore:
    """Local vector database for House quote storage and semantic search.
    
    Uses FAISS for efficient vector similarity search with local file persistence.
    """
    
    def __init__(self, config):
        """Initialize RAG store with local vector storage.
        
        Args:
            config: Configuration object with storage settings
            
        Raises:
            VectorStoreError: If vector storage initialization fails
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Storage configuration
        self.data_dir = Path(getattr(config, 'rag_data_dir', 'data/rag'))
        self.vector_dimensions = getattr(config, 'rag_vector_dimensions', 384)
        self.embedding_model_name = getattr(config, 'rag_embedding_model', 'all-MiniLM-L6-v2')
        
        # File paths
        self.quotes_file = self.data_dir / 'quotes.json'
        self.index_file = self.data_dir / 'faiss_index.bin'
        self.metadata_file = self.data_dir / 'metadata.json'
        
        # Initialize storage
        self._init_storage()
        self._init_embedding_model()
        self._load_or_create_index()
    
    def _init_storage(self):
        """Initialize local storage directory."""
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Initialized local RAG storage at {self.data_dir}")
            
        except Exception as e:
            raise VectorStoreError(f"Failed to initialize storage: {e}")
    
    def _init_embedding_model(self):
        """Initialize sentence transformer model for embeddings."""
        if not VECTOR_AVAILABLE:
            self.logger.warning("FAISS/sentence-transformers not available - using mock mode")
            self.embedding_model = None
            self.mock_mode = True
            return
        
        try:
            self.embedding_model = sentence_transformers.SentenceTransformer(self.embedding_model_name)
            self.mock_mode = False
            self.logger.info(f"Loaded embedding model: {self.embedding_model_name}")
            
        except Exception as e:
            self.logger.warning(f"Using mock embeddings: {e}")
            self.embedding_model = None
            self.mock_mode = True
    
    def _load_or_create_index(self):
        """Load existing FAISS index or create new one."""
        if self.mock_mode:
            self.quotes_db = {}
            self.quote_metadata = {}
            return
        
        try:
            # Load existing index and data
            if self.index_file.exists() and self.quotes_file.exists():
                self.index = faiss.read_index(str(self.index_file))
                
                with open(self.quotes_file, 'r', encoding='utf-8') as f:
                    self.quotes_db = json.load(f)
                
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    self.quote_metadata = json.load(f)
                
                self.logger.info(f"Loaded existing index with {len(self.quotes_db)} quotes")
            else:
                # Create new index
                self.index = faiss.IndexFlatIP(self.vector_dimensions)  # Inner product for cosine similarity
                self.quotes_db = {}
                self.quote_metadata = {}
                
                self.logger.info("Created new FAISS index")
                
        except Exception as e:
            raise VectorStoreError(f"Failed to initialize vector index: {e}")
    
    def add_quote(self, quote) -> str:
        """Add House quote to vector database with embedding generation.
        
        Args:
            quote: HouseQuote object to store
            
        Returns:
            Unique ID of stored quote
        """
        if not isinstance(quote, HouseQuote):
            raise ValidationError("Quote must be a HouseQuote object")
        
        # Generate embedding if not present
        if quote.embedding_vector is None:
            quote.embedding_vector = self.get_query_embedding(quote.text)
        
        quote_id = str(uuid.uuid4())
        
        if self.mock_mode:
            self.quotes_db[quote_id] = quote
            return quote_id
        
        try:
            # Check for duplicates (skip for now since we don't have speaker)
            # duplicate_id = self._find_duplicate(quote.text, "House")
            # if duplicate_id:
            #     raise DuplicateQuoteError(f"Quote already exists: {quote.text[:50]}...")
            
            # Add to FAISS index
            vector = np.array([quote.embedding_vector], dtype=np.float32)
            # Normalize for cosine similarity
            faiss.normalize_L2(vector)
            self.index.add(vector)
            
            # Store quote data
            self.quotes_db[quote_id] = {
                'text': quote.text,
                'speaker': 'House',  # Default speaker since model doesn't have this field
                'episode_info': quote.episode_info,
                'context': quote.context,
                'emotional_tone': quote.emotional_tone,
                'sarcasm_level': quote.sarcasm_level,
                'medical_topic': quote.medical_topic,
                'character_target': quote.character_target,
                'created_at': datetime.now().isoformat()
            }
            
            # Store metadata
            self.quote_metadata[quote_id] = {
                'vector_index': self.index.ntotal - 1,  # Last added index
                'embedding_model': self.embedding_model_name
            }
            
            # Persist to disk
            self._save_data()
            
            self.logger.info(f"Added quote: {quote_id}")
            return quote_id
            
        except Exception as e:
            if isinstance(e, DuplicateQuoteError):
                raise
            raise VectorStoreError(f"Failed to add quote: {e}")
    
    def search_quotes(self, query: str, top_k: int = 5) -> List:
        """Search for most relevant House quotes using semantic similarity.
        
        Args:
            query: Search query text
            top_k: Maximum number of results to return
            
        Returns:
            List of similar HouseQuote objects
        """
        if not query or not query.strip():
            raise ValidationError("Query cannot be empty")
        
        if top_k <= 0 or top_k > 100:
            raise ValidationError("top_k must be between 1 and 100")
        
        if self.mock_mode:
            # Return mock results for testing
            return [
                HouseQuote(
                    text="Everybody lies.",
                    speaker="House",
                    episode="Pilot",
                    season=1,
                    emotional_tone="cynical",
                    sarcasm_level=4,
                    similarity_score=0.9
                ),
                HouseQuote(
                    text="It's not lupus.",
                    speaker="House", 
                    episode="The Right Stuff",
                    season=1,
                    emotional_tone="matter-of-fact",
                    sarcasm_level=2,
                    similarity_score=0.7
                )
            ][:top_k]
        
        try:
            # Generate query embedding
            query_embedding = self.get_query_embedding(query)
            query_vector = np.array([query_embedding], dtype=np.float32)
            faiss.normalize_L2(query_vector)
            
            # Search FAISS index
            similarities, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))
            
            # Convert results to HouseQuote objects
            results = []
            quote_ids = list(self.quotes_db.keys())
            
            for similarity, idx in zip(similarities[0], indices[0]):
                if idx == -1:  # No more results
                    break
                
                # Find quote by vector index
                quote_id = None
                for qid, metadata in self.quote_metadata.items():
                    if metadata['vector_index'] == idx:
                        quote_id = qid
                        break
                
                if quote_id and quote_id in self.quotes_db:
                    quote_data = self.quotes_db[quote_id]
                    quote = HouseQuote(
                        text=quote_data['text'],
                        episode_info=quote_data.get('episode_info'),
                        context=quote_data.get('context'),
                        emotional_tone=quote_data.get('emotional_tone'),
                        sarcasm_level=quote_data.get('sarcasm_level', 3),
                        medical_topic=quote_data.get('medical_topic'),
                        character_target=quote_data.get('character_target'),
                        similarity_score=float(similarity)
                    )
                    quote.quote_id = quote_id
                    results.append(quote)
            
            return results
            
        except Exception as e:
            raise VectorStoreError(f"Search error: {e}")
    
    def search_quotes_with_context(self, query: str, conversation_history: List[str], top_k: int = 5) -> List:
        """Search quotes considering conversation context."""
        # Combine query with recent conversation context
        context_window = conversation_history[-3:] if conversation_history else []
        combined_query = f"{' '.join(context_window)} {query}".strip()
        
        return self.search_quotes(combined_query, top_k)
    
    def get_quote_by_id(self, quote_id: str):
        """Retrieve House quote by UUID.
        
        Args:
            quote_id: Unique quote identifier
            
        Returns:
            HouseQuote object if found, None otherwise
        """
        if not quote_id or not isinstance(quote_id, str):
            raise ValidationError("Quote ID must be a non-empty string")
        
        if self.mock_mode:
            return self.quotes_db.get(quote_id)
        
        if quote_id not in self.quotes_db:
            return None
        
        quote_data = self.quotes_db[quote_id]
        quote = HouseQuote(
            text=quote_data['text'],
            episode_info=quote_data.get('episode_info'),
            context=quote_data.get('context'),
            emotional_tone=quote_data.get('emotional_tone'),
            sarcasm_level=quote_data.get('sarcasm_level', 3),
            medical_topic=quote_data.get('medical_topic'),
            character_target=quote_data.get('character_target')
        )
        quote.quote_id = quote_id
        
        return quote
    
    def update_quote_metadata(self, quote_id: str, metadata: Dict[str, Any]) -> bool:
        """Update quote metadata without changing embedding.
        
        Args:
            quote_id: Quote to update
            metadata: Dictionary of fields to update
            
        Returns:
            True if quote was updated, False if not found
        """
        if not quote_id:
            raise ValidationError("Quote ID required")
        
        if quote_id not in self.quotes_db:
            return False
        
        # Update allowed fields
        allowed_fields = ['episode_info', 'context', 'emotional_tone', 'sarcasm_level', 'medical_topic', 'character_target']
        quote_data = self.quotes_db[quote_id]
        
        for field, value in metadata.items():
            if field in allowed_fields:
                quote_data[field] = value
        
        quote_data['updated_at'] = datetime.now().isoformat()
        
        if not self.mock_mode:
            self._save_data()
        
        return True
    
    def delete_quote(self, quote_id: str) -> bool:
        """Remove quote from database.
        
        Args:
            quote_id: Quote to delete
            
        Returns:
            True if quote was deleted, False if not found
        """
        if not quote_id:
            raise ValidationError("Quote ID required")
        
        if quote_id not in self.quotes_db:
            return False
        
        # Remove from quotes database
        del self.quotes_db[quote_id]
        
        if quote_id in self.quote_metadata:
            del self.quote_metadata[quote_id]
        
        if not self.mock_mode:
            # Note: FAISS doesn't support efficient deletion, so we keep the vector
            # but remove the metadata. For production, periodic index rebuilding would be needed.
            self._save_data()
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics and health metrics.
        
        Returns:
            Dictionary with database statistics
        """
        total_quotes = len(self.quotes_db)
        speakers = set()
        episodes = set()
        sarcasm_levels = []
        
        for quote_data in self.quotes_db.values():
            if isinstance(quote_data, dict):
                speakers.add(quote_data.get('speaker', 'House'))
                if quote_data.get('episode_info'):
                    episodes.add(quote_data['episode_info'])
                sarcasm_levels.append(quote_data.get('sarcasm_level', 3))
        
        avg_sarcasm = sum(sarcasm_levels) / len(sarcasm_levels) if sarcasm_levels else 0
        
        return {
            'total_quotes': total_quotes,
            'unique_speakers': len(speakers),
            'episodes': len(episodes),
            'avg_sarcasm_level': avg_sarcasm,
            'storage_type': 'local_faiss' if not self.mock_mode else 'mock',
            'embedding_model': self.embedding_model_name
        }
    
    def get_query_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for text query.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        if self.embedding_model is None:
            # Return mock embedding for testing
            return [0.1] * self.vector_dimensions
        
        try:
            embedding = self.embedding_model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
            
        except Exception as e:
            raise VectorStoreError(f"Embedding generation error: {e}")
    
    def _find_duplicate(self, text: str, speaker: str) -> Optional[str]:
        """Find duplicate quote by text and speaker."""
        for quote_id, quote_data in self.quotes_db.items():
            if (isinstance(quote_data, dict) and 
                quote_data.get('text') == text and 
                quote_data.get('speaker') == speaker):
                return quote_id
        return None
    
    def _save_data(self):
        """Persist data to disk."""
        try:
            # Save quotes database
            with open(self.quotes_file, 'w', encoding='utf-8') as f:
                json.dump(self.quotes_db, f, indent=2, default=str)
            
            # Save metadata
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.quote_metadata, f, indent=2, default=str)
            
            # Save FAISS index
            faiss.write_index(self.index, str(self.index_file))
            
        except Exception as e:
            self.logger.error(f"Failed to save data: {e}")
