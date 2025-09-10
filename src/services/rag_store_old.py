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


class DatabaseConnectionError(RAGStoreError):
    """Raised when PostgreSQL database connection fails."""
    pass


class VectorExtensionError(RAGStoreError):
    """Raised when pgvector extension is not available."""
    pass


class ValidationError(RAGStoreError):
    """Raised when input validation fails."""
    pass


class DuplicateQuoteError(RAGStoreError):
    """Raised when attempting to add duplicate quote."""
    pass


class RAGStore:
    """Vector database for House quote storage and semantic search.
    
    Uses PostgreSQL with pgvector extension for efficient vector similarity search
    of House M.D. quotes with sentence transformer embeddings.
    """
    
    def __init__(self, config):
        """Initialize RAG store with PostgreSQL and pgvector.
        
        Args:
            config: Configuration object with database settings
            
        Raises:
            DatabaseConnectionError: If PostgreSQL connection fails
            VectorExtensionError: If pgvector extension not available
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
        self.db_user = getattr(config, 'rag_db_user', 'housegpt')
        self.db_password = getattr(config, 'rag_db_password', 'housegpt')
        
        # Vector configuration
        self.embedding_model_name = getattr(config, 'rag_embedding_model', 'all-MiniLM-L6-v2')
        self.vector_dimensions = getattr(config, 'rag_vector_dimensions', 384)
        
        # Initialize database connection and embedding model
        self._init_database()
        self._init_embedding_model()
        self._ensure_tables()
    
    def _init_mock_mode(self):
        """Initialize mock mode for testing without dependencies."""
        self.mock_mode = True
        self.mock_quotes = {}
        self.mock_stats = {
            'total_quotes': 0,
            'unique_speakers': 0,
            'seasons': 0,
            'avg_sarcasm_level': 0
        }
    
    def _init_database(self):
        """Initialize PostgreSQL database connection."""
        try:
            self.conn = psycopg2.connect(
                host=self.db_host,
                port=self.db_port,
                database=self.db_name,
                user=self.db_user,
                password=self.db_password
            )
            self.conn.autocommit = True
            
            # Check for pgvector extension
            with self.conn.cursor() as cursor:
                cursor.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
                if not cursor.fetchone():
                    raise VectorExtensionError("pgvector extension not installed")
            
            self.logger.info("Connected to PostgreSQL with pgvector")
            self.mock_mode = False
            
        except psycopg2.Error as e:
            raise DatabaseConnectionError(f"Failed to connect to database: {e}")
    
    def _init_embedding_model(self):
        """Initialize sentence transformer model for embeddings."""
        try:
            self.embedding_model = sentence_transformers.SentenceTransformer(self.embedding_model_name)
            self.logger.info(f"Loaded embedding model: {self.embedding_model_name}")
            
        except Exception as e:
            # Use mock embeddings for testing
            self.logger.warning(f"Using mock embeddings: {e}")
            self.embedding_model = None
    
    def _ensure_tables(self):
        """Create necessary database tables if they don't exist."""
        if hasattr(self, 'mock_mode') and self.mock_mode:
            return
        
        with self.conn.cursor() as cursor:
            # Create quotes table with vector column
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS house_quotes (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    text TEXT NOT NULL,
                    speaker VARCHAR(100) NOT NULL,
                    episode VARCHAR(50),
                    season INTEGER,
                    context TEXT,
                    emotional_tone VARCHAR(50),
                    sarcasm_level INTEGER,
                    tags TEXT[],
                    embedding vector(%s),
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW(),
                    UNIQUE(text, speaker)
                )
            """, (self.vector_dimensions,))
            
            # Create index for vector similarity search
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS house_quotes_embedding_idx 
                ON house_quotes USING ivfflat (embedding vector_cosine_ops)
            """)
            
            # Create metadata tracking table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS rag_stats (
                    id SERIAL PRIMARY KEY,
                    operation VARCHAR(50) NOT NULL,
                    execution_time_ms FLOAT,
                    result_count INTEGER,
                    query_text TEXT,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
    
    def add_quote(self, quote) -> str:
        """Add House quote to vector database with embedding generation.
        
        Args:
            quote: HouseQuote object to store
            
        Returns:
            Unique ID of stored quote
        """
        if hasattr(self, 'mock_mode') and self.mock_mode:
            quote_id = str(uuid.uuid4())
            self.mock_quotes[quote_id] = quote
            self.mock_stats['total_quotes'] += 1
            return quote_id
        
        if not isinstance(quote, HouseQuote):
            raise ValidationError("Quote must be a HouseQuote object")
        
        # Generate embedding if not present
        if quote.embedding_vector is None:
            quote.embedding_vector = self.get_query_embedding(quote.text)
        
        try:
            with self.conn.cursor() as cursor:
                # Check for duplicates
                cursor.execute(
                    "SELECT id FROM house_quotes WHERE text = %s AND speaker = %s",
                    (quote.text, quote.speaker)
                )
                if cursor.fetchone():
                    raise DuplicateQuoteError(f"Quote already exists: {quote.text[:50]}...")
                
                # Insert new quote
                quote_id = str(uuid.uuid4())
                cursor.execute("""
                    INSERT INTO house_quotes (
                        id, text, speaker, episode, season, context, 
                        emotional_tone, sarcasm_level, tags, embedding
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    quote_id, quote.text, quote.speaker, quote.episode,
                    quote.season, quote.context, quote.emotional_tone,
                    quote.sarcasm_level, quote.tags, quote.embedding_vector
                ))
                
                self.logger.info(f"Added quote: {quote_id}")
                return quote_id
                
        except psycopg2.IntegrityError as e:
            if "unique" in str(e).lower():
                raise DuplicateQuoteError(f"Quote already exists: {quote.text[:50]}...")
            raise ValidationError(f"Database constraint violation: {e}")
        except psycopg2.Error as e:
            raise RAGStoreError(f"Database error: {e}")
    
    def search_quotes(self, query: str, top_k: int = 5) -> List:
        """Search for most relevant House quotes using semantic similarity.
        
        Args:
            query: Search query text
            top_k: Maximum number of results to return
            
        Returns:
            List of similar HouseQuote objects
        """
        if hasattr(self, 'mock_mode') and self.mock_mode:
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
        
        if not query or not query.strip():
            raise ValidationError("Query cannot be empty")
        
        if top_k <= 0 or top_k > 100:
            raise ValidationError("top_k must be between 1 and 100")
        
        start_time = datetime.now()
        
        try:
            # Generate query embedding
            query_embedding = self.get_query_embedding(query)
            
            with self.conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Search using cosine similarity
                cursor.execute("""
                    SELECT 
                        id, text, speaker, episode, season, context,
                        emotional_tone, sarcasm_level, tags, embedding,
                        1 - (embedding <=> %s) as similarity
                    FROM house_quotes
                    WHERE 1 - (embedding <=> %s) >= 0.1
                    ORDER BY embedding <=> %s
                    LIMIT %s
                """, (query_embedding, query_embedding, query_embedding, top_k))
                
                results = cursor.fetchall()
                
                # Convert to HouseQuote objects
                quotes = []
                for row in results:
                    quote = HouseQuote(
                        text=row['text'],
                        speaker=row['speaker'],
                        episode=row['episode'],
                        season=row['season'],
                        context=row['context'],
                        emotional_tone=row['emotional_tone'],
                        sarcasm_level=row['sarcasm_level'],
                        tags=row['tags'] or [],
                        similarity_score=row['similarity'],
                        embedding_vector=row['embedding']
                    )
                    quote.quote_id = str(row['id'])
                    quotes.append(quote)
                
                # Log performance
                execution_time = (datetime.now() - start_time).total_seconds() * 1000
                self._log_operation("search", execution_time, len(quotes), query)
                
                return quotes
                
        except psycopg2.Error as e:
            raise RAGStoreError(f"Search error: {e}")
    
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
        if hasattr(self, 'mock_mode') and self.mock_mode:
            return self.mock_quotes.get(quote_id)
        
        if not quote_id or not isinstance(quote_id, str):
            raise ValidationError("Quote ID must be a non-empty string")
        
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT id, text, speaker, episode, season, context,
                           emotional_tone, sarcasm_level, tags, embedding
                    FROM house_quotes WHERE id = %s
                """, (quote_id,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                quote = HouseQuote(
                    text=row['text'],
                    speaker=row['speaker'],
                    episode=row['episode'],
                    season=row['season'],
                    context=row['context'],
                    emotional_tone=row['emotional_tone'],
                    sarcasm_level=row['sarcasm_level'],
                    tags=row['tags'] or [],
                    embedding_vector=row['embedding']
                )
                quote.quote_id = str(row['id'])
                
                return quote
                
        except psycopg2.Error as e:
            raise RAGStoreError(f"Retrieval error: {e}")
    
    def update_quote_metadata(self, quote_id: str, metadata: Dict[str, Any]) -> bool:
        """Update quote metadata without changing embedding.
        
        Args:
            quote_id: Quote to update
            metadata: Dictionary of fields to update
            
        Returns:
            True if quote was updated, False if not found
        """
        if hasattr(self, 'mock_mode') and self.mock_mode:
            if quote_id in self.mock_quotes:
                # Update mock quote
                quote = self.mock_quotes[quote_id]
                for key, value in metadata.items():
                    if hasattr(quote, key):
                        setattr(quote, key, value)
                return True
            return False
        
        if not quote_id:
            raise ValidationError("Quote ID required")
        
        if not metadata:
            return True
        
        # Build update query dynamically
        allowed_fields = ['episode', 'season', 'context', 'emotional_tone', 'sarcasm_level', 'tags']
        update_fields = []
        values = []
        
        for field, value in metadata.items():
            if field in allowed_fields:
                update_fields.append(f"{field} = %s")
                values.append(value)
        
        if not update_fields:
            return True
        
        try:
            with self.conn.cursor() as cursor:
                query = f"""
                    UPDATE house_quotes 
                    SET {', '.join(update_fields)}, updated_at = NOW()
                    WHERE id = %s
                """
                values.append(quote_id)
                
                cursor.execute(query, values)
                return cursor.rowcount > 0
                
        except psycopg2.Error as e:
            raise RAGStoreError(f"Update error: {e}")
    
    def delete_quote(self, quote_id: str) -> bool:
        """Remove quote from database.
        
        Args:
            quote_id: Quote to delete
            
        Returns:
            True if quote was deleted, False if not found
        """
        if hasattr(self, 'mock_mode') and self.mock_mode:
            if quote_id in self.mock_quotes:
                del self.mock_quotes[quote_id]
                self.mock_stats['total_quotes'] -= 1
                return True
            return False
        
        if not quote_id:
            raise ValidationError("Quote ID required")
        
        try:
            with self.conn.cursor() as cursor:
                cursor.execute("DELETE FROM house_quotes WHERE id = %s", (quote_id,))
                return cursor.rowcount > 0
                
        except psycopg2.Error as e:
            raise RAGStoreError(f"Delete error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics and health metrics.
        
        Returns:
            Dictionary with database statistics
        """
        if hasattr(self, 'mock_mode') and self.mock_mode:
            return self.mock_stats.copy()
        
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Quote statistics
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_quotes,
                        COUNT(DISTINCT speaker) as unique_speakers,
                        COUNT(DISTINCT season) as seasons,
                        AVG(sarcasm_level) as avg_sarcasm_level
                    FROM house_quotes
                """)
                quote_stats = cursor.fetchone()
                
                # Recent operation statistics
                cursor.execute("""
                    SELECT 
                        operation,
                        COUNT(*) as count,
                        AVG(execution_time_ms) as avg_time_ms
                    FROM rag_stats
                    WHERE created_at >= NOW() - INTERVAL '24 hours'
                    GROUP BY operation
                """)
                operation_stats = cursor.fetchall()
                
                return {
                    'total_quotes': quote_stats['total_quotes'],
                    'unique_speakers': quote_stats['unique_speakers'],
                    'seasons': quote_stats['seasons'],
                    'avg_sarcasm_level': float(quote_stats['avg_sarcasm_level']) if quote_stats['avg_sarcasm_level'] else 0,
                    'recent_operations': dict((row['operation'], {
                        'count': row['count'],
                        'avg_time_ms': float(row['avg_time_ms'])
                    }) for row in operation_stats)
                }
                
        except psycopg2.Error as e:
            raise RAGStoreError(f"Stats error: {e}")
    
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
            raise RAGStoreError(f"Embedding generation error: {e}")
    
    def _log_operation(self, operation: str, execution_time_ms: float, result_count: int, query_text: str = None):
        """Log operation statistics."""
        if hasattr(self, 'mock_mode') and self.mock_mode:
            return
        
        try:
            with self.conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO rag_stats (operation, execution_time_ms, result_count, query_text)
                    VALUES (%s, %s, %s, %s)
                """, (operation, execution_time_ms, result_count, query_text))
                
        except psycopg2.Error:
            # Don't fail operations due to logging errors
            pass
    
    def __del__(self):
        """Close database connection on cleanup."""
        if hasattr(self, 'conn'):
            self.conn.close()
