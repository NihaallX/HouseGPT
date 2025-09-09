# RAG Store Contract

**Component**: `rag_store.py`  
**Purpose**: Vector storage and similarity search for House quotes  
**Version**: 1.0.0

## Interface Definition

### Class: `RAGStore`

#### Constructor
```python
def __init__(self, config: Configuration) -> None
```
**Parameters**:
- `config`: Application configuration object
**Raises**:
- `DatabaseConnectionError`: If PostgreSQL connection fails
- `VectorExtensionError`: If pgvector extension not available

#### Methods

##### `add_quote(quote: HouseQuote) -> str`
Stores a new quote with embedding in vector database.

**Parameters**:
- `quote`: HouseQuote object with text and metadata
**Returns**:
- `quote_id`: Generated UUID for the stored quote
**Preconditions**:
- Database connection must be active
- Quote text must be validated
**Postconditions**:
- Quote stored in database with generated embedding
- Searchable via similarity queries
**Raises**:
- `ValidationError`: If quote fails validation rules
- `DuplicateQuoteError`: If identical quote already exists
- `DatabaseError`: If storage operation fails

##### `search_quotes(query: str, top_k: int = 3) -> List[HouseQuote]`
Finds most relevant quotes using semantic similarity.

**Parameters**:
- `query`: User input text to search against
- `top_k`: Maximum number of quotes to return (default: 3)
**Returns**:
- List of HouseQuote objects ordered by relevance
**Preconditions**:
- Query string must not be empty
- top_k must be between 1 and 10
**Postconditions**:
- Results sorted by cosine similarity score
- Each result includes similarity score in metadata
**Raises**:
- `ValidationError`: If parameters invalid
- `EmbeddingError`: If query embedding generation fails
- `DatabaseError`: If search operation fails

##### `get_quote_by_id(quote_id: str) -> Optional[HouseQuote]`
Retrieves specific quote by UUID.

**Parameters**:
- `quote_id`: UUID string identifier
**Returns**:
- HouseQuote object if found, None otherwise
**Raises**:
- `ValidationError`: If quote_id format invalid
- `DatabaseError`: If retrieval operation fails

##### `update_quote_metadata(quote_id: str, metadata: Dict[str, Any]) -> bool`
Updates quote metadata without changing embedding.

**Parameters**:
- `quote_id`: UUID of quote to update
- `metadata`: Dictionary of fields to update
**Returns**:
- `True` if update successful, `False` if quote not found
**Raises**:
- `ValidationError`: If metadata format invalid
- `DatabaseError`: If update operation fails

##### `delete_quote(quote_id: str) -> bool`
Removes quote from database.

**Parameters**:
- `quote_id`: UUID of quote to delete
**Returns**:
- `True` if deletion successful, `False` if quote not found
**Raises**:
- `DatabaseError`: If deletion operation fails

##### `get_stats() -> Dict[str, Any]`
Returns database statistics and health metrics.

**Returns**:
```python
{
    "total_quotes": int,
    "avg_embedding_time_ms": float,
    "last_search_time_ms": float,
    "database_size_mb": float,
    "index_efficiency": float
}
```

## Embedding Generation

### Process
1. Input text normalization (lowercase, punctuation handling)
2. Generate 384-dimensional vector using sentence-transformers
3. Normalize vector for cosine similarity
4. Store with metadata in PostgreSQL

### Model Contract
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Dimensions**: 384
- **Similarity Metric**: Cosine similarity
- **Normalization**: L2 normalized vectors

## Database Schema Contract

### Table: `house_quotes`
```sql
CREATE TABLE house_quotes (
    quote_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    text TEXT NOT NULL CHECK (length(text) >= 10 AND length(text) <= 200),
    embedding vector(384) NOT NULL,
    episode_info TEXT,
    sarcasm_level INTEGER CHECK (sarcasm_level >= 1 AND sarcasm_level <= 5),
    context_tags TEXT[],
    emotional_tone TEXT CHECK (emotional_tone IN ('angry', 'witty', 'cynical', 'condescending', 'dark')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    similarity_score FLOAT DEFAULT 0.0
);
```

### Indexes Required
```sql
CREATE INDEX idx_quotes_embedding ON house_quotes USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX idx_quotes_sarcasm ON house_quotes (sarcasm_level);
CREATE INDEX idx_quotes_tone ON house_quotes (emotional_tone);
```

## Error Handling

### Exception Hierarchy
```python
class RAGStoreError(Exception): pass
class DatabaseConnectionError(RAGStoreError): pass
class VectorExtensionError(RAGStoreError): pass
class EmbeddingError(RAGStoreError): pass
class DuplicateQuoteError(RAGStoreError): pass
class ValidationError(RAGStoreError): pass
```

### Recovery Strategies
- **Connection lost**: Automatic reconnection with exponential backoff
- **Embedding failure**: Retry with cleaned text, fallback to keyword search
- **Index corruption**: Rebuild vector index automatically
- **Out of memory**: Clear embedding cache and retry

## Performance Contract

- **Search Latency**: < 100ms for similarity search (top-10)
- **Embedding Generation**: < 50ms per quote
- **Memory Usage**: < 500MB for 1000 quotes
- **Concurrent Searches**: Support 10 parallel queries
- **Database Size**: < 50MB for 1000 quotes with embeddings

## Configuration Parameters

```python
{
    "database_url": "postgresql://user:pass@localhost/housegpt",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "similarity_threshold": 0.3,
    "max_results": 10,
    "connection_pool_size": 5,
    "embedding_cache_size": 1000,
    "index_maintenance_interval": 3600
}
```

## Testing Contract

### Unit Tests Required
- Quote storage and retrieval accuracy
- Embedding generation consistency
- Similarity search ranking validation
- Database schema compliance
- Error handling and recovery

### Integration Test Scenarios
- End-to-end quote ingestion pipeline
- Performance testing with 1000+ quotes
- Concurrent access patterns
- Database failover scenarios
- Memory usage under load

### Test Data Requirements
- 100 sample House quotes with known similarity relationships
- Performance baseline measurements
- Edge cases (empty queries, special characters, very long text)
