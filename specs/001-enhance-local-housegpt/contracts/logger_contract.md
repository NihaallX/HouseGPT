# Logger Contract

**Component**: `logger.py`  
**Purpose**: Conversation logging and audit trail  
**Version**: 1.0.0

## Interface Definition

### Class: `ConversationLogger`

#### Constructor
```python
def __init__(self, config: Configuration) -> None
```
**Parameters**:
- `config`: Application configuration object
**Raises**:
- `LoggerInitializationError`: If log directory creation fails
- `DatabaseConnectionError`: If database connection for logs fails

#### Methods

##### `log_conversation(user_input: UserInput, response: ModelResponse) -> str`
Records complete conversation interaction.

**Parameters**:
- `user_input`: User input object with metadata
- `response`: Model response object with processing details
**Returns**:
- `log_id`: UUID of created conversation log entry
**Preconditions**:
- Database connection must be active
- Both input objects must be valid
**Postconditions**:
- Complete interaction logged with timestamp
- Searchable via log_id or timestamp
**Raises**:
- `LoggingError`: If log write operation fails
- `ValidationError`: If input objects invalid

##### `get_conversation_history(session_id: str, limit: int = 50) -> List[ConversationLog]`
Retrieves conversation history for a session.

**Parameters**:
- `session_id`: UUID of conversation session
- `limit`: Maximum number of entries to return
**Returns**:
- List of ConversationLog objects, newest first
**Raises**:
- `ValidationError`: If session_id format invalid
- `DatabaseError`: If retrieval operation fails

##### `search_conversations(query: str, start_date: Optional[datetime] = None, 
                         end_date: Optional[datetime] = None) -> List[ConversationLog]`
Searches conversation logs by content.

**Parameters**:
- `query`: Text to search for in user messages or responses
- `start_date`: Optional earliest timestamp filter
- `end_date`: Optional latest timestamp filter
**Returns**:
- List of matching ConversationLog objects
**Search Scope**:
- User message content
- House response content
- Error details (if any)
**Raises**:
- `ValidationError`: If date parameters invalid
- `DatabaseError`: If search operation fails

##### `get_performance_stats(time_window: int = 24) -> Dict[str, Any]`
Returns conversation performance metrics.

**Parameters**:
- `time_window`: Hours to look back for statistics
**Returns**:
```python
{
    "total_conversations": int,
    "successful_conversations": int,
    "avg_response_time_ms": float,
    "error_rate": float,
    "most_common_errors": List[str],
    "peak_usage_hour": int,
    "total_session_count": int
}
```

##### `cleanup_old_logs(retention_days: int) -> int`
Removes conversation logs older than retention period.

**Parameters**:
- `retention_days`: Number of days to retain logs
**Returns**:
- Number of log entries deleted
**Behavior**:
- Only deletes complete conversation records
- Preserves error logs regardless of age (for debugging)
- Maintains referential integrity
**Raises**:
- `DatabaseError`: If cleanup operation fails

##### `export_logs(output_path: str, format: str = "json", 
               start_date: Optional[datetime] = None,
               end_date: Optional[datetime] = None) -> bool`
Exports conversation logs to file.

**Parameters**:
- `output_path`: File path for exported data
- `format`: Export format ("json", "csv", "txt")
- `start_date`: Optional start of export time range
- `end_date`: Optional end of export time range
**Returns**:
- `True` if export successful, `False` otherwise
**Formats**:
- JSON: Complete structured data
- CSV: Tabular format for analysis
- TXT: Human-readable conversation format
**Raises**:
- `ExportError`: If file creation or writing fails
- `ValidationError`: If parameters invalid

## Logging Schema

### ConversationLog Structure
```python
@dataclass
class ConversationLog:
    log_id: str                    # UUID
    user_message: str              # Original user input
    house_response: str            # Final delivered response
    timestamp: datetime            # Conversation time (UTC)
    session_id: str                # Conversation session UUID
    response_time_ms: int          # Total processing time
    rag_queries: List[str]         # Vector search queries used
    retrieved_quotes: List[str]    # Quote IDs retrieved
    wake_word_confidence: float    # Detection confidence score
    synthesis_success: bool        # Whether voice output succeeded
    success: bool                  # Overall interaction success
    error_details: Optional[str]   # Failure information
    user_input_metadata: Dict      # Additional input context
    response_metadata: Dict        # Additional response context
```

### Database Schema
```sql
CREATE TABLE conversation_logs (
    log_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_message TEXT NOT NULL CHECK (length(user_message) <= 500),
    house_response TEXT NOT NULL CHECK (length(house_response) <= 500),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    session_id UUID NOT NULL,
    response_time_ms INTEGER CHECK (response_time_ms > 0),
    rag_queries TEXT[],
    retrieved_quotes TEXT[],
    wake_word_confidence FLOAT CHECK (wake_word_confidence >= 0.0 AND wake_word_confidence <= 1.0),
    synthesis_success BOOLEAN DEFAULT TRUE,
    success BOOLEAN NOT NULL DEFAULT TRUE,
    error_details TEXT,
    user_input_metadata JSONB,
    response_metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for efficient querying
CREATE INDEX idx_logs_timestamp ON conversation_logs (timestamp);
CREATE INDEX idx_logs_session ON conversation_logs (session_id);
CREATE INDEX idx_logs_success ON conversation_logs (success);
CREATE INDEX idx_logs_text_search ON conversation_logs USING gin(to_tsvector('english', user_message || ' ' || house_response));
```

## Log Levels and Categories

### Log Categories
```python
class LogCategory(Enum):
    CONVERSATION = "conversation"     # Normal user interactions
    ERROR = "error"                  # System errors and failures
    PERFORMANCE = "performance"      # Timing and resource metrics
    SECURITY = "security"           # Authentication and access logs
    SYSTEM = "system"               # Startup, shutdown, configuration
```

### Structured Logging Format
```json
{
    "timestamp": "2025-09-09T15:30:45.123Z",
    "level": "INFO",
    "category": "conversation",
    "session_id": "550e8400-e29b-41d4-a716-446655440000",
    "log_id": "550e8400-e29b-41d4-a716-446655440001",
    "message": "Conversation completed successfully",
    "data": {
        "user_message": "What's wrong with my patient?",
        "house_response": "Whattt? You mean besides your diagnostic skills?",
        "response_time_ms": 3450,
        "components_used": ["wake_word", "rag_store", "house_model", "style_filter", "voice_output"],
        "rag_hits": 3,
        "voice_synthesis_time_ms": 850
    }
}
```

## Privacy and Security

### Data Protection
- **Anonymization**: Option to hash user inputs for privacy
- **Retention Limits**: Automatic cleanup of old conversations
- **Access Control**: Log access restricted to application
- **Encryption**: Sensitive fields encrypted at rest (optional)

### Compliance Features
- **Data Export**: GDPR-compliant data export functionality
- **Right to Deletion**: Ability to remove specific user sessions
- **Audit Trail**: All log operations are themselves logged
- **Integrity Checks**: Tamper detection for log entries

## Error Handling

### Exception Hierarchy
```python
class LoggerError(Exception): pass
class LoggingError(LoggerError): pass
class LoggerInitializationError(LoggerError): pass
class ExportError(LoggerError): pass
class DatabaseError(LoggerError): pass
```

### Error Recovery
- **Database connection lost**: Queue logs in memory, flush when reconnected
- **Disk full**: Rotate to new log file, alert administrator
- **Corruption detected**: Create new log file, preserve corrupted for analysis
- **Permission denied**: Fallback to application-specific directory

## Performance Contract

- **Log Write Speed**: < 10ms per conversation entry
- **Search Performance**: < 100ms for text searches across 10k entries
- **Memory Usage**: < 50MB for log buffers and caches
- **Database Size**: Approximately 1KB per conversation log
- **Export Speed**: > 1000 entries per second for JSON export

## Configuration Parameters

```python
{
    "log_level": "INFO",
    "log_retention_days": 30,
    "database_url": "postgresql://user:pass@localhost/housegpt_logs",
    "log_directory": "./logs",
    "enable_text_search": True,
    "enable_performance_logging": True,
    "anonymize_user_input": False,
    "encrypt_sensitive_fields": False,
    "max_log_file_size_mb": 100,
    "log_rotation_count": 5,
    "export_batch_size": 1000,
    "enable_audit_trail": True
}
```

## Testing Contract

### Unit Tests Required
- Conversation logging accuracy and completeness
- Search functionality with various query types
- Performance metrics calculation
- Data export in all supported formats
- Privacy and anonymization features

### Integration Test Scenarios
- End-to-end logging pipeline with real conversations
- Database failover and recovery scenarios
- Large dataset search performance
- Concurrent logging from multiple sessions
- Log retention and cleanup automation

### Performance Testing
- Sustained logging under high conversation volume
- Search performance with large datasets (10k+ entries)
- Memory usage monitoring during extended operation
- Database query optimization validation
- Export performance with large date ranges

### Data Integrity Testing
- Verify all conversation components are logged
- Validate timestamp accuracy and timezone handling
- Test search indexing effectiveness
- Confirm referential integrity maintenance
- Validate privacy and anonymization correctness
