# Data Model: HouseGPT Implementation

**Feature**: HouseGPT Personality Injector with RAG + Rule-Based Style Filter  
**Date**: 2025-09-09  
**Phase**: 1 - Data Model Design

## Core Entities

### 1. UserInput
Represents incoming user interaction data.

**Attributes**:
- `text_input: str` - Transcribed or typed user message
- `voice_input: Optional[bytes]` - Raw audio data if voice-based
- `wake_word_detected: bool` - Whether "House?" trigger was identified
- `timestamp: datetime` - When input was received
- `confidence_score: float` - Wake word detection confidence (0.0-1.0)

**Validation Rules**:
- Either `text_input` or `voice_input` must be present
- `timestamp` must be valid UTC datetime
- `confidence_score` must be between 0.0 and 1.0
- `text_input` max length: 500 characters

**State Transitions**:
- Created → Validated → Processed → Archived

### 2. HouseQuote
Represents a Gregory House quote in the knowledge base.

**Attributes**:
- `quote_id: str` - Unique identifier (UUID)
- `text: str` - The actual quote content
- `embedding: List[float]` - 384-dimensional vector representation
- `episode_info: Optional[str]` - Source episode reference
- `sarcasm_level: int` - Scale 1-5 for sarcasm intensity
- `context_tags: List[str]` - Categorization tags (medical, personal, etc.)
- `emotional_tone: str` - angry, witty, cynical, etc.
- `created_at: datetime` - When added to database

**Validation Rules**:
- `text` min length: 10 characters, max length: 200 characters
- `embedding` must be exactly 384 dimensions
- `sarcasm_level` must be 1-5 integer
- `emotional_tone` from predefined enum: ["angry", "witty", "cynical", "condescending", "dark"]

**Relationships**:
- One-to-many with ConversationLog (via retrieval)

### 3. ModelResponse
Represents the AI model's output at different processing stages.

**Attributes**:
- `response_id: str` - Unique identifier (UUID)
- `raw_output: str` - Direct model generation
- `filtered_output: str` - After sarcasm filter processing
- `voice_data: Optional[bytes]` - Generated audio bytes
- `processing_time_ms: int` - Total generation time
- `retrieved_quotes: List[str]` - Quote IDs used for context
- `cranky_opener: str` - Applied opener phrase
- `style_modifications: List[str]` - Applied sarcasm transformations

**Validation Rules**:
- `raw_output` and `filtered_output` max length: 300 characters
- `processing_time_ms` must be positive integer
- `cranky_opener` from predefined list: ["Whattt?", "Yahh?", "Give me Vicodin?"]
- `retrieved_quotes` max 5 items

**State Transitions**:
- Generated → Filtered → Synthesized → Delivered

### 4. ConversationLog
Persistent record of user interactions.

**Attributes**:
- `log_id: str` - Unique identifier (UUID)
- `user_message: str` - Original user input
- `house_response: str` - Final delivered response
- `timestamp: datetime` - Conversation time
- `session_id: str` - Grouping for conversation sessions
- `response_time_ms: int` - Total processing time
- `rag_queries: List[str]` - Vector search queries used
- `success: bool` - Whether interaction completed successfully
- `error_details: Optional[str]` - Failure information if applicable

**Validation Rules**:
- `user_message` and `house_response` max length: 500 characters
- `response_time_ms` must be positive integer
- `timestamp` must be valid UTC datetime
- `error_details` only present when `success` is False

**Relationships**:
- Links to UserInput and ModelResponse via timestamps

### 5. Configuration
Application settings and runtime parameters.

**Attributes**:
- `model_path: str` - Path to LoRA model directory
- `base_model_name: str` - Hugging Face base model identifier
- `voice_settings: Dict[str, Any]` - XTTS configuration parameters
- `wake_word_sensitivity: float` - Detection threshold (0.0-1.0)
- `max_response_length: int` - Token limit for generation
- `rag_top_k: int` - Number of quotes to retrieve
- `audio_chunk_size: int` - Audio processing buffer size
- `log_retention_days: int` - Conversation history cleanup

**Validation Rules**:
- `model_path` must exist on filesystem
- `wake_word_sensitivity` between 0.0 and 1.0
- `max_response_length` between 50 and 300
- `rag_top_k` between 1 and 10
- `log_retention_days` must be positive integer

**Default Values**:
```python
{
    "model_path": "./housegpt-lora-large",
    "base_model_name": "microsoft/DialoGPT-medium",
    "wake_word_sensitivity": 0.7,
    "max_response_length": 150,
    "rag_top_k": 3,
    "audio_chunk_size": 1024,
    "log_retention_days": 30
}
```

## Database Schema

### PostgreSQL Tables

```sql
-- Vector extension for embeddings
CREATE EXTENSION IF NOT EXISTS vector;

-- House quotes with embeddings
CREATE TABLE house_quotes (
    quote_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    text TEXT NOT NULL CHECK (length(text) >= 10 AND length(text) <= 200),
    embedding vector(384) NOT NULL,
    episode_info TEXT,
    sarcasm_level INTEGER CHECK (sarcasm_level >= 1 AND sarcasm_level <= 5),
    context_tags TEXT[],
    emotional_tone TEXT CHECK (emotional_tone IN ('angry', 'witty', 'cynical', 'condescending', 'dark')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Conversation logs
CREATE TABLE conversation_logs (
    log_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_message TEXT NOT NULL CHECK (length(user_message) <= 500),
    house_response TEXT NOT NULL CHECK (length(house_response) <= 500),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    session_id UUID NOT NULL,
    response_time_ms INTEGER CHECK (response_time_ms > 0),
    rag_queries TEXT[],
    success BOOLEAN NOT NULL DEFAULT TRUE,
    error_details TEXT
);

-- Indexes for performance
CREATE INDEX idx_quotes_embedding ON house_quotes USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX idx_logs_timestamp ON conversation_logs (timestamp);
CREATE INDEX idx_logs_session ON conversation_logs (session_id);
```

## Entity Relationships

```
UserInput (1) → (1) ModelResponse
ModelResponse (1) → (1) ConversationLog
HouseQuote (N) → (M) ModelResponse (via retrieved_quotes)
Configuration (1) → (N) All Operations
```

## Serialization Formats

### JSON Schema for API Communication

```json
{
  "UserInput": {
    "type": "object",
    "required": ["timestamp", "wake_word_detected"],
    "properties": {
      "text_input": {"type": "string", "maxLength": 500},
      "voice_input": {"type": "string", "format": "base64"},
      "wake_word_detected": {"type": "boolean"},
      "timestamp": {"type": "string", "format": "date-time"},
      "confidence_score": {"type": "number", "minimum": 0.0, "maximum": 1.0}
    }
  },
  "ModelResponse": {
    "type": "object",
    "required": ["response_id", "raw_output", "filtered_output"],
    "properties": {
      "response_id": {"type": "string", "format": "uuid"},
      "raw_output": {"type": "string", "maxLength": 300},
      "filtered_output": {"type": "string", "maxLength": 300},
      "voice_data": {"type": "string", "format": "base64"},
      "processing_time_ms": {"type": "integer", "minimum": 1},
      "retrieved_quotes": {"type": "array", "maxItems": 5, "items": {"type": "string"}},
      "cranky_opener": {"type": "string", "enum": ["Whattt?", "Yahh?", "Give me Vicodin?"]},
      "style_modifications": {"type": "array", "items": {"type": "string"}}
    }
  }
}
```

## Data Flow

1. **Input Processing**: UserInput → Wake Word Detection → Validation
2. **RAG Retrieval**: Query Embedding → Vector Search → HouseQuote Selection
3. **Response Generation**: Context + User Input → Model → Raw Output
4. **Style Processing**: Raw Output → Sarcasm Filter → Filtered Output
5. **Voice Synthesis**: Filtered Output → XTTS → Voice Data
6. **Logging**: Full interaction → ConversationLog → Database

## Migration Strategy

### Initial Data Population
1. Bootstrap quote database with curated House M.D. quotes
2. Generate embeddings for all quotes using sentence-transformers
3. Create initial configuration with default values
4. Set up database indexes for optimal retrieval performance

### Schema Evolution
- Use PostgreSQL migrations for schema changes
- Maintain backward compatibility for conversation logs
- Version configuration schema for feature additions

**Phase 1 Data Model Status**: ✅ COMPLETE - All entities defined with validation rules
