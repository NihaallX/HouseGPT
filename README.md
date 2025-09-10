# HouseGPT: Local Sarcastic Medical AI Assistant

> *"Everybody lies. Except me. I'm always right."* - Dr. Gregory House

HouseGPT is a locally-running AI assistant that embodies the sarcastic, brilliant diagnostic style of Dr. Gregory House from the hit TV series. Powered by fine-tuned language models and retrieval-augmented generation (RAG), HouseGPT provides medical insights with the characteristic wit and abrasiveness that made House MD famous.

## 🎯 Features

### Core Capabilities
- **🎤 Wake Word Detection**: Activates on "House?" for hands-free interaction
- **🧠 Local AI Inference**: Fine-tuned FLAN-T5 model with House MD personality LoRA
- **📚 RAG-Enhanced Responses**: Retrieves relevant quotes and medical knowledge
- **🗣️ Voice Synthesis**: XTTS-powered voice output with House's characteristic delivery
- **💾 Conversation Logging**: Persistent conversation history and context
- **🔒 Privacy-First**: Completely local execution, no external API calls

### Advanced Features
- **🎯 Style Filtering**: Ensures responses maintain House's sarcastic tone
- **⚡ Real-time Processing**: <5s total response time, <2s wake word detection
- **🗄️ Vector Database**: pgvector-powered semantic search for quotes and context
- **🔧 Modular Architecture**: Extensible design for easy customization
- **📊 Performance Monitoring**: Built-in metrics and optimization tools

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- PostgreSQL with pgvector extension
- 8GB+ RAM (16GB recommended)
- CUDA-compatible GPU (optional, for faster inference)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/housegpt.git
   cd housegpt
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up the database**:
   ```bash
   python scripts/setup_database.py
   ```

5. **Download models**:
   ```bash
   python scripts/download_models.py
   ```

6. **Load quote database**:
   ```bash
   python scripts/load_quotes.py
   ```

### First Run

1. **Start HouseGPT**:
   ```bash
   python -m src.cli.housegpt_cli
   ```

2. **Activate with wake word**:
   Say "House?" and wait for the activation sound.

3. **Start conversing**:
   ```
   User: "I have a headache and feel nauseous."
   House: "Well, obviously you have a hangover. Either that, or you have a brain tumor. 
           Let's go with hangover - it's more common and doesn't require surgery."
   ```

## 📖 Documentation

### Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Wake Word     │    │   Text Input    │    │   Voice Input   │
│   Detection     │    │   Processing    │    │   Processing    │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
          ┌─────────────────────────────────────────────────┐
          │              Conversation Pipeline             │
          │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────┐ │
          │  │   RAG   │  │  Model  │  │ Style   │  │TTS  │ │
          │  │Retrieval│  │Inference│  │Filter   │  │Out  │ │
          │  └─────────┘  └─────────┘  └─────────┘  └─────┘ │
          └─────────────────────────────────────────────────┘
                                 │
          ┌─────────────────────────────────────────────────┐
          │              Response Output                    │
          │  ┌─────────┐  ┌─────────┐  ┌─────────┐         │
          │  │  Text   │  │  Voice  │  │Logging  │         │
          │  │Response │  │ Output  │  │& Store  │         │
          │  └─────────┘  └─────────┘  └─────────┘         │
          └─────────────────────────────────────────────────┘
```

### Core Components

#### 1. Services
- **HouseModel** (`src/services/house_model.py`): LoRA-enhanced FLAN-T5 inference
- **StyleFilter** (`src/services/style_filter.py`): Personality consistency enforcement
- **VoiceOutput** (`src/services/voice_output.py`): XTTS voice synthesis
- **WakeWordDetector** (`src/services/wake_word.py`): Audio activation detection
- **RAGStore** (`src/services/rag_store.py`): Vector-based quote retrieval
- **ConversationLogger** (`src/services/conversation_logger.py`): History management

#### 2. Data Models
- **UserInput** (`src/models/user_input.py`): Input representation and validation
- **HouseResponse** (`src/models/house_response.py`): Response structure with metadata
- **ConversationContext** (`src/models/context.py`): Session and history management
- **Configuration** (`src/models/configuration.py`): System settings and parameters

#### 3. Utilities
- **TextUtils** (`src/lib/text_utils.py`): Text processing and validation
- **AudioUtils** (`src/lib/audio_utils.py`): Audio processing and analysis
- **DatabaseUtils** (`src/lib/db_utils.py`): Database connection and query management

## 🔧 Configuration

### Configuration File (`config/housegpt_config.json`)

```json
{
  "model": {
    "base_model": "google/flan-t5-large",
    "lora_path": "./housegpt-lora-large",
    "temperature": 0.8,
    "max_length": 512,
    "device": "auto"
  },
  "audio": {
    "sample_rate": 16000,
    "wake_word": "House",
    "confidence_threshold": 0.8,
    "vad_enabled": true
  },
  "rag": {
    "enabled": true,
    "max_retrieved_quotes": 5,
    "similarity_threshold": 0.7,
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
  },
  "database": {
    "url": "postgresql://user:pass@localhost/housegpt",
    "pool_size": 10,
    "max_overflow": 20
  },
  "voice": {
    "enabled": true,
    "model": "tts_models/en/ljspeech/tacotron2-DDC",
    "speed": 1.0,
    "voice_clone_path": "./voices/house_voice.wav"
  },
  "security": {
    "max_input_length": 1000,
    "rate_limit_per_minute": 30,
    "log_conversations": true
  }
}
```

## 🎭 Usage Examples

### Text Mode
```python
from src.pipeline.conversation_pipeline import ConversationPipeline
from src.models.configuration import HouseGPTConfig

# Initialize
config = HouseGPTConfig.load_from_file("config/housegpt_config.json")
pipeline = ConversationPipeline(config)

# Process conversation
result = await pipeline.process_conversation(
    user_input_text="I have chest pain and shortness of breath",
    session_id="user_session_123"
)

print(f"House: {result.house_response.text}")
# Output: "Chest pain and shortness of breath? Either you're having a heart attack 
#          or you just ran up the stairs. Given that you're talking to me instead 
#          of calling 911, I'm going with the stairs."
```

### Voice Mode
```python
from src.services.wake_word import WakeWordDetector
from src.services.voice_output import VoiceOutput

# Set up voice services
wake_word = WakeWordDetector(config.audio)
voice_output = VoiceOutput(config.voice)

# Listen for wake word
if wake_word.listen_for_wake_word():
    user_speech = wake_word.capture_speech()
    
    # Process through pipeline
    result = await pipeline.process_conversation(user_speech)
    
    # Speak response
    if result.success and result.voice_audio:
        voice_output.play_audio(result.voice_audio)
```

## 🧪 Testing

### Running Tests
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/unit/ -v          # Unit tests
python -m pytest tests/integration/ -v   # Integration tests
python -m pytest tests/performance/ -v   # Performance tests

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

### Performance Testing
```bash
# Run performance benchmarks
python scripts/performance_testing.py --suite all --verbose

# Test specific components
python scripts/performance_testing.py --suite model --output model_perf.md
python scripts/performance_testing.py --suite end_to_end --csv e2e_metrics.csv
```

## 📊 Performance Metrics

### Target Performance
- **Wake Word Detection**: <2 seconds
- **Total Response Time**: <5 seconds
- **Memory Usage**: <4GB RAM
- **CPU Usage**: <80% single core
- **Model Inference**: <3 seconds
- **RAG Retrieval**: <500ms

## 🔧 Development

### Project Structure
```
housegpt/
├── src/                    # Main source code
│   ├── services/          # Core services
│   ├── models/            # Data models
│   ├── pipeline/          # Processing pipeline
│   ├── lib/               # Utilities
│   └── cli/               # Command-line interface
├── tests/                 # Test suites
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   └── performance/       # Performance tests
├── scripts/               # Utility scripts
├── config/                # Configuration files
├── data/                  # Quote database and embeddings
├── models/                # Model weights and checkpoints
├── voices/                # Voice synthesis models
└── docs/                  # Documentation
```

## 🐛 Troubleshooting

### Common Issues

#### Model Loading Issues
```bash
# Check model files
python scripts/validate_models.py

# Re-download models
python scripts/download_models.py --force
```

#### Database Connection Issues
```bash
# Test database connection
python scripts/test_db_connection.py

# Reset database
python scripts/setup_database.py --reset
```

#### Audio Issues
```bash
# Test audio devices
python scripts/test_audio.py --list-devices

# Test wake word detection
python scripts/test_wake_word.py --debug
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Hugh Laurie** and the cast of House MD for inspiration
- **Hugging Face** for transformer models and tools
- **PostgreSQL** and **pgvector** for vector database capabilities
- **XTTS** team for voice synthesis technology
- The open-source AI community for making this possible

---

*"It's not about the symptoms. It's about the differential diagnosis. And mine is always right."* - Dr. House

**Disclaimer**: HouseGPT is for entertainment and educational purposes only. Do not use for actual medical diagnosis or treatment. Always consult qualified medical professionals for health concerns.

## Features

- **Wake Word Detection**: Responds to "House?" using OpenAI Whisper
- **RAG-Enhanced Responses**: Retrieves relevant House M.D. quotes via vector similarity search
- **LoRA Model Integration**: Uses fine-tuned personality model for authentic House responses
- **Sarcasm Filter**: Applies wit, condescension, and cranky openers to responses
- **Voice Synthesis**: XTTS-powered House-like voice output
- **Local Execution**: No internet required after setup, privacy-focused design

## Quick Start

### Prerequisites
- Python 3.11+
- PostgreSQL 15+ with pgvector extension
- 8GB RAM minimum (16GB recommended)
- Microphone and speakers

### Installation
```bash
git clone <repository-url>
cd housegpt-personality-injector
pip install -r requirements.txt
```

### Setup
```bash
# Create database
createdb housegpt
psql housegpt -c "CREATE EXTENSION vector;"

# Initialize environment
cp .env.example .env
# Edit .env with your database credentials

# Setup database and load quotes
python scripts/setup_database.py
python scripts/load_quotes.py data/house_quotes.json

# Download models
python scripts/download_models.py
```

### Usage
```bash
# Interactive mode with wake word detection
python -m src.cli.main --mode interactive

# Text-only mode
python -m src.cli.main --mode text-only

# Voice-only mode (no wake word)
python -m src.cli.main --mode voice-only
```

## Architecture

### Components
- **Wake Word Detection** (`src/services/wake_word.py`): Continuous audio monitoring
- **RAG Store** (`src/services/rag_store.py`): Vector database for quote retrieval
- **House Model** (`src/services/house_model.py`): LoRA model inference
- **Style Filter** (`src/services/style_filter.py`): Personality injection
- **Voice Output** (`src/services/voice_output.py`): Text-to-speech synthesis
- **Logger** (`src/services/logger.py`): Conversation logging

### Data Flow
1. Wake word detection triggers input capture
2. RAG system retrieves relevant House quotes
3. LoRA model generates response with quote context
4. Style filter applies sarcasm and personality traits
5. XTTS synthesizes House-like voice output
6. Complete interaction logged for analytics

## Development

### Setup Development Environment
```bash
pip install -e ".[dev]"
pre-commit install
```

### Running Tests
```bash
# All tests
pytest

# Contract tests only
pytest tests/contract/

# Integration tests only
pytest tests/integration/

# Unit tests only
pytest tests/unit/
```

### Code Quality
```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Run pre-commit hooks
pre-commit run --all-files
```

## Configuration

### Environment Variables
```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/housegpt

# Models
LORA_MODEL_PATH=./housegpt-lora-large
BASE_MODEL_NAME=microsoft/DialoGPT-medium

# Audio
AUDIO_DEVICE_INDEX=0
VOICE_SAMPLE_PATH=./assets/house_voice_sample.wav

# Performance
MAX_RESPONSE_LENGTH=150
WAKE_WORD_SENSITIVITY=0.7
RAG_TOP_K=3
```

### Performance Tuning
- **Wake Word Sensitivity**: Adjust `WAKE_WORD_SENSITIVITY` (0.0-1.0)
- **Response Length**: Modify `MAX_RESPONSE_LENGTH` (50-300 tokens)
- **Quote Retrieval**: Change `RAG_TOP_K` (1-10 quotes per response)
- **Audio Quality**: Adjust chunk sizes and sample rates in config

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Write tests first (TDD approach)
4. Implement the feature
5. Ensure all tests pass
6. Submit a pull request

## Support

For issues and questions:
- Check the [Quickstart Guide](specs/001-enhance-local-housegpt/quickstart.md)
- Review [troubleshooting documentation](docs/troubleshooting.md)
- Open an issue on GitHub
