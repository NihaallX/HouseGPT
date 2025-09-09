# HouseGPT Quick Start Guide

**Feature**: HouseGPT Personality Injector with RAG + Rule-Based Style Filter  
**Version**: 1.0.0  
**Last Updated**: 2025-09-09

## Prerequisites

### System Requirements
- **Operating System**: Windows 10+, macOS 10.15+, or Linux (Ubuntu 20.04+)
- **Python**: 3.11 or higher
- **Memory**: 8GB RAM minimum (16GB recommended)
- **Storage**: 10GB free space for models and dependencies
- **Audio**: Microphone and speakers/headphones

### Software Dependencies
- **PostgreSQL**: 15+ with pgvector extension
- **Git**: For repository management
- **FFmpeg**: For audio processing (auto-installed with dependencies)

## Installation

### 1. Clone Repository
```bash
git clone <repository-url>
cd housegpt-personality-injector
```

### 2. Create Python Environment
```bash
# Using venv
python -m venv housegpt-env
source housegpt-env/bin/activate  # Linux/Mac
# OR
housegpt-env\Scripts\activate  # Windows

# Using conda (alternative)
conda create -n housegpt python=3.11
conda activate housegpt
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up PostgreSQL with pgvector
```bash
# Install PostgreSQL (if not already installed)
# Ubuntu/Debian:
sudo apt install postgresql postgresql-contrib

# macOS with Homebrew:
brew install postgresql

# Windows: Download from postgresql.org

# Install pgvector extension
# Follow instructions at: https://github.com/pgvector/pgvector
```

### 5. Initialize Database
```bash
createdb housegpt
psql housegpt -c "CREATE EXTENSION vector;"
python scripts/setup_database.py
```

### 6. Download Required Models
```bash
# This will download XTTS and sentence-transformers models
python scripts/download_models.py
```

## Configuration

### 1. Environment Variables
Create a `.env` file in the project root:
```bash
# Database Configuration
DATABASE_URL=postgresql://username:password@localhost:5432/housegpt

# Model Paths
LORA_MODEL_PATH=./housegpt-lora-large
BASE_MODEL_NAME=microsoft/DialoGPT-medium

# Audio Settings
AUDIO_DEVICE_INDEX=0  # Use default microphone
VOICE_SAMPLE_PATH=./assets/house_voice_sample.wav

# Performance Settings
MAX_RESPONSE_LENGTH=150
WAKE_WORD_SENSITIVITY=0.7
RAG_TOP_K=3
```

### 2. Voice Sample Setup
Place a high-quality House M.D. voice sample in `./assets/house_voice_sample.wav`:
- Duration: 5-10 seconds
- Format: WAV, 22050 Hz, 16-bit
- Content: Clear dialogue from the show
- Quality: Minimal background noise

### 3. Validate Configuration
```bash
python scripts/validate_setup.py
```
This will check all dependencies, model files, and configuration.

## Initial Data Setup

### 1. Load House Quotes
```bash
# Load curated House M.D. quotes into the vector database
python scripts/load_quotes.py ./data/house_quotes.json
```

### 2. Generate Embeddings
```bash
# Generate vector embeddings for all quotes
python scripts/generate_embeddings.py
```

### 3. Verify RAG Setup
```bash
# Test vector search functionality
python scripts/test_rag.py "What's wrong with this patient?"
```

## First Run

### 1. Start the Application
```bash
python src/cli/main.py --mode interactive
```

### 2. Test Wake Word Detection
1. Application will display "Listening for 'House?'..."
2. Say "House?" clearly into your microphone
3. You should see "Wake word detected!" message
4. Application is now ready for your input

### 3. Test Full Interaction
1. Say: "House?" (wait for detection confirmation)
2. Ask: "What's wrong with my patient who has a fever?"
3. Expect: House-style sarcastic response with voice output

### 4. Test Text-Only Mode
```bash
python src/cli/main.py --mode text-only
# Type: "What's wrong with my patient?"
# Expect: Text response without voice
```

## Validation Checklist

### ✅ Basic Functionality
- [ ] Wake word detection responds to "House?"
- [ ] Text input generates appropriate response
- [ ] RAG system retrieves relevant quotes
- [ ] Style filter applies sarcasm and wit
- [ ] Voice output sounds like House
- [ ] Conversation logging works

### ✅ Performance Validation
- [ ] Wake word detection < 2 seconds
- [ ] Total response time < 5 seconds
- [ ] Memory usage < 4GB
- [ ] No audio dropouts or glitches

### ✅ Error Handling
- [ ] Graceful handling of no microphone
- [ ] Fallback when voice synthesis fails
- [ ] Recovery from database connection issues
- [ ] Proper error messages for invalid input

## Common Issues & Solutions

### Issue: Wake Word Not Detected
**Symptoms**: No response to "House?"
**Solutions**:
1. Check microphone permissions
2. Adjust `WAKE_WORD_SENSITIVITY` in config
3. Verify microphone device index
4. Test with `python scripts/test_audio.py`

### Issue: Poor Voice Quality
**Symptoms**: Robotic or unclear speech
**Solutions**:
1. Verify voice sample quality and format
2. Check XTTS model download completeness
3. Adjust voice synthesis parameters
4. Test with `python scripts/test_tts.py "Hello world"`

### Issue: Slow Responses
**Symptoms**: Response time > 10 seconds
**Solutions**:
1. Check available RAM (close other applications)
2. Verify GPU availability: `python scripts/check_gpu.py`
3. Reduce `MAX_RESPONSE_LENGTH` in config
4. Monitor with `python scripts/performance_monitor.py`

### Issue: Database Connection Errors
**Symptoms**: "Unable to connect to database"
**Solutions**:
1. Verify PostgreSQL service is running
2. Check DATABASE_URL in .env file
3. Confirm pgvector extension installed
4. Test connection: `python scripts/test_db.py`

### Issue: No House Quotes Retrieved
**Symptoms**: Generic responses without House personality
**Solutions**:
1. Verify quotes were loaded: `python scripts/check_quotes.py`
2. Regenerate embeddings: `python scripts/generate_embeddings.py`
3. Test RAG search: `python scripts/test_rag.py "test query"`

## Usage Examples

### Interactive Mode (Default)
```bash
python src/cli/main.py
# Say "House?" -> Ask question -> Get sarcastic response with voice
```

### Text-Only Mode
```bash
python src/cli/main.py --mode text-only
# Type questions directly, get text responses only
```

### Voice-Only Mode (No Wake Word)
```bash
python src/cli/main.py --mode voice-only
# Continuous voice interaction without wake word trigger
```

### Batch Processing
```bash
python src/cli/main.py --batch-file questions.txt --output responses.txt
# Process multiple questions from file
```

### Debug Mode
```bash
python src/cli/main.py --debug
# Detailed logging and component status information
```

## Next Steps

### 1. Customize Voice Settings
Edit voice parameters in the config file to fine-tune House's voice characteristics.

### 2. Add More Quotes
Use `scripts/add_quotes.py` to expand the knowledge base with additional House dialogue.

### 3. Monitor Performance
Set up logging dashboard with `scripts/setup_monitoring.py` to track usage and performance.

### 4. Integration
Explore API mode with `python src/cli/main.py --api-server` for integration with other applications.

## Support

### Logs and Debugging
- Application logs: `./logs/housegpt.log`
- Conversation history: Available via database or export function
- Debug mode: Add `--debug` flag to any command

### Performance Monitoring
```bash
# Real-time performance monitoring
python scripts/monitor.py

# Generate performance report
python scripts/performance_report.py --days 7
```

### Configuration Validation
```bash
# Validate all settings and dependencies
python scripts/validate_config.py

# Test individual components
python scripts/test_component.py --component wake_word
python scripts/test_component.py --component rag_store
python scripts/test_component.py --component house_model
```

**Quick Start Complete!** Your HouseGPT assistant should now be ready to deliver sarcastic medical wisdom with the personality of Dr. Gregory House.
