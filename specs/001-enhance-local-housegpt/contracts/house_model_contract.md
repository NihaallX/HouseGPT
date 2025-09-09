# House Model Contract

**Component**: `house_model.py`  
**Purpose**: LoRA model loading and text generation  
**Version**: 1.0.0

## Interface Definition

### Class: `HouseModel`

#### Constructor
```python
def __init__(self, config: Configuration) -> None
```
**Parameters**:
- `config`: Application configuration object
**Raises**:
- `ModelLoadError`: If LoRA or base model fails to load
- `TokenizerError`: If tokenizer initialization fails
- `GPUError`: If CUDA setup fails (when GPU available)

#### Methods

##### `generate_response(user_input: str, context_quotes: List[HouseQuote]) -> str`
Generates House-style response using LoRA model with RAG context.

**Parameters**:
- `user_input`: User's message text
- `context_quotes`: Retrieved quotes for context injection
**Returns**:
- Generated response text (raw, unfiltered)
**Preconditions**:
- Model must be loaded and ready
- user_input must not be empty
- context_quotes must be valid HouseQuote objects
**Postconditions**:
- Response generated within token limits
- Context quotes influence generation style
**Raises**:
- `GenerationError`: If model generation fails
- `ValidationError`: If input parameters invalid
- `TokenLimitError`: If input exceeds maximum tokens

##### `is_model_loaded() -> bool`
Returns model loading status.

**Returns**:
- `True` if model ready for generation, `False` otherwise

##### `reload_model() -> None`
Reloads LoRA adapter (useful for hot-swapping weights).

**Preconditions**:
- Model path must exist and be valid
**Postconditions**:
- Fresh model loaded with current weights
- Previous model cleared from memory
**Raises**:
- `ModelLoadError`: If reload fails
- `FileNotFoundError`: If model path invalid

##### `get_model_info() -> Dict[str, Any]`
Returns model metadata and configuration.

**Returns**:
```python
{
    "base_model": str,
    "lora_path": str,
    "model_size_mb": float,
    "max_tokens": int,
    "device": str,  # "cpu" or "cuda"
    "load_time_ms": float
}
```

##### `estimate_tokens(text: str) -> int`
Estimates token count for input text.

**Parameters**:
- `text`: Text to analyze
**Returns**:
- Estimated token count
**Uses**: Tokenizer for accurate estimation

##### `set_generation_params(**kwargs) -> None`
Updates generation parameters at runtime.

**Parameters**:
- `max_length`: Maximum tokens to generate
- `temperature`: Randomness control (0.0-2.0)
- `top_p`: Nucleus sampling parameter
- `repetition_penalty`: Penalty for repeated tokens
**Raises**:
- `ValidationError`: If parameters out of valid range

## Generation Process

### Context Injection
1. Format retrieved quotes as conversation history
2. Inject user input as latest message
3. Apply House personality prompt template
4. Generate response with LoRA-enhanced model

### Prompt Template
```python
HOUSE_PROMPT = """
You are Dr. Gregory House, the brilliant but abrasive diagnostician. 
Respond in character with sharp wit, sarcasm, and medical insight.

Previous context from House:
{context_quotes}

Patient/Colleague: {user_input}
House:"""
```

### Generation Parameters
```python
{
    "max_length": 150,
    "temperature": 0.8,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id
}
```

## Model Architecture Contract

### Base Model Requirements
- **Type**: Causal language model (GPT-style)
- **Minimum Size**: 350M parameters
- **Compatible Architectures**: GPT-2, DialoGPT, GPT-J, LLaMA
- **Tokenizer**: Must support special tokens and padding

### LoRA Adapter Specifications
- **Target Modules**: Query, Key, Value projection layers
- **Rank**: 16-64 (configurable)
- **Alpha**: 32 (scaling parameter)
- **Dropout**: 0.1
- **Format**: Compatible with Hugging Face PEFT library

### Memory Requirements
- **Minimum RAM**: 4GB for base model + LoRA
- **GPU Memory**: 2GB VRAM (if GPU acceleration used)
- **Storage**: 1-2GB for model weights

## Error Handling

### Exception Hierarchy
```python
class HouseModelError(Exception): pass
class ModelLoadError(HouseModelError): pass
class TokenizerError(HouseModelError): pass
class GenerationError(HouseModelError): pass
class TokenLimitError(HouseModelError): pass
class GPUError(HouseModelError): pass
```

### Recovery Strategies
- **Out of memory**: Reduce batch size, clear cache, fallback to CPU
- **Generation timeout**: Interrupt generation, return partial result
- **Model corruption**: Reload from checkpoint
- **CUDA errors**: Fallback to CPU inference

## Performance Contract

- **Model Loading**: < 30 seconds on modern hardware
- **Generation Speed**: > 10 tokens/second on CPU, > 50 tokens/second on GPU
- **Memory Usage**: < 4GB RAM during inference
- **Response Latency**: < 5 seconds for 150-token response
- **Concurrent Requests**: Support 3 parallel generations (with proper memory management)

## Configuration Parameters

```python
{
    "model_path": "./housegpt-lora-large",
    "base_model_name": "microsoft/DialoGPT-medium",
    "device": "auto",  # "cpu", "cuda", or "auto"
    "torch_dtype": "float16",  # or "float32"
    "max_response_length": 150,
    "temperature": 0.8,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
    "context_window": 1024
}
```

## Testing Contract

### Unit Tests Required
- Model loading with valid LoRA weights
- Response generation with various inputs
- Context injection functionality
- Parameter validation and error handling
- Token estimation accuracy

### Integration Test Scenarios
- End-to-end generation pipeline with RAG context
- Memory usage monitoring during sustained generation
- Model hot-swapping capabilities
- Performance benchmarking across different hardware
- Error recovery under resource constraints

### Performance Benchmarks
- Generate 100 responses of 150 tokens each
- Measure mean, median, 95th percentile latency
- Monitor memory usage patterns
- Test with different context quote counts
- Validate output quality against reference responses

### Quality Metrics
- **Coherence**: Generated text must be grammatically correct
- **Character Consistency**: Responses should match House's personality
- **Context Relevance**: Should incorporate provided quotes appropriately
- **Length Compliance**: Respect configured token limits
- **Repetition Avoidance**: Minimal repeated phrases or words
