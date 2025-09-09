# Voice Output Contract

**Component**: `voice_output.py`  
**Purpose**: Text-to-speech synthesis with House-like voice  
**Version**: 1.0.0

## Interface Definition

### Class: `VoiceOutput`

#### Constructor
```python
def __init__(self, config: Configuration) -> None
```
**Parameters**:
- `config`: Application configuration object
**Raises**:
- `TTSModelError`: If XTTS model fails to load
- `AudioDeviceError`: If audio output device not available
- `VoiceCloningError`: If House voice sample not found

#### Methods

##### `synthesize_speech(text: str, output_path: Optional[str] = None) -> bytes`
Converts text to House-like speech audio.

**Parameters**:
- `text`: Text content to synthesize
- `output_path`: Optional file path to save audio (WAV format)
**Returns**:
- Audio data as bytes (WAV format)
**Preconditions**:
- Text must not be empty and < 500 characters
- XTTS model must be loaded
**Postconditions**:
- Audio generated with House voice characteristics
- If output_path provided, file saved to disk
**Raises**:
- `SynthesisError`: If speech generation fails
- `ValidationError`: If text parameters invalid
- `FileSystemError`: If file save fails

##### `play_audio(audio_data: bytes) -> None`
Plays audio through system speakers.

**Parameters**:
- `audio_data`: WAV audio bytes to play
**Preconditions**:
- Audio device must be available
- audio_data must be valid WAV format
**Postconditions**:
- Audio played through default output device
**Side Effects**:
- Blocks until playback complete
**Raises**:
- `AudioDeviceError`: If playback device unavailable
- `PlaybackError`: If audio format unsupported

##### `play_text(text: str) -> None`
Convenience method: synthesize and immediately play text.

**Parameters**:
- `text`: Text to convert to speech and play
**Behavior**:
- Combines synthesize_speech() and play_audio()
- Uses temporary audio buffer
**Raises**:
- All exceptions from synthesize_speech() and play_audio()

##### `set_voice_settings(**kwargs) -> None`
Updates voice synthesis parameters.

**Parameters**:
- `speed`: Speech rate multiplier (0.5-2.0)
- `pitch`: Pitch adjustment in semitones (-12 to +12)
- `energy`: Vocal energy/intensity (0.1-2.0)
- `temperature`: Randomness in synthesis (0.1-1.0)
**Raises**:
- `ValidationError`: If parameters out of valid range

##### `is_model_ready() -> bool`
Returns TTS model loading status.

**Returns**:
- `True` if ready for synthesis, `False` otherwise

##### `get_voice_info() -> Dict[str, Any]`
Returns voice model information and settings.

**Returns**:
```python
{
    "model_name": str,
    "sample_rate": int,
    "voice_sample_path": str,
    "model_size_mb": float,
    "supported_languages": List[str],
    "current_settings": Dict[str, float]
}
```

##### `preload_model() -> None`
Preloads TTS model for faster first synthesis.

**Behavior**:
- Loads model into memory
- Initializes voice cloning parameters
- Caches frequently used components
**Raises**:
- `TTSModelError`: If model loading fails

## Voice Synthesis Process

### XTTS Integration
1. Load XTTS v2 model with House voice sample
2. Configure synthesis parameters for character voice
3. Generate speech with proper prosody and intonation
4. Post-process audio for consistency and quality

### House Voice Characteristics
- **Pitch Range**: Lower register (male voice, slight rasp)
- **Speech Rate**: Moderate to fast (confident delivery)
- **Intonation**: Sarcastic rises, dismissive falls
- **Emphasis**: Strong stress on key sarcastic words
- **Pauses**: Strategic pauses before dramatic reveals

### Audio Quality Standards
- **Sample Rate**: 22050 Hz (high quality)
- **Bit Depth**: 16-bit (CD quality)
- **Format**: WAV (uncompressed)
- **Duration**: Matches natural speech timing
- **Noise Floor**: < -40dB (clean audio)

## Voice Sample Requirements

### Reference Audio
- **Source**: High-quality House M.D. dialogue clips
- **Duration**: 5-10 seconds of clear speech
- **Content**: Representative of character voice
- **Quality**: Studio quality, minimal background noise
- **Format**: WAV or high-quality MP3

### Sample Processing
```python
VOICE_SAMPLE_CONFIG = {
    "sample_rate": 22050,
    "trim_silence": True,
    "normalize_volume": True,
    "noise_reduction": True,
    "voice_activity_threshold": 0.5
}
```

## Error Handling

### Exception Hierarchy
```python
class VoiceOutputError(Exception): pass
class TTSModelError(VoiceOutputError): pass
class SynthesisError(VoiceOutputError): pass
class AudioDeviceError(VoiceOutputError): pass
class PlaybackError(VoiceOutputError): pass
class VoiceCloningError(VoiceOutputError): pass
class FileSystemError(VoiceOutputError): pass
```

### Recovery Strategies
- **Model loading failure**: Fallback to system TTS
- **Voice cloning failure**: Use generic male voice
- **Audio device unavailable**: Save to file instead of playing
- **Synthesis timeout**: Cancel and return error
- **Memory overflow**: Clear audio cache and retry

### Graceful Degradation
1. **Primary**: XTTS with House voice cloning
2. **Fallback 1**: XTTS with generic male voice
3. **Fallback 2**: System TTS (Windows SAPI, macOS say, Linux espeak)
4. **Final**: Text-only output (no audio)

## Performance Contract

- **Synthesis Speed**: Real-time or faster (1x speech rate minimum)
- **Model Loading**: < 15 seconds for XTTS initialization
- **Memory Usage**: < 2GB for model and audio buffers
- **Audio Latency**: < 1 second from text to audio start
- **Quality Consistency**: 95% voice similarity across generations
- **File I/O**: < 100ms for audio file operations

## Configuration Parameters

```python
{
    "model_name": "tts_models/multilingual/multi-dataset/xtts_v2",
    "voice_sample_path": "./assets/house_voice_sample.wav",
    "speaker_wav": "./assets/house_voice_sample.wav",
    "language": "en",
    "speed": 1.0,
    "pitch": 0,
    "energy": 1.2,
    "temperature": 0.7,
    "sample_rate": 22050,
    "output_format": "wav",
    "audio_device_index": None,  # Default device
    "enable_voice_cloning": True,
    "cache_audio": True,
    "max_cache_size_mb": 100
}
```

## Testing Contract

### Unit Tests Required
- Text synthesis with various input lengths
- Voice parameter adjustment validation
- Audio format and quality verification
- Error handling for invalid inputs
- Performance benchmarking

### Integration Test Scenarios
- End-to-end text-to-speech pipeline
- Audio playback on different devices
- Voice cloning quality assessment
- Resource usage monitoring
- Cross-platform compatibility testing

### Quality Assurance
- **Voice Similarity**: Compare generated audio to reference samples
- **Intelligibility**: Transcription accuracy test
- **Character Consistency**: Voice matches House personality
- **Audio Quality**: Frequency analysis for artifacts
- **Timing Accuracy**: Speech rate matches expected values

### Performance Benchmarks
- Synthesize 100 responses of varying lengths
- Measure synthesis time vs. audio duration
- Monitor memory usage during sustained operation
- Test concurrent synthesis requests
- Validate quality degradation under load

### Audio Test Suite
- Sample texts representing different House responses
- Edge cases (very short, very long, special characters)
- Stress testing with rapid successive generations
- Quality regression testing with reference audio
