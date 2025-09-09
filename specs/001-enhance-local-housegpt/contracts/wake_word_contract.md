# Wake Word Detection Contract

**Component**: `wake_word.py`  
**Purpose**: Continuous audio monitoring and "House?" detection  
**Version**: 1.0.0

## Interface Definition

### Class: `WakeWordDetector`

#### Constructor
```python
def __init__(self, config: Configuration) -> None
```
**Parameters**:
- `config`: Application configuration object
**Raises**:
- `AudioDeviceError`: If microphone not available
- `ModelLoadError`: If Whisper model fails to load

#### Methods

##### `start_listening() -> None`
Begins continuous audio monitoring in background thread.

**Preconditions**:
- Audio device must be available
- Whisper model must be loaded
**Postconditions**:
- Background thread actively monitoring audio
- Callback registered for wake word events
**Raises**:
- `RuntimeError`: If already listening
- `AudioDeviceError`: If microphone access fails

##### `stop_listening() -> None`
Stops audio monitoring and cleanup resources.

**Preconditions**:
- Must be currently listening
**Postconditions**:
- Audio thread terminated
- Resources cleaned up
**Side Effects**:
- All pending audio buffers cleared

##### `is_listening() -> bool`
Returns current monitoring status.

**Returns**:
- `True` if actively monitoring, `False` otherwise

##### `set_wake_word_callback(callback: Callable[[UserInput], None]) -> None`
Registers callback for wake word detection events.

**Parameters**:
- `callback`: Function to call when "House?" detected
**Raises**:
- `TypeError`: If callback not callable

##### `get_detection_stats() -> Dict[str, Any]`
Returns performance metrics.

**Returns**:
```python
{
    "total_detections": int,
    "false_positives": int,
    "avg_confidence": float,
    "uptime_seconds": int
}
```

## Event Callbacks

### Wake Word Detected
```python
def on_wake_word_detected(user_input: UserInput) -> None
```
**Triggered**: When "House?" detected with confidence > threshold  
**Payload**: UserInput object with audio data and metadata

## Error Handling

### Exception Hierarchy
```python
class WakeWordError(Exception): pass
class AudioDeviceError(WakeWordError): pass
class ModelLoadError(WakeWordError): pass
class DetectionTimeoutError(WakeWordError): pass
```

### Recovery Strategies
- **Audio device lost**: Attempt reconnection every 5 seconds
- **Model error**: Fallback to lower-quality model
- **Memory overflow**: Clear buffers and restart detection

## Performance Contract

- **Detection Latency**: < 2 seconds from speech to callback
- **Memory Usage**: < 100MB sustained
- **CPU Usage**: < 10% average on modern hardware
- **False Positive Rate**: < 5% under normal conditions

## Configuration Parameters

```python
{
    "wake_word_sensitivity": 0.7,      # Detection threshold
    "audio_chunk_size": 1024,          # Buffer size in samples
    "sample_rate": 16000,              # Audio sample rate
    "detection_timeout": 30,           # Max processing time per chunk
    "whisper_model": "base"            # Model size (tiny/base/small)
}
```

## Testing Contract

### Unit Tests Required
- Wake word detection accuracy with test audio files
- False positive rejection with random speech
- Performance under sustained load
- Error recovery scenarios

### Integration Test Scenarios
- End-to-end detection with live microphone
- Callback integration with downstream components
- Resource cleanup on shutdown
- Configuration parameter validation
