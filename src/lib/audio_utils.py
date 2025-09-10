"""Audio utilities for HouseGPT real-time processing.

Provides audio stream management, noise reduction, and format conversion
utilities for wake word detection and voice output.
"""

import logging
import numpy as np
import time
import wave
from typing import Optional, Tuple, Generator, Dict
from pathlib import Path
import io

# Audio processing dependencies
try:
    import soundfile as sf
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

try:
    import webrtcvad
    VAD_AVAILABLE = True
except ImportError:
    VAD_AVAILABLE = False


class AudioStream:
    """Real-time audio stream handler for continuous recording."""
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 channels: int = 1,
                 chunk_size: int = 1024,
                 device: Optional[int] = None):
        """Initialize audio stream.
        
        Args:
            sample_rate: Audio sample rate in Hz
            channels: Number of audio channels
            chunk_size: Size of audio chunks for processing
            device: Audio device ID (None for default)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        self.stream = None
        self.is_recording = False
        
        if not AUDIO_AVAILABLE:
            self.logger.warning("Audio dependencies not available")
    
    def start_recording(self) -> bool:
        """Start continuous audio recording.
        
        Returns:
            True if recording started successfully
        """
        if not AUDIO_AVAILABLE:
            self.logger.error("Cannot start recording: audio dependencies missing")
            return False
        
        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                device=self.device,
                blocksize=self.chunk_size,
                dtype=np.float32
            )
            self.stream.start()
            self.is_recording = True
            self.logger.info(f"Started audio recording at {self.sample_rate}Hz")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start recording: {e}")
            return False
    
    def stop_recording(self):
        """Stop audio recording and cleanup."""
        if self.stream and self.is_recording:
            self.stream.stop()
            self.stream.close()
            self.is_recording = False
            self.logger.info("Stopped audio recording")
    
    def read_chunk(self) -> Optional[np.ndarray]:
        """Read one chunk of audio data.
        
        Returns:
            Audio chunk as numpy array, or None if error
        """
        if not self.stream or not self.is_recording:
            return None
        
        try:
            data, overflowed = self.stream.read(self.chunk_size)
            if overflowed:
                self.logger.warning("Audio buffer overflow detected")
            return data.flatten()
            
        except Exception as e:
            self.logger.error(f"Error reading audio chunk: {e}")
            return None
    
    def read_chunks(self, duration: float) -> Generator[np.ndarray, None, None]:
        """Generate audio chunks for specified duration.
        
        Args:
            duration: Recording duration in seconds
            
        Yields:
            Audio chunks as numpy arrays
        """
        if not self.is_recording:
            if not self.start_recording():
                return
        
        chunks_needed = int(duration * self.sample_rate / self.chunk_size)
        
        for _ in range(chunks_needed):
            chunk = self.read_chunk()
            if chunk is not None:
                yield chunk
            time.sleep(self.chunk_size / self.sample_rate)


class AudioProcessor:
    """Audio signal processing utilities."""
    
    def __init__(self, sample_rate: int = 16000):
        """Initialize audio processor.
        
        Args:
            sample_rate: Audio sample rate for processing
        """
        self.sample_rate = sample_rate
        self.logger = logging.getLogger(__name__)
        
        # Initialize VAD if available
        self.vad = None
        if VAD_AVAILABLE:
            self.vad = webrtcvad.Vad(2)  # Aggressiveness level 2
    
    def detect_voice_activity(self, audio_chunk: np.ndarray) -> bool:
        """Detect if audio chunk contains voice activity.
        
        Args:
            audio_chunk: Audio data as numpy array
            
        Returns:
            True if voice activity detected
        """
        if not VAD_AVAILABLE or self.vad is None:
            # Fallback: simple energy-based detection
            energy = np.sum(audio_chunk ** 2)
            threshold = 0.01  # Adjust based on testing
            return energy > threshold
        
        try:
            # Convert to 16-bit PCM for webrtcvad
            pcm_data = (audio_chunk * 32767).astype(np.int16).tobytes()
            
            # webrtcvad expects specific frame sizes
            frame_duration = 30  # ms
            frame_size = int(self.sample_rate * frame_duration / 1000)
            
            if len(pcm_data) < frame_size * 2:  # *2 for 16-bit
                return False
            
            # Use first frame for detection
            frame = pcm_data[:frame_size * 2]
            return self.vad.is_speech(frame, self.sample_rate)
            
        except Exception as e:
            self.logger.error(f"VAD detection failed: {e}")
            return False
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to prevent clipping.
        
        Args:
            audio: Raw audio data
            
        Returns:
            Normalized audio data
        """
        if len(audio) == 0:
            return audio
        
        # Remove DC offset
        audio = audio - np.mean(audio)
        
        # Normalize to [-1, 1] range
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.95  # Leave some headroom
        
        return audio
    
    def apply_noise_gate(self, audio: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """Apply noise gate to reduce background noise.
        
        Args:
            audio: Input audio data
            threshold: Amplitude threshold for gate
            
        Returns:
            Gated audio data
        """
        # Simple noise gate: zero out samples below threshold
        mask = np.abs(audio) > threshold
        return audio * mask
    
    def convert_to_mono(self, audio: np.ndarray) -> np.ndarray:
        """Convert stereo audio to mono.
        
        Args:
            audio: Input audio (possibly stereo)
            
        Returns:
            Mono audio data
        """
        if audio.ndim == 1:
            return audio
        elif audio.ndim == 2:
            return np.mean(audio, axis=1)
        else:
            self.logger.warning(f"Unexpected audio shape: {audio.shape}")
            return audio.flatten()


class AudioFileHandler:
    """File I/O operations for audio data."""
    
    def __init__(self):
        """Initialize audio file handler."""
        self.logger = logging.getLogger(__name__)
    
    def save_audio(self, 
                   audio: np.ndarray, 
                   filepath: Path, 
                   sample_rate: int = 16000) -> bool:
        """Save audio data to file.
        
        Args:
            audio: Audio data to save
            filepath: Output file path
            sample_rate: Audio sample rate
            
        Returns:
            True if saved successfully
        """
        if not AUDIO_AVAILABLE:
            self.logger.error("Cannot save audio: soundfile not available")
            return False
        
        try:
            sf.write(str(filepath), audio, sample_rate)
            self.logger.info(f"Saved audio to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save audio: {e}")
            return False
    
    def load_audio(self, filepath: Path) -> Tuple[Optional[np.ndarray], Optional[int]]:
        """Load audio data from file.
        
        Args:
            filepath: Audio file path
            
        Returns:
            Tuple of (audio_data, sample_rate) or (None, None) if error
        """
        if not AUDIO_AVAILABLE:
            self.logger.error("Cannot load audio: soundfile not available")
            return None, None
        
        try:
            audio, sample_rate = sf.read(str(filepath))
            self.logger.info(f"Loaded audio from {filepath}")
            return audio, sample_rate
            
        except Exception as e:
            self.logger.error(f"Failed to load audio: {e}")
            return None, None
    
    def audio_to_wav_bytes(self, 
                          audio: np.ndarray, 
                          sample_rate: int = 16000) -> bytes:
        """Convert audio array to WAV format bytes.
        
        Args:
            audio: Audio data
            sample_rate: Audio sample rate
            
        Returns:
            WAV format bytes
        """
        buffer = io.BytesIO()
        
        try:
            # Ensure audio is in correct format for wav
            if audio.dtype != np.int16:
                # Convert float to 16-bit PCM
                audio_int16 = (audio * 32767).astype(np.int16)
            else:
                audio_int16 = audio
            
            # Write WAV file to buffer
            with wave.open(buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_int16.tobytes())
            
            return buffer.getvalue()
            
        except Exception as e:
            self.logger.error(f"Failed to convert audio to WAV: {e}")
            return b""


def get_audio_devices() -> Dict[int, str]:
    """Get list of available audio input devices.
    
    Returns:
        Dictionary mapping device ID to device name
    """
    if not AUDIO_AVAILABLE:
        return {}
    
    try:
        devices = sd.query_devices()
        input_devices = {}
        
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                input_devices[i] = device['name']
        
        return input_devices
        
    except Exception as e:
        logging.error(f"Failed to query audio devices: {e}")
        return {}


def test_audio_setup() -> bool:
    """Test if audio recording setup is working.
    
    Returns:
        True if audio setup is functional
    """
    if not AUDIO_AVAILABLE:
        print("❌ Audio dependencies not available")
        print("📦 Install with: pip install soundfile sounddevice")
        return False
    
    try:
        # Test recording a short sample
        print("🎤 Testing audio recording...")
        stream = AudioStream(sample_rate=16000, chunk_size=1024)
        
        if not stream.start_recording():
            print("❌ Failed to start audio recording")
            return False
        
        # Record for 1 second
        chunks = []
        for chunk in stream.read_chunks(1.0):
            chunks.append(chunk)
        
        stream.stop_recording()
        
        if chunks:
            total_samples = sum(len(chunk) for chunk in chunks)
            print(f"✅ Audio recording successful: {total_samples} samples recorded")
            return True
        else:
            print("❌ No audio data recorded")
            return False
            
    except Exception as e:
        print(f"❌ Audio test failed: {e}")
        return False


if __name__ == "__main__":
    # Test audio setup when run directly
    test_audio_setup()
