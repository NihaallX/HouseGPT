"""Voice Output Service for HouseGPT.

Text-to-speech synthesis with House MD voice cloning capabilities,
using XTTS for high-quality voice synthesis.
"""

import logging
import io
import time
import wave
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass

# Try to import TTS dependencies
try:
    import torch
    import numpy as np
    import soundfile as sf
    from TTS.api import TTS
    from TTS.utils.manage import ModelManager
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

# Try to import audio playback dependencies
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    try:
        import sounddevice as sd
        SOUNDDEVICE_AVAILABLE = True
    except ImportError:
        SOUNDDEVICE_AVAILABLE = False
        PYGAME_AVAILABLE = False

from ..models.house_response import HouseResponse


@dataclass
class AudioMetadata:
    """Metadata for synthesized audio."""
    duration: float
    sample_rate: int
    channels: int
    format: str
    file_size: int
    synthesis_time: float
    model_name: str
    voice_sample_used: str


class AudioDeviceError(Exception):
    """Raised when audio output device cannot be accessed."""
    pass


class TTSError(Exception):
    """Raised when text-to-speech synthesis fails."""
    pass


class PlaybackError(Exception):
    """Raised when audio playback fails."""
    pass


class ValidationError(Exception):
    """Raised when input validation fails."""
    pass


class ModelLoadError(Exception):
    """Raised when TTS model loading fails."""
    pass


class VoiceOutput:
    """Text-to-speech service with House voice cloning.
    
    Provides high-quality speech synthesis using XTTS with voice cloning
    capabilities to match House MD's voice characteristics.
    """
    
    def __init__(self, config):
        """Initialize voice output with TTS model and audio configuration.
        
        Args:
            config: Configuration object with audio settings
            
        Raises:
            AudioDeviceError: If audio output device not available
            TTSError: If TTS model cannot be loaded
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # TTS components
        self.tts_model = None
        self.model_name = None
        self.device = None
        
        # Voice cloning
        self.reference_audio_path = None
        self.voice_loaded = False
        
        # Audio playback
        self.playback_backend = None
        self.audio_device = None
        
        # Performance tracking
        self.synthesis_count = 0
        self.total_synthesis_time = 0.0
        self.total_audio_duration = 0.0
        
        # Model state
        self.is_loaded = False
        self.mock_mode = not TTS_AVAILABLE
        
        if self.mock_mode:
            self.logger.warning("TTS not available - using mock mode")
            self._init_mock_mode()
        else:
            self._init_tts_model()
            self._init_audio_playback()
    
    def _init_mock_mode(self):
        """Initialize mock mode for testing."""
        self.model_name = "mock-tts-model"
        self.is_loaded = True
        self.voice_loaded = True
        self.logger.info("Mock voice output initialized")
    
    def _init_tts_model(self):
        """Initialize TTS model for voice synthesis."""
        try:
            self.logger.info("Initializing TTS model...")
            
            # Determine device
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.logger.info(f"Using device: {self.device}")
            
            # Initialize TTS with XTTS model
            model_name = getattr(self.config.audio, 'tts_model', 'tts_models/multilingual/multi-dataset/xtts_v2')
            
            self.logger.info(f"Loading TTS model: {model_name}")
            self.tts_model = TTS(model_name, progress_bar=False).to(self.device)
            self.model_name = model_name
            
            # Load reference voice if available
            self._load_reference_voice()
            
            self.is_loaded = True
            self.logger.info("TTS model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize TTS model: {e}")
            raise TTSError(f"TTS model initialization failed: {e}")
    
    def _load_reference_voice(self):
        """Load reference voice for cloning."""
        try:
            voice_sample_path = getattr(self.config.audio, 'voice_sample_path', None)
            
            if not voice_sample_path:
                self.logger.warning("No voice sample path configured")
                return
            
            voice_path = Path(voice_sample_path)
            if not voice_path.exists():
                self.logger.warning(f"Voice sample not found: {voice_path}")
                return
            
            # Validate audio format
            try:
                # Check if file is readable as audio
                data, sample_rate = sf.read(str(voice_path))
                duration = len(data) / sample_rate
                
                self.logger.info(f"Voice sample loaded: {duration:.1f}s at {sample_rate}Hz")
                
                self.reference_audio_path = str(voice_path)
                self.voice_loaded = True
                
            except Exception as e:
                self.logger.warning(f"Could not load voice sample: {e}")
                
        except Exception as e:
            self.logger.warning(f"Voice loading failed: {e}")
    
    def _init_audio_playback(self):
        """Initialize audio playback system."""
        try:
            # Try pygame first
            if PYGAME_AVAILABLE:
                pygame.mixer.init()
                self.playback_backend = "pygame"
                self.logger.info("Audio playback initialized with pygame")
                
            elif SOUNDDEVICE_AVAILABLE:
                # Test sounddevice
                devices = sd.query_devices()
                default_output = sd.default.device[1]
                self.audio_device = default_output
                self.playback_backend = "sounddevice"
                self.logger.info(f"Audio playback initialized with sounddevice (device: {default_output})")
                
            else:
                self.logger.warning("No audio playback backend available")
                self.playback_backend = "none"
                
        except Exception as e:
            self.logger.warning(f"Audio playback initialization failed: {e}")
            self.playback_backend = "none"
    
    def synthesize_speech(self, response: HouseResponse) -> bytes:
        """Synthesize speech from response text.
        
        Args:
            response: HouseResponse containing text to synthesize
            
        Returns:
            Audio data as bytes (WAV format)
            
        Raises:
            ValidationError: If response is invalid
            TTSError: If synthesis fails
        """
        if not response or not response.text:
            raise ValidationError("Response text cannot be empty")
        
        if not self.is_loaded:
            raise TTSError("TTS model not loaded")
        
        text = response.text.strip()
        if not text:
            raise ValidationError("Response text cannot be empty after stripping")
        
        start_time = time.time()
        
        try:
            if self.mock_mode:
                # Generate mock audio data
                audio_data = self._generate_mock_audio(text)
                synthesis_time = time.time() - start_time
                
                self.synthesis_count += 1
                self.total_synthesis_time += synthesis_time
                
                return audio_data
            
            # Preprocess text for better synthesis
            processed_text = self._preprocess_text(text)
            
            # Generate speech
            if self.voice_loaded and self.reference_audio_path:
                # Use voice cloning
                self.logger.debug("Synthesizing with voice cloning")
                audio_array = self.tts_model.tts(
                    text=processed_text,
                    speaker_wav=self.reference_audio_path,
                    language="en"
                )
            else:
                # Use default voice
                self.logger.debug("Synthesizing with default voice")
                audio_array = self.tts_model.tts(
                    text=processed_text,
                    language="en"
                )
            
            # Convert to bytes
            audio_data = self._array_to_wav_bytes(audio_array)
            
            synthesis_time = time.time() - start_time
            duration = len(audio_array) / self.tts_model.synthesizer.output_sample_rate
            
            # Update statistics
            self.synthesis_count += 1
            self.total_synthesis_time += synthesis_time
            self.total_audio_duration += duration
            
            self.logger.info(f"Synthesized {duration:.1f}s audio in {synthesis_time:.2f}s")
            
            return audio_data
            
        except Exception as e:
            self.logger.error(f"Speech synthesis failed: {e}")
            raise TTSError(f"Failed to synthesize speech: {e}")
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for better TTS synthesis."""
        # Remove extra whitespace
        processed = " ".join(text.split())
        
        # Expand common abbreviations for better pronunciation
        replacements = {
            "Dr.": "Doctor",
            "Mr.": "Mister",
            "Mrs.": "Missus",
            "Ms.": "Miss",
            "vs.": "versus",
            "etc.": "et cetera",
            "i.e.": "that is",
            "e.g.": "for example",
            "MRI": "M R I",
            "CT": "C T",
            "ICU": "I C U",
            "ER": "E R",
            "MD": "M D"
        }
        
        for abbrev, expansion in replacements.items():
            processed = processed.replace(abbrev, expansion)
        
        # Handle medical terms that might be mispronounced
        medical_replacements = {
            "sarcoidosis": "sar-coy-do-sis",
            "vasculitis": "vas-cu-li-tis",
            "lupus": "lou-pus",
            "vicodin": "vi-co-din"
        }
        
        for term, pronunciation in medical_replacements.items():
            processed = processed.replace(term, pronunciation)
        
        return processed
    
    def _generate_mock_audio(self, text: str) -> bytes:
        """Generate mock audio data for testing."""
        # Create a simple sine wave audio
        sample_rate = 22050
        duration = max(1.0, len(text) * 0.1)  # Roughly 100ms per character
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        # Create a more complex waveform
        frequency = 200  # Base frequency
        audio = np.sin(2 * np.pi * frequency * t) * 0.3
        audio += np.sin(2 * np.pi * frequency * 1.5 * t) * 0.2  # Harmonic
        audio += np.random.normal(0, 0.05, len(audio))  # Add some noise
        
        # Convert to int16
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # Convert to WAV bytes
        with io.BytesIO() as wav_buffer:
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_int16.tobytes())
            
            return wav_buffer.getvalue()
    
    def _array_to_wav_bytes(self, audio_array: np.ndarray) -> bytes:
        """Convert audio array to WAV format bytes."""
        # Ensure audio is in the right format
        if audio_array.dtype != np.int16:
            # Normalize and convert to int16
            if audio_array.max() <= 1.0:
                audio_array = audio_array * 32767
            audio_array = audio_array.astype(np.int16)
        
        # Get sample rate from TTS model
        sample_rate = getattr(self.tts_model.synthesizer, 'output_sample_rate', 22050)
        
        # Create WAV bytes
        with io.BytesIO() as wav_buffer:
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_array.tobytes())
            
            return wav_buffer.getvalue()
    
    def play_audio(self, audio_data: bytes, wait: bool = True) -> bool:
        """Play audio data.
        
        Args:
            audio_data: WAV format audio data
            wait: Whether to wait for playback to complete
            
        Returns:
            True if playback started successfully
            
        Raises:
            PlaybackError: If playback fails
        """
        if not audio_data:
            raise ValidationError("Audio data cannot be empty")
        
        if self.playback_backend == "none":
            self.logger.warning("No audio playback backend available")
            return False
        
        try:
            if self.playback_backend == "pygame":
                return self._play_with_pygame(audio_data, wait)
            elif self.playback_backend == "sounddevice":
                return self._play_with_sounddevice(audio_data, wait)
            else:
                raise PlaybackError(f"Unknown playback backend: {self.playback_backend}")
                
        except Exception as e:
            self.logger.error(f"Audio playback failed: {e}")
            raise PlaybackError(f"Failed to play audio: {e}")
    
    def _play_with_pygame(self, audio_data: bytes, wait: bool) -> bool:
        """Play audio using pygame."""
        try:
            # Load audio data
            audio_buffer = io.BytesIO(audio_data)
            pygame.mixer.music.load(audio_buffer)
            
            # Play audio
            pygame.mixer.music.play()
            
            if wait:
                # Wait for playback to complete
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
            
            return True
            
        except Exception as e:
            raise PlaybackError(f"Pygame playback failed: {e}")
    
    def _play_with_sounddevice(self, audio_data: bytes, wait: bool) -> bool:
        """Play audio using sounddevice."""
        try:
            # Load WAV data
            with io.BytesIO(audio_data) as audio_buffer:
                data, sample_rate = sf.read(audio_buffer)
            
            # Play audio
            sd.play(data, sample_rate, device=self.audio_device)
            
            if wait:
                sd.wait()  # Wait until audio is finished
            
            return True
            
        except Exception as e:
            raise PlaybackError(f"Sounddevice playback failed: {e}")
    
    def speak_response(self, response: HouseResponse, play: bool = True) -> AudioMetadata:
        """Synthesize and optionally play response.
        
        Args:
            response: Response to speak
            play: Whether to play audio immediately
            
        Returns:
            AudioMetadata with synthesis information
            
        Raises:
            TTSError: If synthesis fails
            PlaybackError: If playback fails and play=True
        """
        start_time = time.time()
        
        # Synthesize speech
        audio_data = self.synthesize_speech(response)
        
        # Get audio metadata
        metadata = self._get_audio_metadata(audio_data, time.time() - start_time)
        
        # Play if requested
        if play:
            self.play_audio(audio_data, wait=False)
        
        return metadata
    
    def _get_audio_metadata(self, audio_data: bytes, synthesis_time: float) -> AudioMetadata:
        """Extract metadata from audio data."""
        try:
            with io.BytesIO(audio_data) as audio_buffer:
                with wave.open(audio_buffer, 'rb') as wav_file:
                    sample_rate = wav_file.getframerate()
                    channels = wav_file.getnchannels()
                    frames = wav_file.getnframes()
                    sample_width = wav_file.getsampwidth()
                    
                    duration = frames / sample_rate
                    
                    return AudioMetadata(
                        duration=duration,
                        sample_rate=sample_rate,
                        channels=channels,
                        format=f"WAV {sample_width * 8}-bit",
                        file_size=len(audio_data),
                        synthesis_time=synthesis_time,
                        model_name=self.model_name or "unknown",
                        voice_sample_used=self.reference_audio_path or "default"
                    )
                    
        except Exception as e:
            self.logger.warning(f"Could not extract audio metadata: {e}")
            return AudioMetadata(
                duration=0.0,
                sample_rate=22050,
                channels=1,
                format="WAV 16-bit",
                file_size=len(audio_data),
                synthesis_time=synthesis_time,
                model_name=self.model_name or "unknown",
                voice_sample_used=self.reference_audio_path or "default"
            )
    
    def save_audio(self, audio_data: bytes, filepath: Union[str, Path]) -> bool:
        """Save audio data to file.
        
        Args:
            audio_data: WAV format audio data
            filepath: Path to save audio file
            
        Returns:
            True if save successful
        """
        try:
            output_path = Path(filepath)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'wb') as f:
                f.write(audio_data)
            
            self.logger.info(f"Audio saved to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save audio: {e}")
            return False
    
    def stop_playback(self) -> bool:
        """Stop current audio playback.
        
        Returns:
            True if stop successful
        """
        try:
            if self.playback_backend == "pygame":
                pygame.mixer.music.stop()
                return True
            elif self.playback_backend == "sounddevice":
                sd.stop()
                return True
            else:
                return False
                
        except Exception as e:
            self.logger.warning(f"Failed to stop playback: {e}")
            return False
    
    def get_voice_stats(self) -> Dict[str, Any]:
        """Get voice output performance statistics.
        
        Returns:
            Dictionary with voice statistics
        """
        avg_synthesis_time = (
            self.total_synthesis_time / self.synthesis_count
            if self.synthesis_count > 0 else 0.0
        )
        
        avg_audio_duration = (
            self.total_audio_duration / self.synthesis_count
            if self.synthesis_count > 0 else 0.0
        )
        
        real_time_factor = (
            avg_audio_duration / avg_synthesis_time
            if avg_synthesis_time > 0 else 0.0
        )
        
        stats = {
            'is_loaded': self.is_loaded,
            'voice_loaded': self.voice_loaded,
            'synthesis_count': self.synthesis_count,
            'total_synthesis_time': self.total_synthesis_time,
            'total_audio_duration': self.total_audio_duration,
            'avg_synthesis_time': avg_synthesis_time,
            'avg_audio_duration': avg_audio_duration,
            'real_time_factor': real_time_factor,
            'model_name': self.model_name,
            'device': getattr(self, 'device', 'unknown'),
            'playback_backend': self.playback_backend,
            'voice_sample_path': self.reference_audio_path,
            'mock_mode': self.mock_mode
        }
        
        return stats
        """Synthesize speech audio from House response."""
        raise NotImplementedError("synthesize_speech implementation pending")
    
    def play_audio(self, audio_data: bytes) -> None:
        """Play audio data to configured output device."""
        raise NotImplementedError("play_audio implementation pending")
    
    def speak_response(self, response, async_playback: bool = False) -> None:
        """Synthesize and play House response."""
        raise NotImplementedError("speak_response implementation pending")
    
    def stop_playback(self) -> None:
        """Stop current audio playback."""
        raise NotImplementedError("stop_playback implementation pending")
    
    def get_audio_config(self):
        """Get current audio configuration."""
        raise NotImplementedError("get_audio_config implementation pending")
    
    def update_audio_config(self, config) -> None:
        """Update audio configuration settings."""
        raise NotImplementedError("update_audio_config implementation pending")
    
    def list_audio_devices(self) -> List[Dict[str, Any]]:
        """Get list of available audio output devices."""
        raise NotImplementedError("list_audio_devices implementation pending")
    
    def get_playback_status(self) -> Dict[str, Any]:
        """Get current playback status information."""
        raise NotImplementedError("get_playback_status implementation pending")
    
    def set_volume(self, volume: float) -> None:
        """Set audio output volume level."""
        raise NotImplementedError("set_volume implementation pending")
