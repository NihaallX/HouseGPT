"""Voice output service for HouseGPT text-to-speech synthesis.

Provides voice synthesis capabilities using Edge-TTS with House-like voice characteristics.
Supports audio playback and voice customization for authentic House personality delivery.
"""

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List
import io

try:
    import edge_tts
    import pygame
    import soundfile as sf
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False


class VoiceOutputError(Exception):
    """Base exception for voice output operations."""
    pass


class TTSError(VoiceOutputError):
    """Raised when text-to-speech synthesis fails."""
    pass


class AudioPlaybackError(VoiceOutputError):
    """Raised when audio playback fails."""
    pass


class VoiceOutput:
    """Voice synthesis and audio playback service for House responses.
    
    Uses Edge-TTS for high-quality speech synthesis with House-like voice characteristics.
    Supports real-time playback and voice customization.
    """
    
    def __init__(self, config):
        """Initialize voice output service.
        
        Args:
            config: Configuration object with voice settings
            
        Raises:
            VoiceOutputError: If voice synthesis initialization fails
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Voice configuration
        self.voice_name = getattr(config, 'voice_name', 'en-US-AndrewMultilingualNeural')  # Deep, authoritative male voice
        self.speaking_rate = getattr(config, 'voice_rate', '+0%')  # Normal speed
        self.pitch = getattr(config, 'voice_pitch', '-10%')  # Slightly lower pitch for authority
        self.volume = getattr(config, 'voice_volume', '+0%')  # Normal volume
        
        # Audio settings
        self.sample_rate = getattr(config, 'audio_sample_rate', 24000)
        self.output_format = getattr(config, 'audio_format', 'wav')
        
        # Initialize audio system
        self._init_audio_system()
        
        # Cache for voice samples
        self.voice_cache: Dict[str, bytes] = {}
        self.cache_enabled = getattr(config, 'voice_cache_enabled', True)
        
        if not VOICE_AVAILABLE:
            self.logger.warning("Voice dependencies not available - using mock mode")
            self.mock_mode = True
        else:
            self.mock_mode = False
            self.logger.info(f"Voice output initialized with {self.voice_name}")
    
    def _init_audio_system(self):
        """Initialize pygame audio system."""
        if not VOICE_AVAILABLE:
            return
        
        try:
            pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffer=1024)
            pygame.mixer.init()
            self.logger.info("Audio system initialized")
        except Exception as e:
            self.logger.warning(f"Audio system initialization failed: {e}")
    
    async def synthesize_speech(self, text: str) -> bytes:
        """Synthesize speech from text using Edge-TTS.
        
        Args:
            text: Text to synthesize
            
        Returns:
            Audio data as bytes
            
        Raises:
            TTSError: If synthesis fails
        """
        if self.mock_mode:
            self.logger.info(f"Mock TTS: {text[:50]}...")
            return b"mock_audio_data"
        
        # Check cache first
        cache_key = f"{text}_{self.voice_name}_{self.speaking_rate}_{self.pitch}"
        if self.cache_enabled and cache_key in self.voice_cache:
            self.logger.debug("Using cached voice synthesis")
            return self.voice_cache[cache_key]
        
        try:
            # Create SSML for voice customization
            ssml_text = self._create_ssml(text)
            
            # Create TTS communicator
            communicator = edge_tts.Communicate(ssml_text, self.voice_name)
            
            # Generate audio
            audio_data = b""
            async for chunk in communicator.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]
            
            if not audio_data:
                raise TTSError("No audio data generated")
            
            # Cache the result
            if self.cache_enabled:
                self.voice_cache[cache_key] = audio_data
            
            self.logger.debug(f"Synthesized {len(audio_data)} bytes of audio")
            return audio_data
            
        except Exception as e:
            raise TTSError(f"Speech synthesis failed: {e}")
    
    def _create_ssml(self, text: str) -> str:
        """Create SSML markup for voice customization.
        
        Args:
            text: Plain text to convert
            
        Returns:
            SSML formatted text with House-like voice characteristics
        """
        # Add House-like delivery characteristics
        # - Slower, more deliberate pace for key points
        # - Emphasis on sarcastic phrases
        # - Pauses for dramatic effect
        
        # Process text for House-style delivery
        processed_text = text
        
        # Add emphasis to sarcastic phrases
        sarcastic_phrases = [
            "obviously", "clearly", "brilliant", "genius", "idiot", "moron",
            "fascinating", "interesting", "shocking", "amazing"
        ]
        
        for phrase in sarcastic_phrases:
            if phrase in processed_text.lower():
                processed_text = processed_text.replace(
                    phrase, f'<emphasis level="strong">{phrase}</emphasis>'
                )
        
        # Add pauses after certain punctuation for dramatic effect
        processed_text = processed_text.replace("...", '<break time="1s"/>')
        processed_text = processed_text.replace(". ", '. <break time="300ms"/>')
        processed_text = processed_text.replace("? ", '? <break time="200ms"/>')
        processed_text = processed_text.replace("! ", '! <break time="300ms"/>')
        
        # Create SSML
        ssml = f'''
        <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">
            <voice name="{self.voice_name}">
                <prosody rate="{self.speaking_rate}" pitch="{self.pitch}" volume="{self.volume}">
                    {processed_text}
                </prosody>
            </voice>
        </speak>
        '''
        
        return ssml.strip()
    
    def play_audio(self, audio_data: bytes) -> bool:
        """Play audio data through speakers.
        
        Args:
            audio_data: Audio data to play (MP3 format from Edge-TTS)
            
        Returns:
            True if playback successful, False otherwise
        """
        if self.mock_mode:
            self.logger.info("Mock audio playback")
            return True
        
        try:
            # Save audio to temporary file with .mp3 extension
            # Edge-TTS outputs MP3 format
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_path = temp_file.name
            
            # Play using pygame
            pygame.mixer.music.load(temp_path)
            pygame.mixer.music.play()
            
            # Wait for playback to complete
            while pygame.mixer.music.get_busy():
                pygame.time.wait(100)
            
            # Clean up
            os.unlink(temp_path)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Audio playback failed: {e}")
            return False
    
    async def speak(self, text: str) -> bool:
        """Synthesize and immediately play speech.
        
        Args:
            text: Text to speak
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Synthesize speech
            audio_data = await self.synthesize_speech(text)
            
            # Play audio
            return self.play_audio(audio_data)
            
        except Exception as e:
            self.logger.error(f"Speech output failed: {e}")
            return False
    
    def get_available_voices(self) -> List[str]:
        """Get list of available Edge-TTS voices.
        
        Returns:
            List of voice names suitable for House character
        """
        if self.mock_mode:
            return ["mock-voice"]
        
        # House-like voices (deep, authoritative, male)
        house_voices = [
            "en-US-AndrewMultilingualNeural",  # Deep, authoritative
            "en-US-BrianNeural",              # Mature, serious
            "en-US-ChristopherNeural",        # Professional, confident
            "en-US-EricNeural",               # Calm, authoritative
            "en-US-GuyNeural",                # Deep, mature
            "en-US-RogerNeural",              # Sophisticated, dry
        ]
        
        return house_voices
    
    def set_voice(self, voice_name: str) -> bool:
        """Change the TTS voice.
        
        Args:
            voice_name: Name of the voice to use
            
        Returns:
            True if voice was set successfully
        """
        try:
            available_voices = self.get_available_voices()
            if voice_name in available_voices:
                self.voice_name = voice_name
                self.logger.info(f"Voice changed to {voice_name}")
                return True
            else:
                self.logger.warning(f"Voice {voice_name} not available")
                return False
        except Exception as e:
            self.logger.error(f"Failed to set voice: {e}")
            return False
    
    def adjust_voice_parameters(self, rate: str = None, pitch: str = None, volume: str = None):
        """Adjust voice synthesis parameters.
        
        Args:
            rate: Speaking rate (e.g., "+20%", "-10%")
            pitch: Voice pitch (e.g., "+5%", "-15%") 
            volume: Voice volume (e.g., "+10%", "-5%")
        """
        if rate is not None:
            self.speaking_rate = rate
        if pitch is not None:
            self.pitch = pitch
        if volume is not None:
            self.volume = volume
            
        self.logger.info(f"Voice parameters updated: rate={self.speaking_rate}, pitch={self.pitch}, volume={self.volume}")
    
    def get_voice_stats(self) -> Dict[str, Any]:
        """Get voice synthesis statistics.
        
        Returns:
            Dictionary with voice service statistics
        """
        return {
            'voice_name': self.voice_name,
            'speaking_rate': self.speaking_rate,
            'pitch': self.pitch,
            'volume': self.volume,
            'cache_size': len(self.voice_cache),
            'cache_enabled': self.cache_enabled,
            'mock_mode': self.mock_mode,
            'audio_available': VOICE_AVAILABLE
        }
    
    def clear_cache(self):
        """Clear the voice synthesis cache."""
        self.voice_cache.clear()
        self.logger.info("Voice cache cleared")
    
    def shutdown(self):
        """Shutdown voice output service."""
        if VOICE_AVAILABLE:
            try:
                pygame.mixer.quit()
                self.logger.info("Voice output service shutdown")
            except Exception as e:
                self.logger.error(f"Error during shutdown: {e}")
        
        self.clear_cache()
