"""Contract tests for VoiceOutput service.

These tests validate the contract defined in contracts/voice_output_contract.md.
Tests MUST fail initially before implementation exists.
"""

import pytest
import wave
import io
from unittest.mock import Mock, patch
from src.services.voice_output import VoiceOutput, AudioDeviceError, TTSError, PlaybackError
from src.models.response import HouseResponse
from src.models.audio_config import AudioConfig
from src.lib.config import Configuration


class TestVoiceOutputContract:
    """Test VoiceOutput contract compliance."""
    
    @pytest.fixture
    def config(self):
        """Provide test configuration."""
        config = Configuration()
        config.audio.tts_model = "tts_models/en/ljspeech/tacotron2-DDC"
        config.audio.voice_clone_path = "./voice_samples/house_voice.wav"
        config.audio.sample_rate = 22050
        config.audio.bit_depth = 16
        config.audio.speed_factor = 1.0
        config.audio.pitch_adjustment = 0
        config.audio.output_device = "default"
        return config
    
    @pytest.fixture
    def voice_output(self, config):
        """Provide VoiceOutput instance."""
        return VoiceOutput(config)
    
    def test_constructor_requires_config(self):
        """Test constructor requires Configuration object."""
        with pytest.raises(TypeError):
            VoiceOutput()
    
    def test_constructor_raises_audio_device_error_if_no_output(self, config):
        """Test constructor raises AudioDeviceError if no audio output available."""
        config.audio.output_device = "nonexistent_device"
        with pytest.raises(AudioDeviceError):
            VoiceOutput(config)
    
    def test_constructor_raises_tts_error_if_model_missing(self, config):
        """Test constructor raises TTSError if TTS model not found."""
        config.audio.tts_model = "invalid/nonexistent-model"
        with pytest.raises(TTSError):
            VoiceOutput(config)
    
    def test_synthesize_speech_returns_audio_data(self, voice_output):
        """Test synthesize_speech returns valid audio data."""
        response = HouseResponse(
            text="Everybody lies, especially patients.",
            confidence_score=0.85,
            sarcasm_level=4,
            emotional_tone="condescending",
            house_authenticity=0.9
        )
        
        audio_data = voice_output.synthesize_speech(response)
        
        # Should return bytes representing audio
        assert isinstance(audio_data, bytes)
        assert len(audio_data) > 0
        
        # Should be valid WAV format
        audio_stream = io.BytesIO(audio_data)
        with wave.open(audio_stream, 'rb') as wav_file:
            assert wav_file.getnchannels() in [1, 2]  # Mono or stereo
            assert wav_file.getsampwidth() == 2  # 16-bit
            assert wav_file.getframerate() == 22050  # Sample rate
    
    def test_synthesize_speech_raises_validation_error_for_invalid_response(self, voice_output):
        """Test synthesize_speech raises ValidationError for invalid response."""
        with pytest.raises(Exception):  # Should be ValidationError
            voice_output.synthesize_speech(None)
        
        with pytest.raises(Exception):  # Should be ValidationError
            voice_output.synthesize_speech("invalid response type")
    
    def test_synthesize_speech_raises_tts_error_on_synthesis_failure(self, voice_output):
        """Test synthesize_speech raises TTSError on synthesis failure."""
        response = HouseResponse(
            text="",  # Empty text might cause TTS failure
            confidence_score=0.8,
            sarcasm_level=3,
            emotional_tone="analytical",
            house_authenticity=0.8
        )
        
        with pytest.raises(TTSError):
            voice_output.synthesize_speech(response)
    
    def test_synthesize_speech_applies_voice_cloning(self, voice_output):
        """Test synthesize_speech applies House voice characteristics."""
        response = HouseResponse(
            text="It's never lupus... except when it is.",
            confidence_score=0.9,
            sarcasm_level=5,
            emotional_tone="witty",
            house_authenticity=0.95
        )
        
        audio_data = voice_output.synthesize_speech(response)
        
        # Audio should be generated (can't easily test voice characteristics)
        assert len(audio_data) > 1000  # Should be substantial audio data
    
    def test_synthesize_speech_adjusts_prosody_for_sarcasm(self, voice_output):
        """Test synthesize_speech adjusts prosody based on sarcasm level."""
        high_sarcasm = HouseResponse(
            text="Oh, brilliant diagnosis there.",
            sarcasm_level=5,
            confidence_score=0.8,
            emotional_tone="condescending",
            house_authenticity=0.8
        )
        
        low_sarcasm = HouseResponse(
            text="That's a reasonable observation.",
            sarcasm_level=1,
            confidence_score=0.8,
            emotional_tone="analytical",
            house_authenticity=0.8
        )
        
        high_audio = voice_output.synthesize_speech(high_sarcasm)
        low_audio = voice_output.synthesize_speech(low_sarcasm)
        
        # Both should produce audio (prosody differences hard to test)
        assert len(high_audio) > 0
        assert len(low_audio) > 0
    
    def test_play_audio_outputs_to_speakers(self, voice_output):
        """Test play_audio sends audio to configured output device."""
        # Create minimal valid WAV data
        audio_data = self._create_test_wav_data()
        
        # Should complete without error
        voice_output.play_audio(audio_data)
    
    def test_play_audio_raises_validation_error_for_invalid_data(self, voice_output):
        """Test play_audio raises ValidationError for invalid audio data."""
        with pytest.raises(Exception):  # Should be ValidationError
            voice_output.play_audio(None)
        
        with pytest.raises(Exception):  # Should be ValidationError
            voice_output.play_audio(b"invalid audio data")
    
    def test_play_audio_raises_playback_error_on_device_failure(self, voice_output):
        """Test play_audio raises PlaybackError on audio device failure."""
        audio_data = self._create_test_wav_data()
        
        with patch('pyaudio.PyAudio') as mock_pyaudio:
            mock_stream = Mock()
            mock_stream.write.side_effect = Exception("Audio device disconnected")
            mock_pyaudio.return_value.open.return_value = mock_stream
            
            with pytest.raises(PlaybackError):
                voice_output.play_audio(audio_data)
    
    def test_speak_response_synthesizes_and_plays(self, voice_output):
        """Test speak_response combines synthesis and playback."""
        response = HouseResponse(
            text="The patient is lying, as usual.",
            confidence_score=0.8,
            sarcasm_level=3,
            emotional_tone="analytical",
            house_authenticity=0.85
        )
        
        # Should complete without error
        voice_output.speak_response(response)
    
    def test_speak_response_with_async_playback(self, voice_output):
        """Test speak_response supports asynchronous playback."""
        response = HouseResponse(
            text="Differential diagnosis time.",
            confidence_score=0.9,
            sarcasm_level=2,
            emotional_tone="analytical",
            house_authenticity=0.8
        )
        
        # Should return immediately with async=True
        voice_output.speak_response(response, async_playback=True)
    
    def test_stop_playback_interrupts_current_audio(self, voice_output):
        """Test stop_playback interrupts ongoing audio playback."""
        response = HouseResponse(
            text="This is a long response that should be interrupted mid-playback.",
            confidence_score=0.8,
            sarcasm_level=3,
            emotional_tone="analytical",
            house_authenticity=0.8
        )
        
        # Start async playback
        voice_output.speak_response(response, async_playback=True)
        
        # Stop should not raise error
        voice_output.stop_playback()
    
    def test_get_audio_config_returns_current_settings(self, voice_output):
        """Test get_audio_config returns current audio configuration."""
        config = voice_output.get_audio_config()
        
        assert isinstance(config, AudioConfig)
        assert config.sample_rate == 22050
        assert config.bit_depth == 16
        assert config.channels in [1, 2]
        assert config.buffer_size > 0
        assert isinstance(config.output_device, str)
    
    def test_update_audio_config_changes_settings(self, voice_output):
        """Test update_audio_config modifies audio settings."""
        new_config = AudioConfig(
            sample_rate=44100,
            bit_depth=16,
            channels=2,
            buffer_size=2048,
            output_device="speakers"
        )
        
        voice_output.update_audio_config(new_config)
        
        current_config = voice_output.get_audio_config()
        assert current_config.sample_rate == 44100
        assert current_config.channels == 2
    
    def test_update_audio_config_raises_validation_error_for_invalid_config(self, voice_output):
        """Test update_audio_config raises ValidationError for invalid settings."""
        invalid_config = AudioConfig(
            sample_rate=0,  # Invalid
            bit_depth=7,   # Invalid
            channels=5,    # Invalid
            buffer_size=-1  # Invalid
        )
        
        with pytest.raises(Exception):  # Should be ValidationError
            voice_output.update_audio_config(invalid_config)
    
    def test_list_audio_devices_returns_available_devices(self, voice_output):
        """Test list_audio_devices returns available audio output devices."""
        devices = voice_output.list_audio_devices()
        
        assert isinstance(devices, list)
        assert len(devices) > 0
        
        for device in devices:
            assert isinstance(device, dict)
            assert "id" in device
            assert "name" in device
            assert "channels" in device
            assert "default_sample_rate" in device
    
    def test_get_playback_status_returns_current_state(self, voice_output):
        """Test get_playback_status returns playback information."""
        status = voice_output.get_playback_status()
        
        assert isinstance(status, dict)
        assert "is_playing" in status
        assert "current_position_ms" in status
        assert "total_duration_ms" in status
        assert "volume_level" in status
        
        # Verify types
        assert isinstance(status["is_playing"], bool)
        assert isinstance(status["current_position_ms"], (int, float))
        assert isinstance(status["total_duration_ms"], (int, float))
        assert isinstance(status["volume_level"], float)
        assert 0 <= status["volume_level"] <= 1.0
    
    def test_set_volume_adjusts_output_level(self, voice_output):
        """Test set_volume adjusts audio output volume."""
        # Set volume to 50%
        voice_output.set_volume(0.5)
        
        status = voice_output.get_playback_status()
        assert abs(status["volume_level"] - 0.5) < 0.1
    
    def test_set_volume_raises_validation_error_for_invalid_level(self, voice_output):
        """Test set_volume raises ValidationError for invalid volume level."""
        with pytest.raises(Exception):  # Should be ValidationError
            voice_output.set_volume(-0.1)  # Below 0
        
        with pytest.raises(Exception):  # Should be ValidationError
            voice_output.set_volume(1.1)   # Above 1
    
    def test_performance_contract_synthesis_latency(self, voice_output):
        """Test performance contract: synthesis latency < 2 seconds."""
        import time
        
        response = HouseResponse(
            text="This is a performance test for speech synthesis timing.",
            confidence_score=0.8,
            sarcasm_level=3,
            emotional_tone="analytical",
            house_authenticity=0.8
        )
        
        start_time = time.time()
        audio_data = voice_output.synthesize_speech(response)
        end_time = time.time()
        
        synthesis_time = end_time - start_time
        assert synthesis_time < 2.0  # Should be under 2 seconds
        assert len(audio_data) > 0
    
    def test_performance_contract_playback_latency(self, voice_output):
        """Test performance contract: playback start latency < 200ms."""
        import time
        
        audio_data = self._create_test_wav_data()
        
        start_time = time.time()
        voice_output.play_audio(audio_data)
        # Playback should start immediately (not wait for completion)
        end_time = time.time()
        
        start_latency = end_time - start_time
        assert start_latency < 0.2  # Should be under 200ms
    
    def test_performance_contract_memory_usage(self, voice_output):
        """Test performance contract: audio buffer memory < 100MB."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Generate multiple audio clips
        for i in range(5):
            response = HouseResponse(
                text=f"Performance test audio clip number {i} with substantial content.",
                confidence_score=0.8,
                sarcasm_level=3,
                emotional_tone="analytical",
                house_authenticity=0.8
            )
            voice_output.synthesize_speech(response)
        
        current_memory = process.memory_info().rss
        memory_diff = current_memory - initial_memory
        
        assert memory_diff < 100 * 1024 * 1024  # Should be under 100MB
    
    def _create_test_wav_data(self) -> bytes:
        """Create minimal valid WAV audio data for testing."""
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(22050)  # Sample rate
            
            # Generate 0.1 seconds of silence
            frames = int(0.1 * 22050)
            audio_data = b'\x00\x00' * frames
            wav_file.writeframes(audio_data)
        
        return buffer.getvalue()


@pytest.mark.integration
class TestVoiceOutputIntegration:
    """Integration tests for VoiceOutput with real audio hardware."""
    
    def test_real_tts_model_loading(self):
        """Test with real TTS model files."""
        pytest.skip("Requires TTS model files for testing")
    
    def test_real_audio_device_output(self):
        """Test with real audio output devices."""
        pytest.skip("Requires audio hardware for testing")
    
    def test_voice_cloning_quality(self):
        """Test voice cloning accuracy with House samples."""
        pytest.skip("Requires House voice samples for testing")
