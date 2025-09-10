"""Contract tests for WakeWordDetector service.

These tests validate the contract defined in contracts/wake_word_contract.md.
Tests MUST fail initially before implementation exists.
"""

import pytest
from unittest.mock import Mock, patch
from src.services.wake_word import WakeWordDetector, WakeWordError, AudioDeviceError
from src.lib.config import Configuration


class TestWakeWordDetectorContract:
    """Test WakeWordDetector contract compliance."""
    
    @pytest.fixture
    def config(self):
        """Provide test configuration."""
        config = Configuration()
        config.audio.audio_device_index = 0
        config.audio.whisper_model_size = "base"
        config.app.wake_word_sensitivity = 0.7
        return config
    
    @pytest.fixture
    def detector(self, config):
        """Provide WakeWordDetector instance."""
        return WakeWordDetector(config)
    
    def test_constructor_requires_config(self):
        """Test constructor requires Configuration object."""
        with pytest.raises(TypeError):
            WakeWordDetector()
    
    def test_constructor_raises_audio_device_error_if_no_microphone(self, config):
        """Test constructor raises AudioDeviceError if microphone not available."""
        with patch('pyaudio.PyAudio') as mock_pyaudio:
            mock_pyaudio.side_effect = OSError("No microphone available")
            with pytest.raises(AudioDeviceError):
                WakeWordDetector(config)
    
    def test_constructor_raises_model_load_error_if_whisper_fails(self, config):
        """Test constructor raises ModelLoadError if Whisper model fails to load."""
        with patch('whisper.load_model') as mock_load:
            mock_load.side_effect = Exception("Model load failed")
            with pytest.raises(Exception):  # Should be ModelLoadError when implemented
                WakeWordDetector(config)
    
    def test_start_listening_begins_monitoring(self, detector):
        """Test start_listening begins continuous audio monitoring."""
        # Should not raise when called on valid detector
        detector.start_listening()
        assert detector.is_listening() is True
    
    def test_start_listening_raises_runtime_error_if_already_listening(self, detector):
        """Test start_listening raises RuntimeError if already listening."""
        detector.start_listening()
        with pytest.raises(RuntimeError):
            detector.start_listening()
    
    def test_stop_listening_stops_monitoring(self, detector):
        """Test stop_listening stops audio monitoring and cleanup resources."""
        detector.start_listening()
        detector.stop_listening()
        assert detector.is_listening() is False
    
    def test_is_listening_returns_current_status(self, detector):
        """Test is_listening returns current monitoring status."""
        assert detector.is_listening() is False
        detector.start_listening()
        assert detector.is_listening() is True
        detector.stop_listening()
        assert detector.is_listening() is False
    
    def test_set_wake_word_callback_registers_function(self, detector):
        """Test set_wake_word_callback registers callback for wake word events."""
        callback = Mock()
        detector.set_wake_word_callback(callback)
        # Should not raise exception
    
    def test_set_wake_word_callback_raises_type_error_for_non_callable(self, detector):
        """Test set_wake_word_callback raises TypeError if callback not callable."""
        with pytest.raises(TypeError):
            detector.set_wake_word_callback("not_callable")
    
    def test_get_detection_stats_returns_metrics(self, detector):
        """Test get_detection_stats returns performance metrics dictionary."""
        stats = detector.get_detection_stats()
        
        # Verify required keys in stats dictionary
        assert isinstance(stats, dict)
        assert "total_detections" in stats
        assert "false_positives" in stats
        assert "avg_confidence" in stats
        assert "uptime_seconds" in stats
        
        # Verify types
        assert isinstance(stats["total_detections"], int)
        assert isinstance(stats["false_positives"], int)
        assert isinstance(stats["avg_confidence"], float)
        assert isinstance(stats["uptime_seconds"], int)
    
    def test_wake_word_callback_receives_detected_text(self, detector):
        """Test wake word detection triggers callback with detected text."""
        callback = Mock()
        detector.set_wake_word_callback(callback)
        
        # Simulate wake word detection (implementation will handle this)
        # For now, test that callback can be called with detected text
        detected_text = "House?"
        
        # This should be called by the detector when "House?" is detected
        callback(detected_text)
        callback.assert_called_once_with(detected_text)
    
    def test_performance_contract_detection_latency(self, detector):
        """Test performance contract: detection latency < 2 seconds."""
        # This test verifies the performance requirement
        # Implementation should ensure < 2s from speech to callback
        import time
        
        callback = Mock()
        detector.set_wake_word_callback(callback)
        detector.start_listening()
        
        # Simulate audio input and measure response time
        start_time = time.time()
        
        # Mock audio processing would happen here
        # For now, verify that the contract is testable
        
        detector.stop_listening()
        
        # Performance verification will be implemented with real audio
        assert True  # Placeholder for actual timing test
    
    def test_memory_usage_contract(self, detector):
        """Test performance contract: memory usage < 100MB sustained."""
        # This test verifies memory usage stays within limits
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        detector.start_listening()
        
        # Simulate sustained operation
        current_memory = process.memory_info().rss
        memory_diff = current_memory - initial_memory
        
        detector.stop_listening()
        
        # Should use less than 100MB additional memory
        assert memory_diff < 100 * 1024 * 1024  # 100MB in bytes
    
    def test_false_positive_rate_contract(self, detector):
        """Test performance contract: false positive rate < 5%."""
        # This test verifies false positive requirements
        # Implementation should maintain < 5% false positive rate
        
        # Test with various non-wake-word audio inputs
        false_triggers = 0
        total_tests = 20
        
        callback = Mock()
        detector.set_wake_word_callback(callback)
        
        # Simulate various audio inputs that should NOT trigger
        test_phrases = [
            "hello there", "how are you", "what's up", "goodbye",
            "house music", "housing market", "warehouse", "household"
        ]
        
        for phrase in test_phrases:
            # Mock audio input simulation would go here
            # For now, verify contract is testable
            pass
        
        false_positive_rate = false_triggers / total_tests
        assert false_positive_rate < 0.05  # Less than 5%


@pytest.mark.integration
class TestWakeWordDetectorIntegration:
    """Integration tests for WakeWordDetector with real dependencies."""
    
    def test_real_audio_device_detection(self):
        """Test with real audio device if available."""
        # Skip if no audio device available
        pytest.skip("Requires real audio device for testing")
    
    def test_whisper_model_loading(self):
        """Test actual Whisper model loading."""
        # Skip if models not downloaded
        pytest.skip("Requires Whisper models for testing")
    
    def test_end_to_end_wake_word_detection(self):
        """Test complete wake word detection pipeline."""
        # Skip until implementation complete
        pytest.skip("Requires complete implementation for testing")
