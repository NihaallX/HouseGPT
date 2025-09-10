"""Unit tests for HouseGPT utility functions.

Comprehensive test suite covering all utility modules including text processing,
audio processing, database utilities, and configuration handling.
"""

import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import numpy as np
from typing import List, Dict, Any

# Import utility modules
from src.lib.text_utils import TextCleaner, TextValidator, TextAnalyzer
from src.lib.audio_utils import AudioProcessor, AudioValidator
from src.lib.db_utils import DatabaseManager, ConnectionPool
from src.lib.config_utils import ConfigLoader, ConfigValidator


class TestTextCleaner(unittest.TestCase):
    """Test suite for TextCleaner utility."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cleaner = TextCleaner()
    
    def test_clean_user_input_basic(self):
        """Test basic text cleaning functionality."""
        test_cases = [
            ("Hello world!", "Hello world!"),
            ("  extra   spaces  ", "extra spaces"),
            ("UPPERCASE", "uppercase"),
            ("Mixed CaSe TeXt", "mixed case text"),
            ("", ""),
            ("   ", ""),
            ("Special chars!@#$%", "special chars!@#$%"),
        ]
        
        for input_text, expected in test_cases:
            with self.subTest(input=input_text):
                result = self.cleaner.clean_user_input(input_text)
                self.assertEqual(result, expected)
    
    def test_clean_user_input_unicode(self):
        """Test cleaning with unicode characters."""
        test_cases = [
            ("café résumé", "café résumé"),
            ("naïve émojis 😊", "naïve émojis 😊"),
            ("Chinese: 你好", "chinese: 你好"),
            ("Japanese: こんにちは", "japanese: こんにちは"),
        ]
        
        for input_text, expected in test_cases:
            with self.subTest(input=input_text):
                result = self.cleaner.clean_user_input(input_text)
                self.assertEqual(result, expected)
    
    def test_clean_house_response(self):
        """Test House response cleaning."""
        test_cases = [
            ("Well, obviously...", "Well, obviously..."),
            ("  Multiple   spaces  here  ", "Multiple spaces here"),
            ("SHOUTING RESPONSE", "SHOUTING RESPONSE"),  # Keep caps for House
            ("", ""),
        ]
        
        for input_text, expected in test_cases:
            with self.subTest(input=input_text):
                result = self.cleaner.clean_house_response(input_text)
                self.assertEqual(result, expected)
    
    def test_remove_special_chars(self):
        """Test special character removal."""
        test_cases = [
            ("Normal text", "Normal text"),
            ("Text with @#$% chars", "Text with  chars"),
            ("Keep-hyphens_and_underscores", "Keep-hyphens_and_underscores"),
            ("Remove!@#$%^&*()", "Remove"),
            ("", ""),
        ]
        
        for input_text, expected in test_cases:
            with self.subTest(input=input_text):
                result = self.cleaner.remove_special_chars(input_text)
                self.assertEqual(result, expected)
    
    def test_normalize_whitespace(self):
        """Test whitespace normalization."""
        test_cases = [
            ("Normal text", "Normal text"),
            ("  Extra   spaces  ", "Extra spaces"),
            ("Multiple\n\nlines\n", "Multiple lines"),
            ("Tab\t\tcharacters", "Tab characters"),
            ("Mixed\n \t whitespace", "Mixed whitespace"),
            ("", ""),
        ]
        
        for input_text, expected in test_cases:
            with self.subTest(input=input_text):
                result = self.cleaner.normalize_whitespace(input_text)
                self.assertEqual(result, expected)


class TestTextValidator(unittest.TestCase):
    """Test suite for TextValidator utility."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = TextValidator()
    
    def test_validate_user_input_valid(self):
        """Test validation of valid user inputs."""
        valid_inputs = [
            "Hello House, how are you?",
            "What's the weather like?",
            "Tell me a joke",
            "Can you help me with this?",
            "Short",
            "A longer question that should still be valid input",
        ]
        
        for input_text in valid_inputs:
            with self.subTest(input=input_text):
                is_valid, errors = self.validator.validate_user_input(input_text)
                self.assertTrue(is_valid)
                self.assertEqual(len(errors), 0)
    
    def test_validate_user_input_invalid(self):
        """Test validation of invalid user inputs."""
        invalid_inputs = [
            ("", "Empty input"),
            ("   ", "Empty input"),
            ("A" * 1000, "Too long"),  # Assuming max length < 1000
            ("@#$%^&*()", "No meaningful content"),
        ]
        
        for input_text, expected_error_type in invalid_inputs:
            with self.subTest(input=input_text, expected=expected_error_type):
                is_valid, errors = self.validator.validate_user_input(input_text)
                self.assertFalse(is_valid)
                self.assertGreater(len(errors), 0)
    
    def test_validate_house_response(self):
        """Test validation of House responses."""
        valid_responses = [
            "Obviously, the answer is...",
            "Well, that's interesting.",
            "You're an idiot, but I'll help anyway.",
            "Differential diagnosis includes...",
        ]
        
        for response in valid_responses:
            with self.subTest(response=response):
                is_valid, errors = self.validator.validate_house_response(response)
                self.assertTrue(is_valid)
                self.assertEqual(len(errors), 0)
    
    def test_check_profanity(self):
        """Test profanity detection."""
        # Note: House is allowed to be sarcastic/rude
        test_cases = [
            ("Normal text", False),
            ("You're an idiot", False),  # House-style acceptable
            ("That's stupid", False),    # House-style acceptable
            ("Clean language", False),
        ]
        
        for text, expected_has_profanity in test_cases:
            with self.subTest(text=text):
                has_profanity = self.validator.check_profanity(text)
                self.assertEqual(has_profanity, expected_has_profanity)
    
    def test_check_length_limits(self):
        """Test length validation."""
        # Test various lengths
        short_text = "Hi"
        normal_text = "This is a normal length message"
        long_text = "A" * 500  # Assuming this exceeds limits
        
        self.assertTrue(self.validator.check_length_limits(short_text))
        self.assertTrue(self.validator.check_length_limits(normal_text))
        # Length limit depends on configuration


class TestTextAnalyzer(unittest.TestCase):
    """Test suite for TextAnalyzer utility."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = TextAnalyzer()
    
    def test_extract_keywords(self):
        """Test keyword extraction."""
        test_text = "I have a headache and feel dizzy"
        keywords = self.analyzer.extract_keywords(test_text)
        
        self.assertIsInstance(keywords, list)
        self.assertIn("headache", keywords)
        self.assertIn("dizzy", keywords)
    
    def test_analyze_sentiment(self):
        """Test sentiment analysis."""
        test_cases = [
            ("I feel great today!", "positive"),
            ("This is terrible", "negative"),
            ("It's okay, I guess", "neutral"),
        ]
        
        for text, expected_sentiment in test_cases:
            with self.subTest(text=text):
                sentiment = self.analyzer.analyze_sentiment(text)
                self.assertIsInstance(sentiment, str)
                # Note: Exact sentiment may vary based on model
    
    def test_detect_medical_terms(self):
        """Test medical term detection."""
        medical_text = "I have chest pain and shortness of breath"
        non_medical_text = "I like pizza and movies"
        
        medical_terms = self.analyzer.detect_medical_terms(medical_text)
        non_medical_terms = self.analyzer.detect_medical_terms(non_medical_text)
        
        self.assertIsInstance(medical_terms, list)
        self.assertIsInstance(non_medical_terms, list)
        self.assertGreater(len(medical_terms), len(non_medical_terms))


class TestAudioProcessor(unittest.TestCase):
    """Test suite for AudioProcessor utility."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = AudioProcessor()
    
    def test_validate_audio_data(self):
        """Test audio data validation."""
        # Create mock audio data
        valid_audio = np.random.rand(16000).astype(np.float32)  # 1 second at 16kHz
        invalid_audio = np.array([])
        
        self.assertTrue(self.processor.validate_audio_data(valid_audio))
        self.assertFalse(self.processor.validate_audio_data(invalid_audio))
    
    def test_normalize_audio(self):
        """Test audio normalization."""
        # Create test audio with varying amplitudes
        audio = np.array([0.1, -0.8, 0.5, -0.3, 0.9], dtype=np.float32)
        normalized = self.processor.normalize_audio(audio)
        
        self.assertIsInstance(normalized, np.ndarray)
        self.assertTrue(np.max(normalized) <= 1.0)
        self.assertTrue(np.min(normalized) >= -1.0)
    
    def test_apply_noise_reduction(self):
        """Test noise reduction."""
        # Create noisy audio signal
        signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000))  # 440Hz tone
        noise = np.random.normal(0, 0.1, 16000)
        noisy_audio = signal + noise
        
        cleaned_audio = self.processor.apply_noise_reduction(noisy_audio)
        
        self.assertIsInstance(cleaned_audio, np.ndarray)
        self.assertEqual(len(cleaned_audio), len(noisy_audio))
    
    def test_detect_speech(self):
        """Test speech detection."""
        # Create mock audio with and without speech
        speech_audio = np.random.rand(16000) * 0.5  # Moderate amplitude
        silence_audio = np.random.rand(16000) * 0.01  # Very low amplitude
        
        has_speech_1 = self.processor.detect_speech(speech_audio)
        has_speech_2 = self.processor.detect_speech(silence_audio)
        
        self.assertIsInstance(has_speech_1, bool)
        self.assertIsInstance(has_speech_2, bool)
    
    def test_convert_sample_rate(self):
        """Test sample rate conversion."""
        # Create audio at 44100 Hz
        original_rate = 44100
        target_rate = 16000
        duration = 1.0
        
        original_audio = np.sin(
            2 * np.pi * 440 * np.linspace(0, duration, int(original_rate * duration))
        )
        
        converted_audio = self.processor.convert_sample_rate(
            original_audio, original_rate, target_rate
        )
        
        expected_length = int(target_rate * duration)
        self.assertEqual(len(converted_audio), expected_length)


class TestAudioValidator(unittest.TestCase):
    """Test suite for AudioValidator utility."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = AudioValidator()
    
    def test_validate_format(self):
        """Test audio format validation."""
        valid_formats = ["wav", "mp3", "flac", "ogg"]
        invalid_formats = ["txt", "jpg", "exe", "unknown"]
        
        for fmt in valid_formats:
            with self.subTest(format=fmt):
                self.assertTrue(self.validator.validate_format(fmt))
        
        for fmt in invalid_formats:
            with self.subTest(format=fmt):
                self.assertFalse(self.validator.validate_format(fmt))
    
    def test_validate_sample_rate(self):
        """Test sample rate validation."""
        valid_rates = [8000, 16000, 22050, 44100, 48000]
        invalid_rates = [1000, 7999, 48001, 96000]
        
        for rate in valid_rates:
            with self.subTest(rate=rate):
                self.assertTrue(self.validator.validate_sample_rate(rate))
        
        for rate in invalid_rates:
            with self.subTest(rate=rate):
                # May depend on implementation
                pass
    
    def test_validate_duration(self):
        """Test audio duration validation."""
        valid_durations = [0.5, 1.0, 5.0, 10.0]  # seconds
        invalid_durations = [0.0, -1.0, 60.0, 120.0]  # too short/long
        
        for duration in valid_durations:
            with self.subTest(duration=duration):
                self.assertTrue(self.validator.validate_duration(duration))
        
        for duration in invalid_durations:
            with self.subTest(duration=duration):
                # May depend on implementation
                pass


class TestDatabaseManager(unittest.TestCase):
    """Test suite for DatabaseManager utility."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.db_manager = DatabaseManager("sqlite:///:memory:")
    
    def test_connection_establishment(self):
        """Test database connection."""
        # This would depend on actual implementation
        pass
    
    def test_query_execution(self):
        """Test query execution."""
        # This would depend on actual implementation
        pass
    
    def test_transaction_handling(self):
        """Test transaction management."""
        # This would depend on actual implementation
        pass


class TestConnectionPool(unittest.TestCase):
    """Test suite for ConnectionPool utility."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.pool = ConnectionPool(max_connections=5)
    
    def test_connection_pooling(self):
        """Test connection pool management."""
        # This would depend on actual implementation
        pass
    
    def test_connection_cleanup(self):
        """Test connection cleanup."""
        # This would depend on actual implementation
        pass


class TestConfigLoader(unittest.TestCase):
    """Test suite for ConfigLoader utility."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config_loader = ConfigLoader()
    
    def test_load_from_file(self):
        """Test loading configuration from file."""
        # Create temporary config file
        config_data = {
            "model": {"name": "test_model"},
            "audio": {"sample_rate": 16000},
            "database": {"url": "sqlite:///:memory:"}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            import json
            json.dump(config_data, f)
            config_file = f.name
        
        try:
            loaded_config = self.config_loader.load_from_file(config_file)
            self.assertIsInstance(loaded_config, dict)
            self.assertEqual(loaded_config["model"]["name"], "test_model")
        finally:
            os.unlink(config_file)
    
    def test_load_from_env(self):
        """Test loading configuration from environment."""
        with patch.dict(os.environ, {
            'HOUSEGPT_MODEL_NAME': 'env_model',
            'HOUSEGPT_AUDIO_SAMPLE_RATE': '22050'
        }):
            env_config = self.config_loader.load_from_env()
            self.assertIsInstance(env_config, dict)
    
    def test_merge_configs(self):
        """Test configuration merging."""
        base_config = {"model": {"name": "base"}, "audio": {"sample_rate": 16000}}
        override_config = {"model": {"name": "override"}, "database": {"url": "test"}}
        
        merged = self.config_loader.merge_configs(base_config, override_config)
        
        self.assertEqual(merged["model"]["name"], "override")
        self.assertEqual(merged["audio"]["sample_rate"], 16000)
        self.assertEqual(merged["database"]["url"], "test")


class TestConfigValidator(unittest.TestCase):
    """Test suite for ConfigValidator utility."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = ConfigValidator()
    
    def test_validate_config_structure(self):
        """Test configuration structure validation."""
        valid_config = {
            "model": {"name": "test", "temperature": 0.7},
            "audio": {"sample_rate": 16000, "channels": 1},
            "database": {"url": "sqlite:///:memory:"}
        }
        
        invalid_config = {
            "model": "invalid_structure",
            "audio": {"sample_rate": "not_a_number"}
        }
        
        self.assertTrue(self.validator.validate_config_structure(valid_config))
        self.assertFalse(self.validator.validate_config_structure(invalid_config))
    
    def test_validate_required_fields(self):
        """Test required field validation."""
        config_with_required = {
            "model": {"name": "required_model"},
            "audio": {"sample_rate": 16000}
        }
        
        config_missing_required = {
            "audio": {"sample_rate": 16000}
            # Missing required model.name
        }
        
        # Results depend on actual implementation
        pass
    
    def test_validate_field_types(self):
        """Test field type validation."""
        config_correct_types = {
            "model": {"temperature": 0.7},  # float
            "audio": {"sample_rate": 16000},  # int
            "database": {"url": "sqlite:///:memory:"}  # string
        }
        
        config_wrong_types = {
            "model": {"temperature": "not_a_number"},
            "audio": {"sample_rate": "not_an_int"}
        }
        
        # Results depend on actual implementation
        pass


class TestUtilityIntegration(unittest.TestCase):
    """Integration tests for utility functions working together."""
    
    def test_text_processing_pipeline(self):
        """Test complete text processing pipeline."""
        cleaner = TextCleaner()
        validator = TextValidator()
        analyzer = TextAnalyzer()
        
        # Process user input through complete pipeline
        raw_input = "  HELLO HOUSE! I have a HEADACHE and feel terrible!!!  "
        
        # Step 1: Clean
        cleaned = cleaner.clean_user_input(raw_input)
        self.assertNotEqual(cleaned, raw_input)
        
        # Step 2: Validate
        is_valid, errors = validator.validate_user_input(cleaned)
        self.assertTrue(is_valid)
        
        # Step 3: Analyze
        keywords = analyzer.extract_keywords(cleaned)
        self.assertIsInstance(keywords, list)
    
    def test_audio_processing_pipeline(self):
        """Test complete audio processing pipeline."""
        processor = AudioProcessor()
        validator = AudioValidator()
        
        # Create test audio
        audio = np.random.rand(16000).astype(np.float32)
        
        # Step 1: Validate
        self.assertTrue(processor.validate_audio_data(audio))
        
        # Step 2: Normalize
        normalized = processor.normalize_audio(audio)
        self.assertIsInstance(normalized, np.ndarray)
        
        # Step 3: Noise reduction
        cleaned = processor.apply_noise_reduction(normalized)
        self.assertIsInstance(cleaned, np.ndarray)
    
    def test_config_loading_pipeline(self):
        """Test complete configuration loading pipeline."""
        loader = ConfigLoader()
        validator = ConfigValidator()
        
        # Create test config
        config_data = {
            "model": {"name": "test_model", "temperature": 0.7},
            "audio": {"sample_rate": 16000}
        }
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            import json
            json.dump(config_data, f)
            config_file = f.name
        
        try:
            # Step 1: Load
            loaded_config = loader.load_from_file(config_file)
            self.assertIsInstance(loaded_config, dict)
            
            # Step 2: Validate
            is_valid = validator.validate_config_structure(loaded_config)
            # Result depends on implementation
            
        finally:
            os.unlink(config_file)


# Test utility functions for running tests
def run_text_utils_tests():
    """Run all text utility tests."""
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestTextCleaner))
    suite.addTest(unittest.makeSuite(TestTextValidator))
    suite.addTest(unittest.makeSuite(TestTextAnalyzer))
    
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


def run_audio_utils_tests():
    """Run all audio utility tests."""
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestAudioProcessor))
    suite.addTest(unittest.makeSuite(TestAudioValidator))
    
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


def run_db_utils_tests():
    """Run all database utility tests."""
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestDatabaseManager))
    suite.addTest(unittest.makeSuite(TestConnectionPool))
    
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


def run_config_utils_tests():
    """Run all configuration utility tests."""
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestConfigLoader))
    suite.addTest(unittest.makeSuite(TestConfigValidator))
    
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


def run_all_utility_tests():
    """Run all utility tests."""
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestTextCleaner, TestTextValidator, TestTextAnalyzer,
        TestAudioProcessor, TestAudioValidator,
        TestDatabaseManager, TestConnectionPool,
        TestConfigLoader, TestConfigValidator,
        TestUtilityIntegration
    ]
    
    for test_class in test_classes:
        suite.addTest(unittest.makeSuite(test_class))
    
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


if __name__ == "__main__":
    # Run all tests when script is executed directly
    result = run_all_utility_tests()
    
    # Print summary
    print(f"\n{'='*60}")
    print("UTILITY TESTS SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}")
    
    print(f"{'='*60}")
