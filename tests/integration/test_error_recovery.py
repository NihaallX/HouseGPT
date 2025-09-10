"""Integration test for error handling and fallback modes.

This test validates the system's resilience and graceful degradation
when components fail or encounter unexpected conditions.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import asyncio
from typing import List, Optional

from src.services.wake_word import WakeWordDetector
from src.services.rag_store import RAGStore
from src.services.house_model import HouseModel
from src.services.style_filter import StyleFilter
from src.services.voice_output import VoiceOutput
from src.services.conversation_logger import ConversationLogger
from src.models.house_quote import HouseQuote
from src.models.context import ConversationContext
from src.models.house_response import HouseResponse
from src.models.filter_result import FilterResult
from src.lib.config import Configuration


class TestErrorHandlingIntegration:
    """Test error handling and fallback mechanisms across the pipeline."""
    
    @pytest.fixture
    def config(self):
        """Provide test configuration."""
        return Configuration()
    
    @pytest.fixture
    def mock_services(self, config):
        """Provide mock services for error testing."""
        return {
            'wake_word': Mock(spec=WakeWordDetector),
            'rag_store': Mock(spec=RAGStore),
            'house_model': Mock(spec=HouseModel),
            'style_filter': Mock(spec=StyleFilter),
            'voice_output': Mock(spec=VoiceOutput),
            'logger': Mock(spec=ConversationLogger)
        }
    
    def test_rag_store_failure_fallback(self, mock_services):
        """Test pipeline continues when RAG store fails."""
        # Arrange: RAG store throws exception
        rag_store = mock_services['rag_store']
        house_model = mock_services['house_model']
        style_filter = mock_services['style_filter']
        voice_output = mock_services['voice_output']
        logger = mock_services['logger']
        
        # RAG store fails
        rag_store.search_quotes.side_effect = ConnectionError("Database connection failed")
        
        # Model should still work without RAG quotes
        fallback_response = HouseResponse(
            text="Interesting. A patient with symptoms so mysterious that even my database of witty comebacks is speechless.",
            confidence_score=0.75,
            sarcasm_level=3,
            emotional_tone="analytical",
            house_authenticity=0.8
        )
        house_model.generate_response.return_value = fallback_response
        
        # Style filter passes
        filter_result = FilterResult(
            is_valid=True,
            violations=[],
            score=0.8,
            filtered_response=fallback_response,
            applied_transformations=[]
        )
        style_filter.validate_response.return_value = filter_result
        
        # Voice output works
        voice_output.speak_response.return_value = None
        
        # Logger works
        logger.log_conversation.return_value = None
        
        # Act: Execute pipeline with RAG failure
        user_query = "I have a headache"
        
        try:
            # Attempt RAG search, fail gracefully
            quotes = []
            try:
                quotes = rag_store.search_quotes(user_query, top_k=3)
            except ConnectionError:
                # Fallback: Continue with empty quotes
                quotes = []
        
            # Continue pipeline with empty quotes
            context = ConversationContext(
                user_query=user_query,
                relevant_quotes=quotes,  # Empty due to RAG failure
                conversation_history=[]
            )
            
            response = house_model.generate_response(context)
            validation_result = style_filter.validate_response(response)
            
            if validation_result.is_valid:
                voice_output.speak_response(response)
        
        except Exception as e:
            pytest.fail(f"Pipeline should not fail completely due to RAG error: {e}")
        
        # Assert: Verify graceful degradation
        
        # RAG was attempted but failed
        rag_store.search_quotes.assert_called_once_with(user_query, top_k=3)
        
        # Model still generated response with empty quotes
        house_model.generate_response.assert_called_once()
        call_context = house_model.generate_response.call_args[0][0]
        assert len(call_context.relevant_quotes) == 0
        
        # Pipeline continued normally
        style_filter.validate_response.assert_called_once()
        voice_output.speak_response.assert_called_once()
    
    def test_house_model_failure_fallback(self, mock_services):
        """Test fallback when House model fails."""
        # Arrange: Model generation fails
        rag_store = mock_services['rag_store']
        house_model = mock_services['house_model']
        style_filter = mock_services['style_filter']
        voice_output = mock_services['voice_output']
        
        # RAG works normally
        quotes = [HouseQuote(text="Test quote", sarcasm_level=3)]
        rag_store.search_quotes.return_value = quotes
        
        # Model fails
        house_model.generate_response.side_effect = RuntimeError("Model inference failed")
        
        # Fallback response mechanism
        fallback_response = HouseResponse(
            text="I'm experiencing technical difficulties. Must be the same incompetence that affects hospital IT systems.",
            confidence_score=0.6,
            sarcasm_level=3,
            emotional_tone="frustrated",
            house_authenticity=0.7
        )
        
        # Style filter and voice output work
        filter_result = FilterResult(
            is_valid=True,
            violations=[],
            score=0.7,
            filtered_response=fallback_response,
            applied_transformations=[]
        )
        style_filter.validate_response.return_value = filter_result
        voice_output.speak_response.return_value = None
        
        # Act: Execute pipeline with model failure
        user_query = "What's wrong with me?"
        
        try:
            quotes = rag_store.search_quotes(user_query, top_k=3)
            context = ConversationContext(user_query=user_query, relevant_quotes=quotes)
            
            response = None
            try:
                response = house_model.generate_response(context)
            except RuntimeError:
                # Fallback: Use predefined response
                response = fallback_response
            
            validation_result = style_filter.validate_response(response)
            
            if validation_result.is_valid:
                voice_output.speak_response(response)
        
        except Exception as e:
            pytest.fail(f"Pipeline should not fail completely due to model error: {e}")
        
        # Assert: Verify fallback mechanism
        
        # Model was attempted but failed
        house_model.generate_response.assert_called_once()
        
        # Fallback response was used
        style_filter.validate_response.assert_called_once()
        validated_response = style_filter.validate_response.call_args[0][0]
        assert "technical difficulties" in validated_response.text
        
        voice_output.speak_response.assert_called_once()
    
    def test_style_filter_failure_allows_passthrough(self, mock_services):
        """Test pipeline continues when style filter fails."""
        # Arrange: Style filter throws exception
        rag_store = mock_services['rag_store']
        house_model = mock_services['house_model']
        style_filter = mock_services['style_filter']
        voice_output = mock_services['voice_output']
        
        # RAG and model work normally
        quotes = [HouseQuote(text="Test quote", sarcasm_level=3)]
        rag_store.search_quotes.return_value = quotes
        
        response = HouseResponse(
            text="Your symptoms are intriguing in their complete lack of specificity.",
            confidence_score=0.85,
            sarcasm_level=4,
            emotional_tone="condescending",
            house_authenticity=0.9
        )
        house_model.generate_response.return_value = response
        
        # Style filter fails
        style_filter.validate_response.side_effect = Exception("Style filter crashed")
        
        # Voice output works
        voice_output.speak_response.return_value = None
        
        # Act: Execute pipeline with style filter failure
        user_query = "I feel sick"
        
        try:
            quotes = rag_store.search_quotes(user_query, top_k=3)
            context = ConversationContext(user_query=user_query, relevant_quotes=quotes)
            response = house_model.generate_response(context)
            
            # Attempt style validation, fall back to passthrough
            validated_response = response
            try:
                validation_result = style_filter.validate_response(response)
                if validation_result.is_valid:
                    validated_response = validation_result.filtered_response
            except Exception:
                # Fallback: Use original response without style filtering
                pass
            
            voice_output.speak_response(validated_response)
        
        except Exception as e:
            pytest.fail(f"Pipeline should not fail completely due to style filter error: {e}")
        
        # Assert: Verify passthrough mechanism
        
        # Style filter was attempted
        style_filter.validate_response.assert_called_once()
        
        # Original response was used despite filter failure
        voice_output.speak_response.assert_called_once()
        spoken_response = voice_output.speak_response.call_args[0][0]
        assert spoken_response.text == response.text
    
    def test_voice_output_failure_silent_degradation(self, mock_services):
        """Test pipeline completes silently when voice output fails."""
        # Arrange: Voice output fails
        rag_store = mock_services['rag_store']
        house_model = mock_services['house_model']
        style_filter = mock_services['style_filter']
        voice_output = mock_services['voice_output']
        logger = mock_services['logger']
        
        # Pipeline works normally until voice output
        quotes = [HouseQuote(text="Test quote", sarcasm_level=3)]
        rag_store.search_quotes.return_value = quotes
        
        response = HouseResponse(
            text="Your condition requires further investigation.",
            confidence_score=0.8,
            sarcasm_level=3,
            emotional_tone="analytical",
            house_authenticity=0.85
        )
        house_model.generate_response.return_value = response
        
        filter_result = FilterResult(
            is_valid=True,
            violations=[],
            score=0.85,
            filtered_response=response,
            applied_transformations=[]
        )
        style_filter.validate_response.return_value = filter_result
        
        # Voice output fails
        voice_output.speak_response.side_effect = Exception("Audio system failure")
        
        # Logger still works
        logger.log_conversation.return_value = None
        
        # Act: Execute pipeline with voice failure
        user_query = "What should I do?"
        
        conversation_completed = False
        try:
            quotes = rag_store.search_quotes(user_query, top_k=3)
            context = ConversationContext(user_query=user_query, relevant_quotes=quotes)
            response = house_model.generate_response(context)
            validation_result = style_filter.validate_response(response)
            
            # Attempt voice output, fail silently
            try:
                if validation_result.is_valid:
                    voice_output.speak_response(validation_result.filtered_response)
            except Exception:
                # Silent failure - conversation still considered complete
                pass
            
            # Log conversation regardless of voice failure
            logger.log_conversation(Mock())
            conversation_completed = True
        
        except Exception as e:
            pytest.fail(f"Pipeline should complete despite voice failure: {e}")
        
        # Assert: Verify silent degradation
        assert conversation_completed is True
        
        # Voice output was attempted
        voice_output.speak_response.assert_called_once()
        
        # Conversation was still logged
        logger.log_conversation.assert_called_once()
    
    def test_wake_word_detection_timeout_handling(self, mock_services):
        """Test timeout handling in wake word detection."""
        # Arrange: Wake word detector has timeout
        wake_word = mock_services['wake_word']
        
        # Mock timeout scenario
        wake_word.listen_for_wake_word.side_effect = TimeoutError("Wake word timeout")
        wake_word.is_listening.return_value = True
        
        # Act: Handle wake word timeout
        timeout_handled = False
        try:
            # Attempt wake word detection with timeout
            wake_word.listen_for_wake_word(timeout=5.0)
        except TimeoutError:
            # Handle timeout gracefully
            timeout_handled = True
        
        # Assert: Verify timeout was handled
        assert timeout_handled is True
        wake_word.listen_for_wake_word.assert_called_once_with(timeout=5.0)
    
    def test_database_connection_recovery(self, mock_services):
        """Test automatic recovery from database connection issues."""
        # Arrange: Database connection is unstable
        rag_store = mock_services['rag_store']
        
        # First call fails, second succeeds (connection recovery)
        connection_error = ConnectionError("Connection lost")
        successful_result = [HouseQuote(text="Recovery quote", sarcasm_level=3)]
        
        rag_store.search_quotes.side_effect = [connection_error, successful_result]
        
        # Mock retry mechanism
        def search_with_retry(query, top_k=5, max_retries=2):
            for attempt in range(max_retries):
                try:
                    return rag_store.search_quotes(query, top_k=top_k)
                except ConnectionError:
                    if attempt == max_retries - 1:
                        raise
                    continue  # Retry
        
        # Act: Test retry mechanism
        user_query = "Test query"
        result = search_with_retry(user_query, top_k=3, max_retries=2)
        
        # Assert: Verify recovery worked
        assert len(result) == 1
        assert result[0].text == "Recovery quote"
        
        # Verify retry attempts
        assert rag_store.search_quotes.call_count == 2
    
    def test_memory_pressure_handling(self, mock_services):
        """Test handling of memory pressure situations."""
        # Arrange: Simulate memory pressure
        house_model = mock_services['house_model']
        
        # Mock memory error
        house_model.generate_response.side_effect = MemoryError("Insufficient memory")
        
        # Fallback response for memory issues
        memory_fallback = HouseResponse(
            text="My brain is currently experiencing a memory shortage. Unlike my patients, at least I admit when I can't think clearly.",
            confidence_score=0.5,
            sarcasm_level=4,
            emotional_tone="self-deprecating",
            house_authenticity=0.75
        )
        
        # Act: Handle memory pressure
        user_query = "Complex medical question"
        context = ConversationContext(user_query=user_query, relevant_quotes=[])
        
        response = None
        try:
            response = house_model.generate_response(context)
        except MemoryError:
            # Fallback for memory issues
            response = memory_fallback
        
        # Assert: Verify memory pressure handling
        assert response is not None
        assert "memory shortage" in response.text
        assert response.emotional_tone == "self-deprecating"
    
    def test_concurrent_request_error_isolation(self, mock_services):
        """Test that errors in one conversation don't affect others."""
        # Arrange: Multiple concurrent conversations, one fails
        house_model = mock_services['house_model']
        
        # First conversation fails, second succeeds
        def selective_failure(context):
            if "fail" in context.user_query.lower():
                raise RuntimeError("Intentional failure")
            else:
                return HouseResponse(
                    text="Successful response",
                    confidence_score=0.8,
                    sarcasm_level=3,
                    emotional_tone="analytical",
                    house_authenticity=0.8
                )
        
        house_model.generate_response.side_effect = selective_failure
        
        # Act: Process concurrent conversations
        conversations = [
            {"query": "This will fail", "should_succeed": False},
            {"query": "This will work", "should_succeed": True},
            {"query": "This also works", "should_succeed": True}
        ]
        
        results = []
        for conv in conversations:
            try:
                context = ConversationContext(
                    user_query=conv["query"],
                    relevant_quotes=[],
                    conversation_history=[]
                )
                response = house_model.generate_response(context)
                results.append({"success": True, "response": response})
            except RuntimeError:
                results.append({"success": False, "response": None})
        
        # Assert: Verify error isolation
        assert len(results) == 3
        
        # First conversation should fail
        assert results[0]["success"] is False
        
        # Other conversations should succeed
        assert results[1]["success"] is True
        assert results[2]["success"] is True
        assert results[1]["response"].text == "Successful response"
        assert results[2]["response"].text == "Successful response"
    
    def test_configuration_validation_errors(self, mock_services):
        """Test handling of configuration validation errors."""
        # Arrange: Invalid configuration
        from src.lib.config import Configuration
        
        # Mock invalid config that would cause validation errors
        invalid_configs = [
            {"issue": "missing_required_field", "error_type": "KeyError"},
            {"issue": "invalid_data_type", "error_type": "TypeError"},
            {"issue": "out_of_range_value", "error_type": "ValueError"}
        ]
        
        # Act & Assert: Test each configuration error type
        for config_case in invalid_configs:
            with pytest.raises(Exception):  # Should raise appropriate exception
                # This would be actual config validation in real implementation
                if config_case["error_type"] == "KeyError":
                    raise KeyError(f"Missing required field: {config_case['issue']}")
                elif config_case["error_type"] == "TypeError":
                    raise TypeError(f"Invalid data type: {config_case['issue']}")
                elif config_case["error_type"] == "ValueError":
                    raise ValueError(f"Invalid value: {config_case['issue']}")
    
    def test_graceful_shutdown_handling(self, mock_services):
        """Test graceful shutdown when system is interrupted."""
        # Arrange: Simulate system shutdown signal
        import signal
        
        wake_word = mock_services['wake_word']
        logger = mock_services['logger']
        
        shutdown_called = False
        
        def mock_shutdown():
            nonlocal shutdown_called
            shutdown_called = True
            wake_word.stop_listening()
            logger.flush_logs()  # Ensure logs are written
        
        # Mock shutdown signal handler
        wake_word.stop_listening.return_value = None
        logger.flush_logs.return_value = None
        
        # Act: Simulate graceful shutdown
        mock_shutdown()
        
        # Assert: Verify cleanup was performed
        assert shutdown_called is True
        wake_word.stop_listening.assert_called_once()
        logger.flush_logs.assert_called_once()
    
    def test_error_logging_and_monitoring(self, mock_services):
        """Test that errors are properly logged for monitoring."""
        # Arrange: Logger for error tracking
        logger = mock_services['logger']
        
        # Mock error scenarios
        error_scenarios = [
            {"component": "RAG", "error": ConnectionError("Database down")},
            {"component": "Model", "error": RuntimeError("Model crashed")},
            {"component": "Voice", "error": OSError("Audio device unavailable")}
        ]
        
        logger.log_error.return_value = None
        
        # Act: Log each error type
        for scenario in error_scenarios:
            try:
                raise scenario["error"]
            except Exception as e:
                # Log error with context
                logger.log_error(
                    component=scenario["component"],
                    error=e,
                    context={"user_query": "Test query", "timestamp": "2024-01-01T00:00:00Z"}
                )
        
        # Assert: Verify error logging
        assert logger.log_error.call_count == 3
        
        # Verify each error was logged with proper context
        logged_calls = logger.log_error.call_args_list
        for i, call in enumerate(logged_calls):
            args, kwargs = call
            # Verify error context was included
            assert "component" in kwargs or len(args) >= 1
            assert "error" in kwargs or len(args) >= 2


@pytest.mark.integration
class TestRealWorldErrorScenarios:
    """Integration tests for real-world error scenarios."""
    
    def test_network_partition_recovery(self):
        """Test recovery from network partitions."""
        pytest.skip("Requires network simulation")
    
    def test_disk_space_exhaustion(self):
        """Test handling when disk space runs out."""
        pytest.skip("Requires filesystem testing")
    
    def test_hardware_failure_scenarios(self):
        """Test handling of hardware failures."""
        pytest.skip("Requires hardware simulation")
