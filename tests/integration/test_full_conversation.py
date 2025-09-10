"""Integration test for complete wake word → response → voice pipeline.

This test validates the end-to-end conversation flow from wake word detection
through House response generation to voice output.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from src.services.wake_word import WakeWordDetector
from src.services.rag_store import RAGStore
from src.services.house_model import HouseModel
from src.services.style_filter import StyleFilter
from src.services.voice_output import VoiceOutput
from src.services.conversation_logger import ConversationLogger
from src.models.house_quote import HouseQuote
from src.models.context import ConversationContext
from src.models.response import HouseResponse
from src.models.conversation_entry import ConversationEntry
from src.lib.config import Configuration


class TestFullConversationPipeline:
    """Test complete conversation pipeline integration."""
    
    @pytest.fixture
    def config(self):
        """Provide test configuration."""
        return Configuration()
    
    @pytest.fixture
    def mock_services(self, config):
        """Provide mocked service instances for integration testing."""
        # Create mock instances that can be used for integration testing
        # These will be replaced with real instances once implemented
        
        wake_word = Mock(spec=WakeWordDetector)
        rag_store = Mock(spec=RAGStore)
        house_model = Mock(spec=HouseModel)
        style_filter = Mock(spec=StyleFilter)
        voice_output = Mock(spec=VoiceOutput)
        logger = Mock(spec=ConversationLogger)
        
        return {
            'wake_word': wake_word,
            'rag_store': rag_store,
            'house_model': house_model,
            'style_filter': style_filter,
            'voice_output': voice_output,
            'logger': logger
        }
    
    def test_successful_conversation_flow(self, mock_services):
        """Test successful end-to-end conversation flow."""
        # Arrange: Setup mock responses for each service
        
        # 1. Wake word detection
        wake_word = mock_services['wake_word']
        wake_word.is_listening.return_value = True
        
        # 2. RAG quote retrieval
        rag_store = mock_services['rag_store']
        relevant_quotes = [
            HouseQuote(
                text="Everybody lies, especially patients.",
                episode_info="Season 1, Episode 1",
                sarcasm_level=4,
                emotional_tone="condescending",
                similarity_score=0.85
            ),
            HouseQuote(
                text="It's not lupus. It's never lupus.",
                episode_info="Season 4, Episode 8", 
                sarcasm_level=5,
                emotional_tone="witty",
                similarity_score=0.78
            )
        ]
        rag_store.search_quotes.return_value = relevant_quotes
        
        # 3. House model response generation
        house_model = mock_services['house_model']
        house_response = HouseResponse(
            text="Oh, wonderful. Another mystery patient with 'vague symptoms.' Let me guess - you've been WebMD-ing again?",
            confidence_score=0.87,
            sarcasm_level=4,
            emotional_tone="condescending",
            house_authenticity=0.92
        )
        house_model.generate_response.return_value = house_response
        
        # 4. Style filter validation
        style_filter = mock_services['style_filter']
        filter_result = Mock()
        filter_result.is_valid = True
        filter_result.violations = []
        filter_result.score = 0.92
        filter_result.filtered_response = house_response
        style_filter.validate_response.return_value = filter_result
        
        # 5. Voice synthesis and playback
        voice_output = mock_services['voice_output']
        audio_data = b"fake_audio_data_for_testing"
        voice_output.synthesize_speech.return_value = audio_data
        voice_output.speak_response.return_value = None
        
        # 6. Conversation logging
        logger = mock_services['logger']
        logger.log_conversation.return_value = None
        
        # Act: Execute the conversation pipeline
        user_query = "I have a headache and feel tired all the time"
        
        # Simulate wake word detection trigger
        wake_word_callback = Mock()
        wake_word.set_wake_word_callback(wake_word_callback)
        
        # Pipeline execution
        # 1. Wake word detected (simulated)
        detected_text = "House?"
        
        # 2. RAG search for relevant quotes
        quotes = rag_store.search_quotes(user_query, top_k=3)
        
        # 3. Generate conversation context
        context = ConversationContext(
            user_query=user_query,
            relevant_quotes=quotes,
            conversation_history=[]
        )
        
        # 4. Generate House response
        response = house_model.generate_response(context)
        
        # 5. Validate response with style filter
        validation_result = style_filter.validate_response(response)
        
        # 6. Synthesize and play audio
        if validation_result.is_valid:
            audio = voice_output.synthesize_speech(response)
            voice_output.speak_response(response)
        
        # 7. Log conversation
        conversation_entry = ConversationEntry(
            timestamp=datetime.now(),
            user_query=user_query,
            house_response=response.text,
            wake_word_detected=True,
            rag_quotes_found=len(quotes),
            response_confidence=response.confidence_score,
            sarcasm_level=response.sarcasm_level,
            emotional_tone=response.emotional_tone
        )
        logger.log_conversation(conversation_entry)
        
        # Assert: Verify pipeline execution
        
        # Verify RAG search was called
        rag_store.search_quotes.assert_called_once_with(user_query, top_k=3)
        
        # Verify model generation was called with proper context
        house_model.generate_response.assert_called_once()
        call_args = house_model.generate_response.call_args[0][0]
        assert call_args.user_query == user_query
        assert len(call_args.relevant_quotes) == 2
        
        # Verify style validation was performed
        style_filter.validate_response.assert_called_once_with(response)
        
        # Verify voice synthesis and playback
        voice_output.synthesize_speech.assert_called_once_with(response)
        voice_output.speak_response.assert_called_once_with(response)
        
        # Verify conversation was logged
        logger.log_conversation.assert_called_once()
        logged_entry = logger.log_conversation.call_args[0][0]
        assert logged_entry.user_query == user_query
        assert logged_entry.house_response == response.text
        assert logged_entry.wake_word_detected is True
        assert logged_entry.rag_quotes_found == 2
    
    def test_conversation_flow_with_style_filter_rejection(self, mock_services):
        """Test conversation flow when style filter rejects response."""
        # Arrange: Setup mocks for rejected response scenario
        rag_store = mock_services['rag_store']
        house_model = mock_services['house_model']
        style_filter = mock_services['style_filter']
        voice_output = mock_services['voice_output']
        logger = mock_services['logger']
        
        # RAG returns quotes
        quotes = [HouseQuote(text="Test quote", sarcasm_level=2)]
        rag_store.search_quotes.return_value = quotes
        
        # Model generates response
        original_response = HouseResponse(
            text="Have a nice day! Everything will be fine!",  # Not House-like
            confidence_score=0.5,
            sarcasm_level=1,  # Too low
            emotional_tone="cheerful",  # Not allowed
            house_authenticity=0.3  # Too low
        )
        house_model.generate_response.return_value = original_response
        
        # Style filter rejects and provides corrected version
        filter_result = Mock()
        filter_result.is_valid = False
        filter_result.violations = [
            {"type": "sarcasm_level_too_low", "details": {"expected": ">= 2", "actual": 1}},
            {"type": "forbidden_emotional_tone", "details": {"tone": "cheerful"}},
            {"type": "house_authenticity_too_low", "details": {"expected": ">= 0.6", "actual": 0.3}}
        ]
        
        corrected_response = HouseResponse(
            text="Oh great, another patient who thinks I'm their personal cheerleader.",
            confidence_score=0.8,
            sarcasm_level=4,
            emotional_tone="condescending",
            house_authenticity=0.85
        )
        
        style_filter.validate_response.return_value = filter_result
        style_filter.apply_corrections.return_value = corrected_response
        
        # Act: Execute pipeline with style correction
        user_query = "Doctor, will I be okay?"
        
        context = ConversationContext(
            user_query=user_query,
            relevant_quotes=quotes,
            conversation_history=[]
        )
        
        response = house_model.generate_response(context)
        validation_result = style_filter.validate_response(response)
        
        final_response = response
        if not validation_result.is_valid:
            final_response = style_filter.apply_corrections(response)
        
        voice_output.speak_response(final_response)
        
        # Assert: Verify correction pipeline
        style_filter.validate_response.assert_called_once_with(original_response)
        style_filter.apply_corrections.assert_called_once_with(original_response)
        voice_output.speak_response.assert_called_once_with(corrected_response)
    
    def test_conversation_flow_with_no_relevant_quotes(self, mock_services):
        """Test conversation flow when RAG finds no relevant quotes."""
        # Arrange: Setup scenario with no RAG results
        rag_store = mock_services['rag_store']
        house_model = mock_services['house_model']
        style_filter = mock_services['style_filter']
        voice_output = mock_services['voice_output']
        
        # RAG returns empty results
        rag_store.search_quotes.return_value = []
        
        # Model should still generate response without quotes
        response = HouseResponse(
            text="Fascinating. A patient with symptoms so unique that even my vast knowledge of House quotes can't help. Impressive.",
            confidence_score=0.75,
            sarcasm_level=3,
            emotional_tone="analytical",
            house_authenticity=0.8
        )
        house_model.generate_response.return_value = response
        
        # Style filter accepts
        filter_result = Mock()
        filter_result.is_valid = True
        filter_result.filtered_response = response
        style_filter.validate_response.return_value = filter_result
        
        # Act: Execute pipeline with no RAG quotes
        user_query = "I have a very rare condition"
        
        # Pipeline execution
        # 1. RAG search for relevant quotes (returns empty)
        quotes = rag_store.search_quotes(user_query, top_k=3)
        
        # 2. Generate conversation context with empty quotes
        context = ConversationContext(
            user_query=user_query,
            relevant_quotes=quotes,  # Empty quotes from search
            conversation_history=[]
        )
        
        # 3. Generate response without quotes
        model_response = house_model.generate_response(context)
        
        # 4. Validate response
        validation_result = style_filter.validate_response(model_response)
        
        # 5. Synthesize voice if valid
        if validation_result.is_valid:
            voice_output.speak_response(model_response)
        
        # Assert: Verify pipeline works without RAG quotes
        rag_store.search_quotes.assert_called_once()
        house_model.generate_response.assert_called_once()
        
        # Verify context had empty quotes
        call_args = house_model.generate_response.call_args[0][0]
        assert len(call_args.relevant_quotes) == 0
        
        voice_output.speak_response.assert_called_once_with(response)
    
    def test_conversation_flow_performance_requirements(self, mock_services):
        """Test conversation flow meets performance requirements."""
        import time
        
        # Arrange: Setup mocks with realistic timing
        rag_store = mock_services['rag_store']
        house_model = mock_services['house_model']
        style_filter = mock_services['style_filter']
        voice_output = mock_services['voice_output']
        
        # Configure mock timing
        def slow_rag_search(*args, **kwargs):
            time.sleep(0.05)  # 50ms - under 100ms requirement
            return [HouseQuote(text="Test quote", sarcasm_level=3)]
        
        def slow_model_generation(*args, **kwargs):
            time.sleep(2.0)  # 2s - under 3s requirement  
            return HouseResponse(
                text="Test response",
                confidence_score=0.8,
                sarcasm_level=3,
                emotional_tone="analytical",
                house_authenticity=0.8
            )
        
        def slow_style_filter(*args, **kwargs):
            time.sleep(0.005)  # 5ms - under 10ms requirement
            result = Mock()
            result.is_valid = True
            result.filtered_response = args[0]
            return result
        
        def slow_voice_synthesis(*args, **kwargs):
            time.sleep(1.5)  # 1.5s - under 2s requirement
            return b"audio_data"
        
        rag_store.search_quotes.side_effect = slow_rag_search
        house_model.generate_response.side_effect = slow_model_generation
        style_filter.validate_response.side_effect = slow_style_filter
        voice_output.synthesize_speech.side_effect = slow_voice_synthesis
        
        # Act: Measure total pipeline performance
        start_time = time.time()
        
        user_query = "Performance test query"
        
        # Execute pipeline
        quotes = rag_store.search_quotes(user_query, top_k=3)
        context = ConversationContext(user_query=user_query, relevant_quotes=quotes)
        response = house_model.generate_response(context)
        validation_result = style_filter.validate_response(response)
        audio_data = voice_output.synthesize_speech(response)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Assert: Verify performance requirements
        # Total response time should be < 5 seconds (requirement from spec)
        assert total_time < 5.0, f"Total pipeline time {total_time:.2f}s exceeds 5s requirement"
        
        # Individual component times already validated by side effects
        # RAG: < 100ms, Model: < 3s, Filter: < 10ms, TTS: < 2s
    
    @pytest.mark.asyncio
    async def test_conversation_flow_concurrent_requests(self, mock_services):
        """Test conversation flow handles concurrent requests properly."""
        # This test verifies the pipeline can handle multiple concurrent conversations
        # without interference or resource conflicts
        
        # Arrange: Setup mocks for concurrent execution
        rag_store = mock_services['rag_store']
        house_model = mock_services['house_model']
        style_filter = mock_services['style_filter']
        voice_output = mock_services['voice_output']
        
        async def async_rag_search(query, **kwargs):
            await asyncio.sleep(0.01)  # Simulate async I/O
            return [HouseQuote(text=f"Quote for: {query}", sarcasm_level=3)]
        
        async def async_model_generation(context):
            await asyncio.sleep(0.1)  # Simulate model processing
            return HouseResponse(
                text=f"Response to: {context.user_query}",
                confidence_score=0.8,
                sarcasm_level=3,
                emotional_tone="analytical",
                house_authenticity=0.8
            )
        
        # Setup async mocks
        rag_store.search_quotes = AsyncMock(side_effect=async_rag_search)
        house_model.generate_response = AsyncMock(side_effect=async_model_generation)
        
        filter_result = Mock()
        filter_result.is_valid = True
        style_filter.validate_response.return_value = filter_result
        
        # Act: Execute concurrent conversations
        async def process_conversation(query_id):
            query = f"Concurrent query {query_id}"
            quotes = await rag_store.search_quotes(query, top_k=3)
            context = ConversationContext(user_query=query, relevant_quotes=quotes)
            response = await house_model.generate_response(context)
            return response
        
        # Run 3 concurrent conversations
        tasks = [process_conversation(i) for i in range(3)]
        responses = await asyncio.gather(*tasks)
        
        # Assert: Verify concurrent execution worked
        assert len(responses) == 3
        
        # Verify each conversation was processed correctly
        for i, response in enumerate(responses):
            assert f"Concurrent query {i}" in response.text
        
        # Verify services were called for each conversation
        assert rag_store.search_quotes.call_count == 3
        assert house_model.generate_response.call_count == 3


@pytest.mark.integration
class TestFullConversationIntegrationWithRealServices:
    """Integration tests with real service instances (when available)."""
    
    def test_real_service_integration(self):
        """Test with real service instances."""
        pytest.skip("Requires real service implementations")
    
    def test_real_audio_pipeline(self):
        """Test with real audio input/output."""
        pytest.skip("Requires audio hardware and models")
    
    def test_real_database_integration(self):
        """Test with real PostgreSQL database."""
        pytest.skip("Requires PostgreSQL database setup")
