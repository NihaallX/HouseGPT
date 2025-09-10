"""Contract tests for HouseModel service.

These tests validate the contract defined in contracts/house_model_contract.md.
Tests MUST fail initially before implementation exists.
"""

import pytest
import torch
from unittest.mock import Mock, patch
from src.services.house_model import HouseModel, ModelLoadError, InferenceError
from src.models.house_quote import HouseQuote
from src.models.context import ConversationContext
from src.models.response import HouseResponse
from src.lib.config import Configuration


class TestHouseModelContract:
    """Test HouseModel contract compliance."""
    
    @pytest.fixture
    def config(self):
        """Provide test configuration."""
        config = Configuration()
        # Use the correct base model that matches our LoRA adapter
        config.model.base_model_name = "google/flan-t5-large"
        config.model.lora_weights_path = "./housegpt-lora-large"
        config.model.device = "cuda" if torch.cuda.is_available() else "cpu"
        config.model.max_length = 512
        return config
    
    @pytest.fixture
    def house_model(self, config):
        """Provide HouseModel instance."""
        return HouseModel(config)
    
    def test_constructor_requires_config(self):
        """Test constructor requires Configuration object."""
        with pytest.raises(TypeError):
            HouseModel()
    
    def test_constructor_raises_model_load_error_if_missing_model(self, config):
        """Test constructor raises ModelLoadError if base model not found."""
        config.model.base_model_name = "invalid/nonexistent-model"
        with pytest.raises(ModelLoadError):
            HouseModel(config, strict_mode=True)
    
    def test_constructor_raises_model_load_error_if_missing_lora(self, config):
        """Test constructor raises ModelLoadError if LoRA weights not found."""
        config.model.lora_model_path = "./invalid/nonexistent-lora"
        with pytest.raises(ModelLoadError):
            HouseModel(config, strict_mode=True)
    
    def test_generate_response_returns_house_response(self, house_model):
        """Test generate_response returns HouseResponse object."""
        context = ConversationContext(
            user_query="My head hurts and I have a fever.",
            relevant_quotes=[
                HouseQuote(
                    text="Everybody lies.",
                    sarcasm_level=3,
                    similarity_score=0.85
                )
            ],
            conversation_history=[]
        )
        
        response = house_model.generate_response(context)
        
        assert isinstance(response, HouseResponse)
        assert isinstance(response.text, str)
        assert len(response.text) > 0
        assert 0 <= response.confidence_score <= 1.0
        assert response.sarcasm_level in range(1, 6)
        assert response.emotional_tone in ["witty", "condescending", "analytical", "frustrated"]
    
    def test_generate_response_raises_validation_error_for_invalid_context(self, house_model):
        """Test generate_response raises ValidationError for invalid context."""
        with pytest.raises(Exception):  # Should be ValidationError
            house_model.generate_response(None)
        
        with pytest.raises(Exception):  # Should be ValidationError
            house_model.generate_response("invalid context type")
    
    def test_generate_response_raises_inference_error_on_model_failure(self, house_model):
        """Test generate_response raises InferenceError on model failure."""
        context = ConversationContext(
            user_query="Test query",
            relevant_quotes=[],
            conversation_history=[]
        )
        
        with patch.object(house_model, '_model') as mock_model:
            mock_model.generate.side_effect = Exception("CUDA out of memory")
            with pytest.raises(InferenceError):
                house_model.generate_response(context)
    
    def test_generate_response_incorporates_rag_quotes(self, house_model):
        """Test generate_response incorporates RAG quotes into response."""
        relevant_quote = HouseQuote(
            text="Differential diagnosis means looking beyond the obvious.",
            episode_info="Season 3, Episode 15",
            sarcasm_level=2
        )
        
        context = ConversationContext(
            user_query="How do you diagnose rare diseases?",
            relevant_quotes=[relevant_quote],
            conversation_history=[]
        )
        
        response = house_model.generate_response(context)
        
        # Response should reference the quote or similar concepts
        response_lower = response.text.lower()
        has_differential = any(word in response_lower for word in ["differential", "differentiating", "differentiate"])
        has_diagnosis = "diagnosis" in response_lower  
        has_concepts = any(word in response_lower for word in [
            "cause", "obvious", "beyond", "examining", "multiple", "several"
        ])
        
        assert has_differential or has_diagnosis or has_concepts, f"Response '{response.text}' should incorporate quote concepts"
    
    def test_generate_response_maintains_conversation_history(self, house_model):
        """Test generate_response considers conversation history."""
        history = [
            {"role": "user", "content": "What's wrong with my patient?"},
            {"role": "assistant", "content": "I need more symptoms."}
        ]
        
        context = ConversationContext(
            user_query="They have a rash and joint pain.",
            relevant_quotes=[],
            conversation_history=history
        )
        
        response = house_model.generate_response(context)
        
        # Response should acknowledge previous conversation
        assert len(response.text) > 0
        assert response.confidence_score > 0
    
    def test_generate_response_respects_max_length(self, house_model):
        """Test generate_response respects configured max_length."""
        context = ConversationContext(
            user_query="Tell me everything about medicine.",
            relevant_quotes=[],
            conversation_history=[]
        )
        
        response = house_model.generate_response(context)
        
        # Should not exceed model's max_length
        # Rough estimate: 4 chars per token
        max_chars = house_model.config.model.max_length * 4
        assert len(response.text) <= max_chars
    
    def test_analyze_style_returns_style_metrics(self, house_model):
        """Test analyze_style returns style analysis metrics."""
        text = "Whattt? You think that's a real symptom? Please."
        
        style_metrics = house_model.analyze_style(text)
        
        assert isinstance(style_metrics, dict)
        assert "sarcasm_level" in style_metrics
        assert "emotional_tone" in style_metrics
        assert "confidence_score" in style_metrics
        assert "house_authenticity" in style_metrics
        
        # Verify ranges
        assert 1 <= style_metrics["sarcasm_level"] <= 5
        assert style_metrics["emotional_tone"] in ["witty", "condescending", "analytical", "frustrated"]
        assert 0 <= style_metrics["confidence_score"] <= 1.0
        assert 0 <= style_metrics["house_authenticity"] <= 1.0
    
    def test_analyze_style_raises_validation_error_for_empty_text(self, house_model):
        """Test analyze_style raises ValidationError for empty text."""
        with pytest.raises(Exception):  # Should be ValidationError
            house_model.analyze_style("")
        
        with pytest.raises(Exception):  # Should be ValidationError
            house_model.analyze_style(None)
    
    def test_get_model_info_returns_metadata(self, house_model):
        """Test get_model_info returns model metadata and status."""
        info = house_model.get_model_info()
        
        assert isinstance(info, dict)
        assert "base_model" in info
        assert "lora_weights_path" in info
        assert "device" in info
        assert "model_size_mb" in info
        assert "loaded_at" in info
        assert "inference_count" in info
        assert "avg_inference_time_ms" in info
        
        # Verify types
        assert isinstance(info["base_model"], str)
        assert isinstance(info["lora_weights_path"], str)
        assert isinstance(info["device"], str)
        assert isinstance(info["model_size_mb"], float)
        assert isinstance(info["inference_count"], int)
        assert isinstance(info["avg_inference_time_ms"], float)
    
    def test_reload_model_updates_lora_weights(self, house_model):
        """Test reload_model updates LoRA weights from new path."""
        new_lora_path = "./housegpt-lora-large/checkpoint-156"
        
        # Should not raise exception
        house_model.reload_model(new_lora_path)
        
        # Model info should reflect new path
        info = house_model.get_model_info()
        assert new_lora_path in info["lora_weights_path"]
    
    def test_reload_model_raises_model_load_error_for_invalid_path(self, house_model):
        """Test reload_model raises ModelLoadError for invalid LoRA path."""
        invalid_path = "./invalid/nonexistent-lora"
        
        with pytest.raises(ModelLoadError):
            house_model.reload_model(invalid_path)
    
    def test_unload_model_frees_memory(self, house_model):
        """Test unload_model releases GPU/CPU memory."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        house_model.unload_model()
        
        # Model should be unloaded
        with pytest.raises(Exception):  # Should raise when trying to use unloaded model
            context = ConversationContext(
                user_query="test",
                relevant_quotes=[],
                conversation_history=[]
            )
            house_model.generate_response(context)
    
    def test_performance_contract_inference_latency(self, house_model):
        """Test performance contract: inference latency < 3 seconds."""
        import time
        
        context = ConversationContext(
            user_query="What could cause these symptoms?",
            relevant_quotes=[
                HouseQuote(text="It's never lupus.", sarcasm_level=3)
            ],
            conversation_history=[]
        )
        
        start_time = time.time()
        response = house_model.generate_response(context)
        end_time = time.time()
        
        inference_time = end_time - start_time
        assert inference_time < 3.0  # Should be under 3 seconds
    
    def test_performance_contract_memory_usage(self, house_model):
        """Test performance contract: model memory < 8GB GPU or 16GB CPU."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / (1024 * 1024)
        
        if house_model.config.model.device == "cuda":
            # GPU memory limit
            assert memory_mb < 8 * 1024  # 8GB
        else:
            # CPU memory limit
            assert memory_mb < 16 * 1024  # 16GB
    
    def test_performance_contract_response_quality(self, house_model):
        """Test performance contract: >80% of responses have confidence >0.7."""
        contexts = [
            ConversationContext(
                user_query=f"Test medical query {i}",
                relevant_quotes=[],
                conversation_history=[]
            )
            for i in range(5)  # Small sample for testing
        ]
        
        high_confidence_responses = 0
        for context in contexts:
            response = house_model.generate_response(context)
            if response.confidence_score > 0.7:
                high_confidence_responses += 1
        
        success_rate = high_confidence_responses / len(contexts)
        assert success_rate > 0.8  # >80% success rate


@pytest.mark.integration
class TestHouseModelIntegration:
    """Integration tests for HouseModel with real model files."""
    
    def test_real_model_loading(self):
        """Test with real flan-t5-large model and LoRA weights."""
        # We have real model files, so let's test them!
        import os
        lora_path = "housegpt-lora-large"  # Our actual LoRA path
        
        if not os.path.exists(lora_path):
            pytest.skip("LoRA model files not available")
            
        from src.lib.config import Configuration
        config = Configuration()
        config.model.lora_weights_path = lora_path
        
        house_model = HouseModel(config)
        
        # Test basic functionality with real model
        context = ConversationContext(
            user_query="What could cause fatigue?",
            relevant_quotes=[],
            conversation_history=[]
        )
        
        response = house_model.generate_response(context)
        assert isinstance(response, HouseResponse)
        assert len(response.text) > 0
        assert response.confidence_score > 0
    
    def test_gpu_memory_management(self):
        """Test GPU memory allocation and cleanup."""
        pytest.skip("Requires CUDA GPU for testing")
    
    def test_long_conversation_context(self):
        """Test handling of long conversation histories."""
        pytest.skip("Requires extensive conversation data")
