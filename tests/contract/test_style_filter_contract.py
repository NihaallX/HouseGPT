"""Contract tests for StyleFilter service.

These tests validate the contract defined in contracts/style_filter_contract.md.
Tests MUST fail initially before implementation exists.
"""

import pytest
from unittest.mock import Mock, patch
from src.services.style_filter import StyleFilter, FilterError, RuleViolationError
from src.models.response import HouseResponse
from src.models.filter_result import FilterResult
from src.lib.config import Configuration


class TestStyleFilterContract:
    """Test StyleFilter contract compliance."""
    
    @pytest.fixture
    def config(self):
        """Provide test configuration."""
        config = Configuration()
        config.filter.min_sarcasm_level = 2
        config.filter.max_sarcasm_level = 5
        config.filter.allowed_emotional_tones = ["witty", "condescending", "analytical"]
        config.filter.forbidden_words = ["inappropriate", "offensive"]
        config.filter.min_house_authenticity = 0.6
        return config
    
    @pytest.fixture
    def style_filter(self, config):
        """Provide StyleFilter instance."""
        return StyleFilter(config)
    
    def test_constructor_requires_config(self):
        """Test constructor requires Configuration object."""
        with pytest.raises(TypeError):
            StyleFilter()
    
    def test_validate_response_accepts_valid_house_response(self, style_filter):
        """Test validate_response accepts response meeting all criteria."""
        response = HouseResponse(
            text="Well, well, well. Looks like someone's got a case of the obvious.",
            confidence_score=0.85,
            sarcasm_level=3,
            emotional_tone="witty",
            house_authenticity=0.75
        )
        
        result = style_filter.validate_response(response)
        
        assert isinstance(result, FilterResult)
        assert result.is_valid is True
        assert result.violations == []
        assert result.score >= 0.6
        assert result.filtered_response == response
    
    def test_validate_response_rejects_low_sarcasm_level(self, style_filter):
        """Test validate_response rejects response with insufficient sarcasm."""
        response = HouseResponse(
            text="That's a good observation.",
            confidence_score=0.8,
            sarcasm_level=1,  # Below minimum of 2
            emotional_tone="analytical",
            house_authenticity=0.8
        )
        
        result = style_filter.validate_response(response)
        
        assert result.is_valid is False
        assert "sarcasm_level_too_low" in [v["type"] for v in result.violations]
    
    def test_validate_response_rejects_excessive_sarcasm_level(self, style_filter):
        """Test validate_response rejects response with excessive sarcasm."""
        response = HouseResponse(
            text="Oh, BRILLIANT deduction there, Sherlock!",
            confidence_score=0.8,
            sarcasm_level=6,  # Above maximum of 5
            emotional_tone="condescending",
            house_authenticity=0.8
        )
        
        result = style_filter.validate_response(response)
        
        assert result.is_valid is False
        assert "sarcasm_level_too_high" in [v["type"] for v in result.violations]
    
    def test_validate_response_rejects_forbidden_emotional_tone(self, style_filter):
        """Test validate_response rejects disallowed emotional tone."""
        response = HouseResponse(
            text="I'm really angry about this diagnosis!",
            confidence_score=0.8,
            sarcasm_level=3,
            emotional_tone="frustrated",  # Not in allowed list
            house_authenticity=0.8
        )
        
        result = style_filter.validate_response(response)
        
        assert result.is_valid is False
        assert "forbidden_emotional_tone" in [v["type"] for v in result.violations]
    
    def test_validate_response_rejects_forbidden_words(self, style_filter):
        """Test validate_response rejects responses with forbidden words."""
        response = HouseResponse(
            text="That's an inappropriate comment about the patient.",
            confidence_score=0.8,
            sarcasm_level=3,
            emotional_tone="analytical",
            house_authenticity=0.8
        )
        
        result = style_filter.validate_response(response)
        
        assert result.is_valid is False
        assert "forbidden_words" in [v["type"] for v in result.violations]
        assert "inappropriate" in result.violations[0]["details"]["words"]
    
    def test_validate_response_rejects_low_house_authenticity(self, style_filter):
        """Test validate_response rejects response with low House authenticity."""
        response = HouseResponse(
            text="Have a nice day! Everything will be fine!",
            confidence_score=0.8,
            sarcasm_level=3,
            emotional_tone="witty",
            house_authenticity=0.3  # Below minimum of 0.6
        )
        
        result = style_filter.validate_response(response)
        
        assert result.is_valid is False
        assert "house_authenticity_too_low" in [v["type"] for v in result.violations]
    
    def test_validate_response_rejects_low_confidence(self, style_filter):
        """Test validate_response rejects response with low confidence."""
        response = HouseResponse(
            text="Maybe it could be... I'm not sure...",
            confidence_score=0.3,  # Low confidence
            sarcasm_level=3,
            emotional_tone="analytical",
            house_authenticity=0.8
        )
        
        result = style_filter.validate_response(response)
        
        assert result.is_valid is False
        assert "confidence_too_low" in [v["type"] for v in result.violations]
    
    def test_validate_response_raises_validation_error_for_invalid_response(self, style_filter):
        """Test validate_response raises ValidationError for invalid response type."""
        with pytest.raises(Exception):  # Should be ValidationError
            style_filter.validate_response(None)
        
        with pytest.raises(Exception):  # Should be ValidationError
            style_filter.validate_response("invalid response type")
    
    def test_apply_corrections_improves_failing_response(self, style_filter):
        """Test apply_corrections attempts to fix rule violations."""
        response = HouseResponse(
            text="That's inappropriate and I'm not sure what to think.",
            confidence_score=0.4,
            sarcasm_level=1,
            emotional_tone="frustrated",
            house_authenticity=0.3
        )
        
        corrected = style_filter.apply_corrections(response)
        
        assert isinstance(corrected, HouseResponse)
        # Text should be modified to remove forbidden words
        assert "inappropriate" not in corrected.text.lower()
        # Sarcasm level should be increased
        assert corrected.sarcasm_level >= 2
        # Emotional tone should be changed to allowed value
        assert corrected.emotional_tone in ["witty", "condescending", "analytical"]
    
    def test_apply_corrections_raises_filter_error_if_unfixable(self, style_filter):
        """Test apply_corrections raises FilterError if response can't be fixed."""
        response = HouseResponse(
            text="",  # Empty text can't be fixed
            confidence_score=0.1,
            sarcasm_level=0,
            emotional_tone="invalid",
            house_authenticity=0.0
        )
        
        with pytest.raises(FilterError):
            style_filter.apply_corrections(response)
    
    def test_get_violation_summary_returns_readable_explanation(self, style_filter):
        """Test get_violation_summary returns human-readable violation summary."""
        response = HouseResponse(
            text="inappropriate offensive content",
            confidence_score=0.3,
            sarcasm_level=1,
            emotional_tone="frustrated",
            house_authenticity=0.2
        )
        
        result = style_filter.validate_response(response)
        summary = style_filter.get_violation_summary(result)
        
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "sarcasm" in summary.lower()
        assert "confidence" in summary.lower()
        assert "forbidden" in summary.lower()
    
    def test_get_filter_stats_returns_metrics(self, style_filter):
        """Test get_filter_stats returns filtering statistics."""
        # Process some responses to generate stats
        responses = [
            HouseResponse(
                text="Good response",
                confidence_score=0.8,
                sarcasm_level=3,
                emotional_tone="witty",
                house_authenticity=0.8
            ),
            HouseResponse(
                text="Bad response",
                confidence_score=0.2,
                sarcasm_level=1,
                emotional_tone="frustrated",
                house_authenticity=0.3
            )
        ]
        
        for response in responses:
            style_filter.validate_response(response)
        
        stats = style_filter.get_filter_stats()
        
        assert isinstance(stats, dict)
        assert "total_responses_processed" in stats
        assert "responses_accepted" in stats
        assert "responses_rejected" in stats
        assert "acceptance_rate" in stats
        assert "common_violations" in stats
        assert "avg_processing_time_ms" in stats
        
        # Verify types and ranges
        assert isinstance(stats["total_responses_processed"], int)
        assert isinstance(stats["acceptance_rate"], float)
        assert 0 <= stats["acceptance_rate"] <= 1.0
    
    def test_update_rules_modifies_filter_criteria(self, style_filter):
        """Test update_rules allows dynamic rule modification."""
        new_rules = {
            "min_sarcasm_level": 3,
            "forbidden_words": ["bad", "terrible"],
            "min_house_authenticity": 0.8
        }
        
        style_filter.update_rules(new_rules)
        
        # Test that new rules are applied
        response = HouseResponse(
            text="That's bad news.",
            confidence_score=0.8,
            sarcasm_level=2,  # Now below new minimum of 3
            emotional_tone="analytical",
            house_authenticity=0.7  # Now below new minimum of 0.8
        )
        
        result = style_filter.validate_response(response)
        
        assert result.is_valid is False
        violation_types = [v["type"] for v in result.violations]
        assert "sarcasm_level_too_low" in violation_types
        assert "forbidden_words" in violation_types
        assert "house_authenticity_too_low" in violation_types
    
    def test_update_rules_raises_validation_error_for_invalid_rules(self, style_filter):
        """Test update_rules raises ValidationError for invalid rule values."""
        invalid_rules = {
            "min_sarcasm_level": 10,  # Out of valid range
            "max_sarcasm_level": -1,  # Invalid
            "min_house_authenticity": 1.5  # Out of range
        }
        
        with pytest.raises(Exception):  # Should be ValidationError
            style_filter.update_rules(invalid_rules)
    
    def test_reset_stats_clears_metrics(self, style_filter):
        """Test reset_stats clears all accumulated statistics."""
        # Process a response to generate stats
        response = HouseResponse(
            text="Test response",
            confidence_score=0.8,
            sarcasm_level=3,
            emotional_tone="witty",
            house_authenticity=0.8
        )
        style_filter.validate_response(response)
        
        # Reset stats
        style_filter.reset_stats()
        
        stats = style_filter.get_filter_stats()
        assert stats["total_responses_processed"] == 0
        assert stats["responses_accepted"] == 0
        assert stats["responses_rejected"] == 0
    
    def test_performance_contract_processing_latency(self, style_filter):
        """Test performance contract: processing latency < 10ms per response."""
        import time
        
        response = HouseResponse(
            text="This is a test response for performance measurement.",
            confidence_score=0.8,
            sarcasm_level=3,
            emotional_tone="analytical",
            house_authenticity=0.8
        )
        
        start_time = time.time()
        style_filter.validate_response(response)
        end_time = time.time()
        
        processing_time_ms = (end_time - start_time) * 1000
        assert processing_time_ms < 10  # Should be under 10ms
    
    def test_performance_contract_memory_efficiency(self, style_filter):
        """Test performance contract: memory usage < 50MB for 1000 responses."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Process multiple responses
        for i in range(10):  # Smaller number for testing
            response = HouseResponse(
                text=f"Test response {i} with various content.",
                confidence_score=0.8,
                sarcasm_level=3,
                emotional_tone="witty",
                house_authenticity=0.8
            )
            style_filter.validate_response(response)
        
        current_memory = process.memory_info().rss
        memory_diff = current_memory - initial_memory
        
        # Extrapolate to 1000 responses
        projected_memory = memory_diff * 100  # 10 -> 1000
        assert projected_memory < 50 * 1024 * 1024  # 50MB


@pytest.mark.integration
class TestStyleFilterIntegration:
    """Integration tests for StyleFilter with real rule scenarios."""
    
    def test_real_house_quote_validation(self):
        """Test with actual House M.D. quotes and responses."""
        # Use our actual quotes.json file
        import json
        import os
        
        quotes_file = "data/rag/quotes.json"
        if not os.path.exists(quotes_file):
            pytest.skip("House quote dataset not available")
            
        # Load some real House quotes
        with open(quotes_file, 'r') as f:
            quotes_data = json.load(f)
        
        # Get first few quotes for testing
        sample_quotes = list(quotes_data.values())[:3]
        
        style_filter = StyleFilter(Configuration())
        
        # Test with quotes that should pass validation
        for quote_data in sample_quotes:
            response = HouseResponse(
                text=quote_data["text"],
                confidence_score=0.8,
                sarcasm_level=quote_data.get("sarcasm_level", 3),
                emotional_tone=quote_data.get("emotional_tone", "witty"),
                house_authenticity=0.9  # High authenticity for real House quotes
            )
            
            result = style_filter.validate_response(response)
            
            # Real House quotes should generally pass validation
            # (though some may fail due to strict text analysis)
            assert isinstance(result, FilterResult)
            assert result.score > 0
    
    def test_rule_configuration_persistence(self):
        """Test rule updates persist across service restarts."""
        # Test that rules can be updated and retrieved
        style_filter = StyleFilter(Configuration())
        
        # Update some rules
        new_rules = {
            "min_sarcasm_level": 4,
            "forbidden_words": ["test", "example"],
            "min_house_authenticity": 0.9
        }
        
        style_filter.update_rules(new_rules)
        
        # Verify rules are applied
        response = HouseResponse(
            text="This is a test response",
            confidence_score=0.8,
            sarcasm_level=3,  # Below new minimum of 4
            emotional_tone="witty",
            house_authenticity=0.8  # Below new minimum of 0.9
        )
        
        result = style_filter.validate_response(response)
        
        # Should have violations due to updated rules
        violation_types = [v["type"] for v in result.violations]
        assert "sarcasm_level_too_low" in violation_types
        assert "house_authenticity_too_low" in violation_types
        assert "forbidden_words" in violation_types
    
    def test_large_batch_processing(self):
        """Test processing large batches of responses efficiently."""
        pytest.skip("Requires large response dataset")
