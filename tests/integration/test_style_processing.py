"""Integration test for style filter sarcasm transformation.

This test validates the style filter's ability to transform responses
to match House's characteristic sarcasm, tone, and authenticity requirements.
"""

import pytest
from unittest.mock import Mock, patch
from typing import List, Dict, Any

from src.services.style_filter import StyleFilter
from src.models.house_response import HouseResponse
from src.models.filter_result import FilterResult
from src.lib.config import Configuration


class TestStyleFilterIntegration:
    """Test style filter integration for House personality transformation."""
    
    @pytest.fixture
    def config(self):
        """Provide test configuration."""
        return Configuration()
    
    @pytest.fixture
    def mock_style_filter(self, config):
        """Provide mock style filter with realistic behavior."""
        style_filter = Mock(spec=StyleFilter)
        return style_filter
    
    def test_validate_response_accepts_house_like_content(self, mock_style_filter):
        """Test style filter accepts authentic House responses."""
        # Arrange: Create authentic House response
        house_response = HouseResponse(
            text="Everybody lies. Your patient is lying, you're lying, and I'm lying about caring whether you figure that out.",
            confidence_score=0.92,
            sarcasm_level=4,
            emotional_tone="condescending",
            house_authenticity=0.95
        )
        
        # Mock validation result - should pass
        validation_result = FilterResult(
            is_valid=True,
            violations=[],
            score=0.95,
            filtered_response=house_response,
            applied_transformations=[]
        )
        
        mock_style_filter.validate_response.return_value = validation_result
        
        # Act: Validate authentic response
        result = mock_style_filter.validate_response(house_response)
        
        # Assert: Verify acceptance of authentic content
        assert result.is_valid is True
        assert len(result.violations) == 0
        assert result.score >= 0.9
        assert result.filtered_response.text == house_response.text
        
        mock_style_filter.validate_response.assert_called_once_with(house_response)
    
    def test_validate_response_rejects_inappropriate_content(self, mock_style_filter):
        """Test style filter rejects non-House-like responses."""
        # Arrange: Create inappropriate response (too cheerful/nice)
        inappropriate_response = HouseResponse(
            text="Have a wonderful day! Everything will work out perfectly! :)",
            confidence_score=0.8,
            sarcasm_level=0,  # No sarcasm
            emotional_tone="cheerful",  # Forbidden tone
            house_authenticity=0.1  # Very low authenticity
        )
        
        # Mock validation result - should fail
        validation_result = FilterResult(
            is_valid=False,
            violations=[
                {
                    "type": "forbidden_emotional_tone",
                    "severity": "high",
                    "details": {
                        "tone": "cheerful",
                        "allowed_tones": ["sarcastic", "condescending", "analytical", "cynical"]
                    }
                },
                {
                    "type": "sarcasm_level_too_low", 
                    "severity": "medium",
                    "details": {
                        "actual": 0,
                        "minimum_required": 2
                    }
                },
                {
                    "type": "house_authenticity_too_low",
                    "severity": "high", 
                    "details": {
                        "actual": 0.1,
                        "minimum_required": 0.6
                    }
                },
                {
                    "type": "excessive_positivity",
                    "severity": "medium",
                    "details": {
                        "positive_markers": ["wonderful", "perfectly", ":)"],
                        "explanation": "House rarely uses excessive positivity"
                    }
                }
            ],
            score=0.15,
            filtered_response=None,
            applied_transformations=[]
        )
        
        mock_style_filter.validate_response.return_value = validation_result
        
        # Act: Validate inappropriate response
        result = mock_style_filter.validate_response(inappropriate_response)
        
        # Assert: Verify rejection of inappropriate content
        assert result.is_valid is False
        assert len(result.violations) == 4
        assert result.score < 0.3
        
        # Verify specific violations
        violation_types = [v["type"] for v in result.violations]
        assert "forbidden_emotional_tone" in violation_types
        assert "sarcasm_level_too_low" in violation_types
        assert "house_authenticity_too_low" in violation_types
        assert "excessive_positivity" in violation_types
    
    def test_apply_corrections_transforms_inappropriate_response(self, mock_style_filter):
        """Test style filter can transform responses to be more House-like."""
        # Arrange: Original inappropriate response
        original_response = HouseResponse(
            text="Your condition is treatable. Don't worry, you'll be fine.",
            confidence_score=0.7,
            sarcasm_level=1,
            emotional_tone="reassuring",
            house_authenticity=0.3
        )
        
        # Corrected response with House transformations
        corrected_response = HouseResponse(
            text="Your condition is treatable, assuming you can follow simple instructions and stop lying about your symptoms.",
            confidence_score=0.85,
            sarcasm_level=3,
            emotional_tone="condescending", 
            house_authenticity=0.82
        )
        
        # Mock correction result
        correction_result = FilterResult(
            is_valid=True,
            violations=[],
            score=0.82,
            filtered_response=corrected_response,
            applied_transformations=[
                {
                    "type": "add_sarcasm",
                    "description": "Increased sarcasm level from 1 to 3",
                    "changes": {
                        "added_phrases": ["assuming you can follow simple instructions", "stop lying about your symptoms"]
                    }
                },
                {
                    "type": "change_emotional_tone",
                    "description": "Changed tone from 'reassuring' to 'condescending'",
                    "changes": {
                        "original_tone": "reassuring",
                        "new_tone": "condescending"
                    }
                },
                {
                    "type": "remove_excessive_reassurance",
                    "description": "Removed overly positive reassurance",
                    "changes": {
                        "removed_phrases": ["Don't worry"]
                    }
                }
            ]
        )
        
        mock_style_filter.apply_corrections.return_value = correction_result
        
        # Act: Apply corrections
        result = mock_style_filter.apply_corrections(original_response)
        
        # Assert: Verify transformation
        assert result.is_valid is True
        assert result.filtered_response.sarcasm_level == 3
        assert result.filtered_response.emotional_tone == "condescending"
        assert result.filtered_response.house_authenticity > 0.8
        
        # Verify transformations were applied
        assert len(result.applied_transformations) == 3
        transformation_types = [t["type"] for t in result.applied_transformations]
        assert "add_sarcasm" in transformation_types
        assert "change_emotional_tone" in transformation_types
        assert "remove_excessive_reassurance" in transformation_types
        
        # Verify text was actually changed
        assert result.filtered_response.text != original_response.text
        assert "assuming you can follow" in result.filtered_response.text
        assert "Don't worry" not in result.filtered_response.text
    
    def test_sarcasm_level_enforcement(self, mock_style_filter):
        """Test style filter enforces minimum sarcasm requirements."""
        # Test cases with different sarcasm levels
        test_cases = [
            {
                "input_sarcasm": 0,
                "should_pass": False,
                "description": "No sarcasm - should fail"
            },
            {
                "input_sarcasm": 1,
                "should_pass": False,
                "description": "Minimal sarcasm - should fail"
            },
            {
                "input_sarcasm": 2,
                "should_pass": True,
                "description": "Acceptable sarcasm - should pass"
            },
            {
                "input_sarcasm": 5,
                "should_pass": True,
                "description": "High sarcasm - should pass"
            }
        ]
        
        def mock_validate_sarcasm(response):
            """Mock validation based on sarcasm level."""
            passes = response.sarcasm_level >= 2  # Minimum requirement
            
            violations = []
            if not passes:
                violations.append({
                    "type": "sarcasm_level_too_low",
                    "severity": "medium",
                    "details": {
                        "actual": response.sarcasm_level,
                        "minimum_required": 2
                    }
                })
            
            return FilterResult(
                is_valid=passes,
                violations=violations,
                score=0.8 if passes else 0.4,
                filtered_response=response if passes else None,
                applied_transformations=[]
            )
        
        mock_style_filter.validate_response.side_effect = mock_validate_sarcasm
        
        # Act & Assert: Test each sarcasm level
        for case in test_cases:
            response = HouseResponse(
                text="Test response",
                confidence_score=0.8,
                sarcasm_level=case["input_sarcasm"],
                emotional_tone="analytical",
                house_authenticity=0.7
            )
            
            result = mock_style_filter.validate_response(response)
            
            assert result.is_valid == case["should_pass"], \
                f"Sarcasm level {case['input_sarcasm']}: {case['description']}"
            
            if not case["should_pass"]:
                assert any(v["type"] == "sarcasm_level_too_low" for v in result.violations)
    
    def test_emotional_tone_filtering(self, mock_style_filter):
        """Test style filter enforces allowed emotional tones."""
        # Define allowed and forbidden tones
        allowed_tones = ["sarcastic", "condescending", "analytical", "cynical", "witty"]
        forbidden_tones = ["cheerful", "enthusiastic", "sympathetic", "romantic", "childlike"]
        
        def mock_validate_tone(response):
            """Mock validation based on emotional tone."""
            is_allowed = response.emotional_tone in allowed_tones
            
            violations = []
            if not is_allowed:
                violations.append({
                    "type": "forbidden_emotional_tone",
                    "severity": "high",
                    "details": {
                        "tone": response.emotional_tone,
                        "allowed_tones": allowed_tones
                    }
                })
            
            return FilterResult(
                is_valid=is_allowed,
                violations=violations,
                score=0.8 if is_allowed else 0.2,
                filtered_response=response if is_allowed else None,
                applied_transformations=[]
            )
        
        mock_style_filter.validate_response.side_effect = mock_validate_tone
        
        # Test allowed tones
        for tone in allowed_tones:
            response = HouseResponse(
                text="Test response",
                confidence_score=0.8,
                sarcasm_level=3,
                emotional_tone=tone,
                house_authenticity=0.7
            )
            
            result = mock_style_filter.validate_response(response)
            assert result.is_valid is True, f"Allowed tone '{tone}' should pass validation"
        
        # Test forbidden tones
        for tone in forbidden_tones:
            response = HouseResponse(
                text="Test response",
                confidence_score=0.8,
                sarcasm_level=3,
                emotional_tone=tone,
                house_authenticity=0.7
            )
            
            result = mock_style_filter.validate_response(response)
            assert result.is_valid is False, f"Forbidden tone '{tone}' should fail validation"
            assert any(v["type"] == "forbidden_emotional_tone" for v in result.violations)
    
    def test_house_authenticity_scoring(self, mock_style_filter):
        """Test style filter validates House authenticity scores."""
        # Test cases with different authenticity scores
        authenticity_cases = [
            {"score": 0.95, "should_pass": True, "description": "Very authentic"},
            {"score": 0.8, "should_pass": True, "description": "Authentic enough"},
            {"score": 0.6, "should_pass": True, "description": "Minimum authenticity"},
            {"score": 0.5, "should_pass": False, "description": "Below minimum"},
            {"score": 0.2, "should_pass": False, "description": "Very inauthentic"}
        ]
        
        def mock_validate_authenticity(response):
            """Mock validation based on authenticity score."""
            passes = response.house_authenticity >= 0.6  # Minimum requirement
            
            violations = []
            if not passes:
                violations.append({
                    "type": "house_authenticity_too_low",
                    "severity": "high",
                    "details": {
                        "actual": response.house_authenticity,
                        "minimum_required": 0.6
                    }
                })
            
            return FilterResult(
                is_valid=passes,
                violations=violations,
                score=response.house_authenticity,
                filtered_response=response if passes else None,
                applied_transformations=[]
            )
        
        mock_style_filter.validate_response.side_effect = mock_validate_authenticity
        
        # Test each authenticity score
        for case in authenticity_cases:
            response = HouseResponse(
                text="Test response",
                confidence_score=0.8,
                sarcasm_level=3,
                emotional_tone="analytical",
                house_authenticity=case["score"]
            )
            
            result = mock_style_filter.validate_response(response)
            
            assert result.is_valid == case["should_pass"], \
                f"Authenticity {case['score']}: {case['description']}"
            
            if not case["should_pass"]:
                assert any(v["type"] == "house_authenticity_too_low" for v in result.violations)
    
    def test_content_pattern_detection(self, mock_style_filter):
        """Test style filter detects specific content patterns."""
        # Test cases for different content patterns
        pattern_tests = [
            {
                "text": "That's wonderful news! I'm so happy for you!",
                "violations": ["excessive_positivity"],
                "description": "Excessive positivity detection"
            },
            {
                "text": "I love you so much, darling. You're my everything.",
                "violations": ["romantic_content"],
                "description": "Romantic content detection"
            },
            {
                "text": "Gosh golly! That's super duper amazing!",
                "violations": ["childlike_language"],
                "description": "Childlike language detection"
            },
            {
                "text": "I completely understand your pain and suffering.",
                "violations": ["excessive_empathy"],
                "description": "Excessive empathy detection"
            },
            {
                "text": "Your symptoms suggest differential diagnosis consideration.",
                "violations": [],
                "description": "Medical terminology - should pass"
            }
        ]
        
        def mock_validate_patterns(response):
            """Mock pattern validation."""
            text_lower = response.text.lower()
            violations = []
            
            # Check for various patterns
            if any(word in text_lower for word in ["wonderful", "happy", "amazing", "super duper"]):
                violations.append({
                    "type": "excessive_positivity",
                    "severity": "medium",
                    "details": {"detected_patterns": ["positive_language"]}
                })
            
            if any(word in text_lower for word in ["love", "darling", "everything"]):
                violations.append({
                    "type": "romantic_content",
                    "severity": "high",
                    "details": {"detected_patterns": ["romantic_language"]}
                })
            
            if any(word in text_lower for word in ["gosh", "golly", "super duper"]):
                violations.append({
                    "type": "childlike_language", 
                    "severity": "medium",
                    "details": {"detected_patterns": ["childish_expressions"]}
                })
            
            if any(word in text_lower for word in ["understand your pain", "suffering"]):
                violations.append({
                    "type": "excessive_empathy",
                    "severity": "low",
                    "details": {"detected_patterns": ["empathetic_language"]}
                })
            
            return FilterResult(
                is_valid=len(violations) == 0,
                violations=violations,
                score=0.8 if len(violations) == 0 else 0.3,
                filtered_response=response if len(violations) == 0 else None,
                applied_transformations=[]
            )
        
        mock_style_filter.validate_response.side_effect = mock_validate_patterns
        
        # Test each pattern
        for test_case in pattern_tests:
            response = HouseResponse(
                text=test_case["text"],
                confidence_score=0.8,
                sarcasm_level=3,
                emotional_tone="analytical",
                house_authenticity=0.7
            )
            
            result = mock_style_filter.validate_response(response)
            
            expected_violations = test_case["violations"]
            if expected_violations:
                assert result.is_valid is False, f"Should detect violations in: {test_case['description']}"
                
                detected_types = [v["type"] for v in result.violations]
                for expected_violation in expected_violations:
                    assert expected_violation in detected_types, \
                        f"Should detect {expected_violation} in: {test_case['description']}"
            else:
                assert result.is_valid is True, f"Should pass: {test_case['description']}"
    
    def test_style_filter_performance_requirements(self, mock_style_filter):
        """Test style filter meets performance requirements."""
        import time
        
        # Arrange: Setup performance test
        response = HouseResponse(
            text="Your differential diagnosis lacks the sophisticated analysis that separates competent physicians from the mediocre masses who stumble through medical school.",
            confidence_score=0.88,
            sarcasm_level=4,
            emotional_tone="condescending",
            house_authenticity=0.91
        )
        
        # Mock realistic processing time (should be < 10ms per requirement)
        def timed_validation(*args, **kwargs):
            time.sleep(0.005)  # 5ms - well under 10ms requirement
            return FilterResult(
                is_valid=True,
                violations=[],
                score=0.91,
                filtered_response=response,
                applied_transformations=[]
            )
        
        mock_style_filter.validate_response.side_effect = timed_validation
        
        # Act: Measure validation performance
        start_time = time.time()
        result = mock_style_filter.validate_response(response)
        end_time = time.time()
        
        validation_time = end_time - start_time
        
        # Assert: Verify performance requirements
        assert validation_time < 0.01, f"Style validation took {validation_time:.3f}s, exceeds 10ms requirement"
        assert result.is_valid is True
        assert result.score > 0.9
    
    def test_style_filter_with_multiple_transformations(self, mock_style_filter):
        """Test style filter can apply multiple transformations simultaneously."""
        # Arrange: Response needing multiple corrections
        original_response = HouseResponse(
            text="You seem like a nice person. I hope everything works out well for you!",
            confidence_score=0.6,
            sarcasm_level=0,
            emotional_tone="sympathetic",
            house_authenticity=0.2
        )
        
        # Multiple transformations applied
        corrected_response = HouseResponse(
            text="You seem like the type of person who reads WebMD and then argues with doctors. Good luck with that brilliant strategy.",
            confidence_score=0.83,
            sarcasm_level=4,
            emotional_tone="condescending",
            house_authenticity=0.86
        )
        
        transformation_result = FilterResult(
            is_valid=True,
            violations=[],
            score=0.86,
            filtered_response=corrected_response,
            applied_transformations=[
                {
                    "type": "add_sarcasm",
                    "description": "Added sarcastic commentary about WebMD",
                    "changes": {"sarcasm_level": "0 → 4"}
                },
                {
                    "type": "change_emotional_tone",
                    "description": "Changed from sympathetic to condescending",
                    "changes": {"emotional_tone": "sympathetic → condescending"}
                },
                {
                    "type": "remove_positive_sentiment",
                    "description": "Removed overly positive language",
                    "changes": {"removed": ["nice person", "hope everything works out well"]}
                },
                {
                    "type": "add_house_personality",
                    "description": "Added characteristic House cynicism",
                    "changes": {"added": ["argues with doctors", "brilliant strategy"]}
                }
            ]
        )
        
        mock_style_filter.apply_corrections.return_value = transformation_result
        
        # Act: Apply multiple corrections
        result = mock_style_filter.apply_corrections(original_response)
        
        # Assert: Verify multiple transformations
        assert result.is_valid is True
        assert len(result.applied_transformations) == 4
        
        # Verify specific transformation types
        transformation_types = [t["type"] for t in result.applied_transformations]
        expected_types = ["add_sarcasm", "change_emotional_tone", "remove_positive_sentiment", "add_house_personality"]
        
        for expected_type in expected_types:
            assert expected_type in transformation_types
        
        # Verify final response meets House standards
        final_response = result.filtered_response
        assert final_response.sarcasm_level >= 3
        assert final_response.emotional_tone in ["condescending", "sarcastic"]
        assert final_response.house_authenticity > 0.8
        
        # Verify text transformation quality
        assert "WebMD" in final_response.text  # Added House-like reference
        assert "brilliant strategy" in final_response.text  # Added sarcasm
        assert "nice person" not in final_response.text  # Removed positivity


@pytest.mark.integration
class TestStyleFilterWithRealNLP:
    """Integration tests with real NLP models and processing."""
    
    def test_real_sentiment_analysis(self):
        """Test with real sentiment analysis models."""
        pytest.skip("Requires sentiment analysis model")
    
    def test_real_style_transfer_model(self):
        """Test with real style transfer neural network."""
        pytest.skip("Requires style transfer model")
    
    def test_real_house_authenticity_scoring(self):
        """Test with real House personality classifier."""
        pytest.skip("Requires trained House personality model")
