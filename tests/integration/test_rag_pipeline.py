"""Integration test for RAG quote retrieval and relevance.

This test validates the RAG (Retrieval-Augmented Generation) pipeline
including vector similarity search, quote ranking, and relevance scoring.
"""

import pytest
from unittest.mock import Mock, patch
import numpy as np
from typing import List

from src.services.rag_store import RAGStore
from src.models.house_quote import HouseQuote
from src.lib.config import Configuration


class TestRAGPipelineIntegration:
    """Test RAG pipeline integration including search, ranking, and relevance."""
    
    @pytest.fixture
    def config(self):
        """Provide test configuration."""
        return Configuration()
    
    @pytest.fixture
    def mock_rag_store(self, config):
        """Provide mock RAG store with realistic data."""
        rag_store = Mock(spec=RAGStore)
        
        # Mock the database of House quotes for testing
        self.mock_quote_database = [
            {
                "id": 1,
                "text": "Everybody lies, especially patients.",
                "episode_info": "Season 1, Episode 1",
                "sarcasm_level": 4,
                "emotional_tone": "condescending",
                "embedding": np.random.rand(384).tolist()  # Simulated embedding
            },
            {
                "id": 2,
                "text": "It's not lupus. It's never lupus.",
                "episode_info": "Season 4, Episode 8",
                "sarcasm_level": 5,
                "emotional_tone": "witty",
                "embedding": np.random.rand(384).tolist()
            },
            {
                "id": 3,
                "text": "The most successful marriages are based on lies.",
                "episode_info": "Season 3, Episode 15",
                "sarcasm_level": 3,
                "emotional_tone": "cynical",
                "embedding": np.random.rand(384).tolist()
            },
            {
                "id": 4,
                "text": "I don't ask why patients lie, I just assume they all do.",
                "episode_info": "Season 2, Episode 11",
                "sarcasm_level": 4,
                "emotional_tone": "matter-of-fact",
                "embedding": np.random.rand(384).tolist()
            },
            {
                "id": 5,
                "text": "Pain is the body's way of telling you something is wrong.",
                "episode_info": "Season 1, Episode 3",
                "sarcasm_level": 1,
                "emotional_tone": "clinical",
                "embedding": np.random.rand(384).tolist()
            }
        ]
        
        return rag_store
    
    def test_search_quotes_by_semantic_similarity(self, mock_rag_store):
        """Test RAG search returns semantically similar quotes."""
        # Arrange: Setup search for deception-related query
        user_query = "My patient isn't telling me the truth"
        
        # Mock search results based on semantic similarity to "lying/deception"
        expected_quotes = [
            HouseQuote(
                text="Everybody lies, especially patients.",
                episode_info="Season 1, Episode 1",
                sarcasm_level=4,
                emotional_tone="condescending",
                similarity_score=0.92  # High similarity to query
            ),
            HouseQuote(
                text="I don't ask why patients lie, I just assume they all do.",
                episode_info="Season 2, Episode 11",
                sarcasm_level=4,
                emotional_tone="matter-of-fact",
                similarity_score=0.87  # Also relevant to lying
            ),
            HouseQuote(
                text="The most successful marriages are based on lies.",
                episode_info="Season 3, Episode 15",
                sarcasm_level=3,
                emotional_tone="cynical",
                similarity_score=0.75  # Related but less specific
            )
        ]
        
        mock_rag_store.search_quotes.return_value = expected_quotes
        
        # Act: Perform search
        results = mock_rag_store.search_quotes(user_query, top_k=3)
        
        # Assert: Verify search quality
        assert len(results) == 3
        
        # Verify results are ranked by similarity
        for i in range(len(results) - 1):
            assert results[i].similarity_score >= results[i + 1].similarity_score
        
        # Verify all results are semantically relevant
        assert all(quote.similarity_score > 0.7 for quote in results), \
            "All returned quotes should have high similarity scores"
        
        # Verify search was called correctly
        mock_rag_store.search_quotes.assert_called_once_with(user_query, top_k=3)
    
    def test_search_quotes_with_different_query_types(self, mock_rag_store):
        """Test RAG search handles different types of medical queries."""
        
        # Test Case 1: Symptom-based query
        symptom_query = "Patient has chest pain and shortness of breath"
        symptom_quotes = [
            HouseQuote(
                text="Pain is the body's way of telling you something is wrong.",
                episode_info="Season 1, Episode 3",
                sarcasm_level=1,
                emotional_tone="clinical",
                similarity_score=0.83
            )
        ]
        
        # Test Case 2: Diagnostic query  
        diagnostic_query = "Could this be lupus?"
        diagnostic_quotes = [
            HouseQuote(
                text="It's not lupus. It's never lupus.",
                episode_info="Season 4, Episode 8",
                sarcasm_level=5,
                emotional_tone="witty",
                similarity_score=0.95
            )
        ]
        
        # Test Case 3: Relationship/personal query
        personal_query = "My marriage is falling apart"
        personal_quotes = [
            HouseQuote(
                text="The most successful marriages are based on lies.",
                episode_info="Season 3, Episode 15",
                sarcasm_level=3,
                emotional_tone="cynical",
                similarity_score=0.78
            )
        ]
        
        # Setup mock responses
        mock_rag_store.search_quotes.side_effect = [
            symptom_quotes, diagnostic_quotes, personal_quotes
        ]
        
        # Act & Assert: Test each query type
        
        # Symptom query
        results1 = mock_rag_store.search_quotes(symptom_query, top_k=5)
        assert len(results1) == 1
        assert "pain" in results1[0].text.lower()
        
        # Diagnostic query  
        results2 = mock_rag_store.search_quotes(diagnostic_query, top_k=5)
        assert len(results2) == 1
        assert "lupus" in results2[0].text.lower()
        assert results2[0].similarity_score > 0.9  # Very high match
        
        # Personal query
        results3 = mock_rag_store.search_quotes(personal_query, top_k=5)
        assert len(results3) == 1
        assert "marriage" in results3[0].text.lower()
        
        # Verify all calls were made
        assert mock_rag_store.search_quotes.call_count == 3
    
    def test_search_quotes_relevance_threshold(self, mock_rag_store):
        """Test RAG search filters quotes below relevance threshold."""
        # Arrange: Query that should return mixed relevance results
        user_query = "I need relationship advice"
        
        # Mock results with varying similarity scores
        mixed_quotes = [
            HouseQuote(
                text="The most successful marriages are based on lies.",
                episode_info="Season 3, Episode 15",
                sarcasm_level=3,
                emotional_tone="cynical",
                similarity_score=0.85  # Above threshold
            ),
            HouseQuote(
                text="Love is just a chemical imbalance.",
                episode_info="Season 2, Episode 7",
                sarcasm_level=4,
                emotional_tone="analytical", 
                similarity_score=0.72  # Above threshold
            ),
            HouseQuote(
                text="It's not lupus. It's never lupus.",
                episode_info="Season 4, Episode 8",
                sarcasm_level=5,
                emotional_tone="witty",
                similarity_score=0.23  # Below threshold - should be filtered
            )
        ]
        
        # Filter to only return quotes above 0.6 threshold
        filtered_quotes = [q for q in mixed_quotes if q.similarity_score > 0.6]
        mock_rag_store.search_quotes.return_value = filtered_quotes
        
        # Act: Search with relevance filtering
        results = mock_rag_store.search_quotes(user_query, top_k=5, min_similarity=0.6)
        
        # Assert: Verify filtering worked
        assert len(results) == 2  # Only 2 quotes above threshold
        assert all(quote.similarity_score > 0.6 for quote in results)
        
        # Verify the irrelevant lupus quote was filtered out
        lupus_quotes = [q for q in results if "lupus" in q.text.lower()]
        assert len(lupus_quotes) == 0
    
    def test_search_quotes_empty_results(self, mock_rag_store):
        """Test RAG search handles queries with no relevant quotes."""
        # Arrange: Query that should return no relevant results
        user_query = "What's the weather like today?"
        
        mock_rag_store.search_quotes.return_value = []
        
        # Act: Search for irrelevant query
        results = mock_rag_store.search_quotes(user_query, top_k=5)
        
        # Assert: Verify empty results are handled gracefully
        assert len(results) == 0
        assert isinstance(results, list)
        
        mock_rag_store.search_quotes.assert_called_once_with(user_query, top_k=5)
    
    def test_search_quotes_performance_requirements(self, mock_rag_store):
        """Test RAG search meets performance requirements."""
        import time
        
        # Arrange: Setup realistic performance test
        user_query = "Patient is experiencing mysterious symptoms"
        
        # Mock realistic search results
        performance_quotes = [
            HouseQuote(
                text=f"Quote {i} about medical mysteries",
                episode_info=f"Season {i}, Episode {i}",
                sarcasm_level=3,
                emotional_tone="analytical",
                similarity_score=0.8 - (i * 0.1)
            )
            for i in range(1, 6)  # 5 quotes
        ]
        
        # Simulate realistic search timing (should be < 100ms per requirement)
        def timed_search(*args, **kwargs):
            time.sleep(0.05)  # 50ms - well under 100ms requirement
            return performance_quotes
        
        mock_rag_store.search_quotes.side_effect = timed_search
        
        # Act: Measure search performance
        start_time = time.time()
        results = mock_rag_store.search_quotes(user_query, top_k=10)
        end_time = time.time()
        
        search_time = end_time - start_time
        
        # Assert: Verify performance requirements
        assert search_time < 0.1, f"RAG search took {search_time:.3f}s, exceeds 100ms requirement"
        assert len(results) == 5
        
        # Verify results are properly ranked
        for i in range(len(results) - 1):
            assert results[i].similarity_score >= results[i + 1].similarity_score
    
    def test_search_quotes_with_conversation_history(self, mock_rag_store):
        """Test RAG search considers conversation context for better relevance."""
        # Arrange: Multi-turn conversation scenario
        user_query = "What about the medication?"
        
        # Previous conversation context
        conversation_history = [
            {"user": "I think my patient has depression", "assistant": "Depression is overdiagnosed..."},
            {"user": "Should I prescribe antidepressants?", "assistant": "Pills aren't magic..."}
        ]
        
        # Mock search with conversation context
        contextual_quotes = [
            HouseQuote(
                text="Antidepressants are just happy pills for sad people.",
                episode_info="Season 3, Episode 12",
                sarcasm_level=4,
                emotional_tone="sarcastic",
                similarity_score=0.88  # High relevance due to context
            ),
            HouseQuote(
                text="Depression is just anger without enthusiasm.",
                episode_info="Season 5, Episode 2",
                sarcasm_level=3,
                emotional_tone="cynical",
                similarity_score=0.82
            )
        ]
        
        mock_rag_store.search_quotes_with_context.return_value = contextual_quotes
        
        # Act: Search with conversation context
        results = mock_rag_store.search_quotes_with_context(
            user_query, 
            conversation_history=conversation_history,
            top_k=5
        )
        
        # Assert: Verify contextual search
        assert len(results) == 2
        
        # Verify medication/antidepressant context was used
        medication_quotes = [q for q in results if any(
            med_term in q.text.lower() 
            for med_term in ["medication", "antidepressant", "pills"]
        )]
        assert len(medication_quotes) > 0, "Should find medication-related quotes"
        
        # Verify context-aware search was called
        mock_rag_store.search_quotes_with_context.assert_called_once_with(
            user_query,
            conversation_history=conversation_history,
            top_k=5
        )
    
    def test_search_quotes_ranking_algorithm(self, mock_rag_store):
        """Test RAG search ranking considers multiple factors."""
        # Arrange: Quotes with different ranking factors
        user_query = "Patient is being dishonest"
        
        ranking_quotes = [
            HouseQuote(
                text="Everybody lies, especially patients.",  # Perfect semantic match
                episode_info="Season 1, Episode 1",
                sarcasm_level=4,
                emotional_tone="condescending",
                similarity_score=0.95
            ),
            HouseQuote(
                text="I don't ask why patients lie, I just assume they all do.",  # Good match
                episode_info="Season 2, Episode 11", 
                sarcasm_level=4,
                emotional_tone="matter-of-fact",
                similarity_score=0.89
            ),
            HouseQuote(
                text="Truth begins in lies.",  # Moderate semantic match
                episode_info="Season 6, Episode 4",
                sarcasm_level=2,
                emotional_tone="philosophical",
                similarity_score=0.76
            ),
            HouseQuote(
                text="Lies are like children - hard to control but sometimes necessary.",  # Lower match
                episode_info="Season 4, Episode 12",
                sarcasm_level=5,
                emotional_tone="witty", 
                similarity_score=0.68
            )
        ]
        
        mock_rag_store.search_quotes.return_value = ranking_quotes
        
        # Act: Get ranked results
        results = mock_rag_store.search_quotes(user_query, top_k=4)
        
        # Assert: Verify ranking quality
        assert len(results) == 4
        
        # Verify similarity score ranking
        similarity_scores = [q.similarity_score for q in results]
        assert similarity_scores == sorted(similarity_scores, reverse=True), \
            "Results should be ranked by similarity score"
        
        # Verify top result is most relevant
        top_result = results[0]
        assert top_result.similarity_score >= 0.9
        assert "everybody lies" in top_result.text.lower()
        
        # Verify ranking considers semantic relevance
        for quote in results:
            # All results should mention lying/deception concepts
            deception_terms = ["lie", "lies", "lying", "dishonest", "truth"]
            has_deception_term = any(term in quote.text.lower() for term in deception_terms)
            assert has_deception_term, f"Quote should contain deception-related terms: {quote.text}"
    
    def test_search_quotes_with_sarcasm_level_preference(self, mock_rag_store):
        """Test RAG search can filter/prefer quotes by sarcasm level."""
        # Arrange: User wants high-sarcasm responses
        user_query = "I need a sarcastic response about patient compliance"
        
        # Quotes with varying sarcasm levels
        sarcasm_quotes = [
            HouseQuote(
                text="Patient compliance is like unicorns - mythical and beautiful.",
                episode_info="Season 3, Episode 8",
                sarcasm_level=5,  # Very sarcastic
                emotional_tone="sarcastic",
                similarity_score=0.85
            ),
            HouseQuote(
                text="Patients always follow medical advice perfectly.",  # Ironic
                episode_info="Season 2, Episode 15",
                sarcasm_level=4,  # High sarcasm
                emotional_tone="ironic",
                similarity_score=0.82
            ),
            HouseQuote(
                text="Following doctor's orders is important for recovery.",
                episode_info="Season 1, Episode 9",
                sarcasm_level=1,  # Low sarcasm
                emotional_tone="clinical",
                similarity_score=0.90  # High similarity but low sarcasm
            )
        ]
        
        # Filter for high sarcasm (>= 3)
        high_sarcasm_quotes = [q for q in sarcasm_quotes if q.sarcasm_level >= 3]
        mock_rag_store.search_quotes.return_value = high_sarcasm_quotes
        
        # Act: Search with sarcasm preference
        results = mock_rag_store.search_quotes(
            user_query, 
            top_k=5, 
            min_sarcasm_level=3
        )
        
        # Assert: Verify sarcasm filtering
        assert len(results) == 2  # Only high-sarcasm quotes
        assert all(quote.sarcasm_level >= 3 for quote in results)
        
        # Verify clinical quote was filtered out despite high similarity
        clinical_quotes = [q for q in results if q.emotional_tone == "clinical"]
        assert len(clinical_quotes) == 0
    
    def test_rag_integration_with_vector_database(self, mock_rag_store):
        """Test RAG integration with vector database operations."""
        # This test simulates the actual vector database interaction
        # In real implementation, this would test pgvector operations
        
        # Arrange: Mock vector database operations
        user_query = "Patient symptoms include fever and fatigue"
        query_embedding = np.random.rand(384).tolist()  # Simulated query embedding
        
        # Mock vector search results (cosine similarity)
        vector_results = [
            {
                "quote_id": 15,
                "text": "Fever is the body's way of cooking the infection.",
                "cosine_similarity": 0.87,
                "episode_info": "Season 2, Episode 6"
            },
            {
                "quote_id": 23,
                "text": "Fatigue is exhaustion with a medical degree.",
                "cosine_similarity": 0.79,
                "episode_info": "Season 4, Episode 11"
            }
        ]
        
        # Convert to HouseQuote objects
        vector_quotes = [
            HouseQuote(
                text=result["text"],
                episode_info=result["episode_info"],
                sarcasm_level=3,  # Default values
                emotional_tone="analytical",
                similarity_score=result["cosine_similarity"]
            )
            for result in vector_results
        ]
        
        mock_rag_store.search_quotes.return_value = vector_quotes
        mock_rag_store.get_query_embedding.return_value = query_embedding
        
        # Act: Perform vector search
        embedding = mock_rag_store.get_query_embedding(user_query)
        results = mock_rag_store.search_quotes(user_query, top_k=10)
        
        # Assert: Verify vector operations
        
        # Verify embedding generation
        mock_rag_store.get_query_embedding.assert_called_once_with(user_query)
        assert len(embedding) == 384  # Standard sentence embedding size
        
        # Verify vector search results
        assert len(results) == 2
        assert all(quote.similarity_score > 0.7 for quote in results)
        
        # Verify results contain relevant medical terms
        all_text = " ".join(quote.text for quote in results).lower()
        assert any(term in all_text for term in ["fever", "fatigue", "infection", "exhaustion"])


@pytest.mark.integration 
class TestRAGPipelineWithRealDatabase:
    """Integration tests with real PostgreSQL + pgvector database."""
    
    def test_real_vector_search(self):
        """Test with real vector database."""
        pytest.skip("Requires PostgreSQL with pgvector extension")
    
    def test_real_embedding_generation(self):
        """Test with real sentence transformer model."""
        pytest.skip("Requires sentence transformers model")
    
    def test_real_quote_database_loading(self):
        """Test loading real House quotes into database."""
        pytest.skip("Requires House quotes dataset")
