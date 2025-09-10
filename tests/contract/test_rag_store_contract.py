"""Contract tests for RAGStore service.

These tests validate the contract defined in contracts/rag_store_contract.md.
Tests MUST fail initially before implementation exists.
"""

import pytest
from unittest.mock import Mock, patch
from src.services.rag_store import RAGStore, RAGStoreError, DatabaseConnectionError
from src.models.house_quote import HouseQuote
from src.lib.config import Configuration


class TestRAGStoreContract:
    """Test RAGStore contract compliance."""
    
    @pytest.fixture
    def config(self):
        """Provide test configuration."""
        config = Configuration()
        config.database.url = "postgresql://test:test@localhost:5432/test_housegpt"
        return config
    
    @pytest.fixture
    def rag_store(self, config):
        """Provide RAGStore instance."""
        return RAGStore(config)
    
    def test_constructor_requires_config(self):
        """Test constructor requires Configuration object."""
        with pytest.raises(TypeError):
            RAGStore()
    
    def test_constructor_raises_database_connection_error(self, config):
        """Test constructor raises DatabaseConnectionError if PostgreSQL fails."""
        config.database.url = "postgresql://invalid:invalid@invalid:5432/invalid"
        with pytest.raises(DatabaseConnectionError):
            RAGStore(config)
    
    def test_constructor_raises_vector_extension_error(self, config):
        """Test constructor raises VectorExtensionError if pgvector not available."""
        with patch('psycopg2.connect') as mock_connect:
            mock_connect.return_value.cursor.return_value.execute.side_effect = \
                Exception("pgvector extension not found")
            with pytest.raises(Exception):  # Should be VectorExtensionError
                RAGStore(config)
    
    def test_add_quote_stores_with_embedding(self, rag_store):
        """Test add_quote stores HouseQuote with generated embedding."""
        quote = HouseQuote(
            text="Whattt? You think lupus is the answer to everything?",
            episode_info="Season 1, Episode 1",
            sarcasm_level=4,
            emotional_tone="condescending"
        )
        
        quote_id = rag_store.add_quote(quote)
        
        # Should return valid UUID
        assert isinstance(quote_id, str)
        assert len(quote_id) == 36  # UUID format
    
    def test_add_quote_raises_validation_error_for_invalid_quote(self, rag_store):
        """Test add_quote raises ValidationError for invalid quote."""
        invalid_quote = HouseQuote(
            text="",  # Empty text should fail validation
            sarcasm_level=6  # Out of range should fail
        )
        
        with pytest.raises(Exception):  # Should be ValidationError
            rag_store.add_quote(invalid_quote)
    
    def test_add_quote_raises_duplicate_error_for_identical_quote(self, rag_store):
        """Test add_quote raises DuplicateQuoteError for identical quote."""
        quote = HouseQuote(
            text="You know what's interesting about this case?",
            sarcasm_level=2
        )
        
        # Add quote first time
        rag_store.add_quote(quote)
        
        # Adding same quote should raise error
        with pytest.raises(Exception):  # Should be DuplicateQuoteError
            rag_store.add_quote(quote)
    
    def test_search_quotes_returns_relevant_results(self, rag_store):
        """Test search_quotes finds most relevant quotes using semantic similarity."""
        query = "What's wrong with my patient who has a fever?"
        results = rag_store.search_quotes(query, top_k=3)
        
        # Should return list of HouseQuote objects
        assert isinstance(results, list)
        assert len(results) <= 3
        
        for quote in results:
            assert isinstance(quote, HouseQuote)
            # Should have similarity score in metadata
            assert hasattr(quote, 'similarity_score')
    
    def test_search_quotes_raises_validation_error_for_empty_query(self, rag_store):
        """Test search_quotes raises ValidationError for empty query."""
        with pytest.raises(Exception):  # Should be ValidationError
            rag_store.search_quotes("", top_k=3)
    
    def test_search_quotes_raises_validation_error_for_invalid_top_k(self, rag_store):
        """Test search_quotes raises ValidationError for invalid top_k."""
        with pytest.raises(Exception):  # Should be ValidationError
            rag_store.search_quotes("test query", top_k=0)
        
        with pytest.raises(Exception):  # Should be ValidationError
            rag_store.search_quotes("test query", top_k=15)
    
    def test_get_quote_by_id_returns_quote_if_exists(self, rag_store):
        """Test get_quote_by_id returns HouseQuote if found."""
        # Add a quote first
        quote = HouseQuote(text="Everybody lies.", sarcasm_level=3)
        quote_id = rag_store.add_quote(quote)
        
        # Retrieve by ID
        retrieved = rag_store.get_quote_by_id(quote_id)
        
        assert retrieved is not None
        assert isinstance(retrieved, HouseQuote)
        assert retrieved.text == "Everybody lies."
    
    def test_get_quote_by_id_returns_none_if_not_found(self, rag_store):
        """Test get_quote_by_id returns None if quote not found."""
        fake_id = "550e8400-e29b-41d4-a716-446655440000"
        result = rag_store.get_quote_by_id(fake_id)
        assert result is None
    
    def test_get_quote_by_id_raises_validation_error_for_invalid_id(self, rag_store):
        """Test get_quote_by_id raises ValidationError for invalid UUID."""
        with pytest.raises(Exception):  # Should be ValidationError
            rag_store.get_quote_by_id("invalid-uuid")
    
    def test_update_quote_metadata_updates_fields(self, rag_store):
        """Test update_quote_metadata updates quote without changing embedding."""
        # Add quote
        quote = HouseQuote(text="It's not lupus.", sarcasm_level=2)
        quote_id = rag_store.add_quote(quote)
        
        # Update metadata
        metadata = {"sarcasm_level": 4, "emotional_tone": "witty"}
        result = rag_store.update_quote_metadata(quote_id, metadata)
        
        assert result is True
    
    def test_update_quote_metadata_returns_false_if_not_found(self, rag_store):
        """Test update_quote_metadata returns False if quote not found."""
        fake_id = "550e8400-e29b-41d4-a716-446655440000"
        metadata = {"sarcasm_level": 3}
        result = rag_store.update_quote_metadata(fake_id, metadata)
        assert result is False
    
    def test_delete_quote_removes_from_database(self, rag_store):
        """Test delete_quote removes quote from database."""
        # Add quote
        quote = HouseQuote(text="Differential diagnosis time.", sarcasm_level=1)
        quote_id = rag_store.add_quote(quote)
        
        # Delete quote
        result = rag_store.delete_quote(quote_id)
        assert result is True
        
        # Verify deletion
        deleted = rag_store.get_quote_by_id(quote_id)
        assert deleted is None
    
    def test_delete_quote_returns_false_if_not_found(self, rag_store):
        """Test delete_quote returns False if quote not found."""
        fake_id = "550e8400-e29b-41d4-a716-446655440000"
        result = rag_store.delete_quote(fake_id)
        assert result is False
    
    def test_get_stats_returns_database_metrics(self, rag_store):
        """Test get_stats returns database statistics and health metrics."""
        stats = rag_store.get_stats()
        
        # Verify required keys
        assert isinstance(stats, dict)
        assert "total_quotes" in stats
        assert "avg_embedding_time_ms" in stats
        assert "last_search_time_ms" in stats
        assert "database_size_mb" in stats
        assert "index_efficiency" in stats
        
        # Verify types
        assert isinstance(stats["total_quotes"], int)
        assert isinstance(stats["avg_embedding_time_ms"], float)
        assert isinstance(stats["last_search_time_ms"], float)
        assert isinstance(stats["database_size_mb"], float)
        assert isinstance(stats["index_efficiency"], float)
    
    def test_performance_contract_search_latency(self, rag_store):
        """Test performance contract: search latency < 100ms for top-10."""
        import time
        
        query = "What's the diagnosis?"
        start_time = time.time()
        
        results = rag_store.search_quotes(query, top_k=10)
        
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        
        assert latency_ms < 100  # Should be under 100ms
    
    def test_performance_contract_embedding_generation(self, rag_store):
        """Test performance contract: embedding generation < 50ms per quote."""
        import time
        
        quote = HouseQuote(
            text="The most successful diagnostic technique is to look the patient in the eye.",
            sarcasm_level=1
        )
        
        start_time = time.time()
        rag_store.add_quote(quote)
        end_time = time.time()
        
        embedding_time_ms = (end_time - start_time) * 1000
        assert embedding_time_ms < 50  # Should be under 50ms
    
    def test_memory_usage_contract(self, rag_store):
        """Test performance contract: memory usage < 500MB for 1000 quotes."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Add multiple quotes (smaller number for testing)
        for i in range(10):
            quote = HouseQuote(
                text=f"House quote number {i} with medical wisdom.",
                sarcasm_level=3
            )
            rag_store.add_quote(quote)
        
        current_memory = process.memory_info().rss
        memory_diff = current_memory - initial_memory
        
        # Extrapolate to 1000 quotes
        projected_memory = memory_diff * 100  # 10 quotes -> 1000 quotes
        assert projected_memory < 500 * 1024 * 1024  # 500MB


@pytest.mark.integration
class TestRAGStoreIntegration:
    """Integration tests for RAGStore with real dependencies."""
    
    def test_real_postgresql_connection(self):
        """Test with real PostgreSQL database."""
        pytest.skip("Requires PostgreSQL database for testing")
    
    def test_pgvector_extension_functionality(self):
        """Test pgvector extension operations."""
        pytest.skip("Requires pgvector extension for testing")
    
    def test_large_dataset_performance(self):
        """Test performance with large quote dataset."""
        pytest.skip("Requires large dataset for performance testing")
