"""Contract tests for ConversationLogger service.

These tests validate the contract defined in contracts/conversation_logger_contract.md.
Tests MUST fail initially before implementation exists.
"""

import pytest
import json
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from src.services.conversation_logger import ConversationLogger, LoggingError, FileAccessError
from src.models.conversation_entry import ConversationEntry
from src.models.performance_metrics import PerformanceMetrics
from src.lib.config import Configuration


class TestConversationLoggerContract:
    """Test ConversationLogger contract compliance."""
    
    @pytest.fixture
    def config(self):
        """Provide test configuration."""
        config = Configuration()
        config.logging.log_file_path = "./logs/conversations.jsonl"
        config.logging.max_file_size_mb = 100
        config.logging.rotation_count = 5
        config.logging.enable_metrics = True
        config.logging.privacy_mode = False
        return config
    
    @pytest.fixture
    def temp_log_dir(self):
        """Provide temporary directory for test logs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def conversation_logger(self, config, temp_log_dir):
        """Provide ConversationLogger instance."""
        config.logging.log_file_path = os.path.join(temp_log_dir, "test_conversations.jsonl")
        return ConversationLogger(config)
    
    def test_constructor_requires_config(self):
        """Test constructor requires Configuration object."""
        with pytest.raises(TypeError):
            ConversationLogger()
    
    def test_constructor_creates_log_directory(self, config, temp_log_dir):
        """Test constructor creates log directory if it doesn't exist."""
        log_path = os.path.join(temp_log_dir, "new_dir", "conversations.jsonl")
        config.logging.log_file_path = log_path
        
        logger = ConversationLogger(config)
        
        assert os.path.exists(os.path.dirname(log_path))
    
    def test_constructor_raises_file_access_error_for_invalid_path(self, config):
        """Test constructor raises FileAccessError for invalid log path."""
        config.logging.log_file_path = "/invalid/readonly/path/conversations.jsonl"
        
        with pytest.raises(FileAccessError):
            ConversationLogger(config)
    
    def test_log_conversation_writes_entry_to_file(self, conversation_logger, temp_log_dir):
        """Test log_conversation writes conversation entry to JSONL file."""
        entry = ConversationEntry(
            timestamp=datetime.now(),
            user_query="What could cause my symptoms?",
            wake_word_detected=True,
            rag_quotes_found=3,
            house_response="Everybody lies, especially about their symptoms.",
            response_confidence=0.85,
            sarcasm_level=4,
            emotional_tone="condescending",
            synthesis_time_ms=1200,
            playback_duration_ms=3500,
            user_satisfaction=None  # Not rated yet
        )
        
        conversation_logger.log_conversation(entry)
        
        # Verify file was created and contains entry
        log_file = conversation_logger.config.logging.log_file_path
        assert os.path.exists(log_file)
        
        with open(log_file, 'r') as f:
            line = f.readline().strip()
            logged_data = json.loads(line)
            
            assert logged_data["user_query"] == entry.user_query
            assert logged_data["house_response"] == entry.house_response
            assert logged_data["response_confidence"] == entry.response_confidence
    
    def test_log_conversation_raises_validation_error_for_invalid_entry(self, conversation_logger):
        """Test log_conversation raises ValidationError for invalid entry."""
        with pytest.raises(Exception):  # Should be ValidationError
            conversation_logger.log_conversation(None)
        
        with pytest.raises(Exception):  # Should be ValidationError
            conversation_logger.log_conversation("invalid entry type")
    
    def test_log_conversation_raises_logging_error_on_write_failure(self, conversation_logger):
        """Test log_conversation raises LoggingError on file write failure."""
        entry = ConversationEntry(
            timestamp=datetime.now(),
            user_query="Test query",
            house_response="Test response"
        )
        
        with patch('builtins.open', side_effect=IOError("Disk full")):
            with pytest.raises(LoggingError):
                conversation_logger.log_conversation(entry)
    
    def test_log_conversation_handles_privacy_mode(self, conversation_logger):
        """Test log_conversation redacts sensitive data in privacy mode."""
        conversation_logger.config.logging.privacy_mode = True
        
        entry = ConversationEntry(
            timestamp=datetime.now(),
            user_query="My patient John Smith has chest pain.",
            house_response="Let's run some tests on your patient.",
            response_confidence=0.8
        )
        
        conversation_logger.log_conversation(entry)
        
        # Check that sensitive data was redacted
        log_file = conversation_logger.config.logging.log_file_path
        with open(log_file, 'r') as f:
            logged_data = json.loads(f.readline().strip())
            
            # User query should be redacted or anonymized
            assert "John Smith" not in logged_data["user_query"]
            assert "[REDACTED]" in logged_data["user_query"] or logged_data["user_query"] == "[PRIVACY_MODE]"
    
    def test_log_performance_metrics_writes_metrics_to_file(self, conversation_logger):
        """Test log_performance_metrics writes performance data."""
        metrics = PerformanceMetrics(
            timestamp=datetime.now(),
            wake_word_latency_ms=150,
            rag_search_time_ms=80,
            model_inference_time_ms=2500,
            style_filter_time_ms=5,
            tts_synthesis_time_ms=1200,
            total_response_time_ms=3935,
            memory_usage_mb=245,
            cpu_usage_percent=65.5,
            gpu_usage_percent=78.2
        )
        
        conversation_logger.log_performance_metrics(metrics)
        
        # Should be written to performance log or same file with type marker
        log_file = conversation_logger.config.logging.log_file_path
        with open(log_file, 'r') as f:
            lines = f.readlines()
            
        # Find performance metrics entry
        perf_entry = None
        for line in lines:
            data = json.loads(line.strip())
            if data.get("entry_type") == "performance_metrics":
                perf_entry = data
                break
        
        assert perf_entry is not None
        assert perf_entry["total_response_time_ms"] == 3935
        assert perf_entry["memory_usage_mb"] == 245
    
    def test_query_conversations_returns_filtered_entries(self, conversation_logger):
        """Test query_conversations returns conversations matching criteria."""
        # Log multiple entries
        entries = [
            ConversationEntry(
                timestamp=datetime.now() - timedelta(hours=1),
                user_query="Query about fever",
                house_response="It's never lupus.",
                response_confidence=0.8
            ),
            ConversationEntry(
                timestamp=datetime.now(),
                user_query="Query about headache",
                house_response="Everybody lies.",
                response_confidence=0.9
            )
        ]
        
        for entry in entries:
            conversation_logger.log_conversation(entry)
        
        # Query recent conversations
        recent = conversation_logger.query_conversations(
            start_time=datetime.now() - timedelta(minutes=30),
            end_time=datetime.now() + timedelta(minutes=1)
        )
        
        assert len(recent) == 1
        assert recent[0].user_query == "Query about headache"
    
    def test_query_conversations_supports_text_search(self, conversation_logger):
        """Test query_conversations supports text search filtering."""
        entries = [
            ConversationEntry(
                timestamp=datetime.now(),
                user_query="Patient has lupus symptoms",
                house_response="It's never lupus.",
                response_confidence=0.8
            ),
            ConversationEntry(
                timestamp=datetime.now(),
                user_query="Patient has migraine",
                house_response="Take some pills.",
                response_confidence=0.7
            )
        ]
        
        for entry in entries:
            conversation_logger.log_conversation(entry)
        
        # Search for "lupus"
        lupus_results = conversation_logger.query_conversations(search_text="lupus")
        
        assert len(lupus_results) == 1
        assert "lupus" in lupus_results[0].user_query.lower()
    
    def test_query_conversations_supports_confidence_filtering(self, conversation_logger):
        """Test query_conversations filters by confidence score."""
        entries = [
            ConversationEntry(
                timestamp=datetime.now(),
                user_query="High confidence query",
                house_response="Confident response",
                response_confidence=0.9
            ),
            ConversationEntry(
                timestamp=datetime.now(),
                user_query="Low confidence query",
                house_response="Uncertain response",
                response_confidence=0.5
            )
        ]
        
        for entry in entries:
            conversation_logger.log_conversation(entry)
        
        # Filter for high confidence
        high_confidence = conversation_logger.query_conversations(min_confidence=0.8)
        
        assert len(high_confidence) == 1
        assert high_confidence[0].response_confidence == 0.9
    
    def test_query_conversations_raises_validation_error_for_invalid_params(self, conversation_logger):
        """Test query_conversations raises ValidationError for invalid parameters."""
        # End time before start time
        with pytest.raises(Exception):  # Should be ValidationError
            conversation_logger.query_conversations(
                start_time=datetime.now(),
                end_time=datetime.now() - timedelta(hours=1)
            )
        
        # Invalid confidence range
        with pytest.raises(Exception):  # Should be ValidationError
            conversation_logger.query_conversations(min_confidence=1.5)
    
    def test_get_conversation_stats_returns_analytics(self, conversation_logger):
        """Test get_conversation_stats returns conversation analytics."""
        # Log several entries with different characteristics
        entries = [
            ConversationEntry(
                timestamp=datetime.now() - timedelta(hours=2),
                user_query="Query 1",
                house_response="Response 1",
                response_confidence=0.8,
                sarcasm_level=3
            ),
            ConversationEntry(
                timestamp=datetime.now() - timedelta(hours=1),
                user_query="Query 2",
                house_response="Response 2",
                response_confidence=0.9,
                sarcasm_level=4
            )
        ]
        
        for entry in entries:
            conversation_logger.log_conversation(entry)
        
        stats = conversation_logger.get_conversation_stats(
            start_time=datetime.now() - timedelta(hours=3)
        )
        
        assert isinstance(stats, dict)
        assert "total_conversations" in stats
        assert "avg_confidence_score" in stats
        assert "avg_response_time_ms" in stats
        assert "sarcasm_distribution" in stats
        assert "emotional_tone_distribution" in stats
        assert "most_common_queries" in stats
        
        # Verify calculated values
        assert stats["total_conversations"] == 2
        assert abs(stats["avg_confidence_score"] - 0.85) < 0.01  # (0.8 + 0.9) / 2
    
    def test_export_conversations_creates_export_file(self, conversation_logger, temp_log_dir):
        """Test export_conversations creates formatted export file."""
        # Log some entries
        entry = ConversationEntry(
            timestamp=datetime.now(),
            user_query="Export test query",
            house_response="Export test response",
            response_confidence=0.8
        )
        conversation_logger.log_conversation(entry)
        
        # Export to CSV
        export_path = os.path.join(temp_log_dir, "export.csv")
        conversation_logger.export_conversations(
            export_path,
            format="csv",
            start_time=datetime.now() - timedelta(hours=1)
        )
        
        assert os.path.exists(export_path)
        with open(export_path, 'r') as f:
            content = f.read()
            assert "Export test query" in content
            assert "Export test response" in content
    
    def test_export_conversations_supports_json_format(self, conversation_logger, temp_log_dir):
        """Test export_conversations supports JSON export format."""
        entry = ConversationEntry(
            timestamp=datetime.now(),
            user_query="JSON export test",
            house_response="JSON response",
            response_confidence=0.7
        )
        conversation_logger.log_conversation(entry)
        
        export_path = os.path.join(temp_log_dir, "export.json")
        conversation_logger.export_conversations(export_path, format="json")
        
        assert os.path.exists(export_path)
        with open(export_path, 'r') as f:
            exported_data = json.load(f)
            
        assert isinstance(exported_data, list)
        assert len(exported_data) >= 1
        assert exported_data[0]["user_query"] == "JSON export test"
    
    def test_rotate_logs_creates_backup_files(self, conversation_logger, temp_log_dir):
        """Test rotate_logs creates backup files when size limit exceeded."""
        # Set small file size limit for testing
        conversation_logger.config.logging.max_file_size_mb = 0.001  # 1KB
        
        # Log many entries to exceed size limit
        for i in range(100):
            entry = ConversationEntry(
                timestamp=datetime.now(),
                user_query=f"Test query {i} with substantial content to increase file size",
                house_response=f"Test response {i} with additional content for size",
                response_confidence=0.8
            )
            conversation_logger.log_conversation(entry)
        
        # Should have created backup files
        log_dir = os.path.dirname(conversation_logger.config.logging.log_file_path)
        backup_files = [f for f in os.listdir(log_dir) if f.endswith('.1') or f.endswith('.bak')]
        
        assert len(backup_files) > 0
    
    def test_clear_old_logs_removes_expired_entries(self, conversation_logger):
        """Test clear_old_logs removes entries older than retention period."""
        # Log old and new entries
        old_entry = ConversationEntry(
            timestamp=datetime.now() - timedelta(days=31),  # Older than 30 days
            user_query="Old query",
            house_response="Old response",
            response_confidence=0.8
        )
        
        new_entry = ConversationEntry(
            timestamp=datetime.now(),
            user_query="New query",
            house_response="New response",
            response_confidence=0.8
        )
        
        conversation_logger.log_conversation(old_entry)
        conversation_logger.log_conversation(new_entry)
        
        # Clear logs older than 30 days
        removed_count = conversation_logger.clear_old_logs(retention_days=30)
        
        assert removed_count >= 1
        
        # Verify old entry was removed
        recent_conversations = conversation_logger.query_conversations(
            start_time=datetime.now() - timedelta(days=32)
        )
        
        # Should only contain new entry
        assert all(entry.user_query == "New query" for entry in recent_conversations)
    
    def test_performance_contract_logging_latency(self, conversation_logger):
        """Test performance contract: logging latency < 5ms per entry."""
        import time
        
        entry = ConversationEntry(
            timestamp=datetime.now(),
            user_query="Performance test query",
            house_response="Performance test response",
            response_confidence=0.8
        )
        
        start_time = time.time()
        conversation_logger.log_conversation(entry)
        end_time = time.time()
        
        logging_time_ms = (end_time - start_time) * 1000
        assert logging_time_ms < 5  # Should be under 5ms
    
    def test_performance_contract_query_speed(self, conversation_logger):
        """Test performance contract: query speed < 100ms for 1000 entries."""
        import time
        
        # Log multiple entries (smaller number for testing)
        for i in range(10):
            entry = ConversationEntry(
                timestamp=datetime.now() - timedelta(minutes=i),
                user_query=f"Query {i}",
                house_response=f"Response {i}",
                response_confidence=0.8
            )
            conversation_logger.log_conversation(entry)
        
        start_time = time.time()
        results = conversation_logger.query_conversations()
        end_time = time.time()
        
        query_time_ms = (end_time - start_time) * 1000
        
        # Extrapolate to 1000 entries (10 -> 1000 = 100x)
        projected_time = query_time_ms * 100
        assert projected_time < 100  # Should be under 100ms for 1000 entries


@pytest.mark.integration
class TestConversationLoggerIntegration:
    """Integration tests for ConversationLogger with real file system."""
    
    def test_log_file_rotation_with_real_filesystem(self):
        """Test log rotation with actual file operations."""
        pytest.skip("Requires extensive file system testing")
    
    def test_concurrent_logging_safety(self):
        """Test thread safety with concurrent logging operations."""
        pytest.skip("Requires multi-threading test setup")
    
    def test_large_dataset_performance(self):
        """Test performance with large conversation datasets."""
        pytest.skip("Requires large dataset for performance testing")
