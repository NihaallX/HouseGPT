"""Conversation logger service for tracking House interactions.

This module provides the ConversationLogger service for logging conversations,
performance metrics, and analytics.
"""

import json
import logging
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from src.models.conversation_entry import ConversationEntry


class LoggingError(Exception):
    """Base exception for logging operations."""
    pass


class FileAccessError(LoggingError):
    """Raised when log file cannot be accessed or written."""
    pass


class ValidationError(LoggingError):
    """Raised when input validation fails."""
    pass


class ConversationLogger:
    """Logger for conversation tracking and analytics.
    
    Logs conversation entries to JSONL files and provides query capabilities
    for analytics and conversation history retrieval.
    """
    
    def __init__(self, config):
        """Initialize conversation logger with file and database configuration.
        
        Args:
            config: Configuration object with logging settings
            
        Raises:
            FileAccessError: If log file path is invalid or inaccessible
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.log_dir = Path(getattr(config, 'conversation_log_dir', 'logs/conversations'))
        self.max_file_size = getattr(config, 'max_log_file_size', 100 * 1024 * 1024)  # 100MB
        self.retention_days = getattr(config, 'log_retention_days', 30)
        
        # Create log directory
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.current_log_file = self.log_dir / f"conversations_{datetime.now().strftime('%Y%m%d')}.jsonl"
            
            # Test write access
            test_file = self.log_dir / 'test_write'
            test_file.touch()
            test_file.unlink()
            
        except (OSError, PermissionError) as e:
            raise FileAccessError(f"Cannot access log directory {self.log_dir}: {e}")
        
        self.logger.info(f"Initialized conversation logger at {self.log_dir}")
    
    def log_conversation(self, entry) -> None:
        """Log conversation entry to JSONL file.
        
        Args:
            entry: ConversationEntry object to log
            
        Raises:
            ValidationError: If entry validation fails
            LoggingError: If file write fails
        """
        if not isinstance(entry, ConversationEntry):
            raise ValidationError("Entry must be a ConversationEntry object")
        
        try:
            # Rotate log file if needed
            self._rotate_log_if_needed()
            
            # Convert to JSON and write
            log_data = entry.to_dict()
            log_line = json.dumps(log_data, default=str) + '\n'
            
            with open(self.current_log_file, 'a', encoding='utf-8') as f:
                f.write(log_line)
            
            self.logger.debug(f"Logged conversation entry: {entry.session_id}")
            
        except Exception as e:
            raise LoggingError(f"Failed to write log entry: {e}")
    
    def log_performance_metrics(self, metrics) -> None:
        """Log performance metrics data.
        
        Args:
            metrics: Performance metrics dictionary
        """
        try:
            metrics_file = self.log_dir / f"metrics_{datetime.now().strftime('%Y%m%d')}.jsonl"
            
            metric_entry = {
                'timestamp': datetime.now().isoformat(),
                'type': 'performance_metrics',
                'data': metrics
            }
            
            log_line = json.dumps(metric_entry, default=str) + '\n'
            
            with open(metrics_file, 'a', encoding='utf-8') as f:
                f.write(log_line)
                
        except Exception as e:
            self.logger.error(f"Failed to log metrics: {e}")
    
    def log_error(self, error_data: Dict[str, Any]) -> None:
        """Log error information for monitoring.
        
        Args:
            error_data: Dictionary containing error details
        """
        try:
            error_file = self.log_dir / f"errors_{datetime.now().strftime('%Y%m%d')}.jsonl"
            
            error_entry = {
                'timestamp': datetime.now().isoformat(),
                'type': 'error',
                'data': error_data
            }
            
            log_line = json.dumps(error_entry, default=str) + '\n'
            
            with open(error_file, 'a', encoding='utf-8') as f:
                f.write(log_line)
                
        except Exception as e:
            self.logger.error(f"Failed to log error: {e}")
    
    def query_conversations(self, 
                          start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None,
                          search_text: Optional[str] = None,
                          min_confidence: Optional[float] = None,
                          session_id: Optional[str] = None) -> List:
        """Query conversations with filtering criteria.
        
        Args:
            start_time: Filter conversations after this time
            end_time: Filter conversations before this time
            search_text: Search in user input or response text
            min_confidence: Minimum confidence score
            session_id: Filter by specific session
            
        Returns:
            List of matching ConversationEntry objects
        """
        results = []
        
        try:
            # Find relevant log files
            log_files = list(self.log_dir.glob("conversations_*.jsonl"))
            
            for log_file in log_files:
                if not log_file.exists():
                    continue
                    
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            entry = ConversationEntry.from_dict(data)
                            
                            # Apply filters
                            if start_time and entry.timestamp < start_time:
                                continue
                            if end_time and entry.timestamp > end_time:
                                continue
                            if session_id and entry.session_id != session_id:
                                continue
                            if min_confidence and entry.response.confidence_score < min_confidence:
                                continue
                            if search_text:
                                search_content = f"{entry.user_input.text} {entry.response.text}".lower()
                                if search_text.lower() not in search_content:
                                    continue
                            
                            results.append(entry)
                            
                        except (json.JSONDecodeError, ValueError):
                            continue  # Skip malformed entries
            
            return sorted(results, key=lambda x: x.timestamp, reverse=True)
            
        except Exception as e:
            raise LoggingError(f"Query failed: {e}")
    
    def get_conversation_stats(self,
                             start_time: Optional[datetime] = None,
                             end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """Get conversation analytics and statistics.
        
        Args:
            start_time: Start time for statistics period
            end_time: End time for statistics period
            
        Returns:
            Dictionary with conversation statistics
        """
        try:
            stats = {
                'total_conversations': 0,
                'unique_sessions': set(),
                'avg_confidence': 0.0,
                'response_time_avg': 0.0,
                'top_topics': {},
                'error_rate': 0.0
            }
            
            confidence_scores = []
            response_times = []
            error_count = 0
            
            # Analyze conversation logs
            log_files = list(self.log_dir.glob("conversations_*.jsonl"))
            
            for log_file in log_files:
                if not log_file.exists():
                    continue
                    
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            entry = ConversationEntry.from_dict(data)
                            
                            # Apply time filters
                            if start_time and entry.timestamp < start_time:
                                continue
                            if end_time and entry.timestamp > end_time:
                                continue
                            
                            stats['total_conversations'] += 1
                            stats['unique_sessions'].add(entry.session_id)
                            confidence_scores.append(entry.response.confidence_score)
                            
                            if entry.response_time_ms:
                                response_times.append(entry.response_time_ms)
                            
                            if entry.error_details:
                                error_count += 1
                            
                        except (json.JSONDecodeError, ValueError):
                            continue
            
            # Calculate averages
            if confidence_scores:
                stats['avg_confidence'] = sum(confidence_scores) / len(confidence_scores)
            if response_times:
                stats['response_time_avg'] = sum(response_times) / len(response_times)
            
            stats['unique_sessions'] = len(stats['unique_sessions'])
            stats['error_rate'] = error_count / max(stats['total_conversations'], 1)
            
            return stats
            
        except Exception as e:
            raise LoggingError(f"Stats calculation failed: {e}")
    
    def export_conversations(self, 
                           export_path: str,
                           format: str = "csv",
                           start_time: Optional[datetime] = None,
                           end_time: Optional[datetime] = None) -> None:
        """Export conversations to file.
        
        Args:
            export_path: Path for export file
            format: Export format ('json' or 'csv')
            start_time: Start time filter
            end_time: End time filter
        """
        try:
            conversations = self.query_conversations(start_time=start_time, end_time=end_time)
            
            if format.lower() == 'json':
                with open(export_path, 'w', encoding='utf-8') as f:
                    json.dump([conv.to_dict() for conv in conversations], f, 
                             indent=2, default=str)
            elif format.lower() == 'csv':
                # CSV export would be implemented here
                raise ValueError(f"CSV export not yet implemented")
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
        except Exception as e:
            raise LoggingError(f"Export failed: {e}")
    
    def rotate_logs(self) -> None:
        """Rotate log files when size limit exceeded."""
        try:
            current_date = datetime.now().strftime('%Y%m%d')
            
            # Find files older than current date
            log_files = list(self.log_dir.glob("conversations_*.jsonl"))
            
            for log_file in log_files:
                if current_date not in log_file.name:
                    # Archive old file
                    archive_name = log_file.with_suffix('.jsonl.archive')
                    log_file.rename(archive_name)
                    self.logger.info(f"Archived log file: {archive_name}")
            
        except Exception as e:
            self.logger.error(f"Log rotation failed: {e}")
    
    def clear_old_logs(self, retention_days: int) -> int:
        """Clear log entries older than retention period.
        
        Args:
            retention_days: Number of days to retain logs
            
        Returns:
            Number of files deleted
        """
        deleted_count = 0
        
        try:
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            for log_file in self.log_dir.glob("*.jsonl*"):
                if log_file.stat().st_mtime < cutoff_date.timestamp():
                    log_file.unlink()
                    deleted_count += 1
                    self.logger.info(f"Deleted old log file: {log_file}")
            
            return deleted_count
                    
        except Exception as e:
            self.logger.error(f"Log cleanup failed: {e}")
            return deleted_count
    
    def flush_logs(self) -> None:
        """Flush any pending log writes to disk."""
        # In this implementation, we write immediately, so nothing to flush
        pass
    
    def _rotate_log_if_needed(self):
        """Rotate log file if it exceeds size limit."""
        try:
            if self.current_log_file.exists() and self.current_log_file.stat().st_size > self.max_file_size:
                # Create new log file with timestamp
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                new_file = self.log_dir / f"conversations_{timestamp}.jsonl"
                self.current_log_file = new_file
                
        except Exception as e:
            self.logger.error(f"Log rotation check failed: {e}")
