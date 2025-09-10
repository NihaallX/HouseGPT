"""Performance metrics data model for logging.

This module provides the PerformanceMetrics model for performance tracking.
"""

from datetime import datetime


class PerformanceMetrics:
    """Data model for performance metrics tracking.
    
    This is a stub implementation that will be replaced during implementation phase.
    """
    
    def __init__(self,
                 timestamp: datetime,
                 wake_word_latency_ms: float,
                 rag_search_time_ms: float,
                 model_inference_time_ms: float,
                 style_filter_time_ms: float,
                 tts_synthesis_time_ms: float,
                 total_response_time_ms: float,
                 memory_usage_mb: float,
                 cpu_usage_percent: float,
                 gpu_usage_percent: float = 0.0):
        """Initialize performance metrics.
        
        Args:
            timestamp: When metrics were recorded
            wake_word_latency_ms: Wake word detection latency
            rag_search_time_ms: RAG search time
            model_inference_time_ms: Model inference time
            style_filter_time_ms: Style filter processing time
            tts_synthesis_time_ms: TTS synthesis time
            total_response_time_ms: Total response time
            memory_usage_mb: Memory usage in MB
            cpu_usage_percent: CPU usage percentage
            gpu_usage_percent: GPU usage percentage
        """
        self.timestamp = timestamp
        self.wake_word_latency_ms = wake_word_latency_ms
        self.rag_search_time_ms = rag_search_time_ms
        self.model_inference_time_ms = model_inference_time_ms
        self.style_filter_time_ms = style_filter_time_ms
        self.tts_synthesis_time_ms = tts_synthesis_time_ms
        self.total_response_time_ms = total_response_time_ms
        self.memory_usage_mb = memory_usage_mb
        self.cpu_usage_percent = cpu_usage_percent
        self.gpu_usage_percent = gpu_usage_percent
