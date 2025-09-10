"""Configuration data model with environment loading.

This module provides the Configuration model for managing all HouseGPT
settings including model paths, audio configurations, and runtime options.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path
import os
from enum import Enum


class LogLevel(Enum):
    """Logging level enumeration."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AudioDevice(Enum):
    """Audio device enumeration."""
    DEFAULT = "default"
    AUTO = "auto"
    NONE = "none"


@dataclass
class ModelConfiguration:
    """Model-related configuration settings."""
    
    # LoRA Model Settings
    lora_model_path: str = "housegpt-lora-large/"
    base_model_name: str = "google/flan-t5-large"
    model_precision: str = "float16"  # float16, float32, bfloat16
    device_map: str = "auto"  # auto, cpu, cuda
    
    # Generation Parameters
    temperature: float = 0.8
    max_length: int = 150
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    do_sample: bool = True
    
    # Performance Settings
    torch_compile: bool = False
    gradient_checkpointing: bool = False
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    
    def __post_init__(self):
        """Validate model configuration."""
        if self.temperature < 0.0 or self.temperature > 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        if self.max_length < 10 or self.max_length > 1000:
            raise ValueError("Max length must be between 10 and 1000")
        if self.top_p < 0.0 or self.top_p > 1.0:
            raise ValueError("Top_p must be between 0.0 and 1.0")


@dataclass
class RAGConfiguration:
    """RAG store configuration settings."""
    
    # Storage Settings
    data_dir: str = "data/rag"
    vector_dimensions: int = 384
    embedding_model: str = "all-MiniLM-L6-v2"
    
    # Search Parameters
    similarity_threshold: float = 0.3
    max_results: int = 5
    context_window: int = 3
    
    # Performance Settings
    batch_size: int = 32
    index_type: str = "IndexFlatIP"  # FAISS index type
    normalize_embeddings: bool = True
    
    def __post_init__(self):
        """Validate RAG configuration."""
        if self.similarity_threshold < 0.0 or self.similarity_threshold > 1.0:
            raise ValueError("Similarity threshold must be between 0.0 and 1.0")
        if self.max_results < 1 or self.max_results > 100:
            raise ValueError("Max results must be between 1 and 100")


@dataclass
class AudioConfiguration:
    """Audio processing configuration settings."""
    
    # Device Settings
    input_device: str = AudioDevice.DEFAULT.value
    output_device: str = AudioDevice.DEFAULT.value
    sample_rate: int = 16000
    chunk_size: int = 1024
    channels: int = 1  # Mono
    
    # Wake Word Detection
    wake_word_model: str = "base"  # Whisper model size
    wake_word_threshold: float = 0.7
    wake_word_phrases: List[str] = field(default_factory=lambda: ["house", "hey house"])
    silence_timeout: float = 3.0
    
    # Voice Input
    speech_timeout: float = 10.0
    phrase_timeout: float = 1.0
    energy_threshold: int = 300
    dynamic_energy_threshold: bool = True
    
    # Voice Output (TTS)
    tts_model: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    voice_sample_path: str = "voice_samples/house_voice.wav"
    enable_voice_cloning: bool = True
    
    # TTS Parameters
    speed: float = 1.0
    pitch: int = 0  # Semitones adjustment
    energy: float = 1.2
    
    def __post_init__(self):
        """Validate audio configuration."""
        if self.sample_rate not in [8000, 16000, 22050, 44100, 48000]:
            raise ValueError("Sample rate must be a standard audio rate")
        if self.wake_word_threshold < 0.0 or self.wake_word_threshold > 1.0:
            raise ValueError("Wake word threshold must be between 0.0 and 1.0")
        if self.speed < 0.1 or self.speed > 3.0:
            raise ValueError("TTS speed must be between 0.1 and 3.0")


@dataclass
class StyleConfiguration:
    """Style filter configuration settings."""
    
    # Sarcasm Settings
    min_sarcasm_level: int = 2
    max_sarcasm_level: int = 5
    sarcasm_boost: float = 1.2
    
    # House Personality Traits
    cynicism_level: float = 0.8
    condescension_level: float = 0.7
    wit_level: float = 0.9
    
    # Content Filtering
    enable_medical_accuracy: bool = True
    enable_profanity_filter: bool = False
    enable_sensitivity_filter: bool = True
    
    # Response Enhancement
    add_cranky_openers: bool = True
    cranky_opener_probability: float = 0.3
    add_dismissive_closers: bool = True
    dismissive_closer_probability: float = 0.2
    
    # Authenticity Scoring
    min_authenticity_score: float = 0.6
    personality_weight: float = 0.4
    medical_weight: float = 0.3
    sarcasm_weight: float = 0.3
    
    def __post_init__(self):
        """Validate style configuration."""
        if self.min_sarcasm_level < 0 or self.min_sarcasm_level > 5:
            raise ValueError("Min sarcasm level must be between 0 and 5")
        if self.max_sarcasm_level < self.min_sarcasm_level or self.max_sarcasm_level > 5:
            raise ValueError("Max sarcasm level must be >= min sarcasm level and <= 5")


@dataclass
class LoggingConfiguration:
    """Logging configuration settings."""
    
    # General Settings
    log_level: str = LogLevel.INFO.value
    log_to_file: bool = True
    log_to_console: bool = True
    
    # File Settings
    log_dir: str = "logs"
    log_filename: str = "housegpt.log"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    
    # Conversation Logging
    conversation_log_dir: str = "data/conversations"
    conversation_log_format: str = "jsonl"
    log_user_inputs: bool = True
    log_model_outputs: bool = True
    log_audio_files: bool = False
    
    # Performance Logging
    log_response_times: bool = True
    log_model_metrics: bool = True
    log_memory_usage: bool = False
    
    def __post_init__(self):
        """Validate logging configuration."""
        if self.log_level not in [level.value for level in LogLevel]:
            raise ValueError(f"Log level must be one of: {[l.value for l in LogLevel]}")
        if self.max_file_size < 1024:  # 1KB minimum
            raise ValueError("Max file size must be at least 1024 bytes")


@dataclass
class Configuration:
    """Master configuration for HouseGPT system.
    
    Aggregates all configuration categories and provides environment loading.
    """
    
    # Configuration Categories
    model: ModelConfiguration = field(default_factory=ModelConfiguration)
    rag: RAGConfiguration = field(default_factory=RAGConfiguration)
    audio: AudioConfiguration = field(default_factory=AudioConfiguration)
    style: StyleConfiguration = field(default_factory=StyleConfiguration)
    logging: LoggingConfiguration = field(default_factory=LoggingConfiguration)
    
    # Application Settings
    app_name: str = "HouseGPT"
    app_version: str = "1.0.0"
    debug_mode: bool = False
    
    # Runtime Settings
    enable_wake_word: bool = True
    enable_voice_output: bool = True
    enable_rag: bool = True
    enable_style_filter: bool = True
    
    # Performance Settings
    max_concurrent_requests: int = 1
    request_timeout: float = 30.0
    health_check_interval: float = 60.0
    
    @classmethod
    def from_environment(cls) -> 'Configuration':
        """Create configuration from environment variables.
        
        Returns:
            Configuration instance with values from environment
        """
        config = cls()
        
        # Model Configuration
        if model_path := os.getenv('HOUSEGPT_MODEL_PATH'):
            config.model.lora_model_path = model_path
        if base_model := os.getenv('HOUSEGPT_BASE_MODEL'):
            config.model.base_model_name = base_model
        if temp := os.getenv('HOUSEGPT_TEMPERATURE'):
            config.model.temperature = float(temp)
        if device := os.getenv('HOUSEGPT_DEVICE'):
            config.model.device_map = device
        
        # RAG Configuration
        if rag_dir := os.getenv('HOUSEGPT_RAG_DIR'):
            config.rag.data_dir = rag_dir
        if embedding_model := os.getenv('HOUSEGPT_EMBEDDING_MODEL'):
            config.rag.embedding_model = embedding_model
        if threshold := os.getenv('HOUSEGPT_SIMILARITY_THRESHOLD'):
            config.rag.similarity_threshold = float(threshold)
        
        # Audio Configuration
        if voice_sample := os.getenv('HOUSEGPT_VOICE_SAMPLE'):
            config.audio.voice_sample_path = voice_sample
        if tts_model := os.getenv('HOUSEGPT_TTS_MODEL'):
            config.audio.tts_model = tts_model
        if wake_words := os.getenv('HOUSEGPT_WAKE_WORDS'):
            config.audio.wake_word_phrases = wake_words.split(',')
        
        # Style Configuration
        if min_sarcasm := os.getenv('HOUSEGPT_MIN_SARCASM'):
            config.style.min_sarcasm_level = int(min_sarcasm)
        if max_sarcasm := os.getenv('HOUSEGPT_MAX_SARCASM'):
            config.style.max_sarcasm_level = int(max_sarcasm)
        
        # Logging Configuration
        if log_level := os.getenv('HOUSEGPT_LOG_LEVEL'):
            config.logging.log_level = log_level.upper()
        if log_dir := os.getenv('HOUSEGPT_LOG_DIR'):
            config.logging.log_dir = log_dir
        
        # Runtime Settings
        if debug := os.getenv('HOUSEGPT_DEBUG'):
            config.debug_mode = debug.lower() in ('true', '1', 'yes', 'on')
        if no_wake := os.getenv('HOUSEGPT_NO_WAKE_WORD'):
            config.enable_wake_word = not (no_wake.lower() in ('true', '1', 'yes', 'on'))
        if no_voice := os.getenv('HOUSEGPT_NO_VOICE'):
            config.enable_voice_output = not (no_voice.lower() in ('true', '1', 'yes', 'on'))
        
        return config
    
    @classmethod
    def from_file(cls, config_path: str) -> 'Configuration':
        """Load configuration from JSON file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration instance
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file is invalid
        """
        import json
        
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Create nested configurations
            model_config = ModelConfiguration(**data.get('model', {}))
            rag_config = RAGConfiguration(**data.get('rag', {}))
            audio_config = AudioConfiguration(**data.get('audio', {}))
            style_config = StyleConfiguration(**data.get('style', {}))
            logging_config = LoggingConfiguration(**data.get('logging', {}))
            
            # Create main configuration
            main_data = {k: v for k, v in data.items() 
                        if k not in ['model', 'rag', 'audio', 'style', 'logging']}
            
            return cls(
                model=model_config,
                rag=rag_config,
                audio=audio_config,
                style=style_config,
                logging=logging_config,
                **main_data
            )
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file: {e}")
        except Exception as e:
            raise ValueError(f"Error loading config file: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        from dataclasses import asdict
        return asdict(self)
    
    def save_to_file(self, config_path: str) -> None:
        """Save configuration to JSON file.
        
        Args:
            config_path: Path where to save configuration
        """
        import json
        
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
    
    def validate(self) -> None:
        """Validate complete configuration.
        
        Raises:
            ValueError: If any configuration is invalid
        """
        # Validate individual configurations (calls their __post_init__)
        self.model.__post_init__()
        self.rag.__post_init__()
        self.audio.__post_init__()
        self.style.__post_init__()
        self.logging.__post_init__()
        
        # Cross-configuration validation
        if self.enable_rag and not Path(self.rag.data_dir).exists():
            Path(self.rag.data_dir).mkdir(parents=True, exist_ok=True)
        
        if self.enable_voice_output and not Path(self.audio.voice_sample_path).exists():
            raise ValueError(f"Voice sample not found: {self.audio.voice_sample_path}")
        
        if not Path(self.model.lora_model_path).exists():
            raise ValueError(f"LoRA model not found: {self.model.lora_model_path}")
        
        # Create required directories
        Path(self.logging.log_dir).mkdir(parents=True, exist_ok=True)
        Path(self.logging.conversation_log_dir).mkdir(parents=True, exist_ok=True)
    
    def get_model_path(self) -> Path:
        """Get absolute path to LoRA model directory."""
        return Path(self.model.lora_model_path).resolve()
    
    def get_voice_sample_path(self) -> Path:
        """Get absolute path to voice sample file."""
        return Path(self.audio.voice_sample_path).resolve()
    
    def get_rag_data_path(self) -> Path:
        """Get absolute path to RAG data directory."""
        return Path(self.rag.data_dir).resolve()
