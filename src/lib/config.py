"""Configuration management for HouseGPT application."""

import os
from pathlib import Path
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseConfig(BaseSettings):
    """Database configuration settings."""
    
    model_config = SettingsConfigDict(env_prefix="DB_")
    
    url: str = Field(default="postgresql://postgres:password@localhost:5432/housegpt")
    host: str = Field(default="localhost")
    port: int = Field(default=5432)
    name: str = Field(default="housegpt")
    user: str = Field(default="postgres")
    password: str = Field(default="password")
    
    @property
    def connection_url(self) -> str:
        """Get full database connection URL."""
        if self.url.startswith("postgresql://"):
            return self.url
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


class ModelConfig(BaseSettings):
    """AI model configuration settings."""
    
    lora_model_path: str = Field(default="./housegpt-lora-large")
    base_model_name: str = Field(default="google/flan-t5-large")
    lora_weights_path: str = Field(default="./housegpt-lora-large")
    sentence_transformer_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2"
    )
    device: str = Field(default="auto")
    max_length: int = Field(default=512)
    
    @property
    def lora_path(self) -> Path:
        """Get Path object for LoRA model directory."""
        return Path(self.lora_model_path).resolve()


class AudioConfig(BaseSettings):
    """Audio processing configuration."""
    
    audio_device_index: Optional[int] = Field(default=None)
    voice_sample_path: str = Field(default="./assets/house_voice_sample.wav")
    whisper_model_size: str = Field(default="base")
    sample_rate: int = Field(default=16000)
    chunk_size: int = Field(default=1024)
    tts_model: str = Field(default="tts_models/en/ljspeech/tacotron2-DDC")
    voice_clone_path: str = Field(default="./voice_samples/house_voice.wav")
    bit_depth: int = Field(default=16)
    speed_factor: float = Field(default=1.0)
    pitch_adjustment: int = Field(default=0)
    output_device: str = Field(default="default")
    
    @property
    def voice_sample(self) -> Path:
        """Get Path object for voice sample file."""
        return Path(self.voice_sample_path).resolve()


class TTSConfig(BaseSettings):
    """Text-to-speech configuration."""
    
    model_name: str = Field(default="tts_models/multilingual/multi-dataset/xtts_v2")
    enable_voice_cloning: bool = Field(default=True)
    speed: float = Field(default=1.0, ge=0.1, le=3.0)
    pitch: int = Field(default=0, ge=-12, le=12)
    energy: float = Field(default=1.2, ge=0.1, le=2.0)


class ApplicationConfig(BaseSettings):
    """Main application configuration."""
    
    max_response_length: int = Field(default=150, ge=50, le=300)
    wake_word_sensitivity: float = Field(default=0.7, ge=0.0, le=1.0)
    rag_top_k: int = Field(default=3, ge=1, le=10)
    log_level: str = Field(default="INFO")
    debug_mode: bool = Field(default=False)


class PerformanceConfig(BaseSettings):
    """Performance and resource configuration."""
    
    max_memory_gb: int = Field(default=4, ge=1, le=32)
    enable_gpu: str = Field(default="auto")  # "auto", "true", "false"
    torch_device: str = Field(default="auto")  # "auto", "cpu", "cuda"
    cache_size_mb: int = Field(default=100, ge=10, le=1000)


class ConversationConfig(BaseSettings):
    """Conversation and logging configuration."""
    
    session_timeout_minutes: int = Field(default=30, ge=1, le=1440)
    log_retention_days: int = Field(default=30, ge=1, le=365)
    enable_conversation_history: bool = Field(default=True)
    anonymize_logs: bool = Field(default=False)


class FilterConfig(BaseSettings):
    """Style filter configuration."""
    
    min_sarcasm_level: int = Field(default=2, ge=1, le=5)
    max_sarcasm_level: int = Field(default=5, ge=1, le=5)
    allowed_emotional_tones: List[str] = Field(default=["witty", "condescending", "analytical", "frustrated"])
    forbidden_words: List[str] = Field(default=["inappropriate", "offensive"])
    min_house_authenticity: float = Field(default=0.6, ge=0.0, le=1.0)
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class LoggingConfig(BaseSettings):
    """Logging configuration."""
    
    log_file_path: str = Field(default="./logs/conversations.jsonl")
    max_file_size_mb: int = Field(default=100, ge=1, le=1000)
    rotation_count: int = Field(default=5, ge=1, le=10)
    enable_metrics: bool = Field(default=True)
    privacy_mode: bool = Field(default=False)
    log_level: str = Field(default="INFO")


class APIConfig(BaseSettings):
    """API server configuration (for future extensions)."""
    
    host: str = Field(default="localhost")
    port: int = Field(default=8000, ge=1000, le=65535)
    enable_api_server: bool = Field(default=False)
    cors_origins: List[str] = Field(default=["http://localhost:3000"])


class Configuration(BaseSettings):
    """Main configuration class that combines all settings."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Sub-configurations
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    app: ApplicationConfig = Field(default_factory=ApplicationConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    conversation: ConversationConfig = Field(default_factory=ConversationConfig)
    filter: FilterConfig = Field(default_factory=FilterConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    
    def __init__(self, **kwargs):
        """Initialize configuration with environment overrides."""
        super().__init__(**kwargs)
        
        # Load sub-configurations with environment variables
        self.database = DatabaseConfig()
        self.model = ModelConfig()
        self.audio = AudioConfig()
        self.tts = TTSConfig()
        self.app = ApplicationConfig()
        self.performance = PerformanceConfig()
        self.conversation = ConversationConfig()
        self.filter = FilterConfig()
        self.logging = LoggingConfig()
        self.api = APIConfig()
    
    def validate_paths(self) -> List[str]:
        """Validate that required paths exist and return any errors."""
        errors = []
        
        # Check LoRA model path
        if not self.model.lora_path.exists():
            errors.append(f"LoRA model path not found: {self.model.lora_path}")
        
        # Check voice sample path (if voice cloning enabled)
        if self.tts.enable_voice_cloning and not self.audio.voice_sample.exists():
            errors.append(f"Voice sample not found: {self.audio.voice_sample}")
        
        return errors
    
    def get_torch_device(self) -> str:
        """Determine the appropriate PyTorch device."""
        import torch
        
        if self.performance.torch_device != "auto":
            return self.performance.torch_device
        
        if self.performance.enable_gpu == "true" or (
            self.performance.enable_gpu == "auto" and torch.cuda.is_available()
        ):
            return "cuda"
        
        return "cpu"
    
    @classmethod
    def load_from_env(cls, env_file: Optional[str] = None) -> "Configuration":
        """Load configuration from environment file."""
        if env_file:
            # Temporarily set env file path
            original_env_file = os.getenv("ENV_FILE")
            os.environ["ENV_FILE"] = env_file
            config = cls()
            if original_env_file:
                os.environ["ENV_FILE"] = original_env_file
            else:
                os.environ.pop("ENV_FILE", None)
            return config
        
        return cls()


# Global configuration instance
config: Optional[Configuration] = None


def get_config() -> Configuration:
    """Get the global configuration instance."""
    global config
    if config is None:
        config = Configuration.load_from_env()
    return config


def reload_config(env_file: Optional[str] = None) -> Configuration:
    """Reload configuration from environment."""
    global config
    config = Configuration.load_from_env(env_file)
    return config
