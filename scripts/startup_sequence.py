#!/usr/bin/env python3
"""Application startup sequence for HouseGPT.

Handles model preloading, health checks, and system initialization
to ensure all components are ready before accepting user interactions.
"""

import asyncio
import logging
import sys
import time
import signal
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import threading

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import torch
    import psutil
    HAS_MONITORING = True
except ImportError:
    HAS_MONITORING = False

from src.models.configuration import HouseGPTConfig
from src.services.house_model import HouseModel
from src.services.style_filter import StyleFilter
from src.services.voice_output import VoiceOutput
from src.services.wake_word import WakeWordDetector
from src.services.rag_store import RAGStore
from src.services.conversation_logger import ConversationLogger
from src.lib.db_utils import DatabaseManager
from src.lib.audio_utils import AudioProcessor


class StartupPhase(Enum):
    """Startup phases."""
    INITIALIZING = "initializing"
    LOADING_CONFIG = "loading_config"
    CHECKING_DEPENDENCIES = "checking_dependencies"
    INITIALIZING_DATABASE = "initializing_database"
    LOADING_MODELS = "loading_models"
    INITIALIZING_SERVICES = "initializing_services"
    RUNNING_HEALTH_CHECKS = "running_health_checks"
    READY = "ready"
    ERROR = "error"


@dataclass
class StartupStats:
    """Startup performance statistics."""
    total_startup_time: float = 0.0
    config_load_time: float = 0.0
    dependency_check_time: float = 0.0
    database_init_time: float = 0.0
    model_load_time: float = 0.0
    service_init_time: float = 0.0
    health_check_time: float = 0.0
    memory_usage_mb: float = 0.0
    gpu_memory_mb: float = 0.0


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    component: str
    success: bool
    message: str
    duration_ms: float
    details: Optional[Dict[str, Any]] = None


class HouseGPTStartup:
    """Manages HouseGPT application startup sequence."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize startup manager.
        
        Args:
            config_path: Optional path to configuration file
        """
        self.config_path = config_path
        self.config: Optional[HouseGPTConfig] = None
        self.logger = logging.getLogger(__name__)
        
        # Startup state
        self.phase = StartupPhase.INITIALIZING
        self.start_time = time.time()
        self.stats = StartupStats()
        self.health_results: List[HealthCheckResult] = []
        
        # Service instances
        self.house_model: Optional[HouseModel] = None
        self.style_filter: Optional[StyleFilter] = None
        self.voice_output: Optional[VoiceOutput] = None
        self.wake_word_detector: Optional[WakeWordDetector] = None
        self.rag_store: Optional[RAGStore] = None
        self.conversation_logger: Optional[ConversationLogger] = None
        self.database_manager: Optional[DatabaseManager] = None
        
        # Error tracking
        self.startup_errors: List[str] = []
        self.warnings: List[str] = []
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self._shutdown_requested = False
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self._shutdown_requested = True
    
    def _change_phase(self, new_phase: StartupPhase) -> None:
        """Change startup phase.
        
        Args:
            new_phase: New startup phase
        """
        old_phase = self.phase
        self.phase = new_phase
        self.logger.info(f"🔄 Startup phase: {old_phase.value} → {new_phase.value}")
    
    def _record_memory_usage(self) -> None:
        """Record current memory usage."""
        if HAS_MONITORING:
            try:
                process = psutil.Process()
                self.stats.memory_usage_mb = process.memory_info().rss / 1024 / 1024
                
                # GPU memory if available
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
                    self.stats.gpu_memory_mb = gpu_memory
                    
            except Exception as e:
                self.logger.warning(f"Failed to record memory usage: {e}")
    
    async def load_configuration(self) -> bool:
        """Load application configuration.
        
        Returns:
            True if successful, False otherwise
        """
        start_time = time.time()
        self._change_phase(StartupPhase.LOADING_CONFIG)
        
        try:
            self.logger.info("📋 Loading configuration...")
            
            if self.config_path:
                self.config = HouseGPTConfig.from_env(self.config_path)
            else:
                self.config = HouseGPTConfig.from_env()
            
            # Validate configuration
            errors = self.config.validate()
            if errors:
                for error in errors:
                    self.startup_errors.append(f"Config validation: {error}")
                return False
            
            self.stats.config_load_time = time.time() - start_time
            self.logger.info("✅ Configuration loaded successfully")
            return True
            
        except Exception as e:
            self.startup_errors.append(f"Configuration loading failed: {e}")
            self.logger.error(f"Failed to load configuration: {e}")
            return False
    
    async def check_dependencies(self) -> bool:
        """Check system dependencies.
        
        Returns:
            True if all dependencies available, False otherwise
        """
        start_time = time.time()
        self._change_phase(StartupPhase.CHECKING_DEPENDENCIES)
        
        try:
            self.logger.info("🔍 Checking dependencies...")
            
            # Check Python version
            if sys.version_info < (3, 8):
                self.startup_errors.append("Python 3.8+ required")
                return False
            
            # Check required packages
            required_packages = {
                "torch": "PyTorch for model inference",
                "transformers": "Transformers library for language models",
                "psycopg2": "PostgreSQL database connectivity"
            }
            
            missing_packages = []
            for package, description in required_packages.items():
                try:
                    __import__(package)
                except ImportError:
                    missing_packages.append(f"{package} ({description})")
            
            if missing_packages:
                self.startup_errors.append(f"Missing packages: {', '.join(missing_packages)}")
                return False
            
            # Check optional packages
            optional_packages = {
                "sentence_transformers": "Embedding models for RAG",
                "TTS": "Text-to-speech synthesis",
                "sounddevice": "Audio input/output",
                "psutil": "System monitoring"
            }
            
            for package, description in optional_packages.items():
                try:
                    __import__(package)
                except ImportError:
                    self.warnings.append(f"Optional package missing: {package} ({description})")
            
            # Check GPU availability
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                self.logger.info(f"🎮 GPU available: {gpu_name} ({gpu_count} devices)")
            else:
                self.warnings.append("No GPU available - using CPU (slower)")
            
            self.stats.dependency_check_time = time.time() - start_time
            self.logger.info("✅ Dependency check completed")
            return True
            
        except Exception as e:
            self.startup_errors.append(f"Dependency check failed: {e}")
            self.logger.error(f"Failed to check dependencies: {e}")
            return False
    
    async def initialize_database(self) -> bool:
        """Initialize database connection.
        
        Returns:
            True if successful, False otherwise
        """
        start_time = time.time()
        self._change_phase(StartupPhase.INITIALIZING_DATABASE)
        
        try:
            self.logger.info("🗄️ Initializing database...")
            
            self.database_manager = DatabaseManager(self.config.database)
            
            # Test database connection
            with self.database_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    cursor.fetchone()
            
            self.stats.database_init_time = time.time() - start_time
            self.logger.info("✅ Database connection established")
            return True
            
        except Exception as e:
            self.startup_errors.append(f"Database initialization failed: {e}")
            self.logger.error(f"Failed to initialize database: {e}")
            return False
    
    async def load_models(self) -> bool:
        """Load AI models.
        
        Returns:
            True if successful, False otherwise
        """
        start_time = time.time()
        self._change_phase(StartupPhase.LOADING_MODELS)
        
        try:
            self.logger.info("🧠 Loading AI models...")
            
            # Load House model (most critical)
            self.logger.info("Loading House personality model...")
            self.house_model = HouseModel(self.config.model)
            await self.house_model.load_model()
            
            # Load other models in parallel if possible
            model_tasks = []
            
            # Voice output model
            if self.config.voice.provider != "none":
                self.logger.info("Loading voice synthesis model...")
                self.voice_output = VoiceOutput(self.config.voice)
                model_tasks.append(self.voice_output.initialize())
            
            # Wake word detection model
            if self.config.audio.wake_words:
                self.logger.info("Loading wake word detection model...")
                self.wake_word_detector = WakeWordDetector(self.config.audio)
                model_tasks.append(self.wake_word_detector.load_model())
            
            # Wait for parallel model loading
            if model_tasks:
                await asyncio.gather(*model_tasks, return_exceptions=True)
            
            self.stats.model_load_time = time.time() - start_time
            self.logger.info(f"✅ Models loaded in {self.stats.model_load_time:.1f}s")
            return True
            
        except Exception as e:
            self.startup_errors.append(f"Model loading failed: {e}")
            self.logger.error(f"Failed to load models: {e}")
            return False
    
    async def initialize_services(self) -> bool:
        """Initialize application services.
        
        Returns:
            True if successful, False otherwise
        """
        start_time = time.time()
        self._change_phase(StartupPhase.INITIALIZING_SERVICES)
        
        try:
            self.logger.info("⚙️ Initializing services...")
            
            # Initialize style filter
            self.style_filter = StyleFilter(self.config.style)
            
            # Initialize RAG store
            if self.config.rag.enabled:
                self.rag_store = RAGStore(self.config.rag)
                await self.rag_store.initialize()
            
            # Initialize conversation logger
            self.conversation_logger = ConversationLogger(self.config.logging)
            
            self.stats.service_init_time = time.time() - start_time
            self.logger.info("✅ Services initialized")
            return True
            
        except Exception as e:
            self.startup_errors.append(f"Service initialization failed: {e}")
            self.logger.error(f"Failed to initialize services: {e}")
            return False
    
    async def run_health_checks(self) -> bool:
        """Run comprehensive health checks.
        
        Returns:
            True if all checks pass, False otherwise
        """
        start_time = time.time()
        self._change_phase(StartupPhase.RUNNING_HEALTH_CHECKS)
        
        self.logger.info("🏥 Running health checks...")
        
        # Define health checks
        health_checks = [
            ("Database", self._check_database_health),
            ("House Model", self._check_house_model_health),
            ("Style Filter", self._check_style_filter_health),
        ]
        
        # Optional health checks
        if self.voice_output:
            health_checks.append(("Voice Output", self._check_voice_output_health))
        
        if self.wake_word_detector:
            health_checks.append(("Wake Word", self._check_wake_word_health))
        
        if self.rag_store:
            health_checks.append(("RAG Store", self._check_rag_store_health))
        
        # Run health checks
        all_passed = True
        for check_name, check_function in health_checks:
            try:
                check_start = time.time()
                success, message, details = await check_function()
                duration_ms = (time.time() - check_start) * 1000
                
                result = HealthCheckResult(
                    component=check_name,
                    success=success,
                    message=message,
                    duration_ms=duration_ms,
                    details=details
                )
                
                self.health_results.append(result)
                
                if success:
                    self.logger.info(f"✅ {check_name}: {message}")
                else:
                    self.logger.error(f"❌ {check_name}: {message}")
                    all_passed = False
                    
            except Exception as e:
                self.logger.error(f"❌ {check_name}: Health check failed - {e}")
                all_passed = False
        
        self.stats.health_check_time = time.time() - start_time
        
        if all_passed:
            self.logger.info("✅ All health checks passed")
        else:
            self.startup_errors.append("Some health checks failed")
        
        return all_passed
    
    async def _check_database_health(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Check database health."""
        try:
            stats = self.database_manager.get_stats()
            
            if stats["connection_state"] == "connected":
                return True, "Database connection healthy", stats
            else:
                return False, f"Database unhealthy: {stats['last_error']}", stats
                
        except Exception as e:
            return False, f"Database check failed: {e}", {}
    
    async def _check_house_model_health(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Check House model health."""
        try:
            # Test model with simple input
            test_input = "Hello"
            response = await self.house_model.generate_response(test_input)
            
            if response and response.text:
                return True, "Model generating responses", {"test_response_length": len(response.text)}
            else:
                return False, "Model not generating responses", {}
                
        except Exception as e:
            return False, f"Model check failed: {e}", {}
    
    async def _check_style_filter_health(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Check style filter health."""
        try:
            # Test style filter
            test_response = "This is a test response."
            filtered = self.style_filter.apply_style(test_response)
            
            if filtered:
                return True, "Style filter working", {"score": filtered.style_score.overall_score}
            else:
                return False, "Style filter not working", {}
                
        except Exception as e:
            return False, f"Style filter check failed: {e}", {}
    
    async def _check_voice_output_health(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Check voice output health."""
        try:
            # Test voice synthesis
            test_text = "Health check test"
            audio_data = self.voice_output.synthesize(test_text)
            
            if audio_data is not None and len(audio_data) > 0:
                return True, "Voice synthesis working", {"audio_length": len(audio_data)}
            else:
                return False, "Voice synthesis failed", {}
                
        except Exception as e:
            return False, f"Voice output check failed: {e}", {}
    
    async def _check_wake_word_health(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Check wake word detection health."""
        try:
            # Basic wake word detector check
            if self.wake_word_detector.is_model_loaded():
                return True, "Wake word detector ready", {}
            else:
                return False, "Wake word detector not loaded", {}
                
        except Exception as e:
            return False, f"Wake word check failed: {e}", {}
    
    async def _check_rag_store_health(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Check RAG store health."""
        try:
            # Test RAG store query
            test_query = "test query"
            results = await self.rag_store.search_quotes(test_query, max_results=1)
            
            return True, f"RAG store working ({len(results)} quotes available)", {"quote_count": len(results)}
                
        except Exception as e:
            return False, f"RAG store check failed: {e}", {}
    
    async def startup_sequence(self) -> bool:
        """Run complete startup sequence.
        
        Returns:
            True if startup successful, False otherwise
        """
        self.logger.info("🏥 Starting HouseGPT application...")
        self.start_time = time.time()
        
        try:
            # Check for shutdown request between phases
            startup_steps = [
                ("Load Configuration", self.load_configuration),
                ("Check Dependencies", self.check_dependencies),
                ("Initialize Database", self.initialize_database),
                ("Load Models", self.load_models),
                ("Initialize Services", self.initialize_services),
                ("Health Checks", self.run_health_checks)
            ]
            
            for step_name, step_function in startup_steps:
                if self._shutdown_requested:
                    self.logger.info("Startup cancelled by shutdown request")
                    return False
                
                self.logger.info(f"🔄 {step_name}...")
                success = await step_function()
                
                if not success:
                    self._change_phase(StartupPhase.ERROR)
                    self.logger.error(f"Startup failed at: {step_name}")
                    return False
            
            # Record final statistics
            self.stats.total_startup_time = time.time() - self.start_time
            self._record_memory_usage()
            
            self._change_phase(StartupPhase.READY)
            self.logger.info("🎉 HouseGPT startup completed successfully!")
            self._log_startup_summary()
            
            return True
            
        except Exception as e:
            self.startup_errors.append(f"Startup sequence failed: {e}")
            self.logger.error(f"Fatal startup error: {e}")
            self._change_phase(StartupPhase.ERROR)
            return False
    
    def _log_startup_summary(self) -> None:
        """Log startup performance summary."""
        self.logger.info("📊 Startup Performance Summary:")
        self.logger.info(f"  Total time: {self.stats.total_startup_time:.2f}s")
        self.logger.info(f"  Config load: {self.stats.config_load_time:.2f}s")
        self.logger.info(f"  Dependencies: {self.stats.dependency_check_time:.2f}s")
        self.logger.info(f"  Database: {self.stats.database_init_time:.2f}s")
        self.logger.info(f"  Models: {self.stats.model_load_time:.2f}s")
        self.logger.info(f"  Services: {self.stats.service_init_time:.2f}s")
        self.logger.info(f"  Health checks: {self.stats.health_check_time:.2f}s")
        self.logger.info(f"  Memory usage: {self.stats.memory_usage_mb:.1f}MB")
        
        if self.stats.gpu_memory_mb > 0:
            self.logger.info(f"  GPU memory: {self.stats.gpu_memory_mb:.1f}MB")
        
        if self.warnings:
            self.logger.warning(f"Startup warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                self.logger.warning(f"  ⚠️ {warning}")
    
    def get_startup_report(self) -> Dict[str, Any]:
        """Get detailed startup report.
        
        Returns:
            Dictionary with startup information
        """
        return {
            "phase": self.phase.value,
            "success": self.phase == StartupPhase.READY,
            "stats": self.stats.__dict__,
            "health_checks": [
                {
                    "component": result.component,
                    "success": result.success,
                    "message": result.message,
                    "duration_ms": result.duration_ms
                }
                for result in self.health_results
            ],
            "errors": self.startup_errors,
            "warnings": self.warnings,
            "services": {
                "house_model": self.house_model is not None,
                "style_filter": self.style_filter is not None,
                "voice_output": self.voice_output is not None,
                "wake_word_detector": self.wake_word_detector is not None,
                "rag_store": self.rag_store is not None,
                "conversation_logger": self.conversation_logger is not None,
                "database_manager": self.database_manager is not None
            }
        }


async def main():
    """Main function for testing startup sequence."""
    import argparse
    
    parser = argparse.ArgumentParser(description="HouseGPT Startup Test")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    # Run startup sequence
    startup = HouseGPTStartup(args.config)
    success = await startup.startup_sequence()
    
    # Print report
    report = startup.get_startup_report()
    print(f"\nStartup {'SUCCESS' if success else 'FAILED'}")
    print(f"Total time: {report['stats']['total_startup_time']:.2f}s")
    
    if not success:
        print("Errors:")
        for error in report['errors']:
            print(f"  - {error}")
    
    return 0 if success else 1


if __name__ == "__main__":
    import asyncio
    sys.exit(asyncio.run(main()))
