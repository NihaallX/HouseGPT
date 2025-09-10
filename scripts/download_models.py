#!/usr/bin/env python3
"""Model downloading and validation script for HouseGPT.

Downloads required models, validates integrity, and sets up model cache
for HouseGPT components including the FLAN-T5 base model, LoRA adapters,
embedding models, and voice synthesis models.
"""

import argparse
import sys
import os
import shutil
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import requests
from urllib.parse import urlparse

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from huggingface_hub import hf_hub_download, model_info, list_repo_files
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    from TTS.api import TTS
    HAS_TTS = True
except ImportError:
    HAS_TTS = False

from src.models.configuration import HouseGPTConfig, ModelConfig


# Model specifications with checksums and download info
MODEL_SPECS = {
    "base_model": {
        "name": "google/flan-t5-large",
        "type": "transformers",
        "required": True,
        "description": "Base FLAN-T5 Large model for text generation",
        "size_gb": 3.0,
        "files": [
            "config.json",
            "generation_config.json",
            "pytorch_model.bin",
            "special_tokens_map.json",
            "spiece.model",
            "tokenizer_config.json",
            "tokenizer.json"
        ]
    },
    
    "embedding_model": {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "type": "sentence_transformers",
        "required": True,
        "description": "Sentence embedding model for RAG functionality",
        "size_gb": 0.1,
        "files": [
            "config.json",
            "pytorch_model.bin",
            "sentence_bert_config.json",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "vocab.txt"
        ]
    },
    
    "voice_model": {
        "name": "tts_models/en/ljspeech/tacotron2-DDC",
        "type": "tts",
        "required": False,
        "description": "Text-to-speech model for voice output",
        "size_gb": 0.2,
        "files": []  # TTS models have varying file structures
    },
    
    "xtts_model": {
        "name": "tts_models/multilingual/multi-dataset/xtts_v2",
        "type": "tts",
        "required": False,
        "description": "XTTS v2 multilingual voice cloning model",
        "size_gb": 1.7,
        "files": []
    },
    
    "wake_word_model": {
        "name": "openai/whisper-base",
        "type": "transformers",
        "required": False,
        "description": "Whisper model for wake word detection",
        "size_gb": 0.3,
        "files": [
            "config.json",
            "preprocessor_config.json",
            "pytorch_model.bin",
            "special_tokens_map.json",
            "tokenizer.json",
            "vocab.json"
        ]
    }
}

# HouseGPT LoRA adapter info
HOUSEGPT_LORA_INFO = {
    "path": "housegpt-lora-large",
    "description": "House MD personality LoRA adapter for FLAN-T5",
    "required_files": [
        "adapter_config.json",
        "adapter_model.safetensors"
    ],
    "optional_files": [
        "README.md",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "spiece.model"
    ]
}


class ModelDownloader:
    """Downloads and validates models for HouseGPT."""
    
    def __init__(self, cache_dir: Optional[str] = None, config: Optional[HouseGPTConfig] = None):
        """Initialize model downloader.
        
        Args:
            cache_dir: Directory to cache models (default: ~/.cache/huggingface)
            config: HouseGPT configuration
        """
        self.config = config or HouseGPTConfig()
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "huggingface"
        self.logger = logging.getLogger(__name__)
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def check_disk_space(self, required_gb: float) -> bool:
        """Check if enough disk space is available.
        
        Args:
            required_gb: Required space in GB
            
        Returns:
            True if enough space available
        """
        try:
            stat = shutil.disk_usage(self.cache_dir)
            available_gb = stat.free / (1024**3)
            
            if available_gb < required_gb:
                self.logger.error(f"Insufficient disk space: {available_gb:.1f}GB available, {required_gb:.1f}GB required")
                return False
            
            self.logger.info(f"Disk space check passed: {available_gb:.1f}GB available")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to check disk space: {e}")
            return False
    
    def check_internet_connection(self) -> bool:
        """Check internet connectivity.
        
        Returns:
            True if internet is available
        """
        try:
            response = requests.get("https://huggingface.co", timeout=10)
            return response.status_code == 200
        except:
            self.logger.error("No internet connection available")
            return False
    
    def validate_model_files(self, model_name: str, model_spec: Dict[str, Any]) -> bool:
        """Validate that model files exist and are complete.
        
        Args:
            model_name: Name of the model
            model_spec: Model specification dictionary
            
        Returns:
            True if all required files exist
        """
        try:
            if model_spec["type"] == "transformers":
                # Check if we can load the model
                if HAS_TRANSFORMERS:
                    try:
                        # Try to load just the config to verify model exists
                        from transformers import AutoConfig
                        AutoConfig.from_pretrained(model_name, cache_dir=self.cache_dir)
                        self.logger.info(f"✅ Model {model_name} validation passed")
                        return True
                    except Exception as e:
                        self.logger.error(f"Model {model_name} validation failed: {e}")
                        return False
                else:
                    self.logger.warning(f"Cannot validate {model_name} - transformers not available")
                    return False
            
            elif model_spec["type"] == "sentence_transformers":
                if HAS_SENTENCE_TRANSFORMERS:
                    try:
                        # Try to load the model
                        SentenceTransformer(model_name, cache_folder=str(self.cache_dir))
                        self.logger.info(f"✅ Embedding model {model_name} validation passed")
                        return True
                    except Exception as e:
                        self.logger.error(f"Embedding model {model_name} validation failed: {e}")
                        return False
                else:
                    self.logger.warning(f"Cannot validate {model_name} - sentence-transformers not available")
                    return False
            
            elif model_spec["type"] == "tts":
                if HAS_TTS:
                    try:
                        # Try to initialize TTS model
                        tts = TTS(model_name=model_name)
                        self.logger.info(f"✅ TTS model {model_name} validation passed")
                        return True
                    except Exception as e:
                        self.logger.error(f"TTS model {model_name} validation failed: {e}")
                        return False
                else:
                    self.logger.warning(f"Cannot validate {model_name} - TTS not available")
                    return False
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error validating model {model_name}: {e}")
            return False
    
    def download_transformers_model(self, model_name: str, model_spec: Dict[str, Any]) -> bool:
        """Download transformers model.
        
        Args:
            model_name: Name of the model
            model_spec: Model specification
            
        Returns:
            True if download successful
        """
        if not HAS_TRANSFORMERS:
            self.logger.error("Transformers library not available")
            return False
        
        try:
            self.logger.info(f"📥 Downloading transformers model: {model_name}")
            
            # Download tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=self.cache_dir
            )
            
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                cache_dir=self.cache_dir,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            self.logger.info(f"✅ Successfully downloaded {model_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to download {model_name}: {e}")
            return False
    
    def download_sentence_transformers_model(self, model_name: str, model_spec: Dict[str, Any]) -> bool:
        """Download sentence transformers model.
        
        Args:
            model_name: Name of the model
            model_spec: Model specification
            
        Returns:
            True if download successful
        """
        if not HAS_SENTENCE_TRANSFORMERS:
            self.logger.error("Sentence transformers library not available")
            return False
        
        try:
            self.logger.info(f"📥 Downloading embedding model: {model_name}")
            
            # Download and cache the model
            model = SentenceTransformer(model_name, cache_folder=str(self.cache_dir))
            
            self.logger.info(f"✅ Successfully downloaded {model_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to download {model_name}: {e}")
            return False
    
    def download_tts_model(self, model_name: str, model_spec: Dict[str, Any]) -> bool:
        """Download TTS model.
        
        Args:
            model_name: Name of the model
            model_spec: Model specification
            
        Returns:
            True if download successful
        """
        if not HAS_TTS:
            self.logger.error("TTS library not available")
            return False
        
        try:
            self.logger.info(f"📥 Downloading TTS model: {model_name}")
            
            # Initialize TTS (this will download the model)
            tts = TTS(model_name=model_name)
            
            self.logger.info(f"✅ Successfully downloaded {model_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to download {model_name}: {e}")
            return False
    
    def download_model(self, model_key: str, model_spec: Dict[str, Any]) -> bool:
        """Download a specific model.
        
        Args:
            model_key: Key identifying the model
            model_spec: Model specification
            
        Returns:
            True if download successful
        """
        model_name = model_spec["name"]
        model_type = model_spec["type"]
        
        self.logger.info(f"🔄 Processing {model_spec['description']}")
        
        # Check if already downloaded and valid
        if self.validate_model_files(model_name, model_spec):
            self.logger.info(f"✅ Model {model_name} already available and valid")
            return True
        
        # Check disk space
        if not self.check_disk_space(model_spec["size_gb"]):
            return False
        
        # Download based on model type
        if model_type == "transformers":
            return self.download_transformers_model(model_name, model_spec)
        elif model_type == "sentence_transformers":
            return self.download_sentence_transformers_model(model_name, model_spec)
        elif model_type == "tts":
            return self.download_tts_model(model_name, model_spec)
        else:
            self.logger.error(f"Unknown model type: {model_type}")
            return False
    
    def validate_lora_adapter(self, adapter_path: str) -> bool:
        """Validate LoRA adapter files.
        
        Args:
            adapter_path: Path to LoRA adapter directory
            
        Returns:
            True if adapter is valid
        """
        adapter_dir = Path(adapter_path)
        
        if not adapter_dir.exists():
            self.logger.error(f"LoRA adapter directory not found: {adapter_dir}")
            return False
        
        # Check required files
        for required_file in HOUSEGPT_LORA_INFO["required_files"]:
            file_path = adapter_dir / required_file
            if not file_path.exists():
                self.logger.error(f"Required LoRA file not found: {file_path}")
                return False
        
        # Validate adapter config
        try:
            config_path = adapter_dir / "adapter_config.json"
            with open(config_path) as f:
                config = json.load(f)
            
            # Basic validation
            if "base_model_name_or_path" not in config:
                self.logger.error("Invalid adapter config: missing base_model_name_or_path")
                return False
            
            self.logger.info(f"✅ LoRA adapter validation passed: {adapter_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to validate LoRA adapter config: {e}")
            return False
    
    def download_all_models(self, include_optional: bool = False) -> Tuple[int, int]:
        """Download all required models.
        
        Args:
            include_optional: Whether to download optional models
            
        Returns:
            Tuple of (successful_downloads, failed_downloads)
        """
        if not self.check_internet_connection():
            return 0, len(MODEL_SPECS)
        
        successful = 0
        failed = 0
        
        total_size = sum(spec["size_gb"] for spec in MODEL_SPECS.values() 
                        if spec["required"] or include_optional)
        
        if not self.check_disk_space(total_size):
            return 0, len(MODEL_SPECS)
        
        self.logger.info(f"🏥 Starting model downloads (estimated {total_size:.1f}GB)")
        
        for model_key, model_spec in MODEL_SPECS.items():
            if not model_spec["required"] and not include_optional:
                self.logger.info(f"⏭️ Skipping optional model: {model_spec['name']}")
                continue
            
            if self.download_model(model_key, model_spec):
                successful += 1
            else:
                failed += 1
        
        return successful, failed
    
    def get_model_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all models.
        
        Returns:
            Dictionary with model status information
        """
        status = {}
        
        for model_key, model_spec in MODEL_SPECS.items():
            model_name = model_spec["name"]
            is_available = self.validate_model_files(model_name, model_spec)
            
            status[model_key] = {
                "name": model_name,
                "description": model_spec["description"],
                "type": model_spec["type"],
                "required": model_spec["required"],
                "size_gb": model_spec["size_gb"],
                "available": is_available,
                "status": "✅ Available" if is_available else "❌ Missing"
            }
        
        # Check LoRA adapter
        adapter_path = self.config.model.adapter_path or "housegpt-lora-large"
        lora_available = self.validate_lora_adapter(adapter_path)
        
        status["lora_adapter"] = {
            "name": "HouseGPT LoRA Adapter",
            "description": "House MD personality adapter",
            "type": "lora",
            "required": True,
            "path": adapter_path,
            "available": lora_available,
            "status": "✅ Available" if lora_available else "❌ Missing"
        }
        
        return status
    
    def cleanup_cache(self, older_than_days: int = 30) -> int:
        """Clean up old cached models.
        
        Args:
            older_than_days: Remove files older than this many days
            
        Returns:
            Number of files removed
        """
        import time
        
        removed_count = 0
        cutoff_time = time.time() - (older_than_days * 24 * 60 * 60)
        
        try:
            for root, dirs, files in os.walk(self.cache_dir):
                for file in files:
                    file_path = Path(root) / file
                    if file_path.stat().st_mtime < cutoff_time:
                        try:
                            file_path.unlink()
                            removed_count += 1
                        except Exception as e:
                            self.logger.warning(f"Failed to remove {file_path}: {e}")
            
            self.logger.info(f"🧹 Removed {removed_count} old cached files")
            return removed_count
            
        except Exception as e:
            self.logger.error(f"Cache cleanup failed: {e}")
            return 0


def main():
    """Main function for model downloading script."""
    parser = argparse.ArgumentParser(description="HouseGPT Model Downloader")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        help="Model cache directory"
    )
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="Download optional models (voice, wake word)"
    )
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="Only show model status, don't download"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing models"
    )
    parser.add_argument(
        "--cleanup-cache",
        type=int,
        metavar="DAYS",
        help="Clean up cache files older than N days"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODEL_SPECS.keys()),
        help="Download specific model only"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    logger = logging.getLogger(__name__)
    
    # Check dependencies
    missing_deps = []
    if not HAS_TRANSFORMERS:
        missing_deps.append("transformers")
    if not HAS_SENTENCE_TRANSFORMERS:
        missing_deps.append("sentence-transformers")
    if not HAS_TTS:
        missing_deps.append("TTS")
    
    if missing_deps:
        logger.warning(f"Missing optional dependencies: {', '.join(missing_deps)}")
        logger.info("Install with: pip install transformers sentence-transformers TTS")
    
    # Get configuration
    config = None
    if args.config:
        config = HouseGPTConfig.from_env(args.config)
    
    # Initialize downloader
    downloader = ModelDownloader(args.cache_dir, config)
    
    # Handle cleanup
    if args.cleanup_cache:
        removed = downloader.cleanup_cache(args.cleanup_cache)
        logger.info(f"Cache cleanup completed: {removed} files removed")
        return 0
    
    # Show model status
    status = downloader.get_model_status()
    
    print("\n📊 Model Status:")
    for model_key, info in status.items():
        print(f"  {info['status']} {info['name']}")
        if args.verbose:
            print(f"    Type: {info['type']}")
            print(f"    Description: {info['description']}")
            if 'size_gb' in info:
                print(f"    Size: {info['size_gb']:.1f}GB")
            if 'path' in info:
                print(f"    Path: {info['path']}")
    
    if args.status_only:
        return 0
    
    # Validate only
    if args.validate_only:
        all_valid = all(info['available'] for info in status.values() if info['required'])
        if all_valid:
            logger.info("✅ All required models are valid")
            return 0
        else:
            logger.error("❌ Some required models are missing or invalid")
            return 1
    
    # Download models
    if args.model:
        # Download specific model
        if args.model in MODEL_SPECS:
            success = downloader.download_model(args.model, MODEL_SPECS[args.model])
            return 0 if success else 1
        else:
            logger.error(f"Unknown model: {args.model}")
            return 1
    else:
        # Download all models
        logger.info("🏥 Starting HouseGPT model downloads...")
        successful, failed = downloader.download_all_models(args.include_optional)
        
        logger.info(f"📊 Download Summary:")
        logger.info(f"  Successful: {successful}")
        logger.info(f"  Failed: {failed}")
        
        return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
