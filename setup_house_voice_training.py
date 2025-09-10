"""House voice cloning setup using SO-VITS-SVC.

This script sets up the proper training pipeline to clone House's voice
from your voice sample using SO-VITS-SVC.
"""

import asyncio
import logging
import sys
import subprocess
from pathlib import Path
import shutil
import os

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import pygame


class HouseVoiceTrainer:
    """Set up House voice cloning training with SO-VITS-SVC."""
    
    def __init__(self):
        """Initialize voice trainer."""
        # Initialize pygame for audio playback
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Paths
        self.voice_sample_path = Path("voice_samples/house_voice.wav")
        self.training_dir = Path("house_voice_training")
        self.dataset_dir = self.training_dir / "dataset_raw" / "house"
        
        self.logger.info("🎭 House Voice Trainer with SO-VITS-SVC initialized")
    
    def setup_training_structure(self):
        """Set up the proper folder structure for SO-VITS-SVC training."""
        try:
            self.logger.info("📁 Setting up training folder structure...")
            
            # Create directories
            self.dataset_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy House voice sample to training dataset
            if self.voice_sample_path.exists():
                target_path = self.dataset_dir / "house_voice.wav"
                shutil.copy2(self.voice_sample_path, target_path)
                self.logger.info(f"✅ Copied voice sample to: {target_path}")
            else:
                self.logger.error(f"❌ Voice sample not found: {self.voice_sample_path}")
                return False
            
            self.logger.info("✅ Training structure set up successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to set up training structure: {e}")
            return False
    
    async def run_preprocessing(self):
        """Run SO-VITS-SVC preprocessing steps."""
        try:
            self.logger.info("🔄 Running SO-VITS-SVC preprocessing...")
            
            # Change to training directory
            os.chdir(self.training_dir)
            
            # Step 1: Resample audio
            self.logger.info("1️⃣  Step 1: Resampling audio...")
            result = subprocess.run(
                ["svc", "pre-resample"],
                capture_output=True,
                text=True,
                cwd=self.training_dir
            )
            
            if result.returncode == 0:
                self.logger.info("✅ Resampling completed")
            else:
                self.logger.error(f"❌ Resampling failed: {result.stderr}")
                return False
            
            # Step 2: Generate config
            self.logger.info("2️⃣  Step 2: Generating config...")
            result = subprocess.run(
                ["svc", "pre-config"],
                capture_output=True,
                text=True,
                cwd=self.training_dir
            )
            
            if result.returncode == 0:
                self.logger.info("✅ Config generation completed")
            else:
                self.logger.error(f"❌ Config generation failed: {result.stderr}")
                return False
            
            # Step 3: HuBERT feature extraction
            self.logger.info("3️⃣  Step 3: HuBERT feature extraction...")
            self.logger.info("⚠️  This step downloads models and takes time...")
            
            result = subprocess.run(
                ["svc", "pre-hubert"],
                capture_output=True,
                text=True,
                cwd=self.training_dir
            )
            
            if result.returncode == 0:
                self.logger.info("✅ HuBERT feature extraction completed")
            else:
                self.logger.error(f"❌ HuBERT failed: {result.stderr}")
                return False
            
            self.logger.info("🎉 All preprocessing steps completed!")
            return True
            
        except Exception as e:
            self.logger.error(f"Preprocessing failed: {e}")
            return False
        finally:
            # Change back to original directory
            os.chdir(project_root)
    
    async def start_training(self):
        """Start training the House voice model."""
        try:
            self.logger.info("🏋️  Starting House voice model training...")
            self.logger.info("⚠️  This will take a long time (hours) and requires GPU for best results")
            
            # Change to training directory
            os.chdir(self.training_dir)
            
            # Start training
            result = subprocess.run(
                ["svc", "train"],
                capture_output=True,
                text=True,
                cwd=self.training_dir
            )
            
            if result.returncode == 0:
                self.logger.info("🎉 Training started successfully!")
                self.logger.info("📈 Monitor training progress in the logs/44k/ directory")
                return True
            else:
                self.logger.error(f"❌ Training failed to start: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Training startup failed: {e}")
            return False
        finally:
            # Change back to original directory
            os.chdir(project_root)
    
    async def test_inference(self, text: str = "Everybody lies. That's my working theory."):
        """Test inference with the trained model (if available)."""
        try:
            self.logger.info(f"🗣️  Testing House voice inference: '{text}'")
            
            # Check if trained model exists
            model_dir = self.training_dir / "logs" / "44k"
            if not model_dir.exists():
                self.logger.warning("❌ No trained model found. Run training first.")
                return False
            
            # Look for the latest model file
            model_files = list(model_dir.glob("G_*.pth"))
            if not model_files:
                self.logger.warning("❌ No model checkpoint files found.")
                return False
            
            latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
            self.logger.info(f"📦 Using model: {latest_model.name}")
            
            # For now, just indicate what would happen
            self.logger.info("💡 Inference would use the svc infer command")
            self.logger.info("💡 This would generate House's voice saying the text")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Inference test failed: {e}")
            return False
    
    async def setup_full_pipeline(self):
        """Set up the complete House voice cloning pipeline."""
        self.logger.info("🏥 Setting up complete House voice cloning pipeline...")
        
        # Step 1: Set up training structure
        if not self.setup_training_structure():
            self.logger.error("❌ Failed to set up training structure")
            return
        
        # Step 2: Ask user about preprocessing
        self.logger.info("🤔 Ready to start preprocessing...")
        self.logger.info("⚠️  This will:")
        self.logger.info("   - Resample your House voice sample")
        self.logger.info("   - Generate training configuration") 
        self.logger.info("   - Extract HuBERT features (downloads models)")
        self.logger.info("   - Take several minutes to complete")
        
        # For demo, let's show what would happen
        self.logger.info("💡 To continue with real voice cloning:")
        self.logger.info(f"   1. cd {self.training_dir}")
        self.logger.info("   2. svc pre-resample")
        self.logger.info("   3. svc pre-config") 
        self.logger.info("   4. svc pre-hubert")
        self.logger.info("   5. svc train")
        self.logger.info("   6. svc infer --input-path <text> --output-path <output.wav>")
        
        self.logger.info("🎭 House voice cloning setup complete!")
        self.logger.info(f"📁 Training directory: {self.training_dir.absolute()}")


async def main():
    """Run the House voice training setup."""
    trainer = HouseVoiceTrainer()
    await trainer.setup_full_pipeline()


if __name__ == "__main__":
    asyncio.run(main())
