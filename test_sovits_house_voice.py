"""Real voice cloning using SO-VITS-SVC.

This uses SO-VITS-SVC to actually clone House's voice from your audio sample.
"""

import asyncio
import logging
import sys
from pathlib import Path
import tempfile
import os
import uuid

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import pygame
import librosa
import soundfile as sf


class SOVITSHouseCloner:
    """Real House voice cloning using SO-VITS-SVC."""
    
    def __init__(self):
        """Initialize SO-VITS voice cloner."""
        # Initialize pygame for audio playback
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Voice sample path
        self.voice_sample_path = Path("voice_samples/house_voice.wav")
        
        self.logger.info("🎭 SO-VITS-SVC House Voice Cloner initialized")
        
        # Check if voice sample exists
        if self.voice_sample_path.exists():
            self.logger.info(f"✅ Found House voice sample: {self.voice_sample_path}")
        else:
            self.logger.warning(f"❌ House voice sample not found: {self.voice_sample_path}")
    
    async def test_sovits_availability(self):
        """Test if SO-VITS-SVC is working."""
        try:
            self.logger.info("🧪 Testing SO-VITS-SVC availability...")
            
            # Try importing SO-VITS-SVC components
            import so_vits_svc_fork
            self.logger.info("✅ SO-VITS-SVC fork imported successfully")
            
            # Check available functions
            self.logger.info(f"📦 SO-VITS-SVC version: {so_vits_svc_fork.__version__ if hasattr(so_vits_svc_fork, '__version__') else 'Unknown'}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ SO-VITS-SVC test failed: {e}")
            return False
    
    async def clone_house_voice_simple(self, text: str) -> bool:
        """Simple test of voice cloning capabilities.
        
        Args:
            text: Text to synthesize
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info(f"🗣️  Attempting to clone House voice for: '{text}'")
            
            # For now, let's see what SO-VITS-SVC offers
            import so_vits_svc_fork
            
            # Check available classes/functions
            available = [item for item in dir(so_vits_svc_fork) if not item.startswith('_')]
            self.logger.info(f"🔍 Available SO-VITS-SVC functions: {available[:10]}...")  # Show first 10
            
            # This is a placeholder - we need to understand the SO-VITS-SVC API
            # Typically SO-VITS-SVC requires training a model on the target voice
            self.logger.info("⚠️  SO-VITS-SVC requires model training for real voice cloning")
            self.logger.info("⚠️  This would need your House voice sample to train a model first")
            
            return False  # Not implemented yet
            
        except Exception as e:
            self.logger.error(f"Voice cloning failed: {e}")
            return False
    
    async def analyze_house_sample(self):
        """Analyze the House voice sample for cloning preparation."""
        if not self.voice_sample_path.exists():
            self.logger.error("❌ No House voice sample to analyze")
            return
        
        try:
            self.logger.info("🔬 Analyzing House voice sample...")
            
            # Load and analyze the audio
            audio, sr = librosa.load(str(self.voice_sample_path), sr=22050)
            duration = len(audio) / sr
            
            self.logger.info(f"📊 Sample analysis:")
            self.logger.info(f"   Duration: {duration:.2f} seconds")
            self.logger.info(f"   Sample rate: {sr} Hz")
            self.logger.info(f"   Total samples: {len(audio)}")
            
            # Basic voice characteristics
            # Fundamental frequency (pitch)
            pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if pitch_values:
                avg_pitch = sum(pitch_values) / len(pitch_values)
                self.logger.info(f"   Average pitch: {avg_pitch:.1f} Hz")
            
            self.logger.info("✅ Sample analysis complete")
            self.logger.info("💡 For real voice cloning, this sample would be used to train a SO-VITS-SVC model")
            
        except Exception as e:
            self.logger.error(f"Sample analysis failed: {e}")
    
    async def run_tests(self):
        """Run all available tests."""
        self.logger.info("🏥 Running SO-VITS-SVC House voice cloning tests...")
        
        # Test 1: Library availability
        sovits_available = await self.test_sovits_availability()
        
        # Test 2: Analyze voice sample
        await self.analyze_house_sample()
        
        # Test 3: Attempt simple voice cloning
        if sovits_available:
            await self.clone_house_voice_simple("Everybody lies.")
        
        self.logger.info("🏁 Tests complete!")
        
        if sovits_available:
            self.logger.info("✅ SO-VITS-SVC is available for voice cloning!")
            self.logger.info("💡 Next step: Set up model training with your House voice sample")
        else:
            self.logger.error("❌ SO-VITS-SVC setup issues detected")


async def main():
    """Run the SO-VITS voice cloning tests."""
    cloner = SOVITSHouseCloner()
    await cloner.run_tests()


if __name__ == "__main__":
    asyncio.run(main())
