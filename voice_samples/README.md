# Voice Samples Setup for HouseGPT

This directory contains voice samples for the HouseGPT personality injector's voice cloning functionality.

## Required Voice Sample

Place your House M.D. voice sample file in this directory with the following specifications:

**File Name:** `house_voice.wav`

**Requirements:**
- Format: WAV audio file
- Sample Rate: 16 kHz (16,000 Hz)
- Bit Depth: 16-bit
- Channels: Mono (single channel)
- Duration: 10-30 seconds recommended
- Quality: Clear speech, minimal background noise

## Voice Sample Content

For best results, the voice sample should contain:
- Clear pronunciation of various phonemes
- Natural speaking pace (not too slow or fast)
- House-like tone and cadence
- Minimal background noise or music
- Consistent volume level

## Example Setup

1. Place your voice sample as: `voice_samples/house_voice.wav`
2. Verify the file meets the technical requirements above
3. Test the voice cloning by running the HouseGPT system

## Configuration

The voice sample is configured in `src/lib/config.py`:
- `voice_clone_path`: Points to `./voice_samples/house_voice.wav`
- `sample_rate`: Set to 16000 Hz
- `bit_depth`: Set to 16-bit

## Troubleshooting

If voice cloning doesn't work properly:
1. Check file format (must be WAV)
2. Verify sample rate is 16 kHz
3. Ensure mono channel (not stereo)
4. Check for audio clarity and minimal noise
5. Try a different voice sample if needed

## Privacy Note

Voice samples are stored locally and are not transmitted or shared. The XTTS model uses them for voice cloning during text-to-speech generation.
