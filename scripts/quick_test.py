#!/usr/bin/env python3
import sys
import os
import numpy as np  # Add this import at the top
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_imports():
    """Test if all required packages can be imported"""
    try:
        import torch
        print("✅ PyTorch imported successfully")
        
        import transformers
        print("✅ Transformers imported successfully")
        
        import librosa
        print("✅ Librosa imported successfully")
        
        import numpy as np
        print("✅ NumPy imported successfully")
        
        import whisper  # Add this test
        print("✅ OpenAI Whisper imported successfully")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_model_loading():
    """Test basic model loading"""
    try:
        from models.whisper_robust import RobustWhisperModel
        print("✅ RobustWhisperModel class imported successfully")
        
        # Test model initialization (this might take a moment)
        print("🔄 Loading Whisper model (this may take a few minutes first time)...")
        model = RobustWhisperModel()
        print("✅ Whisper model loaded successfully")
        
        return True
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        return False

def test_noise_augmentation():
    """Test noise augmentation functionality"""
    try:
        from data.noise_augmentation import NoiseAugmenter
        
        augmenter = NoiseAugmenter()
        test_audio = np.random.random(1000)  # Dummy audio
        noisy_audio = augmenter.add_noise(test_audio)
        print("✅ Noise augmentation working")
        
        return True
    except Exception as e:
        print(f"❌ Noise augmentation error: {e}")
        return False

def main():
    print("🚀 Testing Noise-Robust ASR Setup...")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n❌ Setup incomplete - install requirements first:")
        print("pip install -r requirements.txt")
        print("pip install openai-whisper")
        return
    
    # Test model functionality
    print("\n🔄 Testing model components...")
    if test_model_loading():
        print("✅ Model loading successful!")
    
    # Test noise augmentation
    print("\n🔄 Testing noise augmentation...")
    if test_noise_augmentation():
        print("✅ Noise augmentation successful!")
    
    print("\n🎉 All tests passed! Setup is complete!")
    print("📁 Repository structure ready for development")
    print("\n🚀 Next steps:")
    print("1. Start with: jupyter notebook notebooks/01_initial_experiments.ipynb")
    print("2. Or run: python -c \"from src.models.whisper_robust import RobustWhisperModel; print('Ready!')\"")

if __name__ == "__main__":
    main()
