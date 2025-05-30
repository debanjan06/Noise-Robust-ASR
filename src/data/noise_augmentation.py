import librosa
import numpy as np
import soundfile as sf
import os
from typing import List, Tuple

class NoiseAugmenter:
    """Class for adding various types of noise to audio signals"""
    
    def __init__(self):
        self.noise_types = ['white', 'pink', 'brown', 'traffic', 'crowd']
    
    def add_white_noise(self, audio: np.ndarray, snr_db: float = 10) -> np.ndarray:
        """Add white noise to audio signal"""
        signal_power = np.mean(audio ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
        return audio + noise
    
    def add_noise(self, audio: np.ndarray, noise_type: str = 'white', snr_db: float = 10) -> np.ndarray:
        """Add specified type of noise to audio"""
        if noise_type == 'white':
            return self.add_white_noise(audio, snr_db)
        else:
            # For now, default to white noise - expand later
            return self.add_white_noise(audio, snr_db)
    
    def create_noisy_dataset(self, clean_audio_paths: List[str], output_dir: str, 
                           noise_levels: List[float] = [0, 5, 10, 15, 20]):
        """Generate noisy versions of clean audio files"""
        os.makedirs(output_dir, exist_ok=True)
        
        for audio_path in clean_audio_paths:
            # Load clean audio
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # Generate noisy versions at different SNR levels
            for snr in noise_levels:
                noisy_audio = self.add_noise(audio, snr_db=snr)
                
                # Save noisy audio
                filename = os.path.basename(audio_path)
                name, ext = os.path.splitext(filename)
                output_path = os.path.join(output_dir, f"{name}_snr{snr}{ext}")
                sf.write(output_path, noisy_audio, sr)
        
        return True