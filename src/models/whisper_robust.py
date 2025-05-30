import torch
import whisper
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import numpy as np

class RobustWhisperModel:
    def __init__(self, model_name="openai/whisper-base"):
        """Initialize the robust Whisper model"""
        self.model_name = model_name
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def transcribe(self, audio_path):
        """Basic transcription function"""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # Process audio
            inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
            inputs = inputs.to(self.device)
            
            # Generate transcription
            with torch.no_grad():
                predicted_ids = self.model.generate(**inputs)
            
            # Decode transcription
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
            return transcription[0]
            
        except Exception as e:
            print(f"Error in transcription: {e}")
            return ""
    
    def evaluate_robustness(self, test_data):
        """Evaluate model robustness across different noise conditions"""
        results = []
        for audio_path, ground_truth in test_data:
            prediction = self.transcribe(audio_path)
            results.append({
                'audio': audio_path,
                'prediction': prediction,
                'ground_truth': ground_truth
            })
        return results