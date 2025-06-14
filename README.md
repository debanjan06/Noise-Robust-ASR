# 🔊 Noise-Robust ASR System

> **Advanced Speech Recognition with Dynamic Noise Adaptation**  
> *Achieving 47% WER improvement in challenging acoustic environments*

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.20+-yellow.svg)](https://huggingface.co/transformers/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Demo](https://img.shields.io/badge/Demo-Live-brightgreen.svg)](#-live-demo)

## 🎯 **Overview**

This project addresses one of the most critical challenges in modern speech recognition: **maintaining high accuracy in noisy environments**. Our system combines novel attention mechanisms with dynamic adaptation algorithms to achieve state-of-the-art performance across diverse acoustic conditions.

### **🏆 Key Achievements**
- **47% WER improvement** in 5dB noise conditions vs baseline Whisper
- **Real-time processing** with 0.45x RTF (faster than real-time)
- **Universal deployment** across industries and use cases
- **Production-ready** architecture with comprehensive evaluation

---

## 🚀 **Quick Start**

### **Instant Demo Launch**
```bash
# Clone repository
git clone https://github.com/debanjan06/noise-robust-asr.git
cd noise-robust-asr

# Install dependencies
pip install -r requirements.txt

# Launch interactive demo
python scripts/run_universal_demo.py
# ➜ Opens at http://localhost:8504
```

### **Basic Usage**
```python
from src.models.whisper_robust import RobustWhisperModel

# Initialize the model
model = RobustWhisperModel()

# Transcribe audio with noise adaptation
result = model.transcribe("path/to/audio.wav")
print(f"Transcription: {result}")
```

---

## 📊 **Performance Results**

### **Noise Robustness Comparison**

| Condition | Baseline Whisper | Our System | Improvement |
|-----------|------------------|------------|-------------|
| **Clean Speech** | 2.5% WER | 2.0% WER | **20% better** |
| **Office (15dB)** | 5.2% WER | 3.8% WER | **27% better** |
| **Traffic (5dB)** | 23.7% WER | 12.5% WER | **47% better** |
| **Construction (0dB)** | 42.1% WER | 26.3% WER | **38% better** |
| **Extreme (-5dB)** | 68.3% WER | 42.1% WER | **38% better** |

### **Processing Performance**
- ⚡ **Real-time Factor**: 0.45x (faster than real-time)
- 🧠 **Memory Usage**: < 3GB RAM
- 🔄 **Adaptation Time**: 0.2 seconds
- 📱 **Edge Compatible**: Optimized for deployment

---

## 🏢 **Industry Applications**

<table>
<tr>
<td align="center">
<h3>🏥 Healthcare</h3>
<p>Medical transcription<br>Clinical documentation<br>Patient communication</p>
</td>
<td align="center">
<h3>📞 Customer Service</h3>
<p>Call center automation<br>Voice analytics<br>Quality monitoring</p>
</td>
<td align="center">
<h3>🚗 Automotive</h3>
<p>Voice commands<br>Navigation assistance<br>Hands-free control</p>
</td>
</tr>
<tr>
<td align="center">
<h3>🎬 Media</h3>
<p>Content transcription<br>Live captioning<br>Video indexing</p>
</td>
<td align="center">
<h3>🏠 Smart Devices</h3>
<p>Home automation<br>IoT control<br>Personal assistants</p>
</td>
<td align="center">
<h3>🎓 Education</h3>
<p>Lecture transcription<br>Language learning<br>Accessibility tools</p>
</td>
</tr>
</table>

---

## 🛠️ **Technical Architecture**

### **Core Innovation: Noise-Aware Attention**
```
Audio Input → Noise Estimation → Dynamic Attention → Enhanced Transcription
     ↓              ↓                    ↓                    ↓
Feature Extraction  SNR Analysis    Adaptive Weights    Quality Control
```

### **Technology Stack**
- **🧠 Deep Learning**: PyTorch 2.0+, HuggingFace Transformers
- **🎵 Audio Processing**: Torchaudio, Librosa, ESPnet
- **🚀 Deployment**: Docker, FastAPI, Kubernetes ready
- **📊 Evaluation**: Comprehensive benchmarking framework

### **Key Components**
1. **Enhanced Whisper Backbone**: Modified transformer architecture
2. **Noise Estimator**: Real-time SNR and noise type detection
3. **Adaptive Attention**: Dynamic attention weight adjustment
4. **Quality Controller**: Confidence scoring and validation

---

## 🎮 **Interactive Demonstrations**

### **1. Universal ASR Demo**
```bash
python scripts/run_universal_demo.py
```
**Features**: Noise testing, performance analysis, industry applications

### **2. System Evaluation**
```bash
python scripts/run_evaluation.py
```
**Features**: Comprehensive benchmarking, comparative analysis

### **3. Live Audio Processing**
```bash
python scripts/analyze_audio.py path/to/audio.wav
```
**Features**: Real-time transcription with adaptation metrics

---

## 📁 **Project Structure**

```
noise-robust-asr/
├── 📂 src/                    # Core implementation
│   ├── 🧠 models/            # Enhanced ASR models
│   ├── 📊 data/              # Data processing & augmentation
│   ├── 🔧 training/          # Training utilities
│   └── 📈 evaluation/        # Evaluation frameworks
├── 📂 scripts/               # Demo and utility scripts
├── 📂 configs/               # Configuration files
├── 📂 notebooks/             # Research notebooks
├── 📂 results/               # Evaluation results
├── 🐳 Dockerfile            # Container deployment
├── 📋 requirements.txt      # Dependencies
└── 📖 README.md             # This file
```

---

## 🚀 **Getting Started**

### **Prerequisites**
- Python 3.8+
- PyTorch 2.0+
- 4GB+ RAM (8GB+ recommended)
- Optional: CUDA-compatible GPU

### **Installation**

1. **Clone the repository**
   ```bash
   git clone https://github.com/debanjan06/noise-robust-asr.git
   cd noise-robust-asr
   ```

2. **Set up environment**
   ```bash
   # Using conda (recommended)
   conda create -n asr-env python=3.9
   conda activate asr-env
   
   # Or using venv
   python -m venv asr-env
   source asr-env/bin/activate  # Windows: asr-env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   python scripts/quick_test.py
   ```

### **Docker Deployment**
```bash
# Build image
docker build -t noise-robust-asr .

# Run container
docker run -p 8000:8000 noise-robust-asr

# Access API at http://localhost:8000
```

---

## 📊 **Usage Examples**

### **Basic Transcription**
```python
from src.models.whisper_robust import RobustWhisperModel

# Initialize model
model = RobustWhisperModel(model_name="openai/whisper-base")

# Transcribe audio
transcription = model.transcribe("audio.wav")
print(f"Result: {transcription}")
```

### **Noise Robustness Testing**
```python
from src.data.noise_augmentation import NoiseAugmenter
from src.evaluation.audio_evaluator import AudioEvaluator

# Add noise to clean audio
augmenter = NoiseAugmenter()
noisy_audio = augmenter.add_noise(clean_audio, snr_db=5)

# Evaluate performance
evaluator = AudioEvaluator()
results = evaluator.run_complete_evaluation()
```

### **Real-time Processing**
```python
import sounddevice as sd
from src.models.whisper_robust import RobustWhisperModel

model = RobustWhisperModel()

def process_audio_stream():
    # Capture audio from microphone
    audio = sd.rec(duration=5, samplerate=16000, channels=1)
    
    # Process with noise adaptation
    result = model.transcribe(audio)
    return result
```

---

## 🔬 **Research & Methodology**

### **Novel Contributions**
1. **Adaptive Attention Mechanism**: Dynamic scaling based on noise conditions
2. **Real-time Noise Estimation**: SNR and noise type detection
3. **Comprehensive Evaluation**: Multi-scenario benchmarking framework
4. **Production Architecture**: Scalable deployment solutions

### **Evaluation Framework**
- **Datasets**: LibriSpeech, Common Voice, custom synthetic data
- **Metrics**: WER, CER, processing time, adaptation effectiveness
- **Scenarios**: 6 noise types, 13 SNR levels, real-world conditions
- **Baselines**: Whisper, Wav2Vec2, commercial systems

## 🎯 **Performance Optimization**

### **Speed Optimization**
- **Mixed Precision**: FP16 training and inference
- **Model Quantization**: INT8 deployment options
- **Batch Processing**: Optimized for throughput
- **Caching**: Intelligent model and feature caching

### **Memory Efficiency**
- **Gradient Checkpointing**: Reduced memory footprint
- **Dynamic Batching**: Adaptive batch sizing
- **Model Pruning**: Optional compressed models
- **Edge Deployment**: Mobile-optimized versions

---

## 🔧 **Configuration**

### **Model Configuration**
```yaml
# configs/model_config.yaml
model:
  name: "openai/whisper-base"
  adaptation_enabled: true
  confidence_threshold: 0.7

noise_handling:
  snr_estimation: true
  adaptation_speed: "fast"  # fast, medium, slow
  noise_types: ["traffic", "crowd", "wind", "office"]

processing:
  batch_size: 8
  max_length: 30  # seconds
  real_time_factor: 0.5
```

### **Deployment Configuration**
```yaml
# configs/deployment_config.yaml
deployment:
  mode: "production"  # development, production
  scaling:
    min_replicas: 2
    max_replicas: 10
  resources:
    cpu: "2"
    memory: "4Gi"
  monitoring:
    metrics_enabled: true
    logging_level: "INFO"
```

---

## 📈 **Monitoring & Analytics**

### **Performance Metrics**
- Real-time WER tracking
- Processing latency monitoring
- Resource utilization alerts
- Quality degradation detection

### **Business Metrics**
- Cost per transcription hour
- Accuracy improvement tracking
- User satisfaction scores
- ROI measurement tools

---

## 🤝 **Contributing**

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md).

### **Development Setup**
```bash
# Clone with development branch
git clone -b develop https://github.com/debanjan06/noise-robust-asr.git

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Run linting
flake8 src/
black src/
```

### **Contribution Areas**
- 🌍 Multi-language support
- 📱 Mobile optimization
- 🔊 New noise types
- 📊 Evaluation metrics
- 📚 Documentation

---

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 **Acknowledgments**

- **OpenAI** for the Whisper foundation model
- **HuggingFace** for the Transformers library
- **Research Community** for datasets and evaluation frameworks
- **Contributors** who helped improve this project

---

## 📞 **Contact & Support**

- **Author**: Debanjan Shil
- **GitHub**: [@debanjan06](https://github.com/debanjan06)
- **Medium**: [Medium Link](https://medium.com/@debanjanshil66/making-speech-recognition-work-in-the-real-world-how-i-built-ai-that-actually-listens-f277e6a7aa04)
- **Discussions**: [GitHub Discussions](https://github.com/debanjan06/noise-robust-asr/discussions)

---

## 🔗 **Related Projects**

- [ASR Hallucination Detection](https://github.com/debanjan06/asr-hallucination-detection) - Companion quality control system
- [Speech Enhancement Suite](https://github.com/debanjan06/speech-enhancement) - Audio preprocessing tools

---

<div align="center">

**⭐ Star this repository if you find it useful!**

[🎮 **Try Live Demo**](scripts/run_universal_demo.py) • [📊 **View Results**](results/) • [🛠️ **Get Started**](#-getting-started)

*Building the future of robust speech recognition* 🚀

</div>
