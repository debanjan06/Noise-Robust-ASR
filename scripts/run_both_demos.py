#!/usr/bin/env python3
"""
Launch both ASR demos for comprehensive demonstration
"""

import subprocess
import sys
import time
import threading
from pathlib import Path

def launch_demo(app_name, port, app_path):
    """Launch a specific demo on a specific port"""
    print(f"🚀 Starting {app_name} on port {port}...")
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(app_path),
            "--server.address", "localhost",
            "--server.port", str(port),
            "--browser.gatherUsageStats", "false"
        ])
    except Exception as e:
        print(f"❌ Error launching {app_name}: {e}")

def main():
    print("🎯 Launching Complete ASR Demo Suite")
    print("=" * 60)
    print("This will start TWO comprehensive demos:")
    print("🔊 Noise-Robust ASR Demo     → http://localhost:8502")
    print("🧠 Hallucination Detection  → http://localhost:8501")
    print("=" * 60)
    
    # Check if both demo files exist
    noise_demo = Path("app/streamlit_demo.py")
    hall_demo = Path("../asr-hallucination-detection/app/streamlit_demo.py")
    
    if not noise_demo.exists():
        print("❌ Noise-robust demo not found")
        return
    
    if not hall_demo.exists():
        print("❌ Hallucination detection demo not found")
        print("Note: Run this from the noise-robust-asr directory")
        return
    
    print("✅ Both demos found, starting parallel launch...")
    
    # Launch noise-robust demo in thread
    noise_thread = threading.Thread(
        target=launch_demo,
        args=("Noise-Robust ASR", 8502, noise_demo)
    )
    
    # Launch hallucination demo in thread  
    hall_thread = threading.Thread(
        target=launch_demo,
        args=("Hallucination Detection", 8501, hall_demo)
    )
    
    try:
        noise_thread.start()
        time.sleep(2)  # Stagger startup
        hall_thread.start()
        
        print("\n🌐 Both demos are starting...")
        print("📱 Open these URLs in separate browser tabs:")
        print("   • Noise-Robust ASR: http://localhost:8502")
        print("   • Hallucination Detection: http://localhost:8501")
        print("\n🛑 Press Ctrl+C to stop both demos")
        
        # Wait for threads
        noise_thread.join()
        hall_thread.join()
        
    except KeyboardInterrupt:
        print("\n👋 Stopping all demos...")

if __name__ == "__main__":
    main()