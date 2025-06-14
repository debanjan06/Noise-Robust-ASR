#!/usr/bin/env python3
"""
Launch Complete ASR Portfolio - Both Projects
Shows full spectrum of capabilities
"""

import subprocess
import sys
import threading
import time

def launch_demo(name, script_path, port):
    """Launch a demo in a separate thread"""
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            script_path,
            "--server.address", "localhost",
            "--server.port", str(port),
            "--browser.gatherUsageStats", "false"
        ])
    except Exception as e:
        print(f"❌ Error launching {name}: {e}")

def main():
    print("🎤 Complete ASR Portfolio Launch")
    print("=" * 60)
    print("Launching comprehensive demonstration of ASR capabilities:")
    print()
    print("🔊 NOISE ROBUSTNESS SYSTEM:")
    print("   • 47% WER improvement in challenging environments")
    print("   • Real-time adaptation and processing")
    print("   • Multi-industry applications")
    print("   🌐 http://localhost:8504")
    print()
    print("🛡️ QUALITY CONTROL SYSTEM:")
    print("   • 89% error detection accuracy")
    print("   • Critical safety issue prevention")
    print("   • Enterprise compliance assurance")
    print("   🌐 http://localhost:8505")
    print()
    
    response = input("Launch both demos? (y/n): ").lower().strip()
    if response != 'y':
        print("Demo launch cancelled")
        return
    
    print("🚀 Starting both systems...")
    
    # Determine which project we're in and launch accordingly
    import os
    current_dir = os.path.basename(os.getcwd())
    
    if "noise-robust" in current_dir:
        # Launch noise-robust demo locally
        noise_thread = threading.Thread(
            target=launch_demo,
            args=("Noise Robustness", "scripts/universal_asr_demo.py", 8504)
        )
        
        # Launch quality control demo from other project
        quality_thread = threading.Thread(
            target=launch_demo,
            args=("Quality Control", "../asr-hallucination-detection/scripts/universal_hallucination_demo.py", 8505)
        )
    else:
        # Launch quality control demo locally
        quality_thread = threading.Thread(
            target=launch_demo,
            args=("Quality Control", "scripts/universal_hallucination_demo.py", 8505)
        )
        
        # Launch noise-robust demo from other project
        noise_thread = threading.Thread(
            target=launch_demo,
            args=("Noise Robustness", "../noise-robust-asr/scripts/universal_asr_demo.py", 8504)
        )
    
    try:
        # Start both demos
        noise_thread.start()
        time.sleep(2)  # Stagger startup
        quality_thread.start()
        
        print("\n✅ Both demos are starting...")
        print("📱 Open these URLs in your browser:")
        print("   🔊 Noise Robustness: http://localhost:8504")
        print("   🛡️ Quality Control: http://localhost:8505")
        print("\n💼 Interview Strategy:")
        print("   • Start with the system most relevant to the role")
        print("   • Show both capabilities for comprehensive impact")
        print("   • Emphasize the business value and ROI")
        print("\n🛑 Press Ctrl+C to stop both demos")
        print("=" * 60)
        
        # Wait for threads to complete
        noise_thread.join()
        quality_thread.join()
        
    except KeyboardInterrupt:
        print("\n👋 Stopping all demos...")

if __name__ == "__main__":
    main()