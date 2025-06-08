#!/usr/bin/env python3
"""
Launch the Noise-Robust ASR Interactive Demo
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    print("🔊 Launching Noise-Robust ASR Demo")
    print("=" * 50)
    
    # Check dependencies
    try:
        import streamlit
        import plotly
        print("✅ Demo dependencies available")
    except ImportError:
        print("❌ Installing demo dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit", "plotly"])
        print("✅ Dependencies installed successfully")
    
    # Get the app path
    app_path = Path(__file__).parent.parent / "app" / "streamlit_demo.py"
    
    if not app_path.exists():
        print(f"❌ Demo app not found at: {app_path}")
        return
    
    print("🌐 Starting Noise-Robust ASR Demo...")
    print("📱 Opening in browser at: http://localhost:8502")
    print("\n🎯 Demo Features:")
    print("  🎚️ Interactive noise simulation")
    print("  📊 Real-time performance analysis")
    print("  🔄 Dynamic adaptation visualization")
    print("  🌍 Real-world scenario testing")
    print("  📈 Comparative benchmarking")
    print("\n🛑 Press Ctrl+C to stop")
    print("=" * 50)
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(app_path),
            "--server.address", "localhost",
            "--server.port", "8502",  # Different port from hallucination demo
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
       print("\n👋 Demo stopped by user")
    except Exception as e:
       print(f"❌ Error launching demo: {e}")
       print("\nTroubleshooting:")
       print("1. Ensure port 8502 is available")
       print("2. Try: streamlit run app/streamlit_demo.py --server.port 8503")
       print("3. Check firewall settings")

if __name__ == "__main__":
   main()