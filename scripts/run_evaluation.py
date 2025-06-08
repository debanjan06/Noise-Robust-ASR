#!/usr/bin/env python3
"""
Run comprehensive evaluation of ASR systems
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from evaluation.audio_evaluator import AudioEvaluator

def main():
    print("🚀 ASR System Comprehensive Evaluation")
    print("=" * 50)
    print("This evaluation will:")
    print("  • Test noise robustness across SNR levels")
    print("  • Evaluate hallucination detection accuracy")
    print("  • Generate performance visualizations")
    print("  • Create detailed analysis report")
    print()
    
    try:
        # Run evaluation
        evaluator = AudioEvaluator()
        results, report = evaluator.run_complete_evaluation()
        
        print("\n🎉 Evaluation completed successfully!")
        print("\n📊 Key Results:")
        
        # Print quick summary
        if "noise_robustness" in results:
            noise_results = [r for r in results["noise_robustness"] if r.get("success", False)]
            if noise_results:
                avg_wer = sum(r.get("wer", 0) for r in noise_results) / len(noise_results)
                print(f"  • Average WER across conditions: {avg_wer:.3f}")
        
        if "hallucination_detection" in results:
            hall_results = [r for r in results["hallucination_detection"] if r.get("success", False)]
            if hall_results:
                avg_f1 = sum(r.get("f1", 0) for r in hall_results) / len(hall_results)
                print(f"  • Average hallucination detection F1: {avg_f1:.3f}")
        
        print(f"\n📁 Check results folder for detailed analysis and plots!")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        print("Make sure all dependencies are installed:")
        print("pip install -r requirements.txt")

if __name__ == "__main__":
    main()