import os
import sys
import io
from pathlib import Path

# Prevent UnicodeEncodeError on Windows console
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add project root (parent of scripts/) to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from core.feedback_engine import feedback_engine

def test_feedback():
    print("Testing FeedbackEngine...")
    # Mock project name based on existing files
    project_name = "abcd1" 
    
    # Test path finding
    paths = feedback_engine._find_project_json_files(project_name)
    print(f"Found paths: {paths}")
    
    if not paths:
        print("❌ Could not find JSON files for project abcd1. This confirms the path bug.")
    else:
        print("✅ Found JSON files.")
        
    # Test feedback generation
    feedback = feedback_engine.generate_feedback(project_name)
    print("\n--- Feedback ---")
    print(feedback)
    
    # Test timeline feedback
    timeline = feedback_engine.generate_timeline_feedback([], project_name)
    print("\n--- Timeline Feedback ---")
    print(timeline)

if __name__ == "__main__":
    test_feedback()
