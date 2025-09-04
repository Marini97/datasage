#!/usr/bin/env python3
"""
Simple test script to verify web interface can be imported.
Used by CI to ensure web app dependencies are working.
"""

import sys
import os

# Add the datasage package to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_web_imports():
    """Test that web_app can be imported without errors."""
    try:
        from datasage.web_app import main, display_dataframe_info
        print("✅ Successfully imported web_app components")
        return True
    except Exception as e:
        print(f"❌ Failed to import web_app: {e}")
        return False

def main():
    """Run the web interface import test."""
    print("🌐 Testing Web Interface Imports")
    print("=" * 35)
    
    success = test_web_imports()
    
    if success:
        print("🎉 Web interface imports test passed!")
    else:
        print("⚠️ Web interface imports test failed!")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
