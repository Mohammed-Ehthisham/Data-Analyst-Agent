#!/usr/bin/env python3
"""
Test script to verify the Data Analyst Agent setup
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        import fastapi
        print("✅ FastAPI imported successfully")
    except ImportError as e:
        print(f"❌ FastAPI import failed: {e}")
        return False
    
    try:
        import uvicorn
        print("✅ Uvicorn imported successfully")
    except ImportError as e:
        print(f"❌ Uvicorn import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✅ Pandas imported successfully")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✅ Matplotlib imported successfully")
    except ImportError as e:
        print(f"❌ Matplotlib import failed: {e}")
        return False
    
    try:
        from src.config import get_settings
        print("✅ Config module imported successfully")
    except ImportError as e:
        print(f"❌ Config import failed: {e}")
        return False
    
    try:
        from main import app
        print("✅ Main app imported successfully")
    except ImportError as e:
        print(f"❌ Main app import failed: {e}")
        return False
    
    return True

def test_config():
    """Test configuration loading"""
    print("\n🔧 Testing configuration...")
    
    try:
        from src.config import get_settings
        settings = get_settings()
        
        print(f"✅ Settings loaded")
        print(f"   - OpenAI Model: {settings.openai_model}")
        print(f"   - Debug Mode: {settings.debug}")
        print(f"   - Log Level: {settings.log_level}")
        
        if settings.openai_api_key.startswith('sk-'):
            print("✅ OpenAI API key format looks correct")
            return True
        else:
            print("⚠️ OpenAI API key format incorrect (should start with 'sk-')")
            print("   Please get your API key from: https://platform.openai.com/api-keys")
            return False
            
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_app_creation():
    """Test FastAPI app creation"""
    print("\n🚀 Testing FastAPI app...")
    
    try:
        from main import app
        
        # Check if app has the expected routes
        routes = [route.path for route in app.routes]
        expected_routes = ["/", "/health", "/api/", "/api/test"]
        
        for route in expected_routes:
            if route in routes:
                print(f"✅ Route {route} found")
            else:
                print(f"❌ Route {route} missing")
                return False
        
        print("✅ FastAPI app created successfully")
        return True
        
    except Exception as e:
        print(f"❌ App creation test failed: {e}")
        return False

def test_data_processing():
    """Test basic data processing functionality"""
    print("\n📊 Testing data processing...")
    
    try:
        import pandas as pd
        import numpy as np
        
        # Create sample data
        df = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': [2, 4, 6, 8, 10]
        })
        
        # Test correlation
        correlation = df['x'].corr(df['y'])
        print(f"✅ Data correlation calculated: {correlation}")
        
        # Test visualization (without actually displaying)
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots()
        ax.scatter(df['x'], df['y'])
        plt.close(fig)
        print("✅ Basic plot creation successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Data processing test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Data Analyst Agent - System Test")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Configuration Test", test_config),
        ("FastAPI App Test", test_app_creation),
        ("Data Processing Test", test_data_processing)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        if test_func():
            passed += 1
        else:
            print(f"\n⚠️ {test_name} had issues")
    
    print("\n" + "=" * 50)
    print(f"📋 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your setup is ready.")
        print("\n📝 Next steps:")
        print("1. Get your OpenAI API key from: https://platform.openai.com/api-keys")
        print("2. Update .env file with your actual API key")
        print("3. Start the server: uvicorn main:app --reload")
        print("4. Test the API: python test_api.py")
        return 0
    else:
        print(f"\n❌ {total - passed} tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
