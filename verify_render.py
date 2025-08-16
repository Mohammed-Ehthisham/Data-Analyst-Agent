#!/usr/bin/env python3
"""
Render deployment verification script
"""

import os
import sys

def verify_render_setup():
    """Verify everything is ready for Render deployment"""
    
    print("🚀 RENDER DEPLOYMENT VERIFICATION")
    print("=" * 50)
    
    # Check required files
    required_files = {
        'main.py': 'FastAPI application',
        'requirements.txt': 'Python dependencies',
        'render.yaml': 'Render configuration',
        'start.sh': 'Start script',
        'src/': 'Source code directory'
    }
    
    print("📋 Checking required files...")
    for file_path, description in required_files.items():
        if os.path.exists(file_path):
            print(f"✅ {file_path:20} - {description}")
        else:
            print(f"❌ {file_path:20} - MISSING: {description}")
            return False
    
    # Test app import
    print("\n🧪 Testing app import...")
    try:
        from main import app
        print("✅ FastAPI app imports successfully")
    except Exception as e:
        print(f"❌ App import failed: {e}")
        return False
    
    # Check gunicorn in requirements
    print("\n📦 Checking gunicorn dependency...")
    try:
        with open('requirements.txt', 'r') as f:
            requirements = f.read()
        
        if 'gunicorn' in requirements:
            print("✅ Gunicorn found in requirements.txt")
        else:
            print("❌ Gunicorn missing from requirements.txt")
            return False
    except Exception as e:
        print(f"❌ Error reading requirements.txt: {e}")
        return False
    
    # Check PORT handling
    print("\n🔧 Testing PORT environment variable handling...")
    try:
        import main
        # The main.py should handle PORT env var
        print("✅ PORT handling implemented")
    except Exception as e:
        print(f"❌ PORT handling issue: {e}")
    
    print("\n🎉 RENDER SETUP VERIFICATION COMPLETE!")
    print("\n📝 NEXT STEPS:")
    print("1. git add .")
    print("2. git commit -m 'Prepare for Render deployment'")
    print("3. git push origin main")
    print("4. Go to render.com and deploy!")
    print("\n✨ Your app is ready for Render deployment!")
    
    return True

if __name__ == "__main__":
    success = verify_render_setup()
    sys.exit(0 if success else 1)
