#!/usr/bin/env python3
"""
Vercel deployment preparation script
"""

import os
import sys
from pathlib import Path

def check_environment():
    """Check if environment is ready for deployment"""
    print("🔍 Checking environment...")
    
    # Check if OpenAI API key is available
    openai_key = os.getenv('OPENAI_API_KEY')
    if openai_key:
        print("✅ OPENAI_API_KEY found in environment")
    else:
        print("⚠️  OPENAI_API_KEY not found in local environment")
        print("   Make sure to set it in Vercel dashboard")
    
    return True

def check_required_files():
    """Check if all required files exist"""
    print("📋 Checking required files...")
    
    required_files = {
        'main.py': 'FastAPI application entry point',
        'requirements.txt': 'Python dependencies',
        'vercel.json': 'Vercel configuration',
        'runtime.txt': 'Python version specification',
        'src/agent.py': 'Main agent implementation',
        'src/config.py': 'Configuration settings'
    }
    
    missing_files = []
    
    for file_path, description in required_files.items():
        if Path(file_path).exists():
            print(f"✅ {file_path} - {description}")
        else:
            print(f"❌ {file_path} - {description}")
            missing_files.append(file_path)
    
    return len(missing_files) == 0

def check_dependencies():
    """Check if dependencies are properly specified"""
    print("📦 Checking dependencies...")
    
    try:
        with open('requirements.txt', 'r') as f:
            deps = f.read().strip().split('\n')
        
        essential_deps = ['fastapi', 'openai', 'pandas', 'uvicorn']
        found_deps = []
        
        for dep in deps:
            if dep and not dep.startswith('#'):
                # Handle uvicorn[standard] format
                dep_name = dep.split('==')[0].split('>=')[0].split('<=')[0].split('[')[0]
                if dep_name in essential_deps:
                    found_deps.append(dep_name)
        
        for dep in essential_deps:
            if dep in found_deps:
                print(f"✅ {dep} specified")
            else:
                print(f"❌ {dep} missing")
        
        return len(found_deps) >= len(essential_deps)
        
    except Exception as e:
        print(f"❌ Error reading requirements.txt: {e}")
        return False

def validate_configuration():
    """Validate configuration settings"""
    print("⚙️  Validating configuration...")
    
    try:
        # Test if we can import and initialize settings
        sys.path.append('.')
        from src.config import get_settings
        
        settings = get_settings()
        
        # Check if settings can be loaded
        print("✅ Configuration imports successfully")
        
        # Check critical settings
        if hasattr(settings, 'openai_api_key'):
            print("✅ openai_api_key field exists")
        else:
            print("❌ openai_api_key field missing")
            return False
            
        if hasattr(settings, 'openai_model'):
            print("✅ openai_model field exists")
        else:
            print("❌ openai_model field missing")
            return False
        
        # Test environment variable mapping
        os.environ['TEST_OPENAI_KEY'] = 'test-key'
        test_settings = get_settings()
        if test_settings.openai_model == 'gpt-4':
            print("✅ Default values working")
        else:
            print("⚠️  Default values might have issues")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False

def validate_vercel_config():
    """Validate Vercel configuration"""
    print("🔧 Validating Vercel configuration...")
    
    try:
        import json
        with open('vercel.json', 'r') as f:
            config = json.load(f)
        
        # Check essential configuration
        if 'builds' in config and config['builds']:
            print("✅ Build configuration found")
        else:
            print("❌ Build configuration missing")
            return False
        
        if 'routes' in config and config['routes']:
            print("✅ Route configuration found")
        else:
            print("❌ Route configuration missing")
            return False
        
        # Check if main.py is the entry point
        builds = config.get('builds', [])
        if any(build.get('src') == 'main.py' for build in builds):
            print("✅ main.py configured as entry point")
        else:
            print("⚠️  main.py not found in builds configuration")
        
        return True
        
    except Exception as e:
        print(f"❌ Error validating vercel.json: {e}")
        return False
    """Validate Vercel configuration"""
    print("⚙️  Validating Vercel configuration...")
    
    try:
        import json
        with open('vercel.json', 'r') as f:
            config = json.load(f)
        
        # Check essential configuration
        if 'builds' in config and config['builds']:
            print("✅ Build configuration found")
        else:
            print("❌ Build configuration missing")
            return False
        
        if 'routes' in config and config['routes']:
            print("✅ Route configuration found")
        else:
            print("❌ Route configuration missing")
            return False
        
        # Check if main.py is the entry point
        builds = config.get('builds', [])
        if any(build.get('src') == 'main.py' for build in builds):
            print("✅ main.py configured as entry point")
        else:
            print("⚠️  main.py not found in builds configuration")
        
        return True
        
    except Exception as e:
        print(f"❌ Error validating vercel.json: {e}")
        return False

def show_deployment_instructions():
    """Show deployment instructions"""
    print("\n🚀 DEPLOYMENT INSTRUCTIONS")
    print("=" * 50)
    
    print("\n1. COMMIT YOUR CHANGES:")
    print("   git add .")
    print("   git commit -m 'Prepare for Vercel deployment'")
    print("   git push origin main")
    
    print("\n2. DEPLOY TO VERCEL:")
    print("   • Go to https://vercel.com/dashboard")
    print("   • Click 'Import Project'")
    print("   • Connect your GitHub repository")
    print("   • Select 'Data-Analyst-Agent' repository")
    print("   • Click 'Deploy'")
    
    print("\n3. SET ENVIRONMENT VARIABLES:")
    print("   After deployment, go to Project Settings > Environment Variables")
    print("   Add these variables:")
    print("   • OPENAI_API_KEY = (your OpenAI API key)")
    print("   • OPENAI_MODEL = gpt-4")
    print("   • DEBUG = False")
    
    print("\n4. TEST YOUR DEPLOYMENT:")
    print("   • https://your-app.vercel.app/ (health check)")
    print("   • https://your-app.vercel.app/health (detailed health)")
    print("   • POST to https://your-app.vercel.app/api/ (main endpoint)")
    
    print("\n✨ Your app should be ready to use!")

def main():
    """Main deployment preparation function"""
    print("🚀 DATA ANALYST AGENT - VERCEL DEPLOYMENT PREPARATION")
    print("=" * 60)
    
    # Run all checks
    checks = [
        check_environment(),
        check_required_files(),
        check_dependencies(),
        validate_configuration(),
        validate_vercel_config()
    ]
    
    if all(checks):
        print("\n✅ ALL CHECKS PASSED!")
        print("Your project is ready for Vercel deployment.")
        show_deployment_instructions()
        return True
    else:
        print("\n❌ SOME CHECKS FAILED!")
        print("Please fix the issues above before deploying.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
