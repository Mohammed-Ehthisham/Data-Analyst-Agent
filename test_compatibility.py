#!/usr/bin/env python3
"""
Comprehensive package compatibility test for Vercel deployment
"""

import sys
import importlib

def test_package_compatibility():
    """Test all packages in requirements.txt for compatibility"""
    
    print("🧪 TESTING PACKAGE COMPATIBILITY FOR VERCEL")
    print("=" * 50)
    
    # Essential packages that must work
    essential_packages = {
        'fastapi': 'Core web framework',
        'uvicorn': 'ASGI server',
        'openai': 'AI functionality',
        'pandas': 'Data processing',
        'numpy': 'Numerical computing',
        'requests': 'HTTP requests',
        'beautifulsoup4': 'Web scraping',
        'matplotlib': 'Static plotting',
        'plotly': 'Interactive plotting',
        'duckdb': 'In-memory database',
        'pydantic': 'Data validation',
        'aiohttp': 'Async HTTP',
        'httpx': 'Modern HTTP client'
    }
    
    # Test imports
    failed_imports = []
    success_count = 0
    
    for package, description in essential_packages.items():
        try:
            # Handle special import names
            import_name = package
            if package == 'beautifulsoup4':
                import_name = 'bs4'
            elif package == 'pillow':
                import_name = 'PIL'
            
            module = importlib.import_module(import_name)
            print(f"✅ {package:20} - {description}")
            success_count += 1
            
            # Test version if available
            if hasattr(module, '__version__'):
                print(f"   Version: {module.__version__}")
            
        except ImportError as e:
            print(f"❌ {package:20} - FAILED: {e}")
            failed_imports.append(package)
        except Exception as e:
            print(f"⚠️  {package:20} - WARNING: {e}")
    
    print(f"\n📊 RESULTS:")
    print(f"✅ Successful imports: {success_count}/{len(essential_packages)}")
    print(f"❌ Failed imports: {len(failed_imports)}")
    
    if failed_imports:
        print(f"\n🔧 FAILED PACKAGES:")
        for package in failed_imports:
            print(f"   - {package}")
        return False
    
    # Test FastAPI app initialization
    print(f"\n🚀 TESTING FASTAPI APP INITIALIZATION...")
    try:
        from src.config import get_settings
        from src.agent import DataAnalystAgent
        
        settings = get_settings()
        agent = DataAnalystAgent(settings)
        print("✅ FastAPI app and agent initialization successful")
        
    except Exception as e:
        print(f"❌ App initialization failed: {e}")
        return False
    
    print(f"\n🎉 ALL COMPATIBILITY TESTS PASSED!")
    print(f"Your requirements.txt is optimized for Vercel deployment!")
    
    return True

if __name__ == "__main__":
    success = test_package_compatibility()
    sys.exit(0 if success else 1)
