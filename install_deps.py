#!/usr/bin/env python3
"""
Install all dependencies and verify the setup
"""

import subprocess
import sys

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} installed successfully")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package}")
        return False

def verify_import(module_name, package_name=None):
    """Verify that a module can be imported"""
    if package_name is None:
        package_name = module_name
    
    try:
        __import__(module_name)
        print(f"✅ {package_name} imported successfully")
        return True
    except ImportError:
        print(f"❌ {package_name} import failed")
        return False

def main():
    print("🔧 Installing and verifying Data Analyst Agent dependencies...")
    print("=" * 60)
    
    # Critical packages that must be installed
    critical_packages = [
        "fastapi",
        "uvicorn[standard]",
        "openai",
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn", 
        "plotly",
        "requests",
        "beautifulsoup4",
        "aiohttp",
        "pydantic-settings",
        "python-multipart",
        "python-dotenv",
        "scikit-learn",
        "pillow"
    ]
    
    print("📦 Installing packages...")
    for package in critical_packages:
        install_package(package)
    
    print("\n🧪 Verifying imports...")
    
    # Test imports
    imports_to_test = [
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("openai", "OpenAI"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("matplotlib.pyplot", "Matplotlib"),
        ("seaborn", "Seaborn"),
        ("plotly.graph_objects", "Plotly"),
        ("requests", "Requests"),
        ("bs4", "BeautifulSoup4"),
        ("aiohttp", "aiohttp"),
        ("pydantic_settings", "pydantic-settings"),
        ("sklearn", "scikit-learn"),
        ("PIL", "Pillow")
    ]
    
    failed_imports = []
    for module, name in imports_to_test:
        if not verify_import(module, name):
            failed_imports.append(name)
    
    print("\n🔍 Testing application modules...")
    
    app_modules = [
        ("src.config", "Config"),
        ("src.data_processor", "Data Processor"),
        ("src.visualizer", "Visualizer"),
        ("src.web_scraper", "Web Scraper"),
        ("src.agent", "Agent")
    ]
    
    for module, name in app_modules:
        if not verify_import(module, name):
            failed_imports.append(name)
    
    # Try to import main app
    print("\n🚀 Testing main application...")
    if verify_import("main", "Main App"):
        print("✅ Main application ready!")
    else:
        failed_imports.append("Main App")
    
    print("\n" + "=" * 60)
    if not failed_imports:
        print("🎉 All dependencies installed and verified successfully!")
        print("\n📝 Next steps:")
        print("1. Make sure your .env file has a valid OpenAI API key")
        print("2. Run: python start_server.py")
        print("3. Visit: http://127.0.0.1:8000")
        return True
    else:
        print(f"❌ {len(failed_imports)} modules failed verification:")
        for module in failed_imports:
            print(f"   - {module}")
        print("\nTry running this script again or install missing packages manually.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
