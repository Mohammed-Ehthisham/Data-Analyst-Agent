#!/usr/bin/env python3
"""
Complete setup script for the Data Analyst Agent project
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def print_step(step, description):
    """Print a step description"""
    print(f"\n{step}. {description}")

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_git():
    """Check if git is available"""
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
        print("✅ Git is available")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Git is not available")
        return False

def check_docker():
    """Check if Docker is available"""
    try:
        subprocess.run(["docker", "--version"], capture_output=True, check=True)
        print("✅ Docker is available")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️ Docker is not available (optional)")
        return False

def setup_environment():
    """Set up the Python environment"""
    print_step(1, "Setting up Python environment")
    
    # Check if virtual environment exists
    if not Path("venv").exists():
        print("📦 Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", "venv"])
        print("✅ Virtual environment created")
    else:
        print("✅ Virtual environment already exists")
    
    # Install dependencies
    print("📦 Installing dependencies...")
    venv_python = "venv/Scripts/python" if os.name == "nt" else "venv/bin/python"
    
    try:
        subprocess.run([venv_python, "-m", "pip", "install", "--upgrade", "pip"], check=True)
        subprocess.run([venv_python, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def setup_env_file():
    """Set up environment file"""
    print_step(2, "Setting up environment configuration")
    
    if not Path(".env").exists():
        if Path(".env.example").exists():
            shutil.copy(".env.example", ".env")
            print("✅ .env file created from template")
        else:
            # Create basic .env file
            env_content = """# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4

# Application Settings
DEBUG=False
LOG_LEVEL=INFO

# API Configuration
MAX_REQUEST_SIZE=10485760
TIMEOUT_SECONDS=180

# Data Processing Settings
MAX_PLOT_SIZE=100000
DEFAULT_FIGURE_WIDTH=10
DEFAULT_FIGURE_HEIGHT=6
"""
            with open(".env", "w") as f:
                f.write(env_content)
            print("✅ .env file created")
        
        print("🔑 Please edit .env and add your OpenAI API key!")
        return False
    else:
        print("✅ .env file already exists")
        return True

def run_tests():
    """Run basic tests"""
    print_step(3, "Running basic tests")
    
    venv_python = "venv/Scripts/python" if os.name == "nt" else "venv/bin/python"
    
    try:
        # Test imports
        test_script = """
import sys
sys.path.append('.')

try:
    from src.config import get_settings
    from src.agent import DataAnalystAgent
    from src.data_processor import DataProcessor
    from src.visualizer import Visualizer
    from src.web_scraper import WebScraper
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
"""
        
        result = subprocess.run([venv_python, "-c", test_script], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print(result.stdout.strip())
            return True
        else:
            print(f"❌ Import test failed: {result.stderr}")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Test failed: {e}")
        return False

def setup_git_hooks():
    """Set up git hooks for development"""
    print_step(4, "Setting up development tools")
    
    if Path(".git").exists():
        # Create pre-commit hook
        hooks_dir = Path(".git/hooks")
        hooks_dir.mkdir(exist_ok=True)
        
        pre_commit_hook = hooks_dir / "pre-commit"
        hook_content = """#!/bin/sh
# Run tests before commit
echo "Running tests..."
python -m pytest tests/ --maxfail=1 --disable-warnings -q
if [ $? -ne 0 ]; then
    echo "Tests failed. Commit aborted."
    exit 1
fi
echo "Tests passed. Proceeding with commit."
"""
        
        with open(pre_commit_hook, "w") as f:
            f.write(hook_content)
        
        # Make it executable (Unix-like systems)
        if os.name != "nt":
            os.chmod(pre_commit_hook, 0o755)
        
        print("✅ Git hooks set up")
    else:
        print("⚠️ Not a git repository, skipping git hooks")

def show_next_steps():
    """Show next steps to the user"""
    print_header("🎉 Setup Complete!")
    
    print("\n📋 Next Steps:")
    print("1. Edit .env file and add your OpenAI API key")
    print("2. Activate virtual environment:")
    if os.name == "nt":
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    
    print("3. Start the development server:")
    print("   python start.py")
    print("   or")
    print("   uvicorn main:app --reload")
    
    print("\n🌐 Access Points:")
    print("   API: http://localhost:8000/api/")
    print("   Docs: http://localhost:8000/docs")
    print("   Health: http://localhost:8000/health")
    
    print("\n🚀 Deployment:")
    print("   Docker: docker-compose up")
    print("   See DEPLOYMENT.md for cloud deployment options")
    
    print("\n🧪 Testing:")
    print("   python test_api.py (after starting server)")
    print("   pytest (run test suite)")

def main():
    """Main setup function"""
    print_header("Data Analyst Agent - Setup Script")
    
    # Check system requirements
    print("🔍 Checking system requirements...")
    
    if not check_python_version():
        return 1
    
    check_git()
    check_docker()
    
    # Run setup steps
    success = True
    
    # Step 1: Environment setup
    if not setup_environment():
        success = False
    
    # Step 2: Environment file
    env_ready = setup_env_file()
    
    # Step 3: Run tests
    if success and env_ready:
        if not run_tests():
            success = False
    
    # Step 4: Development tools
    setup_git_hooks()
    
    # Show results
    if success:
        show_next_steps()
        if not env_ready:
            print("\n⚠️ Don't forget to configure your OpenAI API key in .env!")
        return 0
    else:
        print("\n❌ Setup incomplete. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
