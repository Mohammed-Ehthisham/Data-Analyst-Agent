#!/usr/bin/env python3
"""
Development server startup script
"""

import os
import sys
import subprocess
from pathlib import Path

def check_env_file():
    """Check if .env file exists"""
    if not Path(".env").exists():
        print("❌ .env file not found!")
        print("📝 Creating .env from template...")
        
        if Path(".env.example").exists():
            # Copy example file
            with open(".env.example", "r") as src, open(".env", "w") as dst:
                content = src.read()
                dst.write(content)
            
            print("✅ .env file created from .env.example")
            print("🔑 Please edit .env and add your OpenAI API key!")
            return False
        else:
            print("❌ .env.example not found!")
            return False
    return True

def check_dependencies():
    """Check if dependencies are installed"""
    try:
        import fastapi
        import uvicorn
        import pandas
        print("✅ Dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("📦 Install dependencies with: pip install -r requirements.txt")
        return False

def start_server():
    """Start the development server"""
    try:
        print("🚀 Starting Data Analyst Agent...")
        print("🌐 Server will be available at: http://localhost:8000")
        print("📖 API docs: http://localhost:8000/docs")
        print("💊 Health check: http://localhost:8000/health")
        print("\n" + "="*50)
        
        # Start uvicorn server
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "main:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload",
            "--log-level", "info"
        ])
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

def main():
    """Main startup function"""
    print("🔧 Data Analyst Agent - Development Setup")
    print("="*50)
    
    # Check environment
    if not check_env_file():
        return 1
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Start server
    start_server()
    return 0

if __name__ == "__main__":
    sys.exit(main())
