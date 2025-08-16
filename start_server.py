#!/usr/bin/env python3
"""
Simple server startup script
"""

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Data Analyst Agent server...")
    print("📍 Server will be available at: http://localhost:8000")
    print("📖 API documentation: http://localhost:8000/docs")
    print("💊 Health check: http://localhost:8000/health")
    print("\nPress Ctrl+C to stop the server")
    
    try:
        uvicorn.run(
            "main:app",
            host="127.0.0.1",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
