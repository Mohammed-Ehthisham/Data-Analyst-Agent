from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
import asyncio
from typing import Any, Union
import logging

from src.agent import DataAnalystAgent
from src.config import get_settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Data Analyst Agent",
    description="An intelligent API that uses LLMs to source, prepare, analyze, and visualize data",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the data analyst agent
settings = get_settings()
agent = DataAnalystAgent(settings)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Data Analyst Agent is running", "status": "healthy"}

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "version": "1.0.0"}

@app.post("/api/")
async def analyze_data(file: UploadFile = File(...)) -> Union[list, dict]:
    """
    Main endpoint for data analysis tasks.
    
    Accepts a text file with analysis instructions and returns results in JSON format.
    """
    try:
        # Read the uploaded file
        content = await file.read()
        question = content.decode('utf-8').strip()
        
        logger.info(f"Received analysis request: {question[:200]}...")
        
        # Process the request with timeout (3 minutes)
        try:
            result = await asyncio.wait_for(
                agent.analyze(question), 
                timeout=180  # 3 minutes
            )
            
            logger.info("Analysis completed successfully")
            return result
            
        except asyncio.TimeoutError:
            logger.error("Analysis timed out after 3 minutes")
            raise HTTPException(
                status_code=408, 
                detail="Analysis timed out. Please try with a simpler question."
            )
            
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing analysis request: {str(e)}"
        )

@app.post("/api/test")
async def test_endpoint(question: str) -> Union[list, dict]:
    """
    Test endpoint that accepts a direct string question.
    Useful for development and testing.
    """
    try:
        logger.info(f"Test request: {question[:200]}...")
        
        result = await asyncio.wait_for(
            agent.analyze(question), 
            timeout=180
        )
        
        return result
        
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=408, 
            detail="Analysis timed out after 3 minutes"
        )
    except Exception as e:
        logger.error(f"Error in test endpoint: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )

# Export for Vercel
handler = app
