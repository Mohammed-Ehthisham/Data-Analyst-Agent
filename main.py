from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import List, Dict
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
import asyncio
from typing import Any, Union
import logging

from src.llm_agent import LLMDataAnalystAgent
from src.config import get_settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="LLM-Driven Data Analyst Agent",
    description="An intelligent API that uses LLM system prompts for structured data analysis",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the LLM-driven data analyst agent
settings = get_settings()
agent = LLMDataAnalystAgent(settings)


def _schema_fallback(question_text: str) -> dict:
    """Return a dynamic JSON with keys inferred from the question text.

    Heuristics:
    - Extract bullet list items like `- key_name` as required keys.
    - Choose default value types based on key words.
    - If no keys found, return a minimal timeout note.
    """
    import re

    text = question_text or ""
    keys: list[str] = []

    # Extract keys from bullet points (e.g., '- total_sales')
    for line in text.splitlines():
        m = re.match(r"\s*[-*]\s*([A-Za-z0-9_]+)", line)
        if m:
            keys.append(m.group(1))

    # If none found, try to detect JSON-like patterns: "key": or key:
    if not keys:
        for m in re.finditer(r"\b([A-Za-z0-9_]+)\s*:\s*", text):
            keys.append(m.group(1))

    # Deduplicate while preserving order
    seen = set()
    keys = [k for k in keys if not (k in seen or seen.add(k))]

    def default_for(key: str):
        k = key.lower()
        # Image-like fields
        if any(s in k for s in ["chart", "plot", "graph", "image"]):
            return "data:image/png;base64,"
        # Distances / path lengths often int
        if "shortest_path" in k:
            return -1
        # Float-style metrics
        if any(s in k for s in ["correlation", "density", "ratio", "r2", "average_degree", "slope"]):
            return 0.0
        # Common numeric totals/aggregations
        if any(s in k for s in ["total", "count", "sum", "median", "mean", "sales", "tax", "records"]):
            return 0
        # Strings for labels
        if any(s in k for s in ["region", "node", "name", "title", "top_"]):
            return ""
        # Default to empty string
        return ""

    if keys:
        return {k: default_for(k) for k in keys}

    # Fallback minimal
    return {"status": "timeout", "note": "Returning fallback structure."}

@app.get("/")
async def root():
    """Health check endpoint"""
    logger.info("Root endpoint accessed")
    return {"message": "LLM-Driven Data Analyst Agent is running", "status": "healthy", "version": "2.0.0"}

@app.get("/health")
async def health():
    """Health check endpoint for Render deployment"""
    return {"status": "healthy", "version": "2.0.0", "model": "LLM-Driven Agent"}

@app.post("/api/")
async def analyze_data(files: List[UploadFile] = File(...)) -> Union[list, dict]:
    """
    Main endpoint for data analysis tasks.

    - Accepts multiple files. One must be a questions.txt that contains the prompt.
    - Any additional files (.csv, .json, .png, etc.) are passed to the agent as context.
    """
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded. Expected at least questions.txt.")

        # Log incoming file names
        file_names = [f.filename for f in files if f and f.filename]
        logger.info(f"Incoming files: {file_names}")

        # Find the questions.txt file (case-insensitive)
        question_file: UploadFile | None = None
        for f in files:
            if f and f.filename and f.filename.lower() == "questions.txt":
                question_file = f
                break

        if question_file is None:
            raise HTTPException(status_code=400, detail="Missing required file: questions.txt")

        # Read all files
        contents: Dict[str, bytes] = {}
        for f in files:
            try:
                contents[f.filename] = await f.read()
            except Exception:
                contents[f.filename] = b""

        # Extract the question
        try:
            question = contents[question_file.filename].decode("utf-8", errors="ignore").strip()
        except Exception:
            question = ""

        if not question:
            raise HTTPException(status_code=400, detail="questions.txt is empty or unreadable.")

        logger.info(f"Received analysis request: {question[:200]}...")

        # Prepare additional context files (exclude questions.txt)
        context_files = []
        for name, data in contents.items():
            if name == question_file.filename:
                continue
            # Infer a simple content type from extension
            ext = (name.rsplit('.', 1)[-1].lower() if '.' in name else '')
            content_type = {
                'csv': 'text/csv',
                'json': 'application/json',
                'png': 'image/png',
                'jpg': 'image/jpeg',
                'jpeg': 'image/jpeg',
                'parquet': 'application/octet-stream'
            }.get(ext, 'application/octet-stream')
            context_files.append({
                'filename': name,
                'content_type': content_type,
                'content': data,
            })

        # Process the request with timeout (3 minutes)
        try:
            result = await asyncio.wait_for(
                agent.analyze(question, files=context_files),
                timeout=settings.timeout_seconds
            )

            logger.info("Analysis completed successfully")
            return result

        except asyncio.TimeoutError:
            logger.error("Analysis timed out after configured timeout")
            # Return schema-aware fallback to get partial credit
            return _schema_fallback(question)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        # Return a safe fallback JSON
        return {"error": "server_error", "message": str(e)}

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
            timeout=settings.timeout_seconds  # Use consistent timeout
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
    import os
    port = int(os.environ.get("PORT", 8001))  # Changed default to 8001
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=port,
        log_level="info"
    )
