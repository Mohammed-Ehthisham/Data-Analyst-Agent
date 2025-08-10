#!/bin/bash
# Render start script

# Install dependencies
pip install -r requirements.txt

# Start the FastAPI application with Gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT
