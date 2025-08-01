#!/bin/bash

# Deployment script for Data Analyst Agent

echo "🚀 Deploying Data Analyst Agent..."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ .env file not found. Please copy .env.example to .env and configure your API keys."
    exit 1
fi

# Build and start with Docker Compose
echo "📦 Building Docker image..."
docker-compose build

echo "🔧 Starting services..."
docker-compose up -d

echo "⏳ Waiting for services to start..."
sleep 30

# Health check
echo "🔍 Checking service health..."
response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health)

if [ $response = "200" ]; then
    echo "✅ Service is healthy and running!"
    echo "🌐 API endpoint: http://localhost:8000/api/"
    echo "📖 API docs: http://localhost:8000/docs"
    echo "💊 Health check: http://localhost:8000/health"
else
    echo "❌ Service health check failed. Check logs:"
    docker-compose logs
    exit 1
fi

echo "🎉 Deployment complete!"
