@echo off
REM Deployment script for Data Analyst Agent (Windows)

echo 🚀 Deploying Data Analyst Agent...

REM Check if .env file exists
if not exist .env (
    echo ❌ .env file not found. Please copy .env.example to .env and configure your API keys.
    exit /b 1
)

REM Build and start with Docker Compose
echo 📦 Building Docker image...
docker-compose build

echo 🔧 Starting services...
docker-compose up -d

echo ⏳ Waiting for services to start...
timeout /t 30 /nobreak

REM Health check
echo 🔍 Checking service health...
curl -s -o nul -w "%%{http_code}" http://localhost:8000/health > health_check.txt
set /p response=<health_check.txt
del health_check.txt

if "%response%"=="200" (
    echo ✅ Service is healthy and running!
    echo 🌐 API endpoint: http://localhost:8000/api/
    echo 📖 API docs: http://localhost:8000/docs
    echo 💊 Health check: http://localhost:8000/health
) else (
    echo ❌ Service health check failed. Check logs:
    docker-compose logs
    exit /b 1
)

echo 🎉 Deployment complete!
