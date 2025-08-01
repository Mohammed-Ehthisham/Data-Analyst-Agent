# Makefile for Data Analyst Agent

.PHONY: help setup install test run docker-build docker-run clean lint format

# Default target
help:
	@echo "Data Analyst Agent - Available Commands:"
	@echo "======================================"
	@echo "setup          - Run complete project setup"
	@echo "install        - Install Python dependencies"
	@echo "test           - Run test suite"
	@echo "test-api       - Test API endpoints (requires running server)"
	@echo "run            - Start development server"
	@echo "docker-build   - Build Docker image"
	@echo "docker-run     - Run with Docker Compose"
	@echo "lint           - Run code linting"
	@echo "format         - Format code with black"
	@echo "clean          - Clean up temporary files"
	@echo "deploy         - Deploy to production"

# Setup and installation
setup:
	python setup.py

install:
	pip install -r requirements.txt

# Testing
test:
	pytest tests/ -v

test-api:
	python test_api.py

# Development
run:
	python start.py

run-direct:
	uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Docker commands
docker-build:
	docker build -t data-analyst-agent .

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down

docker-logs:
	docker-compose logs -f

# Code quality
lint:
	flake8 src/ tests/ main.py --max-line-length=100 --ignore=E203,W503

format:
	black src/ tests/ main.py --line-length=100

# Deployment
deploy:
	@echo "Choose deployment method:"
	@echo "1. Railway: git push railway main"
	@echo "2. Render: Connect GitHub repository"
	@echo "3. Docker: make docker-run"
	@echo "4. Cloud Run: gcloud run deploy"

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf build/
	rm -rf dist/

# Production setup
prod-setup:
	cp .env.example .env
	@echo "Please edit .env with your production values"

# Health check
health:
	curl -f http://localhost:8000/health || echo "Service not running"

# Full test including API
test-full: test test-api

# Development workflow
dev: install run

# Production workflow  
prod: docker-build docker-run
