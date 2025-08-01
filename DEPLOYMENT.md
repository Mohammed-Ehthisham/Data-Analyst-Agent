# Data Analyst Agent - Setup and Deployment Guide

## Quick Start

### 1. Environment Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Mohammed-Ehthisham/Data-Analyst-Agent.git
   cd Data-Analyst-Agent
   ```

2. **Set up environment variables:**
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` file and add your OpenAI API key:
   ```env
   OPENAI_API_KEY=your_actual_openai_api_key_here
   ```

### 2. Local Development

#### Option A: Docker (Recommended)
```bash
# Linux/Mac
./deploy.sh

# Windows
deploy.bat
```

#### Option B: Python Virtual Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run the application
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Testing the API

Once running, test the endpoints:

```bash
# Health check
curl http://localhost:8000/health

# Test with sample question
curl -X POST "http://localhost:8000/api/" \
  -F "file=@examples/wikipedia_question.txt"

# Test endpoint (direct string input)
curl -X POST "http://localhost:8000/api/test?question=Analyze sample data"
```

Visit `http://localhost:8000/docs` for interactive API documentation.

## Deployment Options

### 1. Railway (Recommended for Production)

1. **Fork this repository**
2. **Connect to Railway:**
   - Go to [Railway](https://railway.app)
   - Connect your GitHub account
   - Deploy from your forked repository
3. **Set environment variables in Railway dashboard:**
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `OPENAI_MODEL`: gpt-4 (optional)

### 2. Render

1. **Connect repository to Render**
2. **Use these settings:**
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
3. **Set environment variables:**
   - `OPENAI_API_KEY`: Your OpenAI API key

### 3. Google Cloud Run

```bash
# Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/data-analyst-agent

# Deploy to Cloud Run
gcloud run deploy data-analyst-agent \
  --image gcr.io/YOUR_PROJECT_ID/data-analyst-agent \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars OPENAI_API_KEY=your_key_here
```

### 4. AWS Lambda (with Serverless Framework)

```bash
# Install serverless
npm install -g serverless
npm install serverless-python-requirements

# Deploy
serverless deploy
```

### 5. Digital Ocean App Platform

1. **Connect repository**
2. **Use these settings:**
   - Source: Your GitHub repository
   - Type: Web Service
   - Build Command: `pip install -r requirements.txt`
   - Run Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

## Configuration

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Yes | - |
| `OPENAI_MODEL` | OpenAI model to use | No | gpt-4 |
| `DEBUG` | Enable debug mode | No | False |
| `LOG_LEVEL` | Logging level | No | INFO |
| `MAX_REQUEST_SIZE` | Max request size in bytes | No | 10MB |
| `TIMEOUT_SECONDS` | Request timeout | No | 180 |
| `MAX_PLOT_SIZE` | Max plot size in bytes | No | 100KB |

### Supported Data Sources

- **Wikipedia**: Automatic table scraping and analysis
- **DuckDB**: Cloud-based analytical queries
- **APIs**: RESTful API data fetching
- **CSV/JSON**: File-based data processing
- **Web Scraping**: Custom website data extraction

### Output Formats

- **JSON Arrays**: `[answer1, answer2, ...]`
- **JSON Objects**: `{"question1": "answer1", ...}`
- **Base64 Images**: Data URIs for plots and visualizations

## API Reference

### POST `/api/`

Main analysis endpoint that accepts file uploads.

**Request:**
```bash
curl -X POST "https://your-api.com/api/" \
  -F "file=@question.txt"
```

**Response:**
```json
[
  1,
  "Titanic", 
  0.485782,
  "data:image/png;base64,iVBORw0KG..."
]
```

### POST `/api/test`

Test endpoint for development.

**Request:**
```bash
curl -X POST "https://your-api.com/api/test?question=Your question here"
```

### GET `/health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

## Troubleshooting

### Common Issues

1. **OpenAI API Key Error**
   - Ensure your API key is valid
   - Check billing and rate limits

2. **Memory Issues**
   - Increase container memory limits
   - Optimize data processing for large datasets

3. **Timeout Errors**
   - Increase `TIMEOUT_SECONDS` environment variable
   - Break down complex queries into smaller parts

4. **Plot Size Errors**
   - Reduce plot DPI or dimensions
   - Increase `MAX_PLOT_SIZE` if needed

### Performance Optimization

1. **Use efficient data formats:**
   - Parquet for large datasets
   - Polars for faster processing

2. **Implement caching:**
   - Cache frequently accessed data
   - Use Redis for distributed caching

3. **Optimize visualizations:**
   - Use appropriate plot types
   - Limit data points for large datasets

## Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_agent.py

# Run with coverage
pytest --cov=src

# Run integration tests
pytest -m integration
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details.
