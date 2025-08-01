# Data Analyst Agent

An intelligent API that uses LLMs to source, prepare, analyze, and visualize data from various sources.

## Features

- **Multi-source Data Integration**: Web scraping, databases, APIs
- **LLM-powered Analysis**: Intelligent data interpretation and insights
- **Dynamic Visualizations**: Generate charts and plots as base64 images
- **Fast Response**: Optimized for <3 minute response times
- **Flexible Input**: Accept various data analysis task descriptions

## API Usage

```bash
curl "https://your-api-endpoint.com/api/" -X POST -F "@question.txt"
```

## Architecture

- **FastAPI**: High-performance web framework
- **LangChain**: LLM orchestration and data processing
- **Pandas/Polars**: Data manipulation
- **DuckDB**: Fast analytical queries
- **Matplotlib/Plotly**: Data visualization
- **BeautifulSoup**: Web scraping
- **Docker**: Containerized deployment

## Sample Questions Supported

1. Web scraping and analysis (Wikipedia, structured data)
2. Database queries and statistical analysis
3. Data visualization and regression analysis
4. Multi-format data processing (JSON, Parquet, CSV)

## Response Formats

- JSON arrays for multiple answers
- JSON objects for structured responses
- Base64-encoded images for visualizations

## Deployment

The application is containerized and can be deployed on:
- Railway
- Render
- Google Cloud Run
- AWS Lambda
- Any Docker-compatible platform

## License

MIT License - see LICENSE file for details.