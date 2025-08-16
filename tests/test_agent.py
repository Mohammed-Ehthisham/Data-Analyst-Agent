"""
Test suite for the Data Analyst Agent
"""

import pytest
import asyncio
import json
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import pandas as pd

from main import app
from src.agent import DataAnalystAgent
from src.config import get_settings

client = TestClient(app)

@pytest.fixture
def sample_wikipedia_data():
    """Sample Wikipedia table data for testing"""
    return pd.DataFrame({
        'Film': ['Avatar', 'Avengers: Endgame', 'Titanic'],
        'Worldwide gross': ['$2.847 billion', '$2.798 billion', '$2.201 billion'],
        'Year': [2009, 2019, 1997],
        'Rank': [1, 2, 3],
        'Peak': [1, 1, 1]
    })

@pytest.fixture
def sample_court_data():
    """Sample court data for testing"""
    return pd.DataFrame({
        'court': ['Delhi HC', 'Bombay HC', 'Madras HC'],
        'cases_count': [1500, 1200, 1000],
        'year': [2020, 2020, 2020]
    })

class TestAPI:
    """Test the FastAPI endpoints"""
    
    def test_root_endpoint(self):
        """Test the root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        assert "Data Analyst Agent is running" in response.json()["message"]
    
    def test_health_endpoint(self):
        """Test the health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_api_endpoint_with_file(self):
        """Test the main API endpoint with file upload"""
        test_question = "Test analysis question"
        
        with patch('src.agent.DataAnalystAgent.analyze') as mock_analyze:
            mock_analyze.return_value = ["test", "result"]
            
            response = client.post(
                "/api/",
                files={"file": ("question.txt", test_question, "text/plain")}
            )
            
            assert response.status_code == 200
            assert response.json() == ["test", "result"]
    
    def test_test_endpoint(self):
        """Test the test endpoint"""
        with patch('src.agent.DataAnalystAgent.analyze') as mock_analyze:
            mock_analyze.return_value = {"result": "test"}
            
            response = client.post("/api/test?question=test question")
            
            assert response.status_code == 200
            assert response.json() == {"result": "test"}

class TestDataAnalystAgent:
    """Test the main agent functionality"""
    
    @pytest.fixture
    def agent(self):
        """Create agent instance for testing"""
        settings = get_settings()
        return DataAnalystAgent(settings)
    
    @pytest.mark.asyncio
    async def test_parse_task_wikipedia(self, agent):
        """Test task parsing for Wikipedia questions"""
        question = """
        Scrape the list of highest grossing films from Wikipedia. It is at the URL:
        https://en.wikipedia.org/wiki/List_of_highest-grossing_films

        Answer the following questions and respond with a JSON array of strings containing the answer.

        1. How many $2 bn movies were released before 2020?
        2. Which is the earliest film that grossed over $1.5 bn?
        """
        
        task_info = await agent._parse_task(question)
        
        assert task_info["type"] == "wikipedia_scraping"
        assert "https://en.wikipedia.org/wiki/List_of_highest-grossing_films" in task_info["data_sources"]
        assert len(task_info["questions"]) == 2
        assert task_info["output_format"] == "array"
    
    @pytest.mark.asyncio
    async def test_parse_task_database(self, agent):
        """Test task parsing for database questions"""
        question = """
        This DuckDB query counts the number of decisions in the dataset.

        ```sql
        INSTALL httpfs; LOAD httpfs;
        SELECT COUNT(*) FROM read_parquet('s3://test/data.parquet');
        ```

        Answer the following questions and respond with a JSON object containing the answer.
        """
        
        task_info = await agent._parse_task(question)
        
        assert task_info["type"] == "database_analysis"
        assert task_info["output_format"] == "object"
    
    @pytest.mark.asyncio
    async def test_count_high_grossing_films(self, agent, sample_wikipedia_data):
        """Test counting high grossing films"""
        result = await agent._count_high_grossing_films(sample_wikipedia_data, 2.0, 2020)
        
        # Should count Avatar and Titanic (both > $2bn and before 2020)
        assert result == 2
    
    @pytest.mark.asyncio
    async def test_calculate_correlation(self, agent, sample_wikipedia_data):
        """Test correlation calculation"""
        result = await agent._calculate_correlation(sample_wikipedia_data, "Rank", "Peak")
        
        # Correlation should be calculated (exact value depends on data)
        assert isinstance(result, float)
        assert -1 <= result <= 1

class TestWebScraper:
    """Test web scraping functionality"""
    
    @pytest.fixture
    def scraper(self):
        """Create scraper instance for testing"""
        from src.web_scraper import WebScraper
        return WebScraper()
    
    def test_clean_scraped_text(self, scraper):
        """Test text cleaning functionality"""
        dirty_text = "  This   is  a   test[1]  with   $100  and  50%  growth  "
        clean_text = scraper.clean_scraped_text(dirty_text)
        
        assert clean_text == "This is a test with $100 and 50% growth"
    
    @pytest.mark.asyncio
    async def test_scrape_wikipedia_table_mock(self, scraper):
        """Test Wikipedia scraping with mocked response"""
        mock_html = """
        <table class="wikitable">
            <tr><th>Film</th><th>Gross</th></tr>
            <tr><td>Avatar</td><td>$2.847 billion</td></tr>
            <tr><td>Titanic</td><td>$2.201 billion</td></tr>
        </table>
        """
        
        with patch('requests.Session.get') as mock_get:
            mock_response = Mock()
            mock_response.content = mock_html
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response
            
            df = await scraper.scrape_wikipedia_table("https://test.com")
            
            assert df is not None
            assert len(df) == 2
            assert 'Film' in df.columns
            assert 'Avatar' in df['Film'].values

class TestVisualizer:
    """Test visualization functionality"""
    
    @pytest.fixture
    def visualizer(self):
        """Create visualizer instance for testing"""
        from src.visualizer import Visualizer
        settings = get_settings()
        return Visualizer(settings)
    
    @pytest.mark.asyncio
    async def test_create_scatterplot_with_regression(self, visualizer, sample_wikipedia_data):
        """Test scatterplot creation"""
        result = await visualizer.create_scatterplot_with_regression(
            sample_wikipedia_data, 'Rank', 'Peak', 'Test Plot'
        )
        
        assert result.startswith('data:image/png;base64,')
        assert len(result) > 100  # Should be a substantial base64 string
    
    @pytest.mark.asyncio
    async def test_create_bar_chart(self, visualizer, sample_court_data):
        """Test bar chart creation"""
        result = await visualizer.create_bar_chart(
            sample_court_data, 'court', 'cases_count', 'Court Cases'
        )
        
        assert result.startswith('data:image/png;base64,')
        assert len(result) > 100

class TestDataProcessor:
    """Test data processing functionality"""
    
    @pytest.fixture
    def processor(self):
        """Create processor instance for testing"""
        from src.data_processor import DataProcessor
        return DataProcessor()
    
    def test_clean_numeric_column(self, processor):
        """Test numeric column cleaning"""
        dirty_series = pd.Series(['$1,000', '€2,500', '£500'])
        clean_series = processor.clean_numeric_column(dirty_series)
        
        assert clean_series.iloc[0] == 1000
        assert clean_series.iloc[1] == 2500
        assert clean_series.iloc[2] == 500
    
    def test_extract_year_from_text(self, processor):
        """Test year extraction"""
        text_series = pd.Series(['Released in 2020', 'Copyright 2019', 'Year: 2021'])
        year_series = processor.extract_year_from_text(text_series)
        
        assert year_series.iloc[0] == 2020
        assert year_series.iloc[1] == 2019
        assert year_series.iloc[2] == 2021
    
    def test_detect_data_types(self, processor):
        """Test data type detection"""
        test_df = pd.DataFrame({
            'numbers': ['1', '2', '3'],
            'currency': ['$100', '$200', '$300'],
            'dates': ['01/01/2020', '02/01/2020', '03/01/2020'],
            'text': ['apple', 'banana', 'cherry']
        })
        
        types = processor.detect_data_types(test_df)
        
        assert types['numbers'] == 'numeric'
        assert types['currency'] == 'currency'
        assert types['dates'] == 'date'
        assert types['text'] == 'text'

@pytest.mark.integration
class TestIntegration:
    """Integration tests for the complete system"""
    
    @pytest.mark.asyncio
    async def test_full_wikipedia_analysis_flow(self):
        """Test complete Wikipedia analysis flow"""
        settings = get_settings()
        agent = DataAnalystAgent(settings)
        
        question = """
        Scrape the list of highest grossing films from Wikipedia. It is at the URL:
        https://en.wikipedia.org/wiki/List_of_highest-grossing_films

        Answer the following questions and respond with a JSON array of strings containing the answer.

        1. How many $2 bn movies were released before 2020?
        """
        
        # Mock the scraping to avoid actual web requests in tests
        with patch.object(agent.web_scraper, 'scrape_wikipedia_table') as mock_scrape:
            mock_scrape.return_value = pd.DataFrame({
                'Film': ['Avatar', 'Titanic'],
                'Worldwide gross': ['$2.847 billion', '$2.201 billion'],
                'Year': [2009, 1997]
            })
            
            result = await agent.analyze(question)
            
            assert isinstance(result, list)
            assert len(result) >= 1
            assert isinstance(result[0], int)

if __name__ == "__main__":
    pytest.main([__file__])
