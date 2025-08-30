"""
Test script for the enhanced LLM-driven Data Analyst Agent
"""

import asyncio
import json
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm_agent import LLMDataAnalystAgent
from src.config import get_settings

async def test_structure_extraction():
    """Test the JSON structure extraction capability"""
    
    settings = get_settings()
    agent = LLMDataAnalystAgent(settings)
    
    # Test with sales question
    sales_question = """
    Please analyze the attached sales CSV and return exactly these fields:
    - total_sales
    - top_region
    - day_sales_correlation
    - bar_chart (as data:image/png;base64,...)
    - median_sales
    - total_sales_tax (assume 10%)
    - cumulative_sales_chart (as data:image/png;base64,...)
    """
    
    structure = agent._extract_json_structure(sales_question)
    print("Extracted structure for sales question:")
    print(json.dumps(structure, indent=2))
    print()
    
    # Test with network question
    network_question = """
    Analyze the network data and return:
    - edge_count
    - highest_degree_node
    - average_degree
    - network_graph
    - density
    """
    
    structure = agent._extract_json_structure(network_question)
    print("Extracted structure for network question:")
    print(json.dumps(structure, indent=2))
    print()

async def test_chart_generation():
    """Test chart generation capabilities"""
    
    settings = get_settings()
    agent = LLMDataAnalystAgent(settings)
    
    # Test bar chart
    test_data = {"North": 1000, "South": 1500, "East": 800, "West": 1200}
    chart_b64 = agent.chart_generator.create_bar_chart(test_data, "Test Sales by Region")
    
    print(f"Generated bar chart (length: {len(chart_b64)} chars)")
    print(f"Starts with: {chart_b64[:50]}...")
    print()

async def test_fallback_response():
    """Test fallback response generation"""
    
    settings = get_settings()
    agent = LLMDataAnalystAgent(settings)
    
    question = "Return these fields: - total_sales - top_region - chart_data"
    fallback = agent._create_fallback_response(question)
    
    print("Fallback response:")
    print(json.dumps(fallback, indent=2))
    print()

async def main():
    """Run all tests"""
    print("Testing Enhanced LLM-driven Data Analyst Agent\n")
    print("=" * 50)
    
    try:
        await test_structure_extraction()
        await test_chart_generation()
        await test_fallback_response()
        
        print("All tests completed successfully!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
