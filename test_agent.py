"""
Test script for the Data Analyst Agent
"""
import asyncio
import json
from src.agent import DataAnalystAgent
from src.config import get_settings

async def test_network_analysis():
    """Test network analysis functionality"""
    settings = get_settings()
    agent = DataAnalystAgent(settings)
    
    question = """Use the undirected network in `edges.csv`.

Return a JSON object with keys:
- `edge_count`: number
- `highest_degree_node`: string
- `average_degree`: number
- `density`: number
- `shortest_path_alice_eve`: number
- `network_graph`: base64 PNG string under 100kB
- `degree_histogram`: base64 PNG string under 100kB

Answer:
1. How many edges are in the network?
2. Which node has the highest degree?
3. What is the average degree of the network?
4. What is the network density?
5. What is the length of the shortest path between Alice and Eve?
6. Draw the network with nodes labelled and edges shown. Encode as base64 PNG.
7. Plot the degree distribution as a bar chart with green bars. Encode as base64 PNG."""

    try:
        result = await agent.analyze(question)
        print("Network Analysis Result:")
        print(json.dumps(result, indent=2))
        return result
    except Exception as e:
        print(f"Error: {e}")
        return None

async def test_sales_analysis():
    """Test sales analysis functionality"""
    settings = get_settings()
    agent = DataAnalystAgent(settings)
    
    question = """Analyze `sample-sales.csv`.

Return a JSON object with keys:
- `total_sales`: number
- `top_region`: string
- `day_sales_correlation`: number
- `bar_chart`: base64 PNG string under 100kB
- `median_sales`: number
- `total_sales_tax`: number
- `cumulative_sales_chart`: base64 PNG string under 100kB

Answer:
1. What is the total sales across all regions?
2. Which region has the highest total sales?
3. What is the correlation between day of month and sales? (Use the date column.)
4. Plot total sales by region as a bar chart with blue bars. Encode as base64 PNG.
5. What is the median sales amount across all orders?
6. What is the total sales tax if the tax rate is 10%?
7. Plot cumulative sales over time as a line chart with a red line. Encode as base64 PNG."""

    try:
        result = await agent.analyze(question)
        print("Sales Analysis Result:")
        print(json.dumps(result, indent=2))
        return result
    except Exception as e:
        print(f"Error: {e}")
        return None

async def main():
    print("Testing Data Analyst Agent...")
    print("=" * 50)
    
    # Test network analysis
    print("\n1. Testing Network Analysis:")
    network_result = await test_network_analysis()
    
    print("\n" + "=" * 50)
    
    # Test sales analysis
    print("\n2. Testing Sales Analysis:")
    sales_result = await test_sales_analysis()
    
    print("\n" + "=" * 50)
    print("Testing completed!")

if __name__ == "__main__":
    asyncio.run(main())
