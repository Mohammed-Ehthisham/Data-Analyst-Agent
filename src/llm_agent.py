"""
Enhanced LLM-driven Data Analyst Agent
Main focus: System prompt design and LLM orchestration for structured JSON output
"""

import asyncio
import json
import re
import base64
import io
import logging
from typing import Any, Dict, List, Union, Optional
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import openai

from .config import Settings
from .data_processor import DataProcessor
from .visualizer import Visualizer
from .chart_generator import ChartGenerator

logger = logging.getLogger(__name__)

class LLMDataAnalystAgent:
    """
    Enhanced Data Analyst Agent that primarily uses LLM with system prompts
    for structured JSON analysis and minimal hardcoded logic
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.openai_client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
        self.data_processor = DataProcessor()
        self.visualizer = Visualizer(settings)
        self.chart_generator = ChartGenerator()
        
    async def analyze(self, question: str, files: Optional[List[Dict[str, Any]]] = None) -> Union[List[Any], Dict[str, Any]]:
        """
        Main analysis method - uses LLM for most analysis with structured prompts
        """
        try:
            logger.info(f"Starting LLM-driven analysis for: {question[:100]}...")
            
            # Extract required JSON structure from question
            required_structure = self._extract_json_structure(question)
            logger.info(f"Detected required structure: {list(required_structure.keys())}")
            
            # Prepare data context from files
            data_context = await self._prepare_data_context(files)
            
            # For questions requiring charts, we need to do hybrid approach:
            # LLM for analysis + our chart generation
            if self._requires_charts(required_structure):
                return await self._analyze_with_charts(question, required_structure, data_context)
            else:
                # Pure LLM analysis for non-chart responses
                return await self._pure_llm_analyze(question, required_structure, data_context)
                
        except Exception as e:
            logger.error(f"LLM Analysis failed: {e}", exc_info=True)
            # Return fallback structure based on question
            return self._create_fallback_response(question)
    
    def _requires_charts(self, structure: Dict[str, Any]) -> bool:
        """Check if the required structure includes chart fields"""
        return any(isinstance(v, str) and v.startswith("data:image") for v in structure.values())
    
    async def _analyze_with_charts(self, question: str, required_structure: Dict[str, Any], 
                                  data_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle analysis that requires chart generation
        """
        # First, get the LLM to do the data analysis
        analysis_prompt = self._create_analysis_only_prompt(required_structure, data_context)
        
        analysis_result = await self._llm_analyze_data_only(question, analysis_prompt, data_context)
        
        # Then generate charts based on the analysis
        final_result = await self._add_charts_to_result(analysis_result, required_structure, data_context)
        
        return self._validate_and_fix_structure(final_result, required_structure)
    
    async def _pure_llm_analyze(self, question: str, required_structure: Dict[str, Any], 
                               data_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Pure LLM analysis for responses that don't require charts
        """
        system_prompt = self._create_system_prompt(required_structure, data_context)
        result = await self._llm_analyze(question, system_prompt, data_context)
        return self._validate_and_fix_structure(result, required_structure)
    
    def _create_analysis_only_prompt(self, required_structure: Dict[str, Any], 
                                   data_context: Dict[str, Any]) -> str:
        """
        Create a prompt focused on data analysis without chart generation
        """
        # Remove chart fields from structure for analysis
        analysis_structure = {k: v for k, v in required_structure.items() 
                            if not (isinstance(v, str) and v.startswith("data:image"))}
        
        data_description = self._describe_data_context(data_context)
        
        return f"""You are an expert data analyst. Analyze the provided data and return calculated values.

REQUIRED ANALYSIS FIELDS:
{json.dumps(analysis_structure, indent=2)}

CALCULATION GUIDELINES:
1. For totals/sums: Calculate exact values from the data
2. For correlations: Use Pearson correlation coefficient
3. For "top" fields: Find the actual highest value/category
4. For median: Calculate the exact median value
5. For tax calculations: Apply specified percentage (e.g., 10% = multiply by 0.10)
6. Return actual calculated numbers, not placeholders

AVAILABLE DATA:
{data_description}

RESPONSE FORMAT:
Return only the calculated values as a JSON object. Do not include chart fields.
Ensure all numeric calculations are accurate and based on the actual data provided.

Example response format:
{{
    "total_sales": 15420,
    "top_region": "North",
    "day_sales_correlation": 0.23,
    "median_sales": 340,
    "total_sales_tax": 1542.0
}}"""
    
    async def _llm_analyze_data_only(self, question: str, system_prompt: str, 
                                   data_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get LLM to analyze data without generating charts
        """
        try:
            user_message = f"""
ANALYSIS REQUEST: {question}

DATA TO ANALYZE:
{json.dumps(data_context, indent=2, default=str)}

Please perform the calculations and return the exact JSON structure with calculated values.
"""

            # Try with configured model first
            try:
                response = await self.openai_client.chat.completions.create(
                    model=self.settings.openai_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    max_tokens=2000,
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )
            except Exception as model_error:
                logger.warning(f"Model {self.settings.openai_model} failed, trying fallback: {model_error}")
                # Fallback to gpt-4 if configured model fails
                response = await self.openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    max_tokens=2000,
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )
            
            response_text = response.choices[0].message.content
            result = json.loads(response_text)
            
            logger.info(f"LLM data analysis completed: {list(result.keys())}")
            return result
            
        except Exception as e:
            logger.error(f"LLM data analysis failed: {e}")
            return {}
    
    async def _add_charts_to_result(self, analysis_result: Dict[str, Any], 
                                  required_structure: Dict[str, Any], 
                                  data_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate charts and add them to the analysis result
        """
        final_result = analysis_result.copy()
        
        # Find chart fields in required structure
        chart_fields = {k: v for k, v in required_structure.items() 
                       if isinstance(v, str) and v.startswith("data:image")}
        
        for chart_field in chart_fields:
            try:
                chart_data = self._generate_chart_for_field(chart_field, analysis_result, data_context)
                final_result[chart_field] = chart_data
            except Exception as e:
                logger.error(f"Failed to generate chart for {chart_field}: {e}")
                final_result[chart_field] = "data:image/png;base64,"
        
        return final_result
    
    def _generate_chart_for_field(self, field_name: str, analysis_result: Dict[str, Any], 
                                 data_context: Dict[str, Any]) -> str:
        """
        Generate a specific chart based on field name and available data
        """
        field_lower = field_name.lower()
        
        # Load data from context if available
        df = self._load_dataframe_from_context(data_context)
        
        if df is None or df.empty:
            return self.chart_generator._create_error_image("No data available for chart")
        
        # Generate different charts based on field name
        if 'bar_chart' in field_lower:
            return self._create_bar_chart(df, analysis_result)
        elif 'cumulative' in field_lower:
            return self._create_cumulative_chart(df)
        elif 'histogram' in field_lower:
            return self._create_histogram_chart(df)
        elif 'line' in field_lower or 'trend' in field_lower:
            return self._create_line_chart(df)
        else:
            # Default to bar chart
            return self._create_bar_chart(df, analysis_result)
    
    def _load_dataframe_from_context(self, data_context: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """
        Extract DataFrame from data context
        """
        try:
            # Look for CSV data first
            for filename, preview in data_context.get("data_preview", {}).items():
                if preview.get("type") == "csv" and "head" in preview:
                    # Reconstruct basic DataFrame from preview
                    head_data = preview["head"]
                    if head_data:
                        return pd.DataFrame(head_data)
            
            # If no CSV, try JSON
            for filename, preview in data_context.get("data_preview", {}).items():
                if preview.get("type") == "json" and "sample_records" in preview:
                    records = preview["sample_records"]
                    if records:
                        return pd.DataFrame(records)
            
            return None
        except Exception as e:
            logger.error(f"Failed to load DataFrame from context: {e}")
            return None
    
    def _create_bar_chart(self, df: pd.DataFrame, analysis_result: Dict[str, Any]) -> str:
        """Create a bar chart from the data"""
        try:
            # Try to find suitable columns for bar chart
            if 'region' in df.columns and any('sales' in col.lower() for col in df.columns):
                sales_col = next(col for col in df.columns if 'sales' in col.lower())
                region_sales = df.groupby('region')[sales_col].sum()
                return self.chart_generator.create_bar_chart(
                    region_sales.to_dict(), 
                    "Sales by Region", 
                    color="blue"
                )
            elif len(df.columns) >= 2:
                # Use first two columns
                col1, col2 = df.columns[0], df.columns[1]
                if df[col1].dtype == 'object' and pd.api.types.is_numeric_dtype(df[col2]):
                    grouped = df.groupby(col1)[col2].sum().head(10)
                    return self.chart_generator.create_bar_chart(
                        grouped.to_dict(),
                        f"{col2} by {col1}",
                        color="blue"
                    )
            
            return self.chart_generator._create_error_image("Cannot create bar chart from available data")
        except Exception as e:
            return self.chart_generator._create_error_image(f"Bar chart error: {str(e)}")
    
    def _create_cumulative_chart(self, df: pd.DataFrame) -> str:
        """Create a cumulative line chart"""
        try:
            # Find date and sales columns
            date_col = next((col for col in df.columns if 'date' in col.lower()), df.columns[0])
            sales_col = next((col for col in df.columns if 'sales' in col.lower() or 'amount' in col.lower()), df.columns[-1])
            
            # Sort by date and calculate cumulative sum
            df_sorted = df.sort_values(date_col)
            df_sorted['cumulative'] = df_sorted[sales_col].cumsum()
            
            return self.chart_generator.create_line_chart(
                list(range(len(df_sorted))),  # Use index as x-axis for simplicity
                df_sorted['cumulative'].tolist(),
                "Cumulative Sales Over Time",
                color="red"
            )
        except Exception as e:
            return self.chart_generator._create_error_image(f"Cumulative chart error: {str(e)}")
    
    def _create_histogram_chart(self, df: pd.DataFrame) -> str:
        """Create a histogram"""
        try:
            # Find numeric column
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                col = numeric_cols[0]
                return self.chart_generator.create_histogram(
                    df[col].dropna().tolist(),
                    f"Distribution of {col}",
                    color="green"
                )
            
            return self.chart_generator._create_error_image("No numeric data for histogram")
        except Exception as e:
            return self.chart_generator._create_error_image(f"Histogram error: {str(e)}")
    
    def _create_line_chart(self, df: pd.DataFrame) -> str:
        """Create a line chart"""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                x_col, y_col = numeric_cols[0], numeric_cols[1]
                return self.chart_generator.create_line_chart(
                    df[x_col].tolist(),
                    df[y_col].tolist(),
                    f"{y_col} vs {x_col}",
                    color="green"
                )
            elif len(numeric_cols) == 1:
                col = numeric_cols[0]
                return self.chart_generator.create_line_chart(
                    list(range(len(df))),
                    df[col].tolist(),
                    f"{col} Over Time",
                    color="blue"
                )
            
            return self.chart_generator._create_error_image("No suitable data for line chart")
        except Exception as e:
            return self.chart_generator._create_error_image(f"Line chart error: {str(e)}")
    
    def _extract_json_structure(self, question: str) -> Dict[str, Any]:
        """
        Extract the expected JSON structure from the question text
        """
        structure = {}
        
        # Look for bullet points with field names
        bullet_pattern = r'[-*]\s*([a-zA-Z_][a-zA-Z0-9_]*)'
        matches = re.findall(bullet_pattern, question)
        
        for field in matches:
            # Determine field type based on name patterns
            field_lower = field.lower()
            
            if any(keyword in field_lower for keyword in ['chart', 'plot', 'graph', 'image']):
                structure[field] = "data:image/png;base64,"
            elif any(keyword in field_lower for keyword in ['correlation', 'ratio', 'rate', 'percentage']):
                structure[field] = 0.0
            elif any(keyword in field_lower for keyword in ['total', 'count', 'sum', 'number', 'median']):
                structure[field] = 0
            elif any(keyword in field_lower for keyword in ['top', 'best', 'name', 'title', 'region']):
                structure[field] = ""
            elif 'tax' in field_lower:
                structure[field] = 0.0
            else:
                structure[field] = ""
        
        # If no structure found, try to infer from question content
        if not structure:
            if 'sales' in question.lower():
                structure = {
                    "total_sales": 0,
                    "top_region": "",
                    "analysis_result": ""
                }
            elif 'network' in question.lower() or 'graph' in question.lower():
                structure = {
                    "edge_count": 0,
                    "node_count": 0,
                    "analysis_result": ""
                }
            else:
                structure = {"analysis_result": "", "data_summary": ""}
        
        return structure
    
    async def _prepare_data_context(self, files: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Prepare data context from uploaded files
        """
        context = {
            "files_provided": [],
            "data_preview": {},
            "data_summary": {}
        }
        
        if not files:
            return context
        
        for file_info in files:
            filename = file_info.get('filename', '')
            content = file_info.get('content', b'')
            
            context["files_provided"].append(filename)
            
            try:
                # Process different file types
                if filename.lower().endswith('.csv'):
                    df = pd.read_csv(io.BytesIO(content))
                    context["data_preview"][filename] = {
                        "type": "csv",
                        "shape": df.shape,
                        "columns": list(df.columns),
                        "head": df.head(3).to_dict('records'),
                        "dtypes": df.dtypes.to_dict()
                    }
                    context["data_summary"][filename] = {
                        "numeric_summary": df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {},
                        "categorical_summary": {col: df[col].value_counts().head(5).to_dict() 
                                             for col in df.select_dtypes(include=['object']).columns}
                    }
                
                elif filename.lower().endswith('.json'):
                    data = json.loads(content.decode('utf-8', errors='ignore'))
                    if isinstance(data, list) and len(data) > 0:
                        df = pd.json_normalize(data)
                        context["data_preview"][filename] = {
                            "type": "json",
                            "shape": df.shape,
                            "columns": list(df.columns),
                            "sample_records": data[:3]
                        }
                    else:
                        context["data_preview"][filename] = {
                            "type": "json",
                            "content": data
                        }
            
            except Exception as e:
                logger.warning(f"Could not process file {filename}: {e}")
                context["data_preview"][filename] = {"type": "unknown", "error": str(e)}
        
        return context
    
    def _create_system_prompt(self, required_structure: Dict[str, Any], data_context: Dict[str, Any]) -> str:
        """
        Create a comprehensive system prompt for the LLM
        """
        
        structure_description = self._describe_structure(required_structure)
        data_description = self._describe_data_context(data_context)
        
        system_prompt = f"""You are an expert data analyst AI. Your job is to analyze data and return EXACTLY the JSON structure requested.

CRITICAL REQUIREMENTS:
1. You MUST return a valid JSON object with exactly these fields: {list(required_structure.keys())}
2. Each field must have the correct data type as specified
3. For image fields (containing 'chart', 'plot', 'graph'), return "data:image/png;base64," followed by actual base64 image data
4. For numeric fields, return actual calculated numbers (integers or floats as appropriate)
5. For string fields, return meaningful text values
6. Do NOT return placeholder values like 0, "", or null unless the data actually results in those values

EXPECTED OUTPUT STRUCTURE:
{json.dumps(required_structure, indent=2)}

{structure_description}

AVAILABLE DATA:
{data_description}

ANALYSIS GUIDELINES:
1. If data is provided, perform actual calculations and analysis
2. For correlations, use statistical methods (Pearson correlation)
3. For aggregations (total, sum, median), calculate from actual data
4. For "top" fields, find the actual highest/best value
5. For tax calculations, apply the specified percentage (e.g., 10% = multiply by 0.10)
6. For charts/plots:
   - Create meaningful visualizations using matplotlib
   - Convert to base64 PNG format
   - Keep file size reasonable (< 100KB)
   - Use appropriate colors and labels

ERROR HANDLING:
- If data is missing or invalid, return appropriate default values with correct types
- If calculations fail, return 0 for numbers, "" for strings, but log the issue
- Always return the complete JSON structure even if some fields fail

Remember: Your output will be directly parsed as JSON, so ensure it's valid JSON format."""

        return system_prompt
    
    def _describe_structure(self, structure: Dict[str, Any]) -> str:
        """Describe the expected structure in detail"""
        descriptions = []
        
        for field, default_value in structure.items():
            field_desc = f"- {field}: "
            
            if isinstance(default_value, str) and default_value.startswith("data:image"):
                field_desc += "Base64-encoded image (PNG format)"
            elif isinstance(default_value, (int, float)):
                field_desc += f"Numeric value ({'integer' if isinstance(default_value, int) else 'float'})"
            elif isinstance(default_value, str):
                field_desc += "String value"
            else:
                field_desc += f"Value of type {type(default_value).__name__}"
            
            descriptions.append(field_desc)
        
        return "FIELD DESCRIPTIONS:\n" + "\n".join(descriptions)
    
    def _describe_data_context(self, data_context: Dict[str, Any]) -> str:
        """Describe the available data context"""
        if not data_context.get("files_provided"):
            return "No data files provided. Use general knowledge or return appropriate defaults."
        
        description = f"Files provided: {', '.join(data_context['files_provided'])}\n\n"
        
        for filename, preview in data_context.get("data_preview", {}).items():
            description += f"File: {filename}\n"
            description += f"Type: {preview.get('type', 'unknown')}\n"
            
            if 'shape' in preview:
                description += f"Shape: {preview['shape']} (rows x columns)\n"
                description += f"Columns: {preview.get('columns', [])}\n"
            
            if 'head' in preview:
                description += f"Sample data: {preview['head']}\n"
            
            description += "\n"
        
        return description
    
    async def _llm_analyze(self, question: str, system_prompt: str, data_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send the analysis request to the LLM with structured prompt
        """
        try:
            # Prepare user message with question and data context
            user_message = f"""
ANALYSIS REQUEST: {question}

Please analyze the provided data and return the exact JSON structure requested. 

DATA CONTEXT:
{json.dumps(data_context, indent=2, default=str)}

Remember to:
1. Perform actual calculations if data is available
2. Generate real charts/plots for image fields
3. Return the complete JSON structure
4. Ensure all values are meaningful and accurate
"""

            response = await self.openai_client.chat.completions.create(
                model=self.settings.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=4000,
                temperature=0.1,  # Low temperature for consistent, accurate results
                response_format={"type": "json_object"}  # Ensure JSON response
            )
            
            response_text = response.choices[0].message.content
            logger.info(f"LLM response received: {len(response_text)} characters")
            
            # Parse JSON response
            try:
                result = json.loads(response_text)
                return result
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM JSON response: {e}")
                logger.error(f"Response text: {response_text[:500]}")
                # Try to extract JSON from response
                return self._extract_json_from_text(response_text)
                
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            raise
    
    def _extract_json_from_text(self, text: str) -> Dict[str, Any]:
        """Try to extract JSON from malformed text response"""
        try:
            # Look for JSON-like content between braces
            import re
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        # If extraction fails, return empty dict
        return {}
    
    def _validate_and_fix_structure(self, result: Dict[str, Any], required_structure: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the LLM result and fix any missing or incorrect fields
        """
        validated = {}
        
        for field, default_value in required_structure.items():
            if field in result:
                # Validate type and content
                value = result[field]
                
                if isinstance(default_value, str) and default_value.startswith("data:image"):
                    # Ensure it's a proper base64 image
                    if isinstance(value, str) and value.startswith("data:image"):
                        validated[field] = value
                    else:
                        validated[field] = "data:image/png;base64,"  # Empty image placeholder
                
                elif isinstance(default_value, (int, float)):
                    # Ensure numeric type
                    try:
                        if isinstance(default_value, int):
                            validated[field] = int(float(value))
                        else:
                            validated[field] = float(value)
                    except (ValueError, TypeError):
                        validated[field] = default_value
                
                elif isinstance(default_value, str):
                    validated[field] = str(value)
                
                else:
                    validated[field] = value
            else:
                # Field missing, use default
                validated[field] = default_value
        
        return validated
    
    def _create_fallback_response(self, question: str) -> Dict[str, Any]:
        """
        Create a fallback response when analysis fails
        """
        structure = self._extract_json_structure(question)
        
        # Return the structure with default values
        fallback = {}
        for field, default_value in structure.items():
            fallback[field] = default_value
        
        # Add error information if space allows
        if not any("chart" in k or "plot" in k for k in fallback.keys()):
            fallback["error"] = "Analysis failed - returning default structure"
        
        return fallback
