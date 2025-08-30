"""
Chart Generation Helper for LLM Agent
Provides utilities for creating charts that the LLM can use
"""

import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List

class ChartGenerator:
    """Helper class to generate charts for LLM responses"""
    
    def __init__(self, max_size_kb: int = 100):
        self.max_size_kb = max_size_kb
        plt.style.use('default')
    
    def create_bar_chart(self, data: Dict[str, float], title: str = "Bar Chart", 
                        color: str = "blue", figsize: tuple = (6, 4)) -> str:
        """Create a bar chart and return base64 encoded PNG"""
        try:
            plt.figure(figsize=figsize)
            keys = list(data.keys())
            values = list(data.values())
            
            plt.bar(keys, values, color=color)
            plt.title(title)
            plt.xlabel('Category')
            plt.ylabel('Value')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            return self._save_to_base64()
        except Exception as e:
            return self._create_error_image(f"Bar chart error: {str(e)}")
    
    def create_line_chart(self, x_data: List, y_data: List, title: str = "Line Chart", 
                         color: str = "red", figsize: tuple = (6, 4)) -> str:
        """Create a line chart and return base64 encoded PNG"""
        try:
            plt.figure(figsize=figsize)
            plt.plot(x_data, y_data, color=color, linewidth=2, marker='o')
            plt.title(title)
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            return self._save_to_base64()
        except Exception as e:
            return self._create_error_image(f"Line chart error: {str(e)}")
    
    def create_pie_chart(self, data: Dict[str, float], title: str = "Pie Chart", 
                        figsize: tuple = (6, 6)) -> str:
        """Create a pie chart and return base64 encoded PNG"""
        try:
            plt.figure(figsize=figsize)
            labels = list(data.keys())
            sizes = list(data.values())
            
            plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            plt.title(title)
            plt.axis('equal')
            plt.tight_layout()
            
            return self._save_to_base64()
        except Exception as e:
            return self._create_error_image(f"Pie chart error: {str(e)}")
    
    def create_scatter_plot(self, x_data: List, y_data: List, title: str = "Scatter Plot", 
                           color: str = "green", figsize: tuple = (6, 4)) -> str:
        """Create a scatter plot and return base64 encoded PNG"""
        try:
            plt.figure(figsize=figsize)
            plt.scatter(x_data, y_data, color=color, alpha=0.6)
            plt.title(title)
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            return self._save_to_base64()
        except Exception as e:
            return self._create_error_image(f"Scatter plot error: {str(e)}")
    
    def create_histogram(self, data: List, title: str = "Histogram", 
                        color: str = "purple", bins: int = 20, figsize: tuple = (6, 4)) -> str:
        """Create a histogram and return base64 encoded PNG"""
        try:
            plt.figure(figsize=figsize)
            plt.hist(data, bins=bins, color=color, alpha=0.7, edgecolor='black')
            plt.title(title)
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            return self._save_to_base64()
        except Exception as e:
            return self._create_error_image(f"Histogram error: {str(e)}")
    
    def _save_to_base64(self) -> str:
        """Save current matplotlib figure to base64 string"""
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=80, 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        buf.seek(0)
        img_data = buf.getvalue()
        buf.close()
        
        # Check size and compress if necessary
        if len(img_data) > self.max_size_kb * 1024:
            # Try lower DPI
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=60, 
                       facecolor='white', edgecolor='none')
            plt.close()
            buf.seek(0)
            img_data = buf.getvalue()
            buf.close()
        
        b64_string = base64.b64encode(img_data).decode('utf-8')
        return f"data:image/png;base64,{b64_string}"
    
    def _create_error_image(self, error_msg: str) -> str:
        """Create a simple error message image"""
        try:
            plt.figure(figsize=(4, 2))
            plt.text(0.5, 0.5, f"Chart Error:\n{error_msg}", ha='center', va='center', 
                    fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
            plt.axis('off')
            plt.tight_layout()
            
            return self._save_to_base64()
        except:
            # Return minimal base64 placeholder
            return "data:image/png;base64,"
