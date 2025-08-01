"""
Data visualization utilities for the Data Analyst Agent
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.io import to_image
import pandas as pd
import numpy as np
import base64
import io
import logging
from typing import Tuple, Optional, Union
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)

class Visualizer:
    """Handles data visualization and plot generation"""
    
    def __init__(self, settings):
        self.settings = settings
        self.figure_size = settings.default_figure_size
        self.max_plot_size = settings.max_plot_size
        
        # Set matplotlib style
        plt.style.use('default')
        sns.set_palette("husl")
    
    async def create_scatterplot_with_regression(self, df: pd.DataFrame, x_col: str, 
                                               y_col: str, title: str) -> str:
        """Create a scatterplot with regression line and return as base64"""
        try:
            # Clean the data
            df_clean = df.copy()
            df_clean[x_col] = pd.to_numeric(df_clean[x_col], errors='coerce')
            df_clean[y_col] = pd.to_numeric(df_clean[y_col], errors='coerce')
            df_clean = df_clean.dropna(subset=[x_col, y_col])
            
            if len(df_clean) < 2:
                raise ValueError("Insufficient data points for scatter plot")
            
            # Create the plot
            fig, ax = plt.subplots(figsize=self.figure_size)
            
            # Scatter plot
            ax.scatter(df_clean[x_col], df_clean[y_col], alpha=0.6, s=50)
            
            # Regression line
            X = df_clean[x_col].values.reshape(-1, 1)
            y = df_clean[y_col].values
            
            reg_model = LinearRegression().fit(X, y)
            x_range = np.linspace(df_clean[x_col].min(), df_clean[x_col].max(), 100)
            y_pred = reg_model.predict(x_range.reshape(-1, 1))
            
            # Plot regression line as dotted red line
            ax.plot(x_range, y_pred, color='red', linestyle='--', linewidth=2, label='Regression Line')
            
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Convert to base64
            return self._plot_to_base64(fig)
            
        except Exception as e:
            logger.error(f"Error creating scatterplot: {e}")
            return self._create_error_plot("Error creating scatterplot")
    
    async def create_time_series_plot(self, df: pd.DataFrame, x_col: str, 
                                    y_col: str, title: str) -> str:
        """Create a time series plot"""
        try:
            df_clean = df.copy()
            df_clean[x_col] = pd.to_datetime(df_clean[x_col], errors='coerce')
            df_clean[y_col] = pd.to_numeric(df_clean[y_col], errors='coerce')
            df_clean = df_clean.dropna(subset=[x_col, y_col])
            
            fig, ax = plt.subplots(figsize=self.figure_size)
            
            ax.plot(df_clean[x_col], df_clean[y_col], marker='o', linewidth=2)
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            return self._plot_to_base64(fig)
            
        except Exception as e:
            logger.error(f"Error creating time series plot: {e}")
            return self._create_error_plot("Error creating time series plot")
    
    async def create_bar_chart(self, df: pd.DataFrame, x_col: str, 
                             y_col: str, title: str, top_n: int = 10) -> str:
        """Create a bar chart"""
        try:
            df_clean = df.copy()
            df_clean[y_col] = pd.to_numeric(df_clean[y_col], errors='coerce')
            df_clean = df_clean.dropna(subset=[x_col, y_col])
            
            # Take top N items
            df_top = df_clean.nlargest(top_n, y_col)
            
            fig, ax = plt.subplots(figsize=self.figure_size)
            
            bars = ax.bar(range(len(df_top)), df_top[y_col])
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(title)
            ax.set_xticks(range(len(df_top)))
            ax.set_xticklabels(df_top[x_col], rotation=45, ha='right')
            
            # Add value labels on bars
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.0f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            return self._plot_to_base64(fig)
            
        except Exception as e:
            logger.error(f"Error creating bar chart: {e}")
            return self._create_error_plot("Error creating bar chart")
    
    async def create_histogram(self, df: pd.DataFrame, col: str, 
                             title: str, bins: int = 30) -> str:
        """Create a histogram"""
        try:
            df_clean = df.copy()
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            data = df_clean[col].dropna()
            
            fig, ax = plt.subplots(figsize=self.figure_size)
            
            ax.hist(data, bins=bins, alpha=0.7, edgecolor='black')
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            mean_val = data.mean()
            median_val = data.median()
            ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
            ax.axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.2f}')
            ax.legend()
            
            return self._plot_to_base64(fig)
            
        except Exception as e:
            logger.error(f"Error creating histogram: {e}")
            return self._create_error_plot("Error creating histogram")
    
    async def create_correlation_heatmap(self, df: pd.DataFrame, title: str) -> str:
        """Create a correlation heatmap"""
        try:
            # Select only numeric columns
            numeric_df = df.select_dtypes(include=[np.number])
            
            if numeric_df.empty:
                raise ValueError("No numeric columns found for correlation")
            
            correlation_matrix = numeric_df.corr()
            
            fig, ax = plt.subplots(figsize=self.figure_size)
            
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, linewidths=0.5, ax=ax)
            ax.set_title(title)
            
            plt.tight_layout()
            
            return self._plot_to_base64(fig)
            
        except Exception as e:
            logger.error(f"Error creating correlation heatmap: {e}")
            return self._create_error_plot("Error creating correlation heatmap")
    
    async def create_plotly_interactive_plot(self, df: pd.DataFrame, plot_type: str,
                                           x_col: str, y_col: str, title: str) -> str:
        """Create interactive Plotly plots"""
        try:
            if plot_type == "scatter":
                fig = px.scatter(df, x=x_col, y=y_col, title=title)
            elif plot_type == "line":
                fig = px.line(df, x=x_col, y=y_col, title=title)
            elif plot_type == "bar":
                fig = px.bar(df, x=x_col, y=y_col, title=title)
            else:
                raise ValueError(f"Unsupported plot type: {plot_type}")
            
            # Convert to PNG and then to base64
            img_bytes = to_image(fig, format="png", width=800, height=600)
            img_base64 = base64.b64encode(img_bytes).decode()
            
            return f"data:image/png;base64,{img_base64}"
            
        except Exception as e:
            logger.error(f"Error creating Plotly plot: {e}")
            return self._create_error_plot("Error creating interactive plot")
    
    def _plot_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string"""
        try:
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            
            # Check size
            img_size = len(buffer.getvalue())
            if img_size > self.max_plot_size:
                logger.warning(f"Plot size {img_size} exceeds limit {self.max_plot_size}")
                # Try with lower DPI
                buffer = io.BytesIO()
                fig.savefig(buffer, format='png', dpi=50, bbox_inches='tight')
                buffer.seek(0)
            
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close(fig)  # Clean up
            
            return f"data:image/png;base64,{img_base64}"
            
        except Exception as e:
            logger.error(f"Error converting plot to base64: {e}")
            plt.close(fig)  # Clean up even on error
            return self._create_error_plot("Error converting plot")
    
    def _create_error_plot(self, error_message: str) -> str:
        """Create a simple error plot"""
        try:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, error_message, ha='center', va='center',
                   transform=ax.transAxes, fontsize=12, color='red')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            
            return self._plot_to_base64(fig)
            
        except Exception as e:
            logger.error(f"Error creating error plot: {e}")
            # Return minimal base64 image
            return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    
    async def create_regression_analysis_plot(self, df: pd.DataFrame, x_col: str, 
                                            y_col: str, title: str) -> Tuple[str, dict]:
        """Create regression analysis with statistics"""
        try:
            df_clean = df.copy()
            df_clean[x_col] = pd.to_numeric(df_clean[x_col], errors='coerce')
            df_clean[y_col] = pd.to_numeric(df_clean[y_col], errors='coerce')
            df_clean = df_clean.dropna(subset=[x_col, y_col])
            
            X = df_clean[x_col].values.reshape(-1, 1)
            y = df_clean[y_col].values
            
            reg_model = LinearRegression().fit(X, y)
            y_pred = reg_model.predict(X)
            
            # Calculate statistics
            r2 = reg_model.score(X, y)
            slope = reg_model.coef_[0]
            intercept = reg_model.intercept_
            
            # Create plot
            fig, ax = plt.subplots(figsize=self.figure_size)
            
            ax.scatter(df_clean[x_col], df_clean[y_col], alpha=0.6)
            ax.plot(df_clean[x_col], y_pred, color='red', linestyle='--', linewidth=2)
            
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f"{title}\nR² = {r2:.4f}, Slope = {slope:.4f}")
            ax.grid(True, alpha=0.3)
            
            plot_b64 = self._plot_to_base64(fig)
            
            stats = {
                'r_squared': r2,
                'slope': slope,
                'intercept': intercept,
                'n_points': len(df_clean)
            }
            
            return plot_b64, stats
            
        except Exception as e:
            logger.error(f"Error in regression analysis: {e}")
            return self._create_error_plot("Error in regression analysis"), {}
