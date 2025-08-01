"""
Data processing utilities for the Data Analyst Agent
"""

import pandas as pd
import numpy as np
import duckdb
import json
import logging
from typing import Dict, List, Any, Optional, Union
import re
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class DataProcessor:
    """Handles data processing, cleaning, and transformation"""
    
    def __init__(self):
        self.duckdb_conn = duckdb.connect()
        
    def clean_numeric_column(self, series: pd.Series, remove_chars: List[str] = None) -> pd.Series:
        """Clean a series to make it numeric"""
        if remove_chars is None:
            remove_chars = ['$', ',', '%', '₹', '€', '£']
        
        # Convert to string first
        cleaned = series.astype(str)
        
        # Remove specified characters
        for char in remove_chars:
            cleaned = cleaned.str.replace(char, '', regex=False)
        
        # Handle special cases like "1.5 billion" -> 1.5
        cleaned = cleaned.str.replace(r'\s*billion.*', '', regex=True, case=False)
        cleaned = cleaned.str.replace(r'\s*million.*', 'e6', regex=True, case=False)
        cleaned = cleaned.str.replace(r'\s*thousand.*', 'e3', regex=True, case=False)
        
        # Convert to numeric
        return pd.to_numeric(cleaned, errors='coerce')
    
    def extract_year_from_text(self, series: pd.Series) -> pd.Series:
        """Extract year from text/date columns"""
        # Try to extract 4-digit years
        years = series.astype(str).str.extract(r'(\d{4})', expand=False)
        return pd.to_numeric(years, errors='coerce')
    
    def parse_date_column(self, series: pd.Series) -> pd.Series:
        """Parse various date formats"""
        return pd.to_datetime(series, errors='coerce', infer_datetime_format=True)
    
    def calculate_business_days_between(self, start_dates: pd.Series, end_dates: pd.Series) -> pd.Series:
        """Calculate business days between two date series"""
        try:
            start_dt = pd.to_datetime(start_dates, errors='coerce')
            end_dt = pd.to_datetime(end_dates, errors='coerce')
            
            # Calculate business days
            business_days = []
            for start, end in zip(start_dt, end_dt):
                if pd.isna(start) or pd.isna(end):
                    business_days.append(np.nan)
                else:
                    # Use numpy's busday_count for business days
                    try:
                        days = np.busday_count(start.date(), end.date())
                        business_days.append(days)
                    except:
                        business_days.append(np.nan)
            
            return pd.Series(business_days)
        except Exception as e:
            logger.error(f"Error calculating business days: {e}")
            return pd.Series([np.nan] * len(start_dates))
    
    def standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names for easier processing"""
        df_clean = df.copy()
        
        # Convert to lowercase and replace spaces with underscores
        df_clean.columns = (
            df_clean.columns
            .str.lower()
            .str.replace(' ', '_')
            .str.replace('[^a-zA-Z0-9_]', '', regex=True)
        )
        
        return df_clean
    
    def detect_data_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """Detect appropriate data types for each column"""
        type_mapping = {}
        
        for col in df.columns:
            sample = df[col].dropna().astype(str)
            
            if len(sample) == 0:
                type_mapping[col] = 'object'
                continue
            
            # Check for numeric patterns
            if sample.str.match(r'^-?\d+\.?\d*$').all():
                type_mapping[col] = 'numeric'
            # Check for date patterns
            elif sample.str.match(r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}').any():
                type_mapping[col] = 'date'
            # Check for year patterns
            elif sample.str.match(r'^\d{4}$').all():
                type_mapping[col] = 'year'
            # Check for currency patterns
            elif sample.str.contains(r'[$₹€£]').any():
                type_mapping[col] = 'currency'
            else:
                type_mapping[col] = 'text'
        
        return type_mapping
    
    def auto_convert_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Automatically convert column types based on content"""
        df_converted = df.copy()
        type_mapping = self.detect_data_types(df)
        
        for col, dtype in type_mapping.items():
            try:
                if dtype == 'numeric':
                    df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce')
                elif dtype == 'date':
                    df_converted[col] = pd.to_datetime(df_converted[col], errors='coerce')
                elif dtype == 'year':
                    df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce')
                elif dtype == 'currency':
                    df_converted[col] = self.clean_numeric_column(df_converted[col])
            except Exception as e:
                logger.warning(f"Failed to convert column {col} to {dtype}: {e}")
        
        return df_converted
    
    def aggregate_by_group(self, df: pd.DataFrame, group_col: str, 
                          agg_col: str, agg_func: str = 'count') -> pd.DataFrame:
        """Aggregate data by group"""
        try:
            if agg_func == 'count':
                result = df.groupby(group_col)[agg_col].count().reset_index()
            elif agg_func == 'sum':
                result = df.groupby(group_col)[agg_col].sum().reset_index()
            elif agg_func == 'mean':
                result = df.groupby(group_col)[agg_col].mean().reset_index()
            elif agg_func == 'max':
                result = df.groupby(group_col)[agg_col].max().reset_index()
            elif agg_func == 'min':
                result = df.groupby(group_col)[agg_col].min().reset_index()
            else:
                raise ValueError(f"Unsupported aggregation function: {agg_func}")
            
            return result.sort_values(agg_col, ascending=False)
            
        except Exception as e:
            logger.error(f"Error in aggregation: {e}")
            return pd.DataFrame()
    
    def execute_duckdb_query(self, query: str, params: Dict = None) -> pd.DataFrame:
        """Execute a DuckDB query and return results as DataFrame"""
        try:
            if params:
                # Replace parameters in query
                for key, value in params.items():
                    query = query.replace(f":{key}", str(value))
            
            result = self.duckdb_conn.execute(query).fetchdf()
            return result
            
        except Exception as e:
            logger.error(f"Error executing DuckDB query: {e}")
            return pd.DataFrame()
    
    def create_time_series_features(self, df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """Create time-based features from a date column"""
        df_features = df.copy()
        
        try:
            date_series = pd.to_datetime(df_features[date_col], errors='coerce')
            
            df_features[f'{date_col}_year'] = date_series.dt.year
            df_features[f'{date_col}_month'] = date_series.dt.month
            df_features[f'{date_col}_quarter'] = date_series.dt.quarter
            df_features[f'{date_col}_weekday'] = date_series.dt.weekday
            df_features[f'{date_col}_is_weekend'] = date_series.dt.weekday >= 5
            
        except Exception as e:
            logger.error(f"Error creating time series features: {e}")
        
        return df_features
    
    def detect_outliers(self, series: pd.Series, method: str = 'iqr') -> pd.Series:
        """Detect outliers in a numeric series"""
        if method == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return (series < lower_bound) | (series > upper_bound)
        
        elif method == 'zscore':
            z_scores = np.abs((series - series.mean()) / series.std())
            return z_scores > 3
        
        else:
            raise ValueError(f"Unsupported outlier detection method: {method}")
    
    def fill_missing_values(self, df: pd.DataFrame, strategy: Dict[str, str] = None) -> pd.DataFrame:
        """Fill missing values using various strategies"""
        df_filled = df.copy()
        
        if strategy is None:
            strategy = {}
        
        for col in df_filled.columns:
            col_strategy = strategy.get(col, 'drop')
            
            if col_strategy == 'mean' and pd.api.types.is_numeric_dtype(df_filled[col]):
                df_filled[col].fillna(df_filled[col].mean(), inplace=True)
            elif col_strategy == 'median' and pd.api.types.is_numeric_dtype(df_filled[col]):
                df_filled[col].fillna(df_filled[col].median(), inplace=True)
            elif col_strategy == 'mode':
                mode_val = df_filled[col].mode()
                if len(mode_val) > 0:
                    df_filled[col].fillna(mode_val[0], inplace=True)
            elif col_strategy == 'forward':
                df_filled[col].fillna(method='ffill', inplace=True)
            elif col_strategy == 'backward':
                df_filled[col].fillna(method='bfill', inplace=True)
            # 'drop' strategy is handled at the DataFrame level later
        
        return df_filled
