import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # OpenAI API configuration
    openai_api_key: str = ""
    openai_model: str = "gpt-4"
    
    # Application settings
    debug: bool = False
    log_level: str = "INFO"
    
    # API configuration
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    timeout_seconds: int = 180  # 3 minutes
    
    # Data processing settings
    max_plot_size: int = 100000  # 100KB for base64 images
    default_figure_size: tuple = (10, 6)
    
    class Config:
        env_file = ".env"
        case_sensitive = False

def get_settings() -> Settings:
    """Get application settings instance"""
    return Settings()
