import os
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # OpenAI API configuration
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4", alias="OPENAI_MODEL")
    
    # Application settings
    debug: bool = Field(default=False, alias="DEBUG")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    
    # API configuration
    max_request_size: int = Field(default=10 * 1024 * 1024, alias="MAX_REQUEST_SIZE")  # 10MB
    timeout_seconds: int = Field(default=180, alias="TIMEOUT_SECONDS")  # 3 minutes
    
    # Data processing settings
    max_plot_size: int = Field(default=100000, alias="MAX_PLOT_SIZE")  # 100KB for base64 images
    default_figure_size: tuple = (10, 6)
    
    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore"
    }

def get_settings() -> Settings:
    """Get application settings instance"""
    return Settings()
