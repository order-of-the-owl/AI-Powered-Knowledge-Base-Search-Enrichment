"""
Configuration management for the AI Knowledge Base application.
Handles environment variables and application settings.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    
    # Google AI Configuration
    google_api_key: str = Field(..., env="GOOGLE_API_KEY")
    
    # Database Configuration
    chroma_db_path: str = Field(default="./chroma_db", env="CHROMA_DB_PATH")
    
    # Upload Configuration
    upload_dir: str = Field(default="./uploads", env="UPLOAD_DIR")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # External APIs (for auto-enrichment)
    wikipedia_api_url: str = Field(
        default="https://en.wikipedia.org/api/rest_v1", 
        env="WIKIPEDIA_API_URL"
    )
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    
    # CORS Configuration
    cors_origins: list[str] = Field(default=["*"], env="CORS_ORIGINS")
    
    # File Upload Limits
    max_file_size_mb: int = Field(default=50, env="MAX_FILE_SIZE_MB")
    allowed_file_types: list[str] = Field(
        default=[".pdf", ".docx", ".doc", ".txt"], 
        env="ALLOWED_FILE_TYPES"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings."""
    return settings
