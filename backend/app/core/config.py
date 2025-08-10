"""
Configuration management system for Breakpoint LLM stress-testing platform.
Handles all environment variables, API configurations, and application settings.
"""

import os
from typing import List, Optional, Union, Any
from pydantic import validator, Field
from pydantic_settings import BaseSettings
from enum import Enum


class Environment(str, Enum):
    """Environment types for deployment configuration."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Available logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Settings(BaseSettings):
    """Application settings with environment variable parsing and validation."""
    
    # Application Information
    APP_NAME: str = "Breakpoint LLM Stress Testing Platform"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = "Advanced LLM stress testing and vulnerability assessment platform"
    ENVIRONMENT: Environment = Environment.DEVELOPMENT
    DEBUG: bool = Field(default=False, description="Enable debug mode")
    
    # Server Configuration
    HOST: str = Field(default="0.0.0.0", description="Server host address")
    PORT: int = Field(default=8000, description="Server port")
    WORKERS: int = Field(default=1, description="Number of worker processes for production")
    RELOAD: bool = Field(default=True, description="Enable auto-reload for development")
    
    # Database Configuration
    DATABASE_URL: str = Field(
        default="sqlite:///./breakpoint.db",
        description="Database connection URL"
    )
    DATABASE_POOL_SIZE: int = Field(default=10, description="Database connection pool size")
    DATABASE_MAX_OVERFLOW: int = Field(default=20, description="Database connection pool overflow")
    DATABASE_ECHO: bool = Field(default=False, description="Enable SQL query logging")
    
    # Redis Configuration (Optional for caching and sessions)
    REDIS_URL: Optional[str] = Field(default=None, description="Redis connection URL")
    REDIS_PASSWORD: Optional[str] = Field(default=None, description="Redis password")
    REDIS_DB: int = Field(default=0, description="Redis database number")
    REDIS_TTL: int = Field(default=3600, description="Default Redis TTL in seconds")
    
    # LLM Provider API Configurations
    OPENAI_API_KEY: Optional[str] = Field(default=None, description="OpenAI API key")
    OPENAI_ORG_ID: Optional[str] = Field(default=None, description="OpenAI organization ID")
    OPENAI_BASE_URL: str = Field(default="https://api.openai.com/v1", description="OpenAI API base URL")
    OPENAI_MAX_RETRIES: int = Field(default=3, description="OpenAI API max retries")
    OPENAI_TIMEOUT: int = Field(default=60, description="OpenAI API timeout in seconds")
    
    ANTHROPIC_API_KEY: Optional[str] = Field(default=None, description="Anthropic API key")
    ANTHROPIC_BASE_URL: str = Field(default="https://api.anthropic.com", description="Anthropic API base URL")
    ANTHROPIC_MAX_RETRIES: int = Field(default=3, description="Anthropic API max retries")
    ANTHROPIC_TIMEOUT: int = Field(default=60, description="Anthropic API timeout in seconds")
    
    HUGGINGFACE_API_KEY: Optional[str] = Field(default=None, description="HuggingFace API token")
    HUGGINGFACE_BASE_URL: str = Field(default="https://api-inference.huggingface.co", description="HuggingFace API base URL")
    HUGGINGFACE_MAX_RETRIES: int = Field(default=3, description="HuggingFace API max retries")
    HUGGINGFACE_TIMEOUT: int = Field(default=120, description="HuggingFace API timeout in seconds")
    
    # CORS Configuration
    CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000"],
        description="Allowed CORS origins"
    )
    CORS_METHODS: List[str] = Field(
        default=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
        description="Allowed CORS methods"
    )
    CORS_HEADERS: List[str] = Field(
        default=["*"],
        description="Allowed CORS headers"
    )
    CORS_CREDENTIALS: bool = Field(default=True, description="Allow CORS credentials")
    
    # JWT Configuration
    JWT_SECRET_KEY: str = Field(
        default="your-super-secret-jwt-key-change-this-in-production",
        description="JWT secret key for token signing"
    )
    JWT_ALGORITHM: str = Field(default="HS256", description="JWT signing algorithm")
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, description="Access token expiration in minutes")
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=7, description="Refresh token expiration in days")
    
    # Admin Configuration
    ADMIN_USERNAME: str = Field(default="admin", description="Default admin username")
    ADMIN_PASSWORD: str = Field(default="changeme", description="Default admin password")
    API_KEY: str = Field(default="your-api-key-here", description="API key for authentication")
    
    # Convenience aliases for auth service
    @property
    def SECRET_KEY(self) -> str:
        return self.JWT_SECRET_KEY
    
    @property
    def ACCESS_TOKEN_EXPIRE_MINUTES(self) -> int:
        return self.JWT_ACCESS_TOKEN_EXPIRE_MINUTES
    
    # File Upload Configuration
    MAX_FILE_SIZE: int = Field(default=50 * 1024 * 1024, description="Maximum file upload size in bytes (50MB)")
    ALLOWED_FILE_EXTENSIONS: List[str] = Field(
        default=[".txt", ".json", ".csv", ".xlsx", ".pdf"],
        description="Allowed file extensions for upload"
    )
    UPLOAD_DIRECTORY: str = Field(default="./uploads", description="Directory for file uploads")
    
    # Rate Limiting Configuration
    RATE_LIMIT_REQUESTS: int = Field(default=100, description="Rate limit requests per minute")
    RATE_LIMIT_BURST: int = Field(default=20, description="Rate limit burst capacity")
    RATE_LIMIT_WINDOW: int = Field(default=60, description="Rate limit window in seconds")
    
    # WebSocket Configuration
    WEBSOCKET_MAX_CONNECTIONS: int = Field(default=100, description="Maximum WebSocket connections")
    WEBSOCKET_TIMEOUT: int = Field(default=300, description="WebSocket timeout in seconds")
    WEBSOCKET_HEARTBEAT_INTERVAL: int = Field(default=30, description="WebSocket heartbeat interval in seconds")
    
    # Logging Configuration
    LOG_LEVEL: LogLevel = Field(default=LogLevel.INFO, description="Application log level")
    LOG_FORMAT: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string"
    )
    LOG_FILE: Optional[str] = Field(default=None, description="Log file path")
    LOG_ROTATION: str = Field(default="midnight", description="Log rotation schedule")
    LOG_RETENTION: int = Field(default=30, description="Log retention days")
    STRUCTURED_LOGGING: bool = Field(default=False, description="Enable structured JSON logging")
    
    # Email/Notification Configuration
    SMTP_HOST: Optional[str] = Field(default=None, description="SMTP server host")
    SMTP_PORT: int = Field(default=587, description="SMTP server port")
    SMTP_USERNAME: Optional[str] = Field(default=None, description="SMTP username")
    SMTP_PASSWORD: Optional[str] = Field(default=None, description="SMTP password")
    SMTP_TLS: bool = Field(default=True, description="Enable SMTP TLS")
    EMAIL_FROM: Optional[str] = Field(default=None, description="Default sender email")
    
    # Security Configuration
    PASSWORD_HASH_ROUNDS: int = Field(default=12, description="Password hashing rounds")
    SESSION_TIMEOUT: int = Field(default=3600, description="Session timeout in seconds")
    MAX_LOGIN_ATTEMPTS: int = Field(default=5, description="Maximum login attempts before lockout")
    LOCKOUT_DURATION: int = Field(default=900, description="Account lockout duration in seconds")
    
    # Performance Configuration
    REQUEST_TIMEOUT: int = Field(default=30, description="Request timeout in seconds")
    CONNECTION_POOL_SIZE: int = Field(default=20, description="HTTP connection pool size")
    CONNECTION_POOL_MAXSIZE: int = Field(default=50, description="HTTP connection pool max size")
    RETRY_ATTEMPTS: int = Field(default=3, description="Default retry attempts for external APIs")
    RETRY_BACKOFF_FACTOR: float = Field(default=1.0, description="Retry backoff multiplier")
    
    # Data Retention Configuration
    TEST_RESULT_RETENTION_DAYS: int = Field(default=90, description="Test result retention period in days")
    LOG_RETENTION_DAYS: int = Field(default=30, description="Log retention period in days")
    REPORT_RETENTION_DAYS: int = Field(default=365, description="Report retention period in days")
    CACHE_TTL: int = Field(default=3600, description="Default cache TTL in seconds")
    
    # Feature Flags
    ENABLE_CACHING: bool = Field(default=True, description="Enable response caching")
    ENABLE_RATE_LIMITING: bool = Field(default=True, description="Enable API rate limiting")
    ENABLE_WEBSOCKETS: bool = Field(default=True, description="Enable WebSocket support")
    ENABLE_BACKGROUND_TASKS: bool = Field(default=True, description="Enable background task processing")
    ENABLE_METRICS: bool = Field(default=True, description="Enable metrics collection")
    ENABLE_TRACING: bool = Field(default=False, description="Enable distributed tracing")
    
    # Monitoring and Analytics
    METRICS_ENDPOINT: str = Field(default="/metrics", description="Metrics endpoint path")
    HEALTH_CHECK_ENDPOINT: str = Field(default="/health", description="Health check endpoint path")
    MONITORING_ENABLED: bool = Field(default=True, description="Enable monitoring features")
    ANALYTICS_ENABLED: bool = Field(default=False, description="Enable analytics tracking")
    
    # API Configuration
    API_V1_PREFIX: str = Field(default="/api/v1", description="API v1 prefix")
    DOCS_URL: str = Field(default="/docs", description="OpenAPI docs URL")
    REDOC_URL: str = Field(default="/redoc", description="ReDoc URL")
    OPENAPI_URL: str = Field(default="/openapi.json", description="OpenAPI JSON URL")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        
    @validator("CORS_ORIGINS", pre=True)
    def parse_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @validator("ENVIRONMENT")
    def validate_environment(cls, v: str) -> str:
        """Validate environment setting."""
        if v not in [env.value for env in Environment]:
            raise ValueError(f"Invalid environment: {v}")
        return v
    
    @validator("DATABASE_URL")
    def validate_database_url(cls, v: str) -> str:
        """Validate database URL format."""
        if not v.startswith(("sqlite://", "postgresql://", "mysql://")):
            raise ValueError("Invalid database URL format")
        return v
    
    @validator("JWT_SECRET_KEY")
    def validate_jwt_secret(cls, v: str, values: dict) -> str:
        """Validate JWT secret key strength."""
        if values.get("ENVIRONMENT") == Environment.PRODUCTION and len(v) < 32:
            raise ValueError("JWT secret key must be at least 32 characters in production")
        return v
    
    @validator("MAX_FILE_SIZE")
    def validate_file_size(cls, v: int) -> int:
        """Validate file size limits."""
        if v <= 0 or v > 100 * 1024 * 1024:  # 100MB max
            raise ValueError("Invalid file size limit")
        return v
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.ENVIRONMENT == Environment.DEVELOPMENT
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.ENVIRONMENT == Environment.PRODUCTION
    
    @property
    def database_url_sync(self) -> str:
        """Get synchronous database URL."""
        return self.DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://")
    
    @property
    def database_url_async(self) -> str:
        """Get asynchronous database URL."""
        return self.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")


# Global settings instance
settings = Settings()

def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings
