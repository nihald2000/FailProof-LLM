"""
Database initialization and migration utilities.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models.database import Base
from app.core.config import get_settings
import logging

logger = logging.getLogger(__name__)
settings = get_settings()


def get_database_url() -> str:
    """Get database URL from settings."""
    if settings.DATABASE_URL:
        return settings.DATABASE_URL
    
    if settings.DATABASE_TYPE == "postgresql":
        return (
            f"postgresql://{settings.DATABASE_USER}:{settings.DATABASE_PASSWORD}"
            f"@{settings.DATABASE_HOST}:{settings.DATABASE_PORT}/{settings.DATABASE_NAME}"
        )
    else:
        return f"sqlite:///{settings.DATABASE_PATH}"


def create_database_engine():
    """Create database engine."""
    database_url = get_database_url()
    
    if database_url.startswith("sqlite"):
        engine = create_engine(
            database_url,
            connect_args={"check_same_thread": False},
            echo=settings.DEBUG
        )
    else:
        engine = create_engine(
            database_url,
            echo=settings.DEBUG,
            pool_size=settings.DATABASE_POOL_SIZE,
            max_overflow=settings.DATABASE_MAX_OVERFLOW
        )
    
    return engine


def get_session_factory():
    """Get database session factory."""
    engine = create_database_engine()
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_database():
    """Initialize database tables."""
    try:
        engine = create_database_engine()
        Base.metadata.create_all(bind=engine)
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


def get_db():
    """Get database session (dependency injection)."""
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


if __name__ == "__main__":
    # Initialize database when run directly
    init_database()
