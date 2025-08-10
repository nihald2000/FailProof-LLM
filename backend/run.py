"""
Startup script for Breakpoint LLM Stress Testing Platform.
"""

import uvicorn
import argparse
import sys
from pathlib import Path
from app.core.config import get_settings
from app.core.database import init_database
import logging

logger = logging.getLogger(__name__)


def initialize_application():
    """Initialize the application."""
    settings = get_settings()
    
    # Initialize database
    try:
        init_database()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        sys.exit(1)
    
    # Create upload directory
    upload_dir = Path(settings.UPLOAD_DIRECTORY)
    upload_dir.mkdir(exist_ok=True)
    logger.info(f"Upload directory ensured: {upload_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Breakpoint LLM Stress Testing Platform")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--init-db", action="store_true", help="Initialize database only")
    
    args = parser.parse_args()
    
    if args.init_db:
        initialize_application()
        print("Database initialized successfully")
        return
    
    # Initialize application
    initialize_application()
    
    # Start the server
    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()
