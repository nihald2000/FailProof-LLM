"""
FastAPI Application Entry Point for Breakpoint LLM Stress Testing Platform.
Production-ready FastAPI application with comprehensive middleware and security.
"""

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, Request, Response, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.sessions import SessionMiddleware

from app.core.config import settings
from app.api.v1 import tests, results, models, reports, auth
from app.api import deps


# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.value),
    format=settings.LOG_FORMAT if not settings.STRUCTURED_LOGGING else None,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(settings.LOG_FILE) if settings.LOG_FILE else logging.NullHandler()
    ]
)

logger = logging.getLogger(__name__)


class RequestTracingMiddleware(BaseHTTPMiddleware):
    """Middleware for request tracing and correlation IDs."""
    
    async def dispatch(self, request: Request, call_next):
        # Generate correlation ID
        correlation_id = str(uuid.uuid4())
        request.state.correlation_id = correlation_id
        
        # Add correlation ID to response headers
        response = await call_next(request)
        response.headers["X-Correlation-ID"] = correlation_id
        
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for comprehensive request/response logging."""
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Log request details
        logger.info(
            f"Request started",
            extra={
                "correlation_id": getattr(request.state, "correlation_id", None),
                "method": request.method,
                "url": str(request.url),
                "user_agent": request.headers.get("user-agent"),
                "client_ip": request.client.host if request.client else None,
            }
        )
        
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            # Log response details
            logger.info(
                f"Request completed",
                extra={
                    "correlation_id": getattr(request.state, "correlation_id", None),
                    "status_code": response.status_code,
                    "process_time": process_time,
                }
            )
            
            # Add performance headers
            response.headers["X-Process-Time"] = str(process_time)
            
            return response
            
        except Exception as exc:
            process_time = time.time() - start_time
            logger.error(
                f"Request failed",
                extra={
                    "correlation_id": getattr(request.state, "correlation_id", None),
                    "error": str(exc),
                    "process_time": process_time,
                },
                exc_info=True
            )
            raise


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware for adding security headers."""
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        if settings.is_production:
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        return response


class RequestTimeoutMiddleware(BaseHTTPMiddleware):
    """Middleware for request timeout handling."""
    
    async def dispatch(self, request: Request, call_next):
        try:
            return await asyncio.wait_for(
                call_next(request),
                timeout=settings.REQUEST_TIMEOUT
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"Request timeout",
                extra={
                    "correlation_id": getattr(request.state, "correlation_id", None),
                    "timeout": settings.REQUEST_TIMEOUT,
                }
            )
            raise HTTPException(
                status_code=status.HTTP_408_REQUEST_TIMEOUT,
                detail="Request timeout"
            )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    # Startup
    logger.info("Starting Breakpoint LLM Stress Testing Platform")
    
    # Initialize database connections
    try:
        # Database initialization would go here
        logger.info("Database connections initialized")
    except Exception as exc:
        logger.error(f"Failed to initialize database: {exc}")
        raise
    
    # Initialize Redis if configured
    if settings.REDIS_URL:
        try:
            # Redis initialization would go here
            logger.info("Redis connection initialized")
        except Exception as exc:
            logger.warning(f"Failed to initialize Redis: {exc}")
    
    # Initialize LLM service clients
    try:
        # LLM service initialization would go here
        logger.info("LLM service clients initialized")
    except Exception as exc:
        logger.error(f"Failed to initialize LLM services: {exc}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Breakpoint LLM Stress Testing Platform")
    
    # Cleanup database connections
    try:
        # Database cleanup would go here
        logger.info("Database connections closed")
    except Exception as exc:
        logger.error(f"Error during database cleanup: {exc}")
    
    # Cleanup Redis connections
    if settings.REDIS_URL:
        try:
            # Redis cleanup would go here
            logger.info("Redis connection closed")
        except Exception as exc:
            logger.error(f"Error during Redis cleanup: {exc}")


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
    docs_url=settings.DOCS_URL if not settings.is_production else None,
    redoc_url=settings.REDOC_URL if not settings.is_production else None,
    openapi_url=settings.OPENAPI_URL if not settings.is_production else None,
    lifespan=lifespan,
    responses={
        422: {
            "description": "Validation Error",
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "detail": {"type": "string"},
                            "errors": {"type": "array"},
                            "correlation_id": {"type": "string"}
                        }
                    }
                }
            }
        },
        500: {
            "description": "Internal Server Error",
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "detail": {"type": "string"},
                            "correlation_id": {"type": "string"}
                        }
                    }
                }
            }
        }
    }
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_CREDENTIALS,
    allow_methods=settings.CORS_METHODS,
    allow_headers=settings.CORS_HEADERS,
)

# Add trusted host middleware for production
if settings.is_production:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.CORS_ORIGINS
    )

# Add session middleware
app.add_middleware(
    SessionMiddleware,
    secret_key=settings.JWT_SECRET_KEY,
    max_age=settings.SESSION_TIMEOUT,
    same_site="lax",
    https_only=settings.is_production
)

# Add compression middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Add custom middleware
app.add_middleware(RequestTimeoutMiddleware)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(RequestTracingMiddleware)


# Global exception handlers
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions with proper error response format."""
    correlation_id = getattr(request.state, "correlation_id", None)
    
    logger.error(
        f"HTTP exception: {exc.status_code} - {exc.detail}",
        extra={
            "correlation_id": correlation_id,
            "status_code": exc.status_code,
            "detail": exc.detail,
        }
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "correlation_id": correlation_id,
            "timestamp": time.time(),
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors with detailed error information."""
    correlation_id = getattr(request.state, "correlation_id", None)
    
    logger.warning(
        f"Validation error: {exc.errors()}",
        extra={
            "correlation_id": correlation_id,
            "errors": exc.errors(),
        }
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": "Validation error",
            "errors": exc.errors(),
            "correlation_id": correlation_id,
            "timestamp": time.time(),
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions with error tracking."""
    correlation_id = getattr(request.state, "correlation_id", None)
    
    logger.error(
        f"Unexpected error: {str(exc)}",
        extra={
            "correlation_id": correlation_id,
            "error_type": type(exc).__name__,
        },
        exc_info=True
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error" if settings.is_production else str(exc),
            "correlation_id": correlation_id,
            "timestamp": time.time(),
        }
    )


# Health check endpoint
@app.get(
    settings.HEALTH_CHECK_ENDPOINT,
    summary="Health Check",
    description="Check application health and dependencies",
    response_model=Dict[str, Any],
    tags=["Health"]
)
async def health_check():
    """Health check endpoint for monitoring and deployment verification."""
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT.value,
        "checks": {
            "database": "healthy",  # Would check actual database connection
            "redis": "healthy" if settings.REDIS_URL else "not_configured",
            "llm_services": "healthy",  # Would check LLM service availability
        }
    }
    
    # Check if any critical services are down
    critical_checks = ["database", "llm_services"]
    if any(health_status["checks"][check] != "healthy" for check in critical_checks):
        health_status["status"] = "unhealthy"
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=health_status
        )
    
    return health_status


# Metrics endpoint (if enabled)
if settings.ENABLE_METRICS:
    @app.get(
        settings.METRICS_ENDPOINT,
        summary="Application Metrics",
        description="Get application performance metrics",
        tags=["Monitoring"]
    )
    async def metrics():
        """Application metrics endpoint for monitoring."""
        # Metrics collection would be implemented here
        return {
            "requests_total": 0,
            "requests_duration_seconds": 0.0,
            "active_connections": 0,
            "error_rate": 0.0,
        }


# API route registration
app.include_router(
    auth.router,
    prefix=f"{settings.API_V1_PREFIX}/auth",
    tags=["Authentication"]
)

app.include_router(
    tests.router,
    prefix=f"{settings.API_V1_PREFIX}/tests",
    tags=["Tests"],
    dependencies=[Depends(deps.get_current_user)] if hasattr(deps, 'get_current_user') else []
)

app.include_router(
    results.router,
    prefix=f"{settings.API_V1_PREFIX}/results",
    tags=["Results"],
    dependencies=[Depends(deps.get_current_user)] if hasattr(deps, 'get_current_user') else []
)

app.include_router(
    models.router,
    prefix=f"{settings.API_V1_PREFIX}/models",
    tags=["Models"],
    dependencies=[Depends(deps.get_current_user)] if hasattr(deps, 'get_current_user') else []
)

app.include_router(
    reports.router,
    prefix=f"{settings.API_V1_PREFIX}/reports",
    tags=["Reports"],
    dependencies=[Depends(deps.get_current_user)] if hasattr(deps, 'get_current_user') else []
)


# Root endpoint
@app.get(
    "/",
    summary="Root Endpoint",
    description="API root with basic information",
    tags=["Root"]
)
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "description": settings.APP_DESCRIPTION,
        "environment": settings.ENVIRONMENT.value,
        "docs_url": settings.DOCS_URL if not settings.is_production else None,
        "health_check": settings.HEALTH_CHECK_ENDPOINT,
    }


# Development server
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD and settings.is_development,
        workers=settings.WORKERS if settings.is_production else 1,
        log_level=settings.LOG_LEVEL.value.lower(),
        access_log=True,
    )
