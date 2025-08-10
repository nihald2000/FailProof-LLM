"""
Modal Deployment Configuration for Breakpoint LLM Stress Testing Platform
Production-ready FastAPI deployment with comprehensive LLM support and file handling.
"""

import modal
import os
from pathlib import Path

# Define the Modal image with all dependencies
image = (
    modal.Image.debian_slim()
    .pip_install([
        # Core FastAPI dependencies
        "fastapi==0.104.1",
        "uvicorn[standard]==0.24.0",
        "pydantic==2.5.0",
        "pydantic-settings==2.0.3",
        
        # Async HTTP clients
        "aiohttp==3.9.1", 
        "httpx==0.25.2",
        
        # LLM API clients
        "openai==1.3.7",
        "anthropic==0.7.7",
        
        # Authentication and security
        "python-jose[cryptography]==3.3.0",
        "passlib[bcrypt]==1.7.4",
        "python-multipart==0.0.6",
        "itsdangerous==2.1.2",
        
        # Data processing
        "pandas==2.1.4",
        "numpy==1.25.2",
        "jsonschema==4.20.0",
        
        # File handling
        "openpyxl==3.1.2",
        "python-docx==1.1.0",
        "PyPDF2==3.0.1",
        "aiofiles==23.2.0",
        
        # Configuration and utilities
        "python-dotenv==1.0.0",
        "click==8.1.7",
        "python-dateutil==2.8.2",
        "tqdm==4.66.1",
        "rich==13.7.0",
        
        # Monitoring
        "structlog==23.2.0",
        "prometheus-client==0.19.0",
        "psutil==5.9.6",
        
        # Optional: Redis for caching
        "redis==5.0.1",
        "aioredis==2.0.1",
        
        # Optional: Database support
        "sqlalchemy==2.0.23",
        "alembic==1.13.1",
        "asyncpg==0.29.0",
        "aiosqlite==0.19.0",
    ])
    .copy_local_file("requirements.txt", "/app/requirements.txt")
    .copy_local_dir("app", "/app/app")
    .workdir("/app")
    .env({
        "PYTHONPATH": "/app",
        "PYTHONUNBUFFERED": "1",
    })
)

# Create Modal app
app = modal.App(
    name="breakpoint-llm-platform",
    image=image
)

# Define secrets for environment variables
@app.function(
    image=image,
    secrets=[
        modal.Secret.from_dict({
            # Core application settings
            "APP_NAME": "Breakpoint LLM Stress Testing Platform",
            "APP_VERSION": "1.0.0",
            "ENVIRONMENT": "production",
            "DEBUG": "false",
            
            # Server configuration
            "HOST": "0.0.0.0",
            "PORT": "8000",
            "WORKERS": "1",
            "RELOAD": "false",
            
            # CORS settings (wide open for hackathon)
            "CORS_ORIGINS": '["*"]',
            "CORS_CREDENTIALS": "true",
            "CORS_METHODS": '["*"]',
            "CORS_HEADERS": '["*"]',
            
            # JWT and security
            "JWT_SECRET_KEY": "your-super-secret-jwt-key-change-in-production",
            "JWT_ALGORITHM": "HS256",
            "JWT_EXPIRATION_MINUTES": "60",
            "SESSION_TIMEOUT": "3600",
            
            # API settings
            "API_V1_PREFIX": "/api/v1",
            "DOCS_URL": "/docs",
            "REDOC_URL": "/redoc",
            "OPENAPI_URL": "/openapi.json",
            
            # File upload settings
            "MAX_FILE_SIZE": "50",  # MB
            "UPLOAD_DIR": "/tmp/uploads",
            "ALLOWED_FILE_TYPES": '["txt", "json", "csv", "xlsx", "docx", "pdf"]',
            
            # Request settings
            "REQUEST_TIMEOUT": "300",
            "MAX_REQUEST_SIZE": "50",  # MB
            
            # Health check
            "HEALTH_CHECK_ENDPOINT": "/health",
            "METRICS_ENDPOINT": "/metrics",
            "ENABLE_METRICS": "true",
            
            # Logging
            "LOG_LEVEL": "INFO",
            "LOG_FORMAT": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "STRUCTURED_LOGGING": "false",
            
            # Default rate limits
            "DEFAULT_RATE_LIMIT": "100/minute",
            "BURST_RATE_LIMIT": "1000/hour",
        }),
        # Add your API keys as separate secrets
        modal.Secret.from_name("openai-api-key"),  # Contains OPENAI_API_KEY
        modal.Secret.from_name("anthropic-api-key"),  # Contains ANTHROPIC_API_KEY
        modal.Secret.from_name("huggingface-api-key"),  # Contains HUGGINGFACE_API_KEY (optional)
    ],
    timeout=300,  # 5-minute timeout for requests
    container_idle_timeout=300,  # Keep containers warm for 5 minutes
    cpu=2.0,  # 2 vCPUs
    memory=4096,  # 4GB RAM
    allow_concurrent_inputs=100,  # Handle up to 100 concurrent requests
)
def create_app():
    """Initialize and return the FastAPI application."""
    
    # Import here to avoid issues with Modal's import system
    import sys
    sys.path.insert(0, '/app')
    
    from app.main import app as fastapi_app
    
    return fastapi_app


# ASGI application endpoint
@app.asgi_app()
def fastapi_app():
    """Main ASGI application endpoint for Modal."""
    return create_app()


# Health check function (can be called independently)
@app.function(image=image)
def health_check():
    """Standalone health check function."""
    import time
    import json
    
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "service": "breakpoint-llm-platform",
        "version": "1.0.0",
        "environment": "modal-production",
        "checks": {
            "application": "healthy",
            "memory": "healthy",
            "disk": "healthy",
        }
    }
    
    return health_status


# Function to test LLM connectivity
@app.function(
    image=image,
    secrets=[
        modal.Secret.from_name("openai-api-key"),
        modal.Secret.from_name("anthropic-api-key"),
    ]
)
def test_llm_connectivity():
    """Test connectivity to LLM providers."""
    import asyncio
    import aiohttp
    import os
    
    async def test_openai():
        """Test OpenAI connectivity."""
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                return {"provider": "openai", "status": "no_api_key"}
            
            headers = {"Authorization": f"Bearer {api_key}"}
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://api.openai.com/v1/models",
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        return {"provider": "openai", "status": "healthy"}
                    else:
                        return {"provider": "openai", "status": f"error_{response.status}"}
        except Exception as e:
            return {"provider": "openai", "status": f"error: {str(e)}"}
    
    async def test_anthropic():
        """Test Anthropic connectivity."""
        try:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                return {"provider": "anthropic", "status": "no_api_key"}
            
            headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01"}
            async with aiohttp.ClientSession() as session:
                # Anthropic doesn't have a simple endpoint to test, so we'll just validate the key format
                if api_key.startswith("sk-ant-"):
                    return {"provider": "anthropic", "status": "key_format_valid"}
                else:
                    return {"provider": "anthropic", "status": "invalid_key_format"}
        except Exception as e:
            return {"provider": "anthropic", "status": f"error: {str(e)}"}
    
    async def run_tests():
        results = await asyncio.gather(
            test_openai(),
            test_anthropic(),
            return_exceptions=True
        )
        return results
    
    return asyncio.run(run_tests())


# File processing function for large uploads
@app.function(
    image=image,
    timeout=600,  # 10-minute timeout for file processing
    memory=8192,  # 8GB RAM for large file processing
    cpu=4.0,  # 4 vCPUs for file processing
)
def process_large_file(file_content: bytes, filename: str, file_type: str):
    """Process large uploaded files asynchronously."""
    import tempfile
    import os
    from pathlib import Path
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}") as tmp_file:
        tmp_file.write(file_content)
        tmp_path = tmp_file.name
    
    try:
        # Process file based on type
        if file_type.lower() == 'pdf':
            import PyPDF2
            with open(tmp_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text_content = ""
                for page in reader.pages:
                    text_content += page.extract_text()
                return {"type": "pdf", "content": text_content[:10000]}  # Limit content
        
        elif file_type.lower() in ['xlsx', 'xls']:
            import pandas as pd
            df = pd.read_excel(tmp_path)
            return {
                "type": "excel",
                "rows": len(df),
                "columns": len(df.columns),
                "preview": df.head().to_dict()
            }
        
        elif file_type.lower() == 'docx':
            import docx
            doc = docx.Document(tmp_path)
            text_content = ""
            for paragraph in doc.paragraphs:
                text_content += paragraph.text + "\n"
            return {"type": "docx", "content": text_content[:10000]}  # Limit content
        
        elif file_type.lower() == 'csv':
            import pandas as pd
            df = pd.read_csv(tmp_path)
            return {
                "type": "csv",
                "rows": len(df),
                "columns": len(df.columns),
                "preview": df.head().to_dict()
            }
        
        elif file_type.lower() in ['txt', 'json']:
            with open(tmp_path, 'r', encoding='utf-8') as file:
                content = file.read()
                return {"type": file_type, "content": content[:10000]}  # Limit content
        
        else:
            return {"type": "unknown", "error": f"Unsupported file type: {file_type}"}
    
    except Exception as e:
        return {"error": f"Failed to process file: {str(e)}"}
    
    finally:
        # Cleanup temporary file
        try:
            os.unlink(tmp_path)
        except:
            pass


# Batch LLM testing function
@app.function(
    image=image,
    secrets=[
        modal.Secret.from_name("openai-api-key"),
        modal.Secret.from_name("anthropic-api-key"),
    ],
    timeout=1800,  # 30-minute timeout for batch operations
    memory=4096,
    cpu=2.0,
)
def batch_llm_test(test_configs: list, test_prompts: list):
    """Run batch LLM tests across multiple configurations."""
    import asyncio
    import sys
    sys.path.insert(0, '/app')
    
    from app.services.llm_service import universal_llm_service
    from app.models.llm_config import LLMConfig
    
    async def run_batch_tests():
        results = []
        
        for config_data in test_configs:
            config = LLMConfig(**config_data)
            config_results = []
            
            for prompt in test_prompts:
                try:
                    # Test the configuration with the prompt
                    test_result = await universal_llm_service.test_configuration(config)
                    
                    if test_result.success:
                        # Run actual test with prompt
                        response = await universal_llm_service.generate_text(
                            config=config,
                            prompt=prompt,
                            max_tokens=100  # Limit tokens for batch testing
                        )
                        config_results.append({
                            "prompt": prompt[:100],  # Limit prompt length in results
                            "success": True,
                            "response_length": len(response.get("text", "")),
                            "latency": response.get("latency", 0),
                            "tokens_used": response.get("tokens_used", 0)
                        })
                    else:
                        config_results.append({
                            "prompt": prompt[:100],
                            "success": False,
                            "error": test_result.error_message
                        })
                
                except Exception as e:
                    config_results.append({
                        "prompt": prompt[:100],
                        "success": False,
                        "error": str(e)
                    })
            
            results.append({
                "config_name": config.name,
                "provider": config.provider_name,
                "model": config.model_name,
                "test_results": config_results,
                "success_rate": sum(1 for r in config_results if r["success"]) / len(config_results) if config_results else 0
            })
        
        return results
    
    return asyncio.run(run_batch_tests())


# Local development server (for testing)
if __name__ == "__main__":
    # This allows you to run the Modal app locally for testing
    import subprocess
    import sys
    
    print("Starting Modal development server...")
    print("Note: This requires Modal CLI to be installed and authenticated")
    
    try:
        # Run Modal serve command
        subprocess.run([
            sys.executable, "-m", "modal", "serve", __file__
        ])
    except KeyboardInterrupt:
        print("\nShutting down development server...")
    except Exception as e:
        print(f"Error running Modal server: {e}")
        print("\nTo deploy to Modal, run:")
        print("  modal deploy modal_deploy.py")
        print("\nTo run locally, run:")
        print("  modal serve modal_deploy.py")
