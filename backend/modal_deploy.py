# Save as modal_working_fix.py
import modal

app = modal.App("failproof-llm-working")

image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "fastapi>=0.100.0",
    "uvicorn>=0.20.0"
])

@app.function(image=image, timeout=300)
@modal.asgi_app()  # Use asgi_app instead of fastapi_endpoint
def fastapi_app():
    """Working FastAPI application."""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    
    # Create FastAPI app
    fastapi_app = FastAPI(
        title="FailProof LLM Platform",
        description="LLM Stress Testing Platform", 
        version="1.0.0"
    )
    
    # Add CORS
    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Root endpoint
    @fastapi_app.get("/")
    def root():
        return {
            "message": "✅ FailProof LLM Platform - WORKING!",
            "status": "operational",
            "platform": "Modal",
            "timestamp": "2025-08-10",
            "available_endpoints": ["/", "/health", "/api/v1/status"]
        }
    
    # Health endpoint
    @fastapi_app.get("/health")
    def health():
        return {
            "status": "healthy",
            "service": "failproof-llm-platform",
            "version": "1.0.0",
            "checks": {
                "application": "healthy",
                "modal": "operational",
                "fastapi": "working"
            }
        }
    
    # API status endpoint
    @fastapi_app.get("/api/v1/status")
    def api_status():
        return {
            "api_version": "v1",
            "status": "operational", 
            "endpoints": {
                "root": "/",
                "health": "/health",
                "api_status": "/api/v1/status"
            }
        }
    
    # Demo model registration endpoint
    @fastapi_app.post("/api/v1/models/register")
    def register_model(model_data: dict):
        return {
            "status": "success",
            "message": f"Model registered: {model_data.get('name', 'Unknown')}",
            "model_id": "demo-123"
        }
    
    return fastapi_app
