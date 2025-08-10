"""
Simple Modal deployment script for demo
"""
import modal

# Create Modal app
app = modal.App("failproof-llm-working")

# Create image with dependencies
image = modal.Image.debian_slim().pip_install([
    "fastapi",
    "uvicorn",
    "pydantic",
    "pydantic-settings",
    "python-multipart",
    "python-jose[cryptography]",
    "passlib[bcrypt]",
    "aiofiles",
    "httpx",
    "openai",
    "anthropic",
    "requests"
])

@app.function(
    image=image,
    secrets=[],  # No secrets needed for demo
    keep_warm=1
)
@modal.asgi_app()
def fastapi_app():
    from app.main import app
    return app

if __name__ == "__main__":
    modal.run(fastapi_app)
