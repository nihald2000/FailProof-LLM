# Modal Deployment Guide for Breakpoint LLM Platform

## Prerequisites

1. **Install Modal CLI**:
   ```bash
   pip install modal
   ```

2. **Authenticate with Modal**:
   ```bash
   modal token new
   ```

3. **Create Required Secrets** in Modal dashboard or CLI:

   ```bash
   # OpenAI API Key
   modal secret create openai-api-key OPENAI_API_KEY=sk-your-openai-key-here

   # Anthropic API Key  
   modal secret create anthropic-api-key ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here

   # HuggingFace API Key (optional)
   modal secret create huggingface-api-key HUGGINGFACE_API_KEY=hf_your-huggingface-key-here
   ```

## Deployment Steps

1. **Deploy to Modal**:
   ```bash
   cd backend
   modal deploy modal_deploy.py
   ```

2. **Get your deployment URL**:
   After deployment, Modal will provide a URL like:
   ```
   https://your-username--breakpoint-llm-platform-fastapi-app.modal.run
   ```

3. **Test the deployment**:
   ```bash
   # Health check
   curl https://your-app-url.modal.run/health

   # API documentation
   open https://your-app-url.modal.run/docs
   ```

## Local Development with Modal

1. **Run locally with Modal**:
   ```bash
   modal serve modal_deploy.py
   ```

2. **This will start a local server** that mirrors the Modal environment at:
   ```
   http://localhost:8000
   ```

## Configuration

### Environment Variables
All environment variables are configured in the Modal secrets. Key settings:

- **CORS**: Wide open (`["*"]`) for hackathon use
- **File uploads**: Support for PDF, Excel, Word, CSV, JSON, TXT
- **Request timeout**: 5 minutes for long-running operations
- **Memory**: 4GB RAM with auto-scaling
- **CPU**: 2 vCPUs with burst capability

### API Endpoints
- **Health**: `/health`
- **Docs**: `/docs` 
- **API**: `/api/v1/*`
- **Models**: `/api/v1/models/*`
- **Tests**: `/api/v1/tests/*`
- **Results**: `/api/v1/results/*`

### File Upload Support
- **Max file size**: 50MB
- **Supported types**: TXT, JSON, CSV, XLSX, DOCX, PDF
- **Processing**: Async file processing for large files

## Monitoring and Debugging

1. **View logs**:
   ```bash
   modal logs breakpoint-llm-platform
   ```

2. **Check app status**:
   ```bash
   modal app list
   ```

3. **Open Modal dashboard**:
   ```bash
   modal app show breakpoint-llm-platform
   ```

## Production Considerations

1. **Secrets Management**: Store all API keys in Modal secrets
2. **Rate Limiting**: Configure based on your LLM provider limits
3. **Monitoring**: Use Modal's built-in monitoring and logs
4. **Scaling**: Modal auto-scales based on demand
5. **Cold Starts**: Containers stay warm for 5 minutes

## Troubleshooting

### Common Issues:

1. **Import errors**: Ensure all dependencies are in the Modal image
2. **Secret not found**: Check secret names match exactly
3. **Timeout errors**: Increase timeout for long-running operations
4. **CORS issues**: CORS is wide open, shouldn't be an issue

### Debug Commands:

```bash
# Check app status
modal app list

# View recent logs  
modal logs breakpoint-llm-platform --follow

# Test health endpoint
curl https://your-app-url.modal.run/health

# Test LLM connectivity
modal run modal_deploy.py::test_llm_connectivity
```

## API Usage Examples

```python
import requests

# Your Modal deployment URL
BASE_URL = "https://your-app-url.modal.run"

# Test health
response = requests.get(f"{BASE_URL}/health")
print(response.json())

# Register a new LLM model
model_config = {
    "name": "gpt-4-test",
    "provider_name": "openai", 
    "model_name": "gpt-4",
    "endpoint_url": "https://api.openai.com/v1/chat/completions",
    "auth_config": {
        "auth_type": "bearer_token",
        "api_key": "your-openai-key"
    }
}

response = requests.post(f"{BASE_URL}/api/v1/models/register", json=model_config)
print(response.json())
```

## Next Steps

1. **Update frontend**: Point your frontend to the Modal URL
2. **Configure monitoring**: Set up alerts for health checks
3. **Test thoroughly**: Run your full test suite against the deployed API
4. **Scale as needed**: Modal will auto-scale based on traffic

Your FastAPI backend is now deployed and ready for production use! 🚀
