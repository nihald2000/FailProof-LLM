# Breakpoint - LLM Stress Testing Platform Backend

A comprehensive backend system for stress testing and vulnerability assessment of Large Language Models (LLMs). Built with FastAPI, featuring multi-provider LLM integration, real-time monitoring, and advanced security analysis.

## 🚀 Features

### Core Capabilities
- **Multi-Provider LLM Support**: OpenAI GPT, Anthropic Claude, HuggingFace models
- **Adversarial Test Generation**: Sophisticated attack vectors and prompt injection tests
- **Real-time Monitoring**: WebSocket-based progress tracking and live updates
- **Comprehensive Analysis**: AI response vulnerability detection and classification
- **Batch Processing**: Concurrent test execution with performance optimization
- **Report Generation**: Detailed analysis reports in multiple formats (JSON, CSV, PDF)

### Security & Authentication
- **JWT-based Authentication**: Secure token-based authentication system
- **Role-based Access Control**: Admin and user role management
- **API Key Authentication**: Support for API key-based access
- **Rate Limiting**: Configurable request rate limiting
- **CORS Support**: Cross-origin resource sharing configuration

### Data Management
- **Database Support**: PostgreSQL and SQLite support with SQLAlchemy ORM
- **Data Validation**: Comprehensive Pydantic models with validation
- **File Handling**: Upload/download support for test data and reports
- **Caching**: Redis integration for improved performance

## 📋 Requirements

- Python 3.8+
- PostgreSQL 12+ (or SQLite for development)
- Redis 6+ (optional, for caching)
- 2GB+ RAM recommended
- API keys for LLM providers (OpenAI, Anthropic, HuggingFace)

## 🛠️ Installation

### 1. Clone and Setup

```bash
git clone <repository-url>
cd breakthrough/backend
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment Configuration

Create `.env` file from template:

```bash
cp .env.example .env
```

Edit `.env` with your configuration:

```env
# Application
APP_NAME="Breakpoint LLM Stress Testing Platform"
DEBUG=false
ENVIRONMENT=production

# Database
DATABASE_TYPE=postgresql
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_NAME=breakpoint
DATABASE_USER=your_user
DATABASE_PASSWORD=your_password

# LLM Provider API Keys
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
HUGGINGFACE_API_TOKEN=your_huggingface_token

# Authentication
JWT_SECRET_KEY=your-super-secret-jwt-key
ADMIN_USERNAME=admin
ADMIN_PASSWORD=secure_password
API_KEY=your-api-key

# Redis (optional)
REDIS_URL=redis://localhost:6379/0
```

### 5. Initialize Database

```bash
python run.py --init-db
```

## 🚦 Quick Start

### Development Server

```bash
python run.py --reload
```

### Production Server

```bash
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Docker Deployment

```bash
docker build -t breakpoint-backend .
docker run -p 8000:8000 --env-file .env breakpoint-backend
```

## 📚 API Documentation

Once running, access the interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_NAME` | "Breakpoint" | Application name |
| `DEBUG` | false | Enable debug mode |
| `DATABASE_TYPE` | "sqlite" | Database type (postgresql/sqlite) |
| `OPENAI_API_KEY` | - | OpenAI API key |
| `ANTHROPIC_API_KEY` | - | Anthropic API key |
| `HUGGINGFACE_API_TOKEN` | - | HuggingFace API token |
| `JWT_SECRET_KEY` | - | JWT signing secret |
| `RATE_LIMIT_REQUESTS` | 100 | Requests per minute limit |

### LLM Provider Configuration

#### OpenAI
```env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
OPENAI_MAX_TOKENS=4096
OPENAI_TEMPERATURE=0.7
```

#### Anthropic
```env
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-3-sonnet-20240229
ANTHROPIC_MAX_TOKENS=4096
```

#### HuggingFace
```env
HUGGINGFACE_API_TOKEN=hf_...
HUGGINGFACE_MODEL=microsoft/DialoGPT-medium
```

## 🧪 Testing

### Run Tests

```bash
pytest
```

### Test with Coverage

```bash
pytest --cov=app --cov-report=html
```

### Load Testing

```bash
# Install artillery for load testing
npm install -g artillery

# Run load test
artillery run tests/load/basic-load-test.yml
```

## 🏗️ Architecture

### Project Structure

```
backend/
├── app/
│   ├── api/           # API endpoints
│   │   ├── v1/        # API version 1
│   │   └── deps.py    # Dependencies
│   ├── core/          # Core configuration
│   │   ├── config.py  # Settings management
│   │   └── database.py # Database setup
│   ├── models/        # Data models
│   │   ├── test_result.py # Pydantic models
│   │   └── database.py    # SQLAlchemy models
│   ├── services/      # Business logic
│   │   ├── llm_service.py      # LLM integration
│   │   ├── test_generator.py   # Test generation
│   │   ├── test_runner.py      # Test execution
│   │   ├── failure_analyzer.py # Analysis engine
│   │   └── auth_service.py     # Authentication
│   └── utils/         # Utilities
├── requirements.txt   # Dependencies
├── Dockerfile        # Container config
├── .env.example      # Environment template
└── run.py           # Startup script
```

### Key Components

#### LLM Service (`app/services/llm_service.py`)
- Unified interface for multiple LLM providers
- Automatic retry logic and error handling
- Response caching and rate limiting
- Token usage tracking

#### Test Generator (`app/services/test_generator.py`)
- Adversarial prompt generation
- Multiple attack vector categories
- Configurable difficulty levels
- Custom test case support

#### Test Runner (`app/services/test_runner.py`)
- Async test execution engine
- Concurrent processing with worker pools
- Real-time progress tracking
- Performance metrics collection

#### Failure Analyzer (`app/services/failure_analyzer.py`)
- AI response analysis and classification
- Vulnerability pattern detection
- Risk assessment and scoring
- Detailed failure categorization

## 🔒 Security

### Authentication Flow

1. **Login**: POST `/api/v1/auth/login` with username/password
2. **Token**: Receive JWT access token
3. **Authorization**: Include `Authorization: Bearer <token>` header
4. **Refresh**: POST `/api/v1/auth/refresh` to renew token

### API Endpoints Security

- All endpoints require authentication except `/health` and `/docs`
- Admin endpoints require admin role
- Rate limiting applied per IP address
- Input validation on all endpoints

### Best Practices

- Change default admin credentials
- Use strong JWT secret keys
- Enable HTTPS in production
- Regular security updates
- Monitor for suspicious activity

## 📊 Monitoring & Logging

### Health Checks

```bash
curl http://localhost:8000/health
```

### Metrics

- Prometheus metrics at `/metrics`
- Custom application metrics
- Performance tracking
- Error rate monitoring

### Logging

- Structured JSON logging
- Configurable log levels
- File and console output
- Request/response logging

## 🚀 Deployment

### Docker Deployment

```bash
# Build image
docker build -t breakpoint-backend .

# Run container
docker run -d \\
  --name breakpoint-backend \\
  -p 8000:8000 \\
  --env-file .env \\
  breakpoint-backend
```

### Production Considerations

- Use PostgreSQL for production database
- Configure Redis for caching
- Set up reverse proxy (nginx/Apache)
- Enable SSL/TLS certificates
- Configure monitoring and alerting
- Set up log aggregation
- Regular database backups

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:

- Create an issue on GitHub
- Check the documentation
- Review the API documentation at `/docs`

## 🔄 Changelog

### v1.0.0
- Initial release
- Multi-provider LLM support
- Comprehensive test suite
- Real-time monitoring
- Authentication system
- Report generation
