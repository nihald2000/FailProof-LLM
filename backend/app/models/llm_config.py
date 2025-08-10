"""
Generic LLM Configuration Model - Comprehensive Pydantic models for any LLM provider.
Supports current and future models through flexible configuration.
"""

from pydantic import BaseModel, Field, field_validator, SecretStr, HttpUrl
from typing import Optional, Dict, Any, List, Union, Literal
from enum import Enum
from datetime import datetime
import json


class AuthenticationType(str, Enum):
    """Authentication methods supported by LLM providers."""
    BEARER_TOKEN = "bearer_token"
    API_KEY_HEADER = "api_key_header"
    QUERY_PARAM = "query_param"
    CUSTOM_HEADERS = "custom_headers"
    BASIC_AUTH = "basic_auth"
    OAUTH2 = "oauth2"
    NO_AUTH = "no_auth"


class RequestMethod(str, Enum):
    """HTTP methods for API requests."""
    POST = "POST"
    GET = "GET"
    PUT = "PUT"
    PATCH = "PATCH"


class ResponseFormat(str, Enum):
    """Response format types."""
    JSON = "json"
    TEXT = "text"
    STREAM = "stream"
    BINARY = "binary"


class BillingModel(str, Enum):
    """Billing models for cost calculation."""
    PER_TOKEN = "per_token"
    PER_REQUEST = "per_request"
    PER_MINUTE = "per_minute"
    FIXED_RATE = "fixed_rate"
    CUSTOM = "custom"


class LLMConfigStatus(str, Enum):
    """LLM configuration status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    TESTING = "testing"
    DEPRECATED = "deprecated"


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"
    COHERE = "cohere"
    AZURE_OPENAI = "azure_openai"
    GOOGLE = "google"
    CUSTOM = "custom"


class ModelCapability(str, Enum):
    """Model capabilities."""
    CHAT = "chat"
    TEXT_GENERATION = "text_generation"
    FUNCTION_CALLING = "function_calling"
    STREAMING = "streaming"
    EMBEDDINGS = "embeddings"
    IMAGE_GENERATION = "image_generation"
    VISION = "vision"
    AUDIO = "audio"
    CODE_GENERATION = "code_generation"


class AuthConfig(BaseModel):
    """Authentication configuration for LLM providers."""
    auth_type: AuthenticationType = Field(..., description="Authentication method")
    api_key: Optional[SecretStr] = Field(None, description="API key or bearer token")
    header_name: Optional[str] = Field(None, description="Custom header name for API key")
    query_param_name: Optional[str] = Field(None, description="Query parameter name for API key")
    username: Optional[str] = Field(None, description="Username for basic auth")
    password: Optional[SecretStr] = Field(None, description="Password for basic auth")
    custom_headers: Optional[Dict[str, str]] = Field(default={}, description="Custom headers")
    oauth2_config: Optional[Dict[str, Any]] = Field(None, description="OAuth2 configuration")

    @field_validator('header_name')
    def validate_header_name(cls, v, info):
        if info.data.get('auth_type') == AuthenticationType.API_KEY_HEADER and not v:
            raise ValueError('header_name is required for API_KEY_HEADER auth type')
        return v

    @field_validator('query_param_name')
    def validate_query_param_name(cls, v, info):
        if info.data.get('auth_type') == AuthenticationType.QUERY_PARAM and not v:
            raise ValueError('query_param_name is required for QUERY_PARAM auth type')
        return v


class ModelParameters(BaseModel):
    """Model generation parameters with validation."""
    max_tokens: Optional[int] = Field(default=4096, ge=1, le=2000000, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0, description="Top-p sampling")
    top_k: Optional[int] = Field(None, ge=1, le=1000, description="Top-k sampling")
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0, description="Frequency penalty")
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0, description="Presence penalty")
    repetition_penalty: Optional[float] = Field(None, ge=0.0, le=2.0, description="Repetition penalty")
    stop_sequences: Optional[List[str]] = Field(default=[], description="Stop sequences")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    
    # Advanced parameters
    min_tokens: Optional[int] = Field(None, ge=0, description="Minimum tokens to generate")
    length_penalty: Optional[float] = Field(None, description="Length penalty for generation")
    no_repeat_ngram_size: Optional[int] = Field(None, ge=1, description="N-gram size for repetition prevention")
    
    # Custom parameters for specific providers
    custom_parameters: Optional[Dict[str, Any]] = Field(default={}, description="Provider-specific parameters")


class RequestFormat(BaseModel):
    """API request format configuration."""
    method: RequestMethod = Field(default=RequestMethod.POST, description="HTTP method")
    prompt_field_name: str = Field(default="prompt", description="Field name for prompt in request")
    messages_format: bool = Field(default=False, description="Whether to use messages array format")
    message_role_user: str = Field(default="user", description="User role name in messages")
    message_role_assistant: str = Field(default="assistant", description="Assistant role name in messages")
    message_role_system: str = Field(default="system", description="System role name in messages")
    
    # Request structure
    model_field_name: str = Field(default="model", description="Field name for model in request")
    parameters_nested: bool = Field(default=False, description="Whether parameters are nested in sub-object")
    parameters_field_name: Optional[str] = Field(None, description="Field name for parameters if nested")
    
    # Headers and content type
    content_type: str = Field(default="application/json", description="Request content type")
    additional_headers: Optional[Dict[str, str]] = Field(default={}, description="Additional request headers")
    
    # Custom request transformation
    custom_request_template: Optional[str] = Field(None, description="Custom request template (JSON string)")


class ResponseFormat(BaseModel):
    """API response format configuration."""
    response_type: ResponseFormat = Field(default=ResponseFormat.JSON, description="Response format type")
    
    # Field paths for extracting data from response
    content_field_path: str = Field(default="choices.0.text", description="JSONPath to content in response")
    error_field_path: str = Field(default="error.message", description="JSONPath to error message")
    usage_field_path: Optional[str] = Field(default="usage", description="JSONPath to usage information")
    
    # Streaming response configuration
    streaming_format: str = Field(default="sse", description="Streaming format (sse, jsonlines, websocket)")
    streaming_data_prefix: str = Field(default="data: ", description="Prefix for streaming data")
    streaming_end_marker: str = Field(default="[DONE]", description="End marker for streaming")
    streaming_content_path: str = Field(default="choices.0.delta.content", description="JSONPath to streaming content")
    
    # Custom response parsing
    custom_parser: Optional[str] = Field(None, description="Custom response parser function name")


class ProviderLimits(BaseModel):
    """Rate limiting and performance configuration."""
    rate_limit_per_minute: Optional[int] = Field(default=60, ge=1, description="Requests per minute limit")
    rate_limit_per_hour: Optional[int] = Field(None, ge=1, description="Requests per hour limit")
    rate_limit_per_day: Optional[int] = Field(None, ge=1, description="Requests per day limit")
    
    max_concurrent_requests: int = Field(default=5, ge=1, le=100, description="Maximum concurrent requests")
    timeout_seconds: int = Field(default=30, ge=1, le=300, description="Request timeout in seconds")
    retry_attempts: int = Field(default=3, ge=0, le=10, description="Number of retry attempts")
    retry_delay_seconds: float = Field(default=1.0, ge=0.1, le=60.0, description="Delay between retries")
    
    # Backoff configuration
    exponential_backoff: bool = Field(default=True, description="Use exponential backoff for retries")
    max_retry_delay: float = Field(default=60.0, ge=1.0, description="Maximum retry delay")
    
    # Connection pooling
    connection_pool_size: int = Field(default=10, ge=1, le=100, description="Connection pool size")
    keep_alive_timeout: int = Field(default=30, ge=1, description="Keep-alive timeout in seconds")


class CostConfig(BaseModel):
    """Cost calculation configuration."""
    billing_model: BillingModel = Field(..., description="Billing model for cost calculation")
    
    # Token-based pricing
    cost_per_input_token: Optional[float] = Field(None, ge=0, description="Cost per input token")
    cost_per_output_token: Optional[float] = Field(None, ge=0, description="Cost per output token")
    cost_per_token: Optional[float] = Field(None, ge=0, description="Cost per token (if same for input/output)")
    
    # Request-based pricing
    cost_per_request: Optional[float] = Field(None, ge=0, description="Cost per API request")
    
    # Time-based pricing
    cost_per_minute: Optional[float] = Field(None, ge=0, description="Cost per minute of usage")
    
    # Fixed pricing
    fixed_monthly_cost: Optional[float] = Field(None, ge=0, description="Fixed monthly cost")
    
    # Currency and custom pricing
    currency: str = Field(default="USD", description="Currency for cost calculation")
    custom_pricing_formula: Optional[str] = Field(None, description="Custom pricing formula")
    
    # Free tier configuration
    free_tier_requests: Optional[int] = Field(None, ge=0, description="Free requests per month")
    free_tier_tokens: Optional[int] = Field(None, ge=0, description="Free tokens per month")


class CapabilityConfig(BaseModel):
    """Model capability configuration."""
    # Basic capabilities
    supports_streaming: bool = Field(default=False, description="Supports streaming responses")
    supports_functions: bool = Field(default=False, description="Supports function calling")
    supports_vision: bool = Field(default=False, description="Supports image inputs")
    supports_audio: bool = Field(default=False, description="Supports audio inputs")
    supports_video: bool = Field(default=False, description="Supports video inputs")
    supports_documents: bool = Field(default=False, description="Supports document inputs")
    
    # Context and generation limits
    context_window_size: int = Field(default=4096, ge=1, description="Maximum context window size")
    max_output_tokens: int = Field(default=4096, ge=1, description="Maximum output tokens")
    
    # Advanced features
    supports_system_messages: bool = Field(default=True, description="Supports system messages")
    supports_chat_format: bool = Field(default=True, description="Supports chat message format")
    supports_completion_format: bool = Field(default=True, description="Supports completion format")
    supports_custom_stopping: bool = Field(default=True, description="Supports custom stop sequences")
    
    # Moderation and safety
    has_content_filter: bool = Field(default=False, description="Has built-in content filtering")
    supports_safety_settings: bool = Field(default=False, description="Supports safety configuration")
    
    # Performance characteristics
    average_response_time_ms: Optional[int] = Field(None, ge=0, description="Average response time")
    supports_batch_requests: bool = Field(default=False, description="Supports batch processing")
    
    # Custom capabilities
    custom_capabilities: Optional[Dict[str, Any]] = Field(default={}, description="Provider-specific capabilities")


class CustomFields(BaseModel):
    """Provider-specific custom fields and parameters."""
    # Custom API parameters not covered by standard fields
    custom_api_params: Optional[Dict[str, Any]] = Field(default={}, description="Custom API parameters")
    
    # Provider-specific headers
    custom_headers: Optional[Dict[str, str]] = Field(default={}, description="Custom HTTP headers")
    
    # Custom request/response transformations
    request_transformer: Optional[str] = Field(None, description="Custom request transformer function")
    response_transformer: Optional[str] = Field(None, description="Custom response transformer function")
    
    # Provider-specific configuration
    provider_config: Optional[Dict[str, Any]] = Field(default={}, description="Provider-specific configuration")
    
    # Custom validation rules
    validation_rules: Optional[Dict[str, Any]] = Field(default={}, description="Custom validation rules")


class LLMConfigBase(BaseModel):
    """Base LLM configuration model."""
    name: str = Field(..., min_length=1, max_length=100, description="Configuration name")
    display_name: Optional[str] = Field(None, description="Human-readable display name")
    provider_name: str = Field(..., description="Provider name (can be custom)")
    model_name: str = Field(..., description="Model identifier")
    api_endpoint: HttpUrl = Field(..., description="API endpoint URL")
    
    description: Optional[str] = Field(None, max_length=1000, description="Configuration description")
    tags: List[str] = Field(default=[], description="Configuration tags")
    version: str = Field(default="1.0.0", description="Configuration version")
    
    # Status and metadata
    status: LLMConfigStatus = Field(default=LLMConfigStatus.INACTIVE, description="Configuration status")
    created_at: Optional[datetime] = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    
    # Configuration inheritance
    parent_config_id: Optional[str] = Field(None, description="Parent configuration for inheritance")
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            SecretStr: lambda v: v.get_secret_value() if v else None
        }


class LLMConfigCreate(LLMConfigBase):
    """LLM configuration creation model."""
    # Authentication configuration
    auth_config: AuthConfig = Field(..., description="Authentication configuration")
    
    # Model parameters
    model_parameters: ModelParameters = Field(default_factory=ModelParameters, description="Model parameters")
    
    # API format configuration
    request_format: RequestFormat = Field(default_factory=RequestFormat, description="Request format configuration")
    response_format: ResponseFormat = Field(default_factory=ResponseFormat, description="Response format configuration")
    
    # Limits and performance
    provider_limits: ProviderLimits = Field(default_factory=ProviderLimits, description="Provider limits")
    
    # Cost configuration
    cost_config: Optional[CostConfig] = Field(None, description="Cost calculation configuration")
    
    # Capabilities
    capability_config: CapabilityConfig = Field(default_factory=CapabilityConfig, description="Model capabilities")
    
    # Custom fields
    custom_fields: CustomFields = Field(default_factory=CustomFields, description="Custom fields")
    
    # Validation and testing
    auto_test_on_create: bool = Field(default=True, description="Automatically test configuration on creation")
    
    @field_validator('api_endpoint')
    def validate_api_endpoint(cls, v):
        if not str(v).startswith(('http://', 'https://')):
            raise ValueError('API endpoint must be a valid HTTP/HTTPS URL')
        return v
    
    @field_validator('name')
    def validate_name(cls, v):
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Name must contain only alphanumeric characters, underscores, and hyphens')
        return v
    
    # Provider-specific settings
    provider_config: Dict[str, Any] = Field(default={}, description="Provider-specific configuration")
    
    # Rate limiting
    requests_per_minute: Optional[int] = Field(default=60, ge=1, description="Rate limit")
    tokens_per_minute: Optional[int] = Field(default=100000, ge=1, description="Token rate limit")


class LLMConfigUpdate(BaseModel):
    """LLM configuration update model."""
    name: Optional[str] = Field(None, min_length=1, max_length=100, description="Configuration name")
    display_name: Optional[str] = Field(None, description="Human-readable display name")
    description: Optional[str] = Field(None, max_length=1000, description="Configuration description")
    tags: Optional[List[str]] = Field(None, description="Configuration tags")
    status: Optional[LLMConfigStatus] = Field(None, description="Configuration status")
    
    # Partial updates for nested objects
    auth_config: Optional[AuthConfig] = Field(None, description="Authentication configuration")
    model_parameters: Optional[ModelParameters] = Field(None, description="Model parameters")
    request_format: Optional[RequestFormat] = Field(None, description="Request format configuration")
    response_format: Optional[ResponseFormat] = Field(None, description="Response format configuration")
    provider_limits: Optional[ProviderLimits] = Field(None, description="Provider limits")
    cost_config: Optional[CostConfig] = Field(None, description="Cost calculation configuration")
    capability_config: Optional[CapabilityConfig] = Field(None, description="Model capabilities")
    custom_fields: Optional[CustomFields] = Field(None, description="Custom fields")
    
    updated_at: Optional[datetime] = Field(default_factory=datetime.utcnow, description="Last update timestamp")


class LLMConfig(LLMConfigBase):
    """Complete LLM configuration model with all fields."""
    id: str = Field(..., description="Unique configuration ID")
    
    # All configuration sections
    auth_config: AuthConfig = Field(..., description="Authentication configuration")
    model_parameters: ModelParameters = Field(..., description="Model parameters")
    request_format: RequestFormat = Field(..., description="Request format configuration")
    response_format: ResponseFormat = Field(..., description="Response format configuration")
    provider_limits: ProviderLimits = Field(..., description="Provider limits")
    cost_config: Optional[CostConfig] = Field(None, description="Cost calculation configuration")
    capability_config: CapabilityConfig = Field(..., description="Model capabilities")
    custom_fields: CustomFields = Field(..., description="Custom fields")
    
    # Usage statistics
    total_requests: int = Field(default=0, description="Total requests made")
    total_tokens: int = Field(default=0, description="Total tokens processed")
    total_cost: float = Field(default=0.0, description="Total cost incurred")
    last_used_at: Optional[datetime] = Field(None, description="Last usage timestamp")
    
    # Health and performance metrics
    success_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Success rate (0-1)")
    average_response_time: float = Field(default=0.0, ge=0.0, description="Average response time in seconds")
    error_count: int = Field(default=0, description="Total error count")
    last_error: Optional[str] = Field(None, description="Last error message")
    last_health_check: Optional[datetime] = Field(None, description="Last health check timestamp")
    
    # Configuration validation
    is_valid: bool = Field(default=False, description="Whether configuration is valid")
    validation_errors: List[str] = Field(default=[], description="Validation error messages")


class LLMConfigPreset(BaseModel):
    """Preset configuration for popular providers."""
    name: str = Field(..., description="Preset name")
    provider_name: str = Field(..., description="Provider name")
    description: str = Field(..., description="Preset description")
    template: LLMConfigCreate = Field(..., description="Template configuration")
    popular: bool = Field(default=False, description="Whether this is a popular preset")
    category: str = Field(default="general", description="Preset category")
    
    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "name": "openai_gpt4",
                    "provider_name": "OpenAI",
                    "description": "OpenAI GPT-4 configuration",
                    "popular": True,
                    "category": "chat"
                }
            ]
        }


class LLMConfigTest(BaseModel):
    """Configuration test request model."""
    config_id: Optional[str] = Field(None, description="Configuration ID to test")
    config: Optional[LLMConfigCreate] = Field(None, description="Configuration to test")
    test_prompt: str = Field(default="Hello, world!", description="Test prompt")
    expected_response_time: Optional[float] = Field(None, description="Expected response time in seconds")


class LLMConfigTestResult(BaseModel):
    """Configuration test result model."""
    success: bool = Field(..., description="Whether test was successful")
    response_time: float = Field(..., description="Response time in seconds")
    response_content: Optional[str] = Field(None, description="Response content")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    validation_errors: List[str] = Field(default=[], description="Configuration validation errors")
    connectivity_check: bool = Field(..., description="Whether API is reachable")
    authentication_check: bool = Field(..., description="Whether authentication is valid")
    
    # Performance metrics
    tokens_used: Optional[int] = Field(None, description="Tokens used in test")
    cost_estimate: Optional[float] = Field(None, description="Estimated cost for test")
    
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Test timestamp")


class LLMCapabilityDetection(BaseModel):
    """Model capability detection request."""
    config: LLMConfigCreate = Field(..., description="Configuration to analyze")
    perform_deep_analysis: bool = Field(default=False, description="Perform comprehensive capability detection")


class LLMCapabilityDetectionResult(BaseModel):
    """Model capability detection result."""
    detected_capabilities: CapabilityConfig = Field(..., description="Detected capabilities")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence in detection results")
    detection_methods: List[str] = Field(..., description="Methods used for detection")
    warnings: List[str] = Field(default=[], description="Detection warnings")
    recommendations: List[str] = Field(default=[], description="Configuration recommendations")


class LLMConfigImport(BaseModel):
    """Configuration import model."""
    configurations: List[LLMConfigCreate] = Field(..., description="Configurations to import")
    overwrite_existing: bool = Field(default=False, description="Whether to overwrite existing configurations")
    validate_before_import: bool = Field(default=True, description="Validate configurations before importing")


class LLMConfigExport(BaseModel):
    """Configuration export model."""
    config_ids: Optional[List[str]] = Field(None, description="Specific configuration IDs to export")
    include_credentials: bool = Field(default=False, description="Whether to include sensitive credentials")
    export_format: Literal["json", "yaml", "toml"] = Field(default="json", description="Export format")


class LLMConfigBackup(BaseModel):
    """Configuration backup model."""
    backup_name: str = Field(..., description="Backup name")
    configurations: List[LLMConfig] = Field(..., description="Configurations to backup")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Backup creation timestamp")
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="Backup metadata")


class LLMConfigInfo(BaseModel):
    """LLM configuration information model for lists."""
    id: str = Field(..., description="Configuration ID")
    name: str = Field(..., description="Configuration name")
    provider_name: str = Field(..., description="Provider name")
    model_name: str = Field(..., description="Model identifier")
    status: LLMConfigStatus = Field(..., description="Configuration status")
    created_at: datetime = Field(..., description="Creation timestamp")
    success_rate: float = Field(..., description="Success rate")
    total_requests: int = Field(..., description="Total requests made")
    
    class Config:
        use_enum_values = True


# Popular provider presets
PROVIDER_PRESETS = {
    "openai_gpt4": LLMConfigPreset(
        name="openai_gpt4",
        provider_name="OpenAI",
        description="OpenAI GPT-4 configuration with optimal settings",
        popular=True,
        category="chat",
        template=LLMConfigCreate(
            name="openai_gpt4",
            provider_name="OpenAI",
            model_name="gpt-4",
            api_endpoint="https://api.openai.com/v1/chat/completions",
            auth_config=AuthConfig(
                auth_type=AuthenticationType.BEARER_TOKEN,
                api_key=SecretStr("your-api-key-here")
            ),
            request_format=RequestFormat(
                messages_format=True,
                model_field_name="model"
            ),
            response_format=ResponseFormat(
                content_field_path="choices.0.message.content"
            ),
            capability_config=CapabilityConfig(
                supports_streaming=True,
                supports_functions=True,
                context_window_size=8192,
                supports_system_messages=True
            )
        )
    ),
    "anthropic_claude": LLMConfigPreset(
        name="anthropic_claude",
        provider_name="Anthropic",
        description="Anthropic Claude configuration",
        popular=True,
        category="chat",
        template=LLMConfigCreate(
            name="anthropic_claude",
            provider_name="Anthropic",
            model_name="claude-3-sonnet-20240229",
            api_endpoint="https://api.anthropic.com/v1/messages",
            auth_config=AuthConfig(
                auth_type=AuthenticationType.API_KEY_HEADER,
                header_name="x-api-key",
                api_key=SecretStr("your-api-key-here")
            ),
            request_format=RequestFormat(
                messages_format=True,
                additional_headers={"anthropic-version": "2023-06-01"}
            ),
            capability_config=CapabilityConfig(
                context_window_size=200000,
                supports_system_messages=True
            )
        )
    ),
    "huggingface_local": LLMConfigPreset(
        name="huggingface_local",
        provider_name="HuggingFace",
        description="Local HuggingFace model configuration",
        category="local",
        template=LLMConfigCreate(
            name="huggingface_local",
            provider_name="HuggingFace",
            model_name="microsoft/DialoGPT-medium",
            api_endpoint="http://localhost:8000/generate",
            auth_config=AuthConfig(auth_type=AuthenticationType.NO_AUTH),
            request_format=RequestFormat(
                prompt_field_name="inputs"
            ),
            response_format=ResponseFormat(
                content_field_path="generated_text"
            )
        )
    ),
    "ollama_local": LLMConfigPreset(
        name="ollama_local",
        provider_name="Ollama",
        description="Local Ollama model configuration",
        category="local",
        template=LLMConfigCreate(
            name="ollama_local",
            provider_name="Ollama",
            model_name="llama2",
            api_endpoint="http://localhost:11434/api/generate",
            auth_config=AuthConfig(auth_type=AuthenticationType.NO_AUTH),
            request_format=RequestFormat(
                prompt_field_name="prompt",
                model_field_name="model"
            ),
            response_format=ResponseFormat(
                content_field_path="response",
                streaming_format="jsonlines"
            ),
            capability_config=CapabilityConfig(
                supports_streaming=True,
                context_window_size=4096
            )
        )
    ),
    "cohere_command": LLMConfigPreset(
        name="cohere_command",
        provider_name="Cohere",
        description="Cohere Command model configuration",
        category="chat",
        template=LLMConfigCreate(
            name="cohere_command",
            provider_name="Cohere",
            model_name="command",
            api_endpoint="https://api.cohere.ai/v1/generate",
            auth_config=AuthConfig(
                auth_type=AuthenticationType.BEARER_TOKEN,
                api_key=SecretStr("your-api-key-here")
            ),
            request_format=RequestFormat(
                prompt_field_name="prompt",
                model_field_name="model"
            ),
            response_format=ResponseFormat(
                content_field_path="generations.0.text"
            )
        )
    )
}
