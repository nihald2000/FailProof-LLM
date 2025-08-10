"""
Universal LLM Integration Service for Breakpoint Platform.
Generic service that can work with any LLM model regardless of provider through flexible configuration.
"""

import asyncio
import json
import logging
import time
import uuid
import ssl
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, AsyncGenerator, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlparse
import re

import aiohttp
import httpx
from aiohttp import ClientSession, ClientTimeout, ClientError, ClientConnectorError
from pydantic import ValidationError

# Optional imports for specific providers
try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

try:
    from anthropic import AsyncAnthropic
except ImportError:
    AsyncAnthropic = None

from app.models.llm_config import (
    LLMConfig, LLMConfigCreate, AuthConfig, AuthenticationType, 
    ModelParameters, RequestFormat, ResponseFormat, ProviderLimits,
    CostConfig, CapabilityConfig, BillingModel, LLMConfigTestResult,
    LLMCapabilityDetectionResult, LLMProvider, ModelCapability, LLMConfigStatus
)
from app.core.config import settings


logger = logging.getLogger(__name__)


@dataclass
class LLMRequest:
    """Universal LLM request format."""
    prompt: str
    config: LLMConfig
    messages: Optional[List[Dict[str, str]]] = None
    system_message: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stop_sequences: Optional[List[str]] = None
    stream: bool = False
    custom_parameters: Optional[Dict[str, Any]] = None
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class LLMResponse:
    """Universal LLM response format."""
    content: str
    request_id: str
    config_id: str
    provider_name: str
    model_name: str
    
    # Usage information
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    
    # Performance metrics
    response_time: float = 0.0
    cost: Optional[float] = None
    
    # Metadata
    finish_reason: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class LLMError:
    """Universal LLM error format."""
    error_type: str
    error_message: str
    provider_error_code: Optional[str] = None
    retry_after: Optional[int] = None
    is_retryable: bool = False
    request_id: Optional[str] = None


class RateLimiter:
    """Generic rate limiter for API requests."""
    
    def __init__(self, config_id: str, limits: ProviderLimits):
        self.config_id = config_id
        self.limits = limits
        self.requests_per_minute = []
        self.requests_per_hour = []
        self.requests_per_day = []
        self.concurrent_requests = 0
        self.lock = asyncio.Lock()
    
    async def wait_if_needed(self) -> None:
        """Wait if rate limits would be exceeded."""
        async with self.lock:
            now = datetime.utcnow()
            
            # Clean old requests
            cutoff_minute = now - timedelta(minutes=1)
            cutoff_hour = now - timedelta(hours=1)
            cutoff_day = now - timedelta(days=1)
            
            self.requests_per_minute = [t for t in self.requests_per_minute if t > cutoff_minute]
            self.requests_per_hour = [t for t in self.requests_per_hour if t > cutoff_hour]
            self.requests_per_day = [t for t in self.requests_per_day if t > cutoff_day]
            
            # Check limits
            wait_time = 0
            
            if (self.limits.rate_limit_per_minute and 
                len(self.requests_per_minute) >= self.limits.rate_limit_per_minute):
                wait_time = max(wait_time, 60 - (now - self.requests_per_minute[0]).total_seconds())
            
            if (self.limits.rate_limit_per_hour and 
                len(self.requests_per_hour) >= self.limits.rate_limit_per_hour):
                wait_time = max(wait_time, 3600 - (now - self.requests_per_hour[0]).total_seconds())
            
            if (self.limits.rate_limit_per_day and 
                len(self.requests_per_day) >= self.limits.rate_limit_per_day):
                wait_time = max(wait_time, 86400 - (now - self.requests_per_day[0]).total_seconds())
            
            if wait_time > 0:
                logger.warning(f"Rate limit reached for {self.config_id}, waiting {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)
    
    async def acquire(self) -> None:
        """Acquire rate limit slot."""
        await self.wait_if_needed()
        
        async with self.lock:
            # Wait for concurrent request limit
            while self.concurrent_requests >= self.limits.max_concurrent_requests:
                await asyncio.sleep(0.1)
            
            now = datetime.utcnow()
            self.requests_per_minute.append(now)
            self.requests_per_hour.append(now)
            self.requests_per_day.append(now)
            self.concurrent_requests += 1
    
    def release(self) -> None:
        """Release concurrent request slot."""
        self.concurrent_requests = max(0, self.concurrent_requests - 1)


class RequestFormatter:
    """Universal request formatter for different API schemas."""
    
    @staticmethod
    def format_request(request: LLMRequest) -> Dict[str, Any]:
        """Format request according to provider specifications."""
        config = request.config
        format_config = config.request_format
        
        # Start with base request
        formatted_request = {}
        
        # Add model
        formatted_request[format_config.model_field_name] = config.model_name
        
        # Handle prompt vs messages format
        if format_config.messages_format:
            messages = []
            
            # Add system message if supported and provided
            if request.system_message and config.capability_config.supports_system_messages:
                messages.append({
                    "role": format_config.message_role_system,
                    "content": request.system_message
                })
            
            # Add conversation history if provided
            if request.messages:
                messages.extend(request.messages)
            
            # Add current prompt
            messages.append({
                "role": format_config.message_role_user,
                "content": request.prompt
            })
            
            formatted_request["messages"] = messages
        else:
            # Use simple prompt format
            formatted_request[format_config.prompt_field_name] = request.prompt
        
        # Add parameters
        params = RequestFormatter._get_parameters(request)
        
        if format_config.parameters_nested and format_config.parameters_field_name:
            formatted_request[format_config.parameters_field_name] = params
        else:
            formatted_request.update(params)
        
        # Apply custom request template if provided
        if format_config.custom_request_template:
            try:
                template = json.loads(format_config.custom_request_template)
                formatted_request = RequestFormatter._apply_template(formatted_request, template)
            except json.JSONDecodeError:
                logger.warning(f"Invalid custom request template for {config.id}")
        
        return formatted_request
    
    @staticmethod
    def _get_parameters(request: LLMRequest) -> Dict[str, Any]:
        """Extract parameters from request and config."""
        config = request.config
        params = config.model_parameters
        
        result = {}
        
        # Add standard parameters if they have values
        if request.max_tokens is not None:
            result["max_tokens"] = request.max_tokens
        elif params.max_tokens is not None:
            result["max_tokens"] = params.max_tokens
        
        if request.temperature is not None:
            result["temperature"] = request.temperature
        elif params.temperature is not None:
            result["temperature"] = params.temperature
        
        if request.top_p is not None:
            result["top_p"] = request.top_p
        elif params.top_p is not None:
            result["top_p"] = params.top_p
        
        # Add other parameters from config
        if params.top_k is not None:
            result["top_k"] = params.top_k
        if params.frequency_penalty is not None:
            result["frequency_penalty"] = params.frequency_penalty
        if params.presence_penalty is not None:
            result["presence_penalty"] = params.presence_penalty
        if params.repetition_penalty is not None:
            result["repetition_penalty"] = params.repetition_penalty
        
        # Handle stop sequences
        stop_sequences = request.stop_sequences or params.stop_sequences
        if stop_sequences:
            result["stop"] = stop_sequences
        
        # Add streaming parameter
        if request.stream:
            result["stream"] = True
        
        # Add custom parameters
        if params.custom_parameters:
            result.update(params.custom_parameters)
        
        if request.custom_parameters:
            result.update(request.custom_parameters)
        
        return result
    
    @staticmethod
    def _apply_template(data: Dict[str, Any], template: Dict[str, Any]) -> Dict[str, Any]:
        """Apply custom request template."""
        # Simple template application - in production, this could be more sophisticated
        result = template.copy()
        
        # Replace placeholders in template with actual data
        def replace_placeholders(obj, data_dict):
            if isinstance(obj, dict):
                return {k: replace_placeholders(v, data_dict) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_placeholders(item, data_dict) for item in obj]
            elif isinstance(obj, str) and obj.startswith("{{") and obj.endswith("}}"):
                key = obj[2:-2].strip()
                return data_dict.get(key, obj)
            else:
                return obj
        
        return replace_placeholders(result, data)


class ResponseParser:
    """Universal response parser for different API response formats."""
    
    @staticmethod
    def parse_response(raw_response: Dict[str, Any], config: LLMConfig, 
                      request_id: str) -> LLMResponse:
        """Parse response according to provider specifications."""
        format_config = config.response_format
        
        # Extract content using JSONPath
        content = ResponseParser._extract_field(raw_response, format_config.content_field_path)
        
        # Extract usage information if available
        usage_info = None
        if format_config.usage_field_path:
            usage_info = ResponseParser._extract_field(raw_response, format_config.usage_field_path)
        
        # Parse usage tokens
        prompt_tokens = None
        completion_tokens = None
        total_tokens = None
        
        if usage_info:
            if isinstance(usage_info, dict):
                prompt_tokens = usage_info.get("prompt_tokens")
                completion_tokens = usage_info.get("completion_tokens")
                total_tokens = usage_info.get("total_tokens")
        
        # Extract finish reason
        finish_reason = None
        if "choices" in raw_response and raw_response["choices"]:
            choice = raw_response["choices"][0]
            finish_reason = choice.get("finish_reason")
        
        return LLMResponse(
            content=content or "",
            request_id=request_id,
            config_id=config.id,
            provider_name=config.provider_name,
            model_name=config.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            finish_reason=finish_reason,
            raw_response=raw_response
        )
    
    @staticmethod
    def parse_error(raw_response: Dict[str, Any], config: LLMConfig) -> LLMError:
        """Parse error response according to provider specifications."""
        format_config = config.response_format
        
        error_message = ResponseParser._extract_field(raw_response, format_config.error_field_path)
        
        # Extract provider error code if available
        error_code = None
        if "error" in raw_response:
            error_data = raw_response["error"]
            if isinstance(error_data, dict):
                error_code = error_data.get("code") or error_data.get("type")
        
        # Determine if error is retryable
        is_retryable = ResponseParser._is_retryable_error(error_code, error_message)
        
        return LLMError(
            error_type="api_error",
            error_message=error_message or "Unknown error",
            provider_error_code=error_code,
            is_retryable=is_retryable
        )
    
    @staticmethod
    def _extract_field(data: Dict[str, Any], path: str) -> Any:
        """Extract field using JSONPath-like syntax."""
        if not path or not data:
            return None
        
        parts = path.split(".")
        current = data
        
        for part in parts:
            if part.isdigit():
                # Array index
                idx = int(part)
                if isinstance(current, list) and 0 <= idx < len(current):
                    current = current[idx]
                else:
                    return None
            else:
                # Object key
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return None
        
        return current
    
    @staticmethod
    def _is_retryable_error(error_code: Optional[str], error_message: str) -> bool:
        """Determine if error is retryable."""
        if not error_code and not error_message:
            return False
        
        retryable_codes = {
            "rate_limit_exceeded", "too_many_requests", "server_error",
            "timeout", "connection_error", "service_unavailable"
        }
        
        retryable_messages = [
            "rate limit", "too many requests", "server error", "timeout",
            "connection", "unavailable", "overloaded"
        ]
        
        error_text = (error_code or "").lower() + " " + error_message.lower()
        
        return (error_code and error_code.lower() in retryable_codes) or \
               any(msg in error_text for msg in retryable_messages)


class CostCalculator:
    """Universal cost calculator for different billing models."""
    
    @staticmethod
    def calculate_cost(response: LLMResponse, config: LLMConfig) -> float:
        """Calculate cost based on configuration."""
        if not config.cost_config:
            return 0.0
        
        cost_config = config.cost_config
        
        if cost_config.billing_model == BillingModel.PER_TOKEN:
            return CostCalculator._calculate_token_cost(response, cost_config)
        elif cost_config.billing_model == BillingModel.PER_REQUEST:
            return cost_config.cost_per_request or 0.0
        elif cost_config.billing_model == BillingModel.PER_MINUTE:
            # Estimate based on response time
            minutes = response.response_time / 60.0
            return (cost_config.cost_per_minute or 0.0) * minutes
        elif cost_config.billing_model == BillingModel.FIXED_RATE:
            return 0.0  # Fixed costs are handled separately
        else:
            return 0.0
    
    @staticmethod
    def _calculate_token_cost(response: LLMResponse, cost_config: CostConfig) -> float:
        """Calculate token-based cost."""
        total_cost = 0.0
        
        if response.prompt_tokens and cost_config.cost_per_input_token:
            total_cost += response.prompt_tokens * cost_config.cost_per_input_token
        
        if response.completion_tokens and cost_config.cost_per_output_token:
            total_cost += response.completion_tokens * cost_config.cost_per_output_token
        
        # Fallback to total tokens with single rate
        if total_cost == 0.0 and response.total_tokens and cost_config.cost_per_token:
            total_cost = response.total_tokens * cost_config.cost_per_token
        
        return total_cost
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop_sequences: Optional[List[str]] = None
    stream: bool = False
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class LLMResponse:
    """Standardized LLM response format."""
    text: str
    provider: LLMProvider
    model: str
    usage: Optional[Dict[str, int]] = None
    finish_reason: Optional[str] = None
    latency_ms: Optional[float] = None
    cost_estimate: Optional[float] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ModelInfo:
    """Model information and capabilities."""
    name: str
    provider: LLMProvider
    max_tokens: int
    cost_per_token: Optional[float] = None
    capabilities: List[ModelCapability] = None
    supports_streaming: bool = False
    supports_function_calling: bool = False


class LLMError(Exception):
    """Base LLM service error."""
    pass


class RateLimitError(LLMError):
    """Rate limit exceeded error."""
    pass


class AuthenticationError(LLMError):
    """Authentication failed error."""
    pass


class ModelNotFoundError(LLMError):
    """Model not found error."""
    pass


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    def __init__(self, api_key: str, base_url: str, timeout: int = 60):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.session: Optional[ClientSession] = None
        self._rate_limiter = {}  # Simple rate limiting tracker
        
    async def __aenter__(self):
        """Async context manager entry."""
        timeout = ClientTimeout(total=self.timeout)
        self.session = ClientSession(
            timeout=timeout,
            headers={"User-Agent": f"Breakpoint/{settings.APP_VERSION}"}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response for the given request."""
        pass
    
    @abstractmethod
    async def stream_generate(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        """Stream generate response for the given request."""
        pass
    
    @abstractmethod
    def get_model_info(self, model: str) -> ModelInfo:
        """Get model information and capabilities."""
        pass
    
    async def _check_rate_limit(self, model: str) -> bool:
        """Check if rate limit allows the request."""
        current_time = time.time()
        key = f"{self.__class__.__name__}:{model}"
        
        if key not in self._rate_limiter:
            self._rate_limiter[key] = []
        
        # Clean old entries
        self._rate_limiter[key] = [
            timestamp for timestamp in self._rate_limiter[key]
            if current_time - timestamp < 60  # 1 minute window
        ]
        
        # Check if under limit (simplified - real implementation would be more sophisticated)
        if len(self._rate_limiter[key]) >= 60:  # 60 requests per minute
            return False
        
        self._rate_limiter[key].append(current_time)
        return True


class OpenAIClient(BaseLLMClient):
    """OpenAI API client implementation."""
    
    def __init__(self):
        super().__init__(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL,
            timeout=settings.OPENAI_TIMEOUT
        )
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout
        )
        
        self.models = {
            "gpt-4": ModelInfo(
                name="gpt-4",
                provider=LLMProvider.OPENAI,
                max_tokens=8192,
                cost_per_token=0.00003,
                capabilities=[ModelCapability.CHAT, ModelCapability.FUNCTION_CALLING, ModelCapability.STREAMING],
                supports_streaming=True,
                supports_function_calling=True
            ),
            "gpt-4-turbo": ModelInfo(
                name="gpt-4-turbo",
                provider=LLMProvider.OPENAI,
                max_tokens=128000,
                cost_per_token=0.00001,
                capabilities=[ModelCapability.CHAT, ModelCapability.FUNCTION_CALLING, ModelCapability.STREAMING],
                supports_streaming=True,
                supports_function_calling=True
            ),
            "gpt-3.5-turbo": ModelInfo(
                name="gpt-3.5-turbo",
                provider=LLMProvider.OPENAI,
                max_tokens=4096,
                cost_per_token=0.0000015,
                capabilities=[ModelCapability.CHAT, ModelCapability.FUNCTION_CALLING, ModelCapability.STREAMING],
                supports_streaming=True,
                supports_function_calling=True
            )
        }
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response using OpenAI API."""
        start_time = time.time()
        
        try:
            if not await self._check_rate_limit(request.model):
                raise RateLimitError(f"Rate limit exceeded for model {request.model}")
            
            # Prepare messages for chat completion
            messages = [{"role": "user", "content": request.prompt}]
            
            response = await self.client.chat.completions.create(
                model=request.model,
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                frequency_penalty=request.frequency_penalty,
                presence_penalty=request.presence_penalty,
                stop=request.stop_sequences,
                stream=False
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Calculate cost estimate
            usage = response.usage
            cost_estimate = None
            if usage and request.model in self.models:
                model_info = self.models[request.model]
                cost_estimate = (usage.prompt_tokens + usage.completion_tokens) * model_info.cost_per_token
            
            return LLMResponse(
                text=response.choices[0].message.content,
                provider=LLMProvider.OPENAI,
                model=request.model,
                usage={
                    "prompt_tokens": usage.prompt_tokens if usage else 0,
                    "completion_tokens": usage.completion_tokens if usage else 0,
                    "total_tokens": usage.total_tokens if usage else 0
                } if usage else None,
                finish_reason=response.choices[0].finish_reason,
                latency_ms=latency_ms,
                cost_estimate=cost_estimate,
                metadata=request.metadata
            )
            
        except Exception as exc:
            latency_ms = (time.time() - start_time) * 1000
            error_msg = self._parse_openai_error(exc)
            
            logger.error(
                f"OpenAI API error: {error_msg}",
                extra={
                    "model": request.model,
                    "latency_ms": latency_ms,
                    "error": str(exc)
                }
            )
            
            return LLMResponse(
                text="",
                provider=LLMProvider.OPENAI,
                model=request.model,
                latency_ms=latency_ms,
                error=error_msg,
                metadata=request.metadata
            )
    
    async def stream_generate(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        """Stream generate response using OpenAI API."""
        try:
            if not await self._check_rate_limit(request.model):
                raise RateLimitError(f"Rate limit exceeded for model {request.model}")
            
            messages = [{"role": "user", "content": request.prompt}]
            
            stream = await self.client.chat.completions.create(
                model=request.model,
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                frequency_penalty=request.frequency_penalty,
                presence_penalty=request.presence_penalty,
                stop=request.stop_sequences,
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as exc:
            error_msg = self._parse_openai_error(exc)
            logger.error(f"OpenAI streaming error: {error_msg}")
            yield f"Error: {error_msg}"
    
    def get_model_info(self, model: str) -> ModelInfo:
        """Get OpenAI model information."""
        return self.models.get(model, ModelInfo(
            name=model,
            provider=LLMProvider.OPENAI,
            max_tokens=4096,
            capabilities=[ModelCapability.CHAT]
        ))
    
    def _parse_openai_error(self, exc: Exception) -> str:
        """Parse OpenAI API errors."""
        if "rate_limit" in str(exc).lower():
            return "Rate limit exceeded"
        elif "authentication" in str(exc).lower():
            return "Authentication failed"
        elif "not_found" in str(exc).lower():
            return "Model not found"
        else:
            return f"API error: {str(exc)}"


class AnthropicClient(BaseLLMClient):
    """Anthropic Claude API client implementation."""
    
    def __init__(self):
        super().__init__(
            api_key=settings.ANTHROPIC_API_KEY,
            base_url=settings.ANTHROPIC_BASE_URL,
            timeout=settings.ANTHROPIC_TIMEOUT
        )
        self.client = AsyncAnthropic(api_key=self.api_key)
        
        self.models = {
            "claude-3-5-sonnet-20241022": ModelInfo(
                name="claude-3-5-sonnet-20241022",
                provider=LLMProvider.ANTHROPIC,
                max_tokens=200000,
                cost_per_token=0.000003,
                capabilities=[ModelCapability.CHAT, ModelCapability.STREAMING],
                supports_streaming=True
            ),
            "claude-3-haiku-20240307": ModelInfo(
                name="claude-3-haiku-20240307",
                provider=LLMProvider.ANTHROPIC,
                max_tokens=200000,
                cost_per_token=0.00000025,
                capabilities=[ModelCapability.CHAT, ModelCapability.STREAMING],
                supports_streaming=True
            ),
            "claude-3-opus-20240229": ModelInfo(
                name="claude-3-opus-20240229",
                provider=LLMProvider.ANTHROPIC,
                max_tokens=200000,
                cost_per_token=0.000015,
                capabilities=[ModelCapability.CHAT, ModelCapability.STREAMING],
                supports_streaming=True
            )
        }
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response using Anthropic API."""
        start_time = time.time()
        
        try:
            if not await self._check_rate_limit(request.model):
                raise RateLimitError(f"Rate limit exceeded for model {request.model}")
            
            response = await self.client.messages.create(
                model=request.model,
                max_tokens=request.max_tokens or 1024,
                messages=[{"role": "user", "content": request.prompt}],
                temperature=request.temperature,
                top_p=request.top_p,
                stop_sequences=request.stop_sequences
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Calculate cost estimate
            usage = response.usage
            cost_estimate = None
            if usage and request.model in self.models:
                model_info = self.models[request.model]
                cost_estimate = (usage.input_tokens + usage.output_tokens) * model_info.cost_per_token
            
            return LLMResponse(
                text=response.content[0].text if response.content else "",
                provider=LLMProvider.ANTHROPIC,
                model=request.model,
                usage={
                    "prompt_tokens": usage.input_tokens if usage else 0,
                    "completion_tokens": usage.output_tokens if usage else 0,
                    "total_tokens": (usage.input_tokens + usage.output_tokens) if usage else 0
                } if usage else None,
                finish_reason=response.stop_reason,
                latency_ms=latency_ms,
                cost_estimate=cost_estimate,
                metadata=request.metadata
            )
            
        except Exception as exc:
            latency_ms = (time.time() - start_time) * 1000
            error_msg = self._parse_anthropic_error(exc)
            
            logger.error(
                f"Anthropic API error: {error_msg}",
                extra={
                    "model": request.model,
                    "latency_ms": latency_ms,
                    "error": str(exc)
                }
            )
            
            return LLMResponse(
                text="",
                provider=LLMProvider.ANTHROPIC,
                model=request.model,
                latency_ms=latency_ms,
                error=error_msg,
                metadata=request.metadata
            )
    
    async def stream_generate(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        """Stream generate response using Anthropic API."""
        try:
            if not await self._check_rate_limit(request.model):
                raise RateLimitError(f"Rate limit exceeded for model {request.model}")
            
            async with self.client.messages.stream(
                model=request.model,
                max_tokens=request.max_tokens or 1024,
                messages=[{"role": "user", "content": request.prompt}],
                temperature=request.temperature,
                top_p=request.top_p,
                stop_sequences=request.stop_sequences
            ) as stream:
                async for text in stream.text_stream:
                    yield text
                    
        except Exception as exc:
            error_msg = self._parse_anthropic_error(exc)
            logger.error(f"Anthropic streaming error: {error_msg}")
            yield f"Error: {error_msg}"
    
    def get_model_info(self, model: str) -> ModelInfo:
        """Get Anthropic model information."""
        return self.models.get(model, ModelInfo(
            name=model,
            provider=LLMProvider.ANTHROPIC,
            max_tokens=200000,
            capabilities=[ModelCapability.CHAT]
        ))
    
    def _parse_anthropic_error(self, exc: Exception) -> str:
        """Parse Anthropic API errors."""
        if "rate_limit" in str(exc).lower():
            return "Rate limit exceeded"
        elif "authentication" in str(exc).lower():
            return "Authentication failed"
        elif "not_found" in str(exc).lower():
            return "Model not found"
        else:
            return f"API error: {str(exc)}"


class HuggingFaceClient(BaseLLMClient):
    """HuggingFace Inference API client implementation."""
    
    def __init__(self):
        super().__init__(
            api_key=settings.HUGGINGFACE_API_KEY,
            base_url=settings.HUGGINGFACE_BASE_URL,
            timeout=settings.HUGGINGFACE_TIMEOUT
        )
        
        self.models = {
            "meta-llama/Llama-2-7b-chat-hf": ModelInfo(
                name="meta-llama/Llama-2-7b-chat-hf",
                provider=LLMProvider.HUGGINGFACE,
                max_tokens=4096,
                capabilities=[ModelCapability.TEXT_GENERATION]
            ),
            "mistralai/Mistral-7B-Instruct-v0.1": ModelInfo(
                name="mistralai/Mistral-7B-Instruct-v0.1",
                provider=LLMProvider.HUGGINGFACE,
                max_tokens=8192,
                capabilities=[ModelCapability.TEXT_GENERATION]
            )
        }
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response using HuggingFace API."""
        start_time = time.time()
        
        try:
            if not await self._check_rate_limit(request.model):
                raise RateLimitError(f"Rate limit exceeded for model {request.model}")
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "inputs": request.prompt,
                "parameters": {
                    "max_new_tokens": request.max_tokens or 512,
                    "temperature": request.temperature or 0.7,
                    "top_p": request.top_p or 0.9,
                    "do_sample": True,
                    "return_full_text": False
                }
            }
            
            url = f"{self.base_url}/models/{request.model}"
            
            async with self.session.post(url, json=payload, headers=headers) as response:
                if response.status == 200:
                    result = await response.json()
                    latency_ms = (time.time() - start_time) * 1000
                    
                    if isinstance(result, list) and result:
                        generated_text = result[0].get("generated_text", "")
                    else:
                        generated_text = ""
                    
                    return LLMResponse(
                        text=generated_text,
                        provider=LLMProvider.HUGGINGFACE,
                        model=request.model,
                        latency_ms=latency_ms,
                        metadata=request.metadata
                    )
                else:
                    error_text = await response.text()
                    raise ClientError(f"HTTP {response.status}: {error_text}")
                    
        except Exception as exc:
            latency_ms = (time.time() - start_time) * 1000
            error_msg = self._parse_huggingface_error(exc)
            
            logger.error(
                f"HuggingFace API error: {error_msg}",
                extra={
                    "model": request.model,
                    "latency_ms": latency_ms,
                    "error": str(exc)
                }
            )
            
            return LLMResponse(
                text="",
                provider=LLMProvider.HUGGINGFACE,
                model=request.model,
                latency_ms=latency_ms,
                error=error_msg,
                metadata=request.metadata
            )
    
    async def stream_generate(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        """Stream generate (not supported by HuggingFace Inference API)."""
        response = await self.generate(request)
        if response.error:
            yield f"Error: {response.error}"
        else:
            yield response.text
    
    def get_model_info(self, model: str) -> ModelInfo:
        """Get HuggingFace model information."""
        return self.models.get(model, ModelInfo(
            name=model,
            provider=LLMProvider.HUGGINGFACE,
            max_tokens=2048,
            capabilities=[ModelCapability.TEXT_GENERATION]
        ))
    
    def _parse_huggingface_error(self, exc: Exception) -> str:
        """Parse HuggingFace API errors."""
        if "rate" in str(exc).lower():
            return "Rate limit exceeded"
        elif "unauthorized" in str(exc).lower():
            return "Authentication failed"
        elif "not found" in str(exc).lower():
            return "Model not found"
        else:
            return f"API error: {str(exc)}"


class GenericLLMClient:
    """Universal LLM client that can work with any provider."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.rate_limiter = RateLimiter(config.id, config.provider_limits)
        self.session: Optional[ClientSession] = None
        self._setup_session()
    
    def _setup_session(self):
        """Setup HTTP session with configuration."""
        timeout = ClientTimeout(total=self.config.provider_limits.timeout_seconds)
        
        connector = aiohttp.TCPConnector(
            limit=self.config.provider_limits.connection_pool_size,
            limit_per_host=self.config.provider_limits.max_concurrent_requests,
            keepalive_timeout=self.config.provider_limits.keep_alive_timeout,
            ssl=ssl.create_default_context()
        )
        
        self.session = ClientSession(
            timeout=timeout,
            connector=connector,
            headers=self._get_default_headers()
        )
    
    def _get_default_headers(self) -> Dict[str, str]:
        """Get default headers for requests."""
        headers = {
            "Content-Type": self.config.request_format.content_type,
            "User-Agent": f"Breakpoint-LLM-Tester/1.0 ({self.config.provider_name})"
        }
        
        # Add authentication headers
        auth_config = self.config.auth_config
        
        if auth_config.auth_type == AuthenticationType.BEARER_TOKEN:
            if auth_config.api_key:
                headers["Authorization"] = f"Bearer {auth_config.api_key.get_secret_value()}"
        
        elif auth_config.auth_type == AuthenticationType.API_KEY_HEADER:
            if auth_config.api_key and auth_config.header_name:
                headers[auth_config.header_name] = auth_config.api_key.get_secret_value()
        
        elif auth_config.auth_type == AuthenticationType.CUSTOM_HEADERS:
            if auth_config.custom_headers:
                headers.update(auth_config.custom_headers)
        
        # Add additional headers from request format
        if self.config.request_format.additional_headers:
            headers.update(self.config.request_format.additional_headers)
        
        return headers
    
    async def send_request(self, request: LLMRequest) -> LLMResponse:
        """Send request to LLM provider with retry logic."""
        await self.rate_limiter.acquire()
        
        try:
            return await self._send_request_with_retry(request)
        finally:
            self.rate_limiter.release()
    
    async def _send_request_with_retry(self, request: LLMRequest) -> LLMResponse:
        """Send request with exponential backoff retry."""
        last_error = None
        
        for attempt in range(self.config.provider_limits.retry_attempts + 1):
            try:
                start_time = time.time()
                response = await self._send_single_request(request)
                response_time = time.time() - start_time
                
                response.response_time = response_time
                response.cost = CostCalculator.calculate_cost(response, self.config)
                
                return response
                
            except Exception as e:
                last_error = e
                
                if attempt < self.config.provider_limits.retry_attempts:
                    # Calculate retry delay
                    delay = self.config.provider_limits.retry_delay_seconds
                    if self.config.provider_limits.exponential_backoff:
                        delay *= (2 ** attempt)
                        delay = min(delay, self.config.provider_limits.max_retry_delay)
                    
                    logger.warning(
                        f"Request failed (attempt {attempt + 1}), retrying in {delay:.2f}s: {str(e)}"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Request failed after {attempt + 1} attempts: {str(e)}")
        
        # If we get here, all retries failed
        if isinstance(last_error, aiohttp.ClientError):
            raise LLMError(
                error_type="network_error",
                error_message=str(last_error),
                is_retryable=True,
                request_id=request.request_id
            )
        else:
            raise LLMError(
                error_type="unknown_error",
                error_message=str(last_error),
                is_retryable=False,
                request_id=request.request_id
            )
    
    async def _send_single_request(self, request: LLMRequest) -> LLMResponse:
        """Send a single request to the provider."""
        if not self.session:
            self._setup_session()
        
        # Format request according to provider specifications
        request_data = RequestFormatter.format_request(request)
        
        # Add query parameters for auth if needed
        params = {}
        if (self.config.auth_config.auth_type == AuthenticationType.QUERY_PARAM and
            self.config.auth_config.api_key and self.config.auth_config.query_param_name):
            params[self.config.auth_config.query_param_name] = \
                self.config.auth_config.api_key.get_secret_value()
        
        # Send request
        async with self.session.request(
            method=self.config.request_format.method.value,
            url=str(self.config.api_endpoint),
            json=request_data,
            params=params
        ) as response:
            
            response_data = await response.json()
            
            # Check for API errors
            if response.status >= 400:
                error = ResponseParser.parse_error(response_data, self.config)
                raise LLMError(
                    error_type="api_error",
                    error_message=error.error_message,
                    provider_error_code=error.provider_error_code,
                    is_retryable=error.is_retryable,
                    request_id=request.request_id
                )
            
            # Parse successful response
            return ResponseParser.parse_response(response_data, self.config, request.request_id)
    
    async def send_streaming_request(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        """Send streaming request to LLM provider."""
        if not self.config.capability_config.supports_streaming:
            raise LLMError(
                error_type="capability_error",
                error_message=f"Model {self.config.model_name} does not support streaming",
                is_retryable=False,
                request_id=request.request_id
            )
        
        await self.rate_limiter.acquire()
        
        try:
            async for chunk in self._send_streaming_request_with_retry(request):
                yield chunk
        finally:
            self.rate_limiter.release()
    
    async def _send_streaming_request_with_retry(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        """Send streaming request with retry logic."""
        request.stream = True
        
        if not self.session:
            self._setup_session()
        
        request_data = RequestFormatter.format_request(request)
        
        params = {}
        if (self.config.auth_config.auth_type == AuthenticationType.QUERY_PARAM and
            self.config.auth_config.api_key and self.config.auth_config.query_param_name):
            params[self.config.auth_config.query_param_name] = \
                self.config.auth_config.api_key.get_secret_value()
        
        async with self.session.request(
            method=self.config.request_format.method.value,
            url=str(self.config.api_endpoint),
            json=request_data,
            params=params
        ) as response:
            
            if response.status >= 400:
                error_data = await response.json()
                error = ResponseParser.parse_error(error_data, self.config)
                raise LLMError(
                    error_type="api_error",
                    error_message=error.error_message,
                    provider_error_code=error.provider_error_code,
                    is_retryable=error.is_retryable,
                    request_id=request.request_id
                )
            
            # Parse streaming response
            async for chunk in self._parse_streaming_response(response):
                yield chunk
    
    async def _parse_streaming_response(self, response) -> AsyncGenerator[str, None]:
        """Parse streaming response according to provider format."""
        format_config = self.config.response_format
        
        if format_config.streaming_format == "sse":
            # Server-sent events format
            async for line in response.content:
                line = line.decode('utf-8').strip()
                
                if line.startswith(format_config.streaming_data_prefix):
                    data = line[len(format_config.streaming_data_prefix):]
                    
                    if data == format_config.streaming_end_marker:
                        break
                    
                    try:
                        chunk_data = json.loads(data)
                        content = ResponseParser._extract_field(
                            chunk_data, format_config.streaming_content_path
                        )
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue
        
        elif format_config.streaming_format == "jsonlines":
            # JSON lines format
            async for line in response.content:
                line = line.decode('utf-8').strip()
                
                if line:
                    try:
                        chunk_data = json.loads(line)
                        content = ResponseParser._extract_field(
                            chunk_data, format_config.streaming_content_path
                        )
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue
    
    async def test_connectivity(self) -> LLMConfigTestResult:
        """Test connectivity and configuration validity."""
        test_request = LLMRequest(
            prompt="Hello, world!",
            config=self.config,
            max_tokens=10
        )
        
        start_time = time.time()
        
        try:
            response = await self.send_request(test_request)
            response_time = time.time() - start_time
            
            return LLMConfigTestResult(
                success=True,
                response_time=response_time,
                response_content=response.content,
                connectivity_check=True,
                authentication_check=True,
                tokens_used=response.total_tokens,
                cost_estimate=response.cost
            )
            
        except LLMError as e:
            response_time = time.time() - start_time
            
            return LLMConfigTestResult(
                success=False,
                response_time=response_time,
                error_message=e.error_message,
                connectivity_check=e.error_type != "network_error",
                authentication_check=e.provider_error_code != "unauthorized"
            )
        
        except Exception as e:
            response_time = time.time() - start_time
            
            return LLMConfigTestResult(
                success=False,
                response_time=response_time,
                error_message=str(e),
                connectivity_check=False,
                authentication_check=False
            )
    
    async def detect_capabilities(self) -> LLMCapabilityDetectionResult:
        """Detect model capabilities through API introspection."""
        detected_capabilities = CapabilityConfig()
        detection_methods = []
        warnings = []
        recommendations = []
        confidence_score = 0.5
        
        # Test streaming support
        try:
            test_request = LLMRequest(
                prompt="Test streaming",
                config=self.config,
                stream=True,
                max_tokens=5
            )
            
            stream_content = ""
            async for chunk in self.send_streaming_request(test_request):
                stream_content += chunk
                break  # Just test if streaming works
            
            if stream_content:
                detected_capabilities.supports_streaming = True
                detection_methods.append("streaming_test")
                confidence_score += 0.1
        
        except Exception:
            detected_capabilities.supports_streaming = False
            warnings.append("Streaming test failed - model may not support streaming")
        
        # Test function calling (simplified detection)
        if "function" in self.config.model_name.lower() or "tool" in self.config.model_name.lower():
            detected_capabilities.supports_functions = True
            detection_methods.append("name_heuristic")
            confidence_score += 0.1
        
        # Estimate context window based on model name
        model_name = self.config.model_name.lower()
        if "gpt-4" in model_name:
            if "32k" in model_name or "turbo" in model_name:
                detected_capabilities.context_window_size = 32768
            else:
                detected_capabilities.context_window_size = 8192
        elif "gpt-3.5" in model_name:
            detected_capabilities.context_window_size = 4096
        elif "claude" in model_name:
            detected_capabilities.context_window_size = 200000
        elif "llama" in model_name:
            detected_capabilities.context_window_size = 4096
        
        detection_methods.append("model_name_heuristic")
        
        # Add recommendations
        if not detected_capabilities.supports_streaming and self.config.capability_config.supports_streaming:
            recommendations.append("Consider disabling streaming in configuration")
        
        if detected_capabilities.context_window_size != self.config.capability_config.context_window_size:
            recommendations.append(
                f"Consider updating context window size to {detected_capabilities.context_window_size}"
            )
        
        return LLMCapabilityDetectionResult(
            detected_capabilities=detected_capabilities,
            confidence_score=min(confidence_score, 1.0),
            detection_methods=detection_methods,
            warnings=warnings,
            recommendations=recommendations
        )
    
    async def close(self):
        """Close the HTTP session."""
        if self.session:
            await self.session.close()


class UniversalLLMService:
    """Universal LLM service that manages multiple configurations."""
    
    def __init__(self):
        self.clients: Dict[str, GenericLLMClient] = {}
        self.health_monitors: Dict[str, asyncio.Task] = {}
    
    def register_config(self, config: LLMConfig) -> None:
        """Register a new LLM configuration."""
        if config.id in self.clients:
            # Close existing client
            asyncio.create_task(self.clients[config.id].close())
        
        self.clients[config.id] = GenericLLMClient(config)
        
        # Start health monitoring if enabled
        if config.status == "active":
            self._start_health_monitoring(config.id)
    
    def unregister_config(self, config_id: str) -> None:
        """Unregister an LLM configuration."""
        if config_id in self.clients:
            asyncio.create_task(self.clients[config_id].close())
            del self.clients[config_id]
        
        if config_id in self.health_monitors:
            self.health_monitors[config_id].cancel()
            del self.health_monitors[config_id]
    
    def get_client(self, config_id: str) -> Optional[GenericLLMClient]:
        """Get client for a specific configuration."""
        return self.clients.get(config_id)
    
    async def send_request(self, config_id: str, request: LLMRequest) -> LLMResponse:
        """Send request using specific configuration."""
        client = self.get_client(config_id)
        if not client:
            raise LLMError(
                error_type="configuration_error",
                error_message=f"No client found for configuration {config_id}",
                is_retryable=False
            )
        
        # Update request with correct config
        request.config = client.config
        
        return await client.send_request(request)
    
    async def send_streaming_request(self, config_id: str, request: LLMRequest) -> AsyncGenerator[str, None]:
        """Send streaming request using specific configuration."""
        client = self.get_client(config_id)
        if not client:
            raise LLMError(
                error_type="configuration_error",
                error_message=f"No client found for configuration {config_id}",
                is_retryable=False
            )
        
        request.config = client.config
        
        async for chunk in client.send_streaming_request(request):
            yield chunk
    
    async def test_configuration(self, config: LLMConfig) -> LLMConfigTestResult:
        """Test a configuration without registering it."""
        client = GenericLLMClient(config)
        
        try:
            return await client.test_connectivity()
        finally:
            await client.close()
    
    async def detect_model_capabilities(self, config: LLMConfig) -> LLMCapabilityDetectionResult:
        """Detect capabilities for a configuration."""
        client = GenericLLMClient(config)
        
        try:
            return await client.detect_capabilities()
        finally:
            await client.close()
    
    def _start_health_monitoring(self, config_id: str) -> None:
        """Start health monitoring for a configuration."""
        async def health_check():
            while True:
                try:
                    await asyncio.sleep(300)  # Check every 5 minutes
                    
                    client = self.get_client(config_id)
                    if client:
                        result = await client.test_connectivity()
                        
                        # Update health metrics in database (would be implemented)
                        logger.info(f"Health check for {config_id}: {'OK' if result.success else 'FAILED'}")
                
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Health check failed for {config_id}: {str(e)}")
        
        task = asyncio.create_task(health_check())
        self.health_monitors[config_id] = task
    
    async def get_performance_metrics(self, config_id: str) -> Dict[str, Any]:
        """Get performance metrics for a configuration."""
        # This would typically query a database or metrics store
        # For now, return basic information
        client = self.get_client(config_id)
        if not client:
            return {}
        
        return {
            "config_id": config_id,
            "provider_name": client.config.provider_name,
            "model_name": client.config.model_name,
            "status": client.config.status.value,
            "concurrent_requests": client.rate_limiter.concurrent_requests
        }
    
    async def shutdown(self):
        """Shutdown all clients and monitoring tasks."""
        # Cancel all health monitoring tasks
        for task in self.health_monitors.values():
            task.cancel()
        
        # Close all clients
        for client in self.clients.values():
            await client.close()
        
        self.clients.clear()
        self.health_monitors.clear()


# Global service instance
universal_llm_service = UniversalLLMService()

# Compatibility aliases
LLMService = UniversalLLMService
llm_service = universal_llm_service
