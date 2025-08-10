"""
Test Result Data Models for Breakpoint LLM Stress Testing Platform.
Comprehensive Pydantic models with validation and serialization.
"""
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic.types import PositiveInt, NonNegativeFloat, constr


class TestStatus(str, Enum):
    """Test execution status enumeration."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    RETRY = "retry"


class FailureType(str, Enum):
    """Failure classification types."""
    PROMPT_INJECTION = "prompt_injection"
    MALFORMED_OUTPUT = "malformed_output"
    POLICY_VIOLATION = "policy_violation"
    TIMEOUT_ERROR = "timeout_error"
    API_ERROR = "api_error"
    VALIDATION_ERROR = "validation_error"
    UNKNOWN_ERROR = "unknown_error"


class VulnerabilityLevel(str, Enum):
    """Vulnerability severity levels."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ComplianceStatus(str, Enum):
    """Compliance check status."""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    NOT_APPLICABLE = "not_applicable"


class BaseResultModel(BaseModel):
    """Base model with common fields for all result types."""
    
    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique identifier")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(default=None, description="Last update timestamp")
    version: int = Field(default=1, description="Model version for compatibility")
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        validate_assignment = True
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }
    
    def update_timestamp(self):
        """Update the last modified timestamp."""
        self.updated_at = datetime.now()


class PerformanceMetrics(BaseModel):
    """Performance metrics for test execution."""
    
    latency_ms: Optional[NonNegativeFloat] = Field(
        default=None, 
        description="Response latency in milliseconds"
    )
    throughput_rps: Optional[NonNegativeFloat] = Field(
        default=None, 
        description="Throughput in requests per second"
    )
    cpu_usage_percent: Optional[float] = Field(
        default=None, 
        ge=0, le=100, 
        description="CPU usage percentage during execution"
    )
    memory_usage_mb: Optional[NonNegativeFloat] = Field(
        default=None, 
        description="Memory usage in megabytes"
    )
    network_bytes_sent: Optional[int] = Field(
        default=None, 
        ge=0, 
        description="Network bytes sent"
    )
    network_bytes_received: Optional[int] = Field(
        default=None, 
        ge=0, 
        description="Network bytes received"
    )
    token_processing_rate: Optional[NonNegativeFloat] = Field(
        default=None, 
        description="Tokens processed per second"
    )
    
    @field_validator('cpu_usage_percent')
    def validate_cpu_usage(cls, v):
        """Validate CPU usage is within reasonable bounds."""
        if v is not None and (v < 0 or v > 100):
            raise ValueError("CPU usage must be between 0 and 100 percent")
        return v


class TokenUsage(BaseModel):
    """Token usage information."""
    
    prompt_tokens: Optional[int] = Field(
        default=None, 
        ge=0, 
        description="Number of tokens in the prompt"
    )
    completion_tokens: Optional[int] = Field(
        default=None, 
        ge=0, 
        description="Number of tokens in the completion"
    )
    total_tokens: Optional[int] = Field(
        default=None, 
        ge=0, 
        description="Total number of tokens used"
    )
    
    @model_validator(mode='after')
    def validate_token_consistency(self):
        """Validate token count consistency."""
        prompt = self.prompt_tokens
        completion = self.completion_tokens
        total = self.total_tokens
        
        if prompt is not None and completion is not None:
            calculated_total = prompt + completion
            if total is not None and total != calculated_total:
                self.total_tokens = calculated_total
        elif total is not None and prompt is not None:
            self.completion_tokens = total - prompt
        elif total is not None and completion is not None:
            self.prompt_tokens = total - completion
            
        return self


class CostEstimate(BaseModel):
    """Cost estimation for API usage."""
    
    prompt_cost: Optional[float] = Field(
        default=None, 
        ge=0, 
        description="Cost for prompt tokens"
    )
    completion_cost: Optional[float] = Field(
        default=None, 
        ge=0, 
        description="Cost for completion tokens"
    )
    total_cost: Optional[float] = Field(
        default=None, 
        ge=0, 
        description="Total estimated cost"
    )
    currency: str = Field(default="USD", description="Currency for cost estimates")
    rate_per_token: Optional[float] = Field(
        default=None, 
        ge=0, 
        description="Rate per token for this model"
    )
    
    @model_validator(mode='after')
    def calculate_total_cost(self):
        """Calculate total cost if components are provided."""
        prompt_cost = self.prompt_cost or 0
        completion_cost = self.completion_cost or 0
        
        if prompt_cost or completion_cost:
            self.total_cost = prompt_cost + completion_cost
            
        return self


class SecurityAnalysis(BaseModel):
    """Security analysis results."""
    
    vulnerability_detected: bool = Field(
        default=False, 
        description="Whether vulnerabilities were detected"
    )
    vulnerability_type: Optional[str] = Field(
        default=None, 
        description="Type of vulnerability detected"
    )
    severity_level: VulnerabilityLevel = Field(
        default=VulnerabilityLevel.NONE, 
        description="Severity level of detected issues"
    )
    attack_vector: Optional[str] = Field(
        default=None, 
        description="Attack vector used in the test"
    )
    exploitation_risk: Optional[float] = Field(
        default=None, 
        ge=0, le=1, 
        description="Risk score for exploitation (0-1)"
    )
    mitigation_suggestions: List[str] = Field(
        default_factory=list, 
        description="Suggested mitigations"
    )
    confidence_score: Optional[float] = Field(
        default=None, 
        ge=0, le=1, 
        description="Confidence in the security analysis (0-1)"
    )


class ComplianceCheck(BaseModel):
    """Individual compliance check result."""
    
    check_name: str = Field(description="Name of the compliance check")
    status: ComplianceStatus = Field(description="Status of the compliance check")
    description: Optional[str] = Field(
        default=None, 
        description="Description of what was checked"
    )
    details: Optional[str] = Field(
        default=None, 
        description="Detailed results or failure reasons"
    )
    severity: Optional[str] = Field(
        default=None, 
        description="Severity if check failed"
    )
    remediation: Optional[str] = Field(
        default=None, 
        description="Suggested remediation steps"
    )


class ComplianceAnalysis(BaseModel):
    """Compliance analysis results."""
    
    overall_status: ComplianceStatus = Field(
        description="Overall compliance status"
    )
    checks: List[ComplianceCheck] = Field(
        default_factory=list, 
        description="Individual compliance checks"
    )
    policy_violations: List[str] = Field(
        default_factory=list, 
        description="List of policy violations"
    )
    content_warnings: List[str] = Field(
        default_factory=list, 
        description="Content warning flags"
    )
    safety_score: Optional[float] = Field(
        default=None, 
        ge=0, le=1, 
        description="Overall safety score (0-1)"
    )
    
    @model_validator(mode='after')
    def determine_overall_status(self):
        """Determine overall status based on individual checks."""
        checks = self.checks or []
        
        if not checks:
            self.overall_status = ComplianceStatus.NOT_APPLICABLE
            return self
        
        statuses = [check.status for check in checks]
        
        if ComplianceStatus.FAIL in statuses:
            self.overall_status = ComplianceStatus.FAIL
        elif ComplianceStatus.WARNING in statuses:
            self.overall_status = ComplianceStatus.WARNING
        else:
            self.overall_status = ComplianceStatus.PASS
            
        return self


class ExecutionMetadata(BaseModel):
    """Execution metadata and context."""
    
    user_id: Optional[str] = Field(default=None, description="User who initiated the test")
    session_id: Optional[str] = Field(default=None, description="Test session identifier")
    correlation_id: Optional[str] = Field(default=None, description="Request correlation ID")
    test_suite_id: Optional[str] = Field(default=None, description="Test suite identifier")
    environment: Optional[str] = Field(default=None, description="Execution environment")
    user_agent: Optional[str] = Field(default=None, description="Client user agent")
    ip_address: Optional[str] = Field(default=None, description="Client IP address")
    execution_node: Optional[str] = Field(default=None, description="Execution node/worker")
    retry_count: int = Field(default=0, ge=0, description="Number of retry attempts")
    parent_execution_id: Optional[str] = Field(
        default=None, 
        description="Parent execution ID for retries"
    )
    tags: List[str] = Field(default_factory=list, description="Custom tags")
    custom_fields: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Custom metadata fields"
    )


class TestResult(BaseResultModel):
    """Main test result model with comprehensive tracking."""
    
    # Core identification
    test_case_id: str = Field(description="ID of the test case that was executed")
    execution_id: Optional[str] = Field(
        default=None, 
        description="Unique execution identifier"
    )
    
    # Model and provider information
    model_name: str = Field(description="Name of the LLM model used")
    provider: str = Field(description="LLM provider (openai, anthropic, etc.)")
    model_version: Optional[str] = Field(
        default=None, 
        description="Specific model version if available"
    )
    
    # Execution timing
    execution_timestamp: datetime = Field(
        default_factory=datetime.now, 
        description="When the test was executed"
    )
    started_at: Optional[datetime] = Field(
        default=None, 
        description="When execution started"
    )
    completed_at: Optional[datetime] = Field(
        default=None, 
        description="When execution completed"
    )
    
    # Status and results
    status: TestStatus = Field(description="Current execution status")
    success: bool = Field(default=False, description="Whether the test was successful")
    
    # Request and response data
    prompt_text: str = Field(description="The prompt that was sent to the model")
    response_text: Optional[str] = Field(
        default=None, 
        description="The model's response text"
    )
    response_metadata: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Additional response metadata"
    )
    
    # Performance metrics
    response_time_ms: Optional[NonNegativeFloat] = Field(
        default=None, 
        description="Response time in milliseconds"
    )
    performance_metrics: Optional[PerformanceMetrics] = Field(
        default=None, 
        description="Detailed performance metrics"
    )
    
    # Token usage and cost
    token_usage: Optional[TokenUsage] = Field(
        default=None, 
        description="Token usage information"
    )
    cost_estimate: Optional[CostEstimate] = Field(
        default=None, 
        description="Cost estimation for this execution"
    )
    
    # Analysis results
    failure_type: Optional[FailureType] = Field(
        default=None, 
        description="Type of failure if test failed"
    )
    failure_reason: Optional[str] = Field(
        default=None, 
        description="Detailed failure reason"
    )
    confidence_score: Optional[float] = Field(
        default=None, 
        ge=0, le=1, 
        description="Confidence in the analysis results"
    )
    
    # Security and compliance
    security_analysis: Optional[SecurityAnalysis] = Field(
        default=None, 
        description="Security analysis results"
    )
    compliance_analysis: Optional[ComplianceAnalysis] = Field(
        default=None, 
        description="Compliance analysis results"
    )
    
    # Error information
    error_message: Optional[str] = Field(
        default=None, 
        description="Error message if execution failed"
    )
    error_code: Optional[str] = Field(
        default=None, 
        description="Error code for categorization"
    )
    stack_trace: Optional[str] = Field(
        default=None, 
        description="Stack trace for debugging"
    )
    
    # Metadata and context
    metadata: ExecutionMetadata = Field(
        default_factory=ExecutionMetadata, 
        description="Execution metadata"
    )
    
    # Computed properties
    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate execution duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    @property
    def is_completed(self) -> bool:
        """Check if execution is completed (success or failure)."""
        return self.status in [TestStatus.COMPLETED, TestStatus.FAILED, TestStatus.CANCELLED]
    
    @property
    def has_vulnerabilities(self) -> bool:
        """Check if any vulnerabilities were detected."""
        return (self.security_analysis and 
                self.security_analysis.vulnerability_detected)
    
    @property
    def compliance_passed(self) -> bool:
        """Check if compliance checks passed."""
        return (self.compliance_analysis and 
                self.compliance_analysis.overall_status == ComplianceStatus.PASS)
    
    # Validation methods
    @model_validator(mode='after')
    def validate_response_time(self):
        """Validate response time consistency."""
        if self.response_time_ms is None and self.started_at and self.completed_at:
            self.response_time_ms = (self.completed_at - self.started_at).total_seconds() * 1000
        return self
    
    @model_validator(mode='after')
    def determine_success(self):
        """Determine success based on status and other factors."""
        if self.status == TestStatus.COMPLETED:
            # Check for vulnerabilities or compliance failures
            has_critical_vuln = (self.security_analysis and 
                                self.security_analysis.vulnerability_detected and 
                                self.security_analysis.severity_level == VulnerabilityLevel.CRITICAL)
            
            has_compliance_fail = (self.compliance_analysis and 
                                 self.compliance_analysis.overall_status == ComplianceStatus.FAIL)
            
            self.success = not (has_critical_vuln or has_compliance_fail)
        else:
            self.success = self.status == TestStatus.COMPLETED
        
        return self
    
    @model_validator(mode='after')
    def validate_consistency(self):
        """Validate overall model consistency."""
        # If status is failed, ensure we have error information
        if self.status == TestStatus.FAILED:
            if not self.error_message and not self.failure_type:
                self.error_message = "Test failed without specific error details"
                self.failure_type = FailureType.UNKNOWN_ERROR
        
        # If we have error information, ensure status reflects it
        elif self.error_message or self.failure_type:
            if self.status not in [TestStatus.FAILED, TestStatus.TIMEOUT]:
                self.status = TestStatus.FAILED
        
        return self
    
    # Serialization methods
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper serialization."""
        return json.loads(self.json())
    
    def to_csv_row(self) -> Dict[str, Any]:
        """Convert to flat dictionary suitable for CSV export."""
        return {
            'id': self.id,
            'test_case_id': self.test_case_id,
            'execution_timestamp': self.execution_timestamp.isoformat(),
            'status': self.status.value,
            'success': self.success,
            'model_name': self.model_name,
            'provider': self.provider,
            'response_time_ms': self.response_time_ms,
            'prompt_length': len(self.prompt_text),
            'response_length': len(self.response_text) if self.response_text else 0,
            'total_tokens': self.token_usage.total_tokens if self.token_usage else None,
            'total_cost': self.cost_estimate.total_cost if self.cost_estimate else None,
            'has_vulnerabilities': self.has_vulnerabilities,
            'compliance_passed': self.compliance_passed,
            'failure_type': self.failure_type.value if self.failure_type else None,
            'error_message': self.error_message,
        }
    
    # Factory methods
    @classmethod
    def create_from_execution(cls, test_case_id: str, model_name: str, 
                            provider: str, prompt_text: str, **kwargs) -> 'TestResult':
        """Create a new test result from execution parameters."""
        return cls(
            test_case_id=test_case_id,
            model_name=model_name,
            provider=provider,
            prompt_text=prompt_text,
            status=TestStatus.PENDING,
            **kwargs
        )
    
    # Update methods
    def mark_started(self):
        """Mark the test as started."""
        self.status = TestStatus.RUNNING
        self.started_at = datetime.now()
        self.update_timestamp()
    
    def mark_completed(self, response_text: str, **kwargs):
        """Mark the test as completed with results."""
        self.status = TestStatus.COMPLETED
        self.completed_at = datetime.now()
        self.response_text = response_text
        
        # Update any provided fields
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        self.update_timestamp()
    
    def mark_failed(self, error_message: str, failure_type: FailureType = FailureType.UNKNOWN_ERROR):
        """Mark the test as failed with error information."""
        self.status = TestStatus.FAILED
        self.completed_at = datetime.now()
        self.error_message = error_message
        self.failure_type = failure_type
        self.success = False
        self.update_timestamp()
    
    def add_security_analysis(self, analysis: SecurityAnalysis):
        """Add security analysis results."""
        self.security_analysis = analysis
        self.update_timestamp()
    
    def add_compliance_analysis(self, analysis: ComplianceAnalysis):
        """Add compliance analysis results."""
        self.compliance_analysis = analysis
        self.update_timestamp()


class TestResultSummary(BaseModel):
    """Summary statistics for a collection of test results."""
    
    total_tests: PositiveInt = Field(description="Total number of tests")
    successful_tests: int = Field(ge=0, description="Number of successful tests")
    failed_tests: int = Field(ge=0, description="Number of failed tests")
    success_rate: float = Field(ge=0, le=1, description="Success rate (0-1)")
    
    average_response_time_ms: Optional[float] = Field(
        default=None, 
        ge=0, 
        description="Average response time"
    )
    total_cost: Optional[float] = Field(
        default=None, 
        ge=0, 
        description="Total cost for all tests"
    )
    total_tokens: Optional[int] = Field(
        default=None, 
        ge=0, 
        description="Total tokens used"
    )
    
    vulnerability_count: int = Field(
        ge=0, 
        description="Number of tests with vulnerabilities"
    )
    critical_vulnerability_count: int = Field(
        ge=0, 
        description="Number of tests with critical vulnerabilities"
    )
    compliance_failure_count: int = Field(
        ge=0, 
        description="Number of compliance failures"
    )
    
    model_distribution: Dict[str, int] = Field(
        default_factory=dict, 
        description="Distribution by model"
    )
    provider_distribution: Dict[str, int] = Field(
        default_factory=dict, 
        description="Distribution by provider"
    )
    failure_type_distribution: Dict[str, int] = Field(
        default_factory=dict, 
        description="Distribution by failure type"
    )
    
    @model_validator(mode='after')
    def calculate_success_rate(self):
        """Calculate success rate from counts."""
        if self.total_tests > 0:
            self.success_rate = self.successful_tests / self.total_tests
        else:
            self.success_rate = 0.0
        return self
    
    @classmethod
    def from_results(cls, results: List[TestResult]) -> 'TestResultSummary':
        """Create summary from a list of test results."""
        if not results:
            return cls(
                total_tests=0,
                successful_tests=0,
                failed_tests=0,
                vulnerability_count=0,
                critical_vulnerability_count=0,
                compliance_failure_count=0
            )
        
        total_tests = len(results)
        successful_tests = len([r for r in results if r.success])
        failed_tests = total_tests - successful_tests
        
        # Calculate averages
        response_times = [r.response_time_ms for r in results if r.response_time_ms is not None]
        avg_response_time = sum(response_times) / len(response_times) if response_times else None
        
        # Calculate totals
        total_cost = sum([
            r.cost_estimate.total_cost for r in results 
            if r.cost_estimate and r.cost_estimate.total_cost
        ])
        
        total_tokens = sum([
            r.token_usage.total_tokens for r in results 
            if r.token_usage and r.token_usage.total_tokens
        ])
        
        # Count vulnerabilities
        vulnerability_count = len([r for r in results if r.has_vulnerabilities])
        critical_vulnerability_count = len([
            r for r in results 
            if (r.security_analysis and 
                r.security_analysis.vulnerability_detected and 
                r.security_analysis.severity_level == VulnerabilityLevel.CRITICAL)
        ])
        
        # Count compliance failures
        compliance_failure_count = len([
            r for r in results 
            if (r.compliance_analysis and 
                r.compliance_analysis.overall_status == ComplianceStatus.FAIL)
        ])
        
        # Calculate distributions
        model_dist = {}
        provider_dist = {}
        failure_type_dist = {}
        
        for result in results:
            # Model distribution
            model_dist[result.model_name] = model_dist.get(result.model_name, 0) + 1
            
            # Provider distribution
            provider_dist[result.provider] = provider_dist.get(result.provider, 0) + 1
            
            # Failure type distribution
            if result.failure_type:
                failure_type_dist[result.failure_type.value] = failure_type_dist.get(result.failure_type.value, 0) + 1
        
        return cls(
            total_tests=total_tests,
            successful_tests=successful_tests,
            failed_tests=failed_tests,
            average_response_time_ms=avg_response_time,
            total_cost=total_cost if total_cost > 0 else None,
            total_tokens=total_tokens if total_tokens > 0 else None,
            vulnerability_count=vulnerability_count,
            critical_vulnerability_count=critical_vulnerability_count,
            compliance_failure_count=compliance_failure_count,
            model_distribution=model_dist,
            provider_distribution=provider_dist,
            failure_type_distribution=failure_type_dist
        )
