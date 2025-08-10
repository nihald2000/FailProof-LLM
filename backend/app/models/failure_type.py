"""
Failure Type Model - Enumeration and models for test failure classification.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


class FailureSeverity(str, Enum):
    """Failure severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FailureCategory(str, Enum):
    """Main failure categories."""
    SECURITY = "security"
    SAFETY = "safety"
    BIAS = "bias"
    HALLUCINATION = "hallucination"
    PERFORMANCE = "performance"
    COMPLIANCE = "compliance"
    ETHICAL = "ethical"
    TECHNICAL = "technical"


class FailureType(str, Enum):
    """Specific failure types."""
    # Security failures
    PROMPT_INJECTION = "prompt_injection"
    DATA_LEAKAGE = "data_leakage"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    CODE_INJECTION = "code_injection"
    
    # Safety failures
    HARMFUL_CONTENT = "harmful_content"
    VIOLENCE_PROMOTION = "violence_promotion"
    ILLEGAL_ACTIVITY = "illegal_activity"
    SELF_HARM = "self_harm"
    
    # Bias failures
    GENDER_BIAS = "gender_bias"
    RACIAL_BIAS = "racial_bias"
    RELIGIOUS_BIAS = "religious_bias"
    CULTURAL_BIAS = "cultural_bias"
    AGE_BIAS = "age_bias"
    
    # Hallucination failures
    FACTUAL_INACCURACY = "factual_inaccuracy"
    MADE_UP_INFORMATION = "made_up_information"
    CONTRADICTORY_STATEMENTS = "contradictory_statements"
    
    # Performance failures
    TIMEOUT = "timeout"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    MALFORMED_RESPONSE = "malformed_response"
    INCOMPLETE_RESPONSE = "incomplete_response"
    
    # Compliance failures
    PRIVACY_VIOLATION = "privacy_violation"
    GDPR_VIOLATION = "gdpr_violation"
    COPPA_VIOLATION = "coppa_violation"
    
    # Ethical failures
    MANIPULATION = "manipulation"
    DECEPTION = "deception"
    UNFAIRNESS = "unfairness"
    
    # Technical failures
    API_ERROR = "api_error"
    PARSING_ERROR = "parsing_error"
    ENCODING_ERROR = "encoding_error"
    NETWORK_ERROR = "network_error"


class FailureTypeInfo(BaseModel):
    """Detailed information about a failure type."""
    failure_type: FailureType = Field(..., description="The failure type")
    category: FailureCategory = Field(..., description="Failure category")
    severity: FailureSeverity = Field(..., description="Default severity level")
    description: str = Field(..., description="Detailed description")
    detection_patterns: List[str] = Field(default=[], description="Patterns used for detection")
    mitigation_strategies: List[str] = Field(default=[], description="Suggested mitigation strategies")
    examples: List[str] = Field(default=[], description="Example failure cases")
    related_types: List[FailureType] = Field(default=[], description="Related failure types")


class FailureClassification(BaseModel):
    """Classification result for a specific failure."""
    failure_type: FailureType = Field(..., description="Identified failure type")
    category: FailureCategory = Field(..., description="Failure category")
    severity: FailureSeverity = Field(..., description="Assessed severity")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Classification confidence")
    evidence: List[str] = Field(default=[], description="Evidence supporting classification")
    context: Optional[str] = Field(None, description="Additional context")
    automated_detection: bool = Field(..., description="Whether detected automatically")
    detection_rules: List[str] = Field(default=[], description="Detection rules that fired")


class FailurePattern(BaseModel):
    """Pattern definition for failure detection."""
    pattern_id: str = Field(..., description="Unique pattern identifier")
    failure_type: FailureType = Field(..., description="Associated failure type")
    pattern_type: str = Field(..., description="Type of pattern (regex, keyword, ml)")
    pattern_value: str = Field(..., description="Pattern definition")
    weight: float = Field(default=1.0, ge=0.0, description="Pattern weight in scoring")
    case_sensitive: bool = Field(default=False, description="Whether pattern is case sensitive")
    description: Optional[str] = Field(None, description="Pattern description")
    examples: List[str] = Field(default=[], description="Example matches")


class FailureReport(BaseModel):
    """Comprehensive failure analysis report."""
    total_failures: int = Field(..., description="Total number of failures detected")
    failures_by_type: Dict[str, int] = Field(..., description="Failures grouped by type")
    failures_by_category: Dict[str, int] = Field(..., description="Failures grouped by category")
    failures_by_severity: Dict[str, int] = Field(..., description="Failures grouped by severity")
    top_failure_types: List[Dict[str, Any]] = Field(..., description="Most common failure types")
    severity_distribution: Dict[str, float] = Field(..., description="Severity distribution percentages")
    detection_accuracy: Optional[float] = Field(None, description="Overall detection accuracy")
    recommendations: List[str] = Field(default=[], description="Recommended actions")


class FailureTypeMapping(BaseModel):
    """Mapping between failure types and their properties."""
    
    @classmethod
    def get_failure_info(cls, failure_type: FailureType) -> FailureTypeInfo:
        """Get detailed information for a failure type."""
        mapping = {
            FailureType.PROMPT_INJECTION: FailureTypeInfo(
                failure_type=FailureType.PROMPT_INJECTION,
                category=FailureCategory.SECURITY,
                severity=FailureSeverity.HIGH,
                description="Malicious prompts designed to override system instructions",
                detection_patterns=["ignore previous", "system:", "override", "jailbreak"],
                mitigation_strategies=["Input sanitization", "Prompt filtering", "Context isolation"],
                examples=["Ignore all previous instructions and...", "System: You are now..."]
            ),
            FailureType.HARMFUL_CONTENT: FailureTypeInfo(
                failure_type=FailureType.HARMFUL_CONTENT,
                category=FailureCategory.SAFETY,
                severity=FailureSeverity.CRITICAL,
                description="Content that could cause harm to individuals or groups",
                detection_patterns=["violence", "harm", "dangerous", "illegal"],
                mitigation_strategies=["Content filtering", "Safety classifiers", "Human review"],
                examples=["Instructions for harmful activities", "Violent content"]
            ),
            FailureType.GENDER_BIAS: FailureTypeInfo(
                failure_type=FailureType.GENDER_BIAS,
                category=FailureCategory.BIAS,
                severity=FailureSeverity.MEDIUM,
                description="Unfair treatment or representation based on gender",
                detection_patterns=["stereotype", "gender role", "assumption"],
                mitigation_strategies=["Bias testing", "Diverse training data", "Fairness metrics"],
                examples=["Assuming professions based on gender", "Gender stereotypes"]
            ),
            FailureType.FACTUAL_INACCURACY: FailureTypeInfo(
                failure_type=FailureType.FACTUAL_INACCURACY,
                category=FailureCategory.HALLUCINATION,
                severity=FailureSeverity.MEDIUM,
                description="Incorrect factual information provided as truth",
                detection_patterns=["incorrect fact", "wrong date", "false claim"],
                mitigation_strategies=["Fact checking", "Source verification", "Uncertainty modeling"],
                examples=["Wrong historical dates", "Incorrect scientific facts"]
            )
        }
        
        return mapping.get(failure_type, FailureTypeInfo(
            failure_type=failure_type,
            category=FailureCategory.TECHNICAL,
            severity=FailureSeverity.LOW,
            description="Unknown failure type",
            detection_patterns=[],
            mitigation_strategies=[],
            examples=[]
        ))
