"""
Test Case Data Model - Pydantic models for test case representation.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum


class TestCaseDifficulty(str, Enum):
    """Test case difficulty levels."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXTREME = "extreme"


class TestCaseCategory(str, Enum):
    """Test case categories."""
    PROMPT_INJECTION = "prompt_injection"
    MALFORMED_DATA = "malformed_data"
    UNICODE_ATTACKS = "unicode_attacks"
    LONG_INPUTS = "long_inputs"
    ADVERSARIAL = "adversarial"
    JAILBREAK = "jailbreak"
    SAFETY = "safety"
    BIAS = "bias"
    HALLUCINATION = "hallucination"
    PERFORMANCE = "performance"


class TestCaseBase(BaseModel):
    """Base test case model."""
    name: str = Field(..., description="Test case name")
    description: Optional[str] = Field(None, description="Test case description")
    category: TestCaseCategory = Field(..., description="Test case category")
    difficulty: TestCaseDifficulty = Field(default=TestCaseDifficulty.MEDIUM, description="Difficulty level")
    tags: List[str] = Field(default=[], description="Test case tags")
    metadata: Dict[str, Any] = Field(default={}, description="Additional metadata")


class TestCaseCreate(TestCaseBase):
    """Test case creation model."""
    prompt: str = Field(..., description="Test prompt")
    expected_outcome: Optional[str] = Field(None, description="Expected test outcome")
    expected_response: Optional[str] = Field(None, description="Expected response pattern")
    max_tokens: Optional[int] = Field(default=1000, description="Maximum tokens for response")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0, description="Temperature parameter")
    
    @validator('prompt')
    def validate_prompt(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Prompt cannot be empty')
        if len(v) > 50000:  # Reasonable limit
            raise ValueError('Prompt too long (max 50000 characters)')
        return v.strip()


class TestCaseUpdate(BaseModel):
    """Test case update model."""
    name: Optional[str] = Field(None, description="Test case name")
    description: Optional[str] = Field(None, description="Test case description")
    category: Optional[TestCaseCategory] = Field(None, description="Test case category")
    difficulty: Optional[TestCaseDifficulty] = Field(None, description="Difficulty level")
    prompt: Optional[str] = Field(None, description="Test prompt")
    expected_outcome: Optional[str] = Field(None, description="Expected test outcome")
    expected_response: Optional[str] = Field(None, description="Expected response pattern")
    tags: Optional[List[str]] = Field(None, description="Test case tags")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens for response")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Temperature parameter")
    enabled: Optional[bool] = Field(None, description="Whether test case is enabled")


class TestCase(TestCaseBase):
    """Complete test case model."""
    id: str = Field(..., description="Unique test case identifier")
    prompt: str = Field(..., description="Test prompt")
    expected_outcome: Optional[str] = Field(None, description="Expected test outcome")
    expected_response: Optional[str] = Field(None, description="Expected response pattern")
    max_tokens: int = Field(default=1000, description="Maximum tokens for response")
    temperature: float = Field(default=0.7, description="Temperature parameter")
    enabled: bool = Field(default=True, description="Whether test case is enabled")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    created_by: Optional[str] = Field(None, description="Creator user ID")
    usage_count: int = Field(default=0, description="Number of times used")
    success_rate: Optional[float] = Field(None, description="Historical success rate")
    
    class Config:
        from_attributes = True


class TestCaseBatch(BaseModel):
    """Batch test case operations."""
    test_cases: List[TestCaseCreate] = Field(..., description="List of test cases to create")
    batch_name: Optional[str] = Field(None, description="Batch name")
    batch_description: Optional[str] = Field(None, description="Batch description")
    
    @validator('test_cases')
    def validate_batch_size(cls, v):
        if len(v) == 0:
            raise ValueError('Batch cannot be empty')
        if len(v) > 1000:  # Reasonable batch limit
            raise ValueError('Batch too large (max 1000 test cases)')
        return v


class TestCaseTemplate(BaseModel):
    """Test case template for generation."""
    template_name: str = Field(..., description="Template name")
    template_type: TestCaseCategory = Field(..., description="Template category")
    prompt_template: str = Field(..., description="Prompt template with placeholders")
    variables: Dict[str, Any] = Field(default={}, description="Template variables")
    generation_config: Dict[str, Any] = Field(default={}, description="Generation configuration")
    
    @validator('prompt_template')
    def validate_template(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Prompt template cannot be empty')
        return v.strip()


class TestCaseStats(BaseModel):
    """Test case statistics."""
    total_cases: int = Field(..., description="Total number of test cases")
    cases_by_category: Dict[str, int] = Field(..., description="Cases grouped by category")
    cases_by_difficulty: Dict[str, int] = Field(..., description="Cases grouped by difficulty")
    enabled_cases: int = Field(..., description="Number of enabled test cases")
    disabled_cases: int = Field(..., description="Number of disabled test cases")
    average_prompt_length: float = Field(..., description="Average prompt length")
    most_used_tags: List[Dict[str, Union[str, int]]] = Field(..., description="Most frequently used tags")


class TestCaseValidationResult(BaseModel):
    """Test case validation result."""
    is_valid: bool = Field(..., description="Whether test case is valid")
    errors: List[str] = Field(default=[], description="Validation errors")
    warnings: List[str] = Field(default=[], description="Validation warnings")
    suggestions: List[str] = Field(default=[], description="Improvement suggestions")
