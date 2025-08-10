"""
Database Models - SQLAlchemy ORM models for data persistence.
"""

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, Float, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from uuid import uuid4
import datetime

Base = declarative_base()


class TestSession(Base):
    """Test session model."""
    __tablename__ = "test_sessions"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    name = Column(String(255), nullable=False)
    description = Column(Text)
    model_name = Column(String(100), nullable=False)
    provider = Column(String(50), nullable=False)
    status = Column(String(20), default="pending")
    total_tests = Column(Integer, default=0)
    completed_tests = Column(Integer, default=0)
    passed_tests = Column(Integer, default=0)
    failed_tests = Column(Integer, default=0)
    start_time = Column(DateTime, default=func.now())
    end_time = Column(DateTime)
    duration = Column(Float)  # seconds
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    test_results = relationship("TestResult", back_populates="session")


class TestResult(Base):
    """Individual test result model."""
    __tablename__ = "test_results"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    session_id = Column(String, ForeignKey("test_sessions.id"), nullable=False)
    test_type = Column(String(50), nullable=False)
    test_category = Column(String(50))
    test_difficulty = Column(String(20))
    prompt = Column(Text, nullable=False)
    response = Column(Text)
    expected_outcome = Column(String(20))
    actual_outcome = Column(String(20))
    passed = Column(Boolean)
    error_message = Column(Text)
    execution_time = Column(Float)  # seconds
    tokens_used = Column(Integer)
    cost = Column(Float)
    
    # Analysis results
    vulnerability_detected = Column(Boolean, default=False)
    risk_level = Column(String(20))
    confidence_score = Column(Float)
    analysis_details = Column(JSON)
    
    # Timestamps
    started_at = Column(DateTime, default=func.now())
    completed_at = Column(DateTime)
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    session = relationship("TestSession", back_populates="test_results")


class ModelConfig(Base):
    """LLM model configuration model."""
    __tablename__ = "model_configs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    name = Column(String(100), nullable=False, unique=True)
    provider = Column(String(50), nullable=False)
    endpoint_url = Column(String(500))
    api_key_ref = Column(String(100))  # Reference to secure storage
    max_tokens = Column(Integer, default=4096)
    temperature = Column(Float, default=0.7)
    enabled = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


class Report(Base):
    """Generated report model."""
    __tablename__ = "reports"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    name = Column(String(255), nullable=False)
    format = Column(String(20), nullable=False)
    session_ids = Column(JSON)  # List of session IDs
    file_path = Column(String(500))
    file_size = Column(Integer)
    status = Column(String(20), default="generating")
    include_details = Column(Boolean, default=True)
    include_charts = Column(Boolean, default=False)
    created_at = Column(DateTime, default=func.now())
    completed_at = Column(DateTime)


class SystemMetrics(Base):
    """System performance metrics model."""
    __tablename__ = "system_metrics"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=func.now())
    cpu_usage = Column(Float)
    memory_usage = Column(Float)
    active_sessions = Column(Integer, default=0)
    total_requests = Column(Integer, default=0)
    error_rate = Column(Float, default=0.0)
    avg_response_time = Column(Float)
    
    # LLM provider metrics
    openai_requests = Column(Integer, default=0)
    anthropic_requests = Column(Integer, default=0)
    huggingface_requests = Column(Integer, default=0)
    provider_errors = Column(JSON)
