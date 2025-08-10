"""
Test Management API Endpoints for Breakpoint LLM Stress Testing Platform.
Comprehensive REST API with authentication, validation, and error handling.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from uuid import UUID

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query, Path, File, UploadFile, status
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field, validator
import csv
import io

from app.services.test_generator import (
    TestGeneratorService, TestCase, TestCaseCategory, DifficultyLevel, test_generator_service
)
from app.services.test_runner import TestRunner, TestSession, TestExecution, TestPriority, test_runner
from app.services.llm_service import LLMProvider, universal_llm_service
from app.models.test_result import TestResult, TestResultSummary
from app.core.config import settings


logger = logging.getLogger(__name__)

# Create router
router = APIRouter()


# Request/Response Models
class TestRunRequest(BaseModel):
    """Request model for running a single test."""
    prompt: str = Field(..., description="Test prompt to send to the model")
    model: str = Field(..., description="Model name to test")
    provider: LLMProvider = Field(..., description="LLM provider")
    max_tokens: Optional[int] = Field(None, ge=1, le=8192, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(None, ge=0, le=2, description="Temperature for generation")
    top_p: Optional[float] = Field(None, ge=0, le=1, description="Top-p for generation")
    test_case_id: Optional[str] = Field(None, description="Associated test case ID")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @validator('prompt')
    def validate_prompt(cls, v):
        if len(v.strip()) == 0:
            raise ValueError("Prompt cannot be empty")
        if len(v) > 50000:  # 50KB limit
            raise ValueError("Prompt too long (max 50KB)")
        return v


class BatchTestRequest(BaseModel):
    """Request model for running batch tests."""
    test_cases: List[str] = Field(..., description="List of test case IDs")
    models: List[str] = Field(..., description="List of model names")
    providers: List[LLMProvider] = Field(..., description="List of providers")
    session_name: str = Field(..., description="Name for the test session")
    session_description: str = Field(default="", description="Description for the test session")
    priority: TestPriority = Field(default=TestPriority.NORMAL, description="Execution priority")
    max_concurrent: Optional[int] = Field(None, ge=1, le=50, description="Max concurrent executions")
    
    @validator('test_cases')
    def validate_test_cases(cls, v):
        if len(v) == 0:
            raise ValueError("At least one test case is required")
        if len(v) > 1000:
            raise ValueError("Too many test cases (max 1000)")
        return v
    
    @validator('models')
    def validate_models(cls, v):
        if len(v) == 0:
            raise ValueError("At least one model is required")
        return v


class TestGenerationRequest(BaseModel):
    """Request model for generating test cases."""
    category: TestCaseCategory = Field(..., description="Test case category")
    difficulty: DifficultyLevel = Field(default=DifficultyLevel.BASIC, description="Difficulty level")
    count: int = Field(default=5, ge=1, le=100, description="Number of test cases to generate")
    target_task: Optional[str] = Field(None, description="Target task for injection tests")
    format_type: Optional[str] = Field(None, description="Format type for malformed data tests")
    attack_type: Optional[str] = Field(None, description="Attack type for Unicode tests")
    
    @validator('count')
    def validate_count(cls, v):
        if v <= 0:
            raise ValueError("Count must be positive")
        if v > 100:
            raise ValueError("Cannot generate more than 100 test cases at once")
        return v


class TestResponse(BaseModel):
    """Response model for test execution."""
    execution_id: str = Field(..., description="Unique execution ID")
    status: str = Field(..., description="Execution status")
    result: Optional[Dict[str, Any]] = Field(None, description="Test result if completed")
    error: Optional[str] = Field(None, description="Error message if failed")
    created_at: datetime = Field(..., description="Creation timestamp")


class SessionResponse(BaseModel):
    """Response model for test sessions."""
    session_id: str = Field(..., description="Session ID")
    name: str = Field(..., description="Session name")
    status: str = Field(..., description="Session status")
    progress: float = Field(..., description="Progress percentage (0-1)")
    total_tests: int = Field(..., description="Total number of tests")
    completed_tests: int = Field(..., description="Number of completed tests")
    failed_tests: int = Field(..., description="Number of failed tests")
    success_rate: float = Field(..., description="Success rate (0-1)")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")


class ProgressUpdate(BaseModel):
    """WebSocket progress update model."""
    session_id: str
    status: str
    progress: float
    current_test: Optional[str] = None
    tests_completed: int
    tests_total: int
    success_rate: float
    errors: List[str] = []


# API Endpoints

@router.post(
    "/run",
    response_model=TestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Run Single Test",
    description="Execute a single test case against an LLM model"
)
async def run_single_test(
    request: TestRunRequest,
    background_tasks: BackgroundTasks
) -> TestResponse:
    """Run a single test case and return the result."""
    try:
        # Validate model availability
        available_models = universal_llm_service.get_available_models()
        model_names = [model.name for model in available_models if model.provider == request.provider]
        
        if request.model not in model_names:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Model {request.model} not available for provider {request.provider.value}"
            )
        
        # Create test case if not provided
        if request.test_case_id:
            # TODO: Fetch existing test case from database
            test_case = None  # Placeholder
        else:
            # Create ad-hoc test case
            from app.services.test_generator import TestCase
            test_case = TestCase(
                id=f"adhoc_{datetime.now().timestamp()}",
                category=TestCaseCategory.PROMPT_INJECTION,  # Default category
                difficulty=DifficultyLevel.BASIC,
                prompt=request.prompt,
                expected_behavior="Should handle input appropriately",
                description="Ad-hoc test case",
                tags=["adhoc"],
                metadata=request.metadata,
                created_at=datetime.now()
            )
        
        # Prepare LLM configuration
        llm_config = {
            "model": request.model,
            "provider": request.provider.value,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p
        }
        
        # Execute test
        if test_runner is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Test runner not initialized"
            )
        
        execution = await test_runner.run_single_test(test_case, llm_config)
        
        # Prepare response
        result_data = None
        if execution.result:
            result_data = {
                "text": execution.result.text,
                "latency_ms": execution.result.latency_ms,
                "cost_estimate": execution.result.cost_estimate,
                "usage": execution.result.usage,
                "error": execution.result.error
            }
        
        return TestResponse(
            execution_id=execution.id,
            status=execution.status.value,
            result=result_data,
            error=execution.error,
            created_at=execution.created_at
        )
        
    except Exception as exc:
        logger.error(f"Error running single test: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(exc)}"
        )


@router.post(
    "/batch",
    response_model=SessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Run Batch Tests",
    description="Execute multiple test cases in a batch session"
)
async def run_batch_tests(
    request: BatchTestRequest,
    background_tasks: BackgroundTasks
) -> SessionResponse:
    """Run multiple test cases in a batch session."""
    try:
        if test_runner is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Test runner not initialized"
            )
        
        # TODO: Fetch test cases from database
        test_cases = []  # Placeholder
        
        # Create LLM configurations for all model/provider combinations
        llm_configs = []
        for model in request.models:
            for provider in request.providers:
                llm_configs.append({
                    "model": model,
                    "provider": provider.value,
                    "priority": request.priority.value
                })
        
        # Create session
        session_id = await test_runner.create_session(
            name=request.session_name,
            description=request.session_description,
            test_cases=test_cases,
            llm_configs=llm_configs
        )
        
        # Start execution in background
        background_tasks.add_task(test_runner.run_session, session_id)
        
        # Get initial progress
        progress = await test_runner.get_session_progress(session_id)
        
        return SessionResponse(
            session_id=session_id,
            name=request.session_name,
            status=progress["status"],
            progress=progress["progress"],
            total_tests=progress["total_tests"],
            completed_tests=progress["completed_tests"],
            failed_tests=progress["failed_tests"],
            success_rate=progress["success_rate"],
            estimated_completion=progress.get("estimated_completion")
        )
        
    except Exception as exc:
        logger.error(f"Error running batch tests: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(exc)}"
        )


@router.post(
    "/generate",
    response_model=List[Dict[str, Any]],
    summary="Generate Test Cases",
    description="Generate adversarial test cases using built-in generators"
)
async def generate_test_cases(request: TestGenerationRequest) -> List[Dict[str, Any]]:
    """Generate test cases using the test generator service."""
    try:
        # Generate test cases
        test_cases = test_generator_service.generate_multiple(
            category=request.category,
            difficulty=request.difficulty,
            count=request.count,
            target_task=request.target_task,
            format_type=request.format_type,
            attack_type=request.attack_type
        )
        
        # Convert to response format
        return [
            {
                "id": tc.id,
                "category": tc.category.value,
                "difficulty": tc.difficulty.value,
                "prompt": tc.prompt,
                "expected_behavior": tc.expected_behavior,
                "description": tc.description,
                "tags": tc.tags,
                "metadata": tc.metadata,
                "created_at": tc.created_at.isoformat()
            }
            for tc in test_cases
        ]
        
    except Exception as exc:
        logger.error(f"Error generating test cases: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(exc)}"
        )


@router.get(
    "/templates",
    response_model=List[Dict[str, Any]],
    summary="Get Test Templates",
    description="Retrieve pre-built test case templates"
)
async def get_test_templates(
    category: Optional[TestCaseCategory] = Query(None, description="Filter by category"),
    difficulty: Optional[DifficultyLevel] = Query(None, description="Filter by difficulty")
) -> List[Dict[str, Any]]:
    """Get available test case templates."""
    try:
        # Get available categories
        categories = test_generator_service.get_available_categories()
        
        templates = []
        for cat in categories:
            if category is None or cat == category:
                # Generate a sample for each difficulty level
                difficulties = [difficulty] if difficulty else list(DifficultyLevel)
                
                for diff in difficulties:
                    try:
                        sample = test_generator_service.generate_test_case(cat, diff)
                        templates.append({
                            "category": cat.value,
                            "difficulty": diff.value,
                            "description": sample.description,
                            "expected_behavior": sample.expected_behavior,
                            "tags": sample.tags,
                            "sample_prompt": sample.prompt[:200] + "..." if len(sample.prompt) > 200 else sample.prompt
                        })
                    except Exception as e:
                        logger.warning(f"Could not generate sample for {cat.value}/{diff.value}: {e}")
        
        return templates
        
    except Exception as exc:
        logger.error(f"Error getting test templates: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(exc)}"
        )


@router.get(
    "/{test_id}",
    response_model=Dict[str, Any],
    summary="Get Test Details",
    description="Retrieve details for a specific test execution"
)
async def get_test_details(
    test_id: str = Path(..., description="Test execution ID")
) -> Dict[str, Any]:
    """Get details for a specific test execution."""
    try:
        if test_runner is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Test runner not initialized"
            )
        
        # TODO: Implement test lookup in test runner
        # For now, return placeholder
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Test {test_id} not found"
        )
        
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error getting test details: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(exc)}"
        )


@router.get(
    "/",
    response_model=Dict[str, Any],
    summary="List Tests",
    description="List test executions with filtering and pagination"
)
async def list_tests(
    limit: int = Query(default=50, ge=1, le=1000, description="Number of results to return"),
    offset: int = Query(default=0, ge=0, description="Number of results to skip"),
    status: Optional[str] = Query(None, description="Filter by status"),
    model: Optional[str] = Query(None, description="Filter by model"),
    provider: Optional[str] = Query(None, description="Filter by provider"),
    session_id: Optional[str] = Query(None, description="Filter by session ID")
) -> Dict[str, Any]:
    """List test executions with filtering and pagination."""
    try:
        # TODO: Implement test listing with database queries
        # For now, return placeholder
        return {
            "tests": [],
            "total": 0,
            "limit": limit,
            "offset": offset,
            "filters": {
                "status": status,
                "model": model,
                "provider": provider,
                "session_id": session_id
            }
        }
        
    except Exception as exc:
        logger.error(f"Error listing tests: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(exc)}"
        )


@router.put(
    "/{test_id}",
    response_model=Dict[str, Any],
    summary="Update Test",
    description="Update test execution metadata or configuration"
)
async def update_test(
    test_id: str = Path(..., description="Test execution ID"),
    update_data: Dict[str, Any] = {}
) -> Dict[str, Any]:
    """Update test execution metadata."""
    try:
        # TODO: Implement test update functionality
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Test {test_id} not found"
        )
        
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error updating test: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(exc)}"
        )


@router.delete(
    "/{test_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete Test",
    description="Delete a test execution and its results"
)
async def delete_test(
    test_id: str = Path(..., description="Test execution ID")
):
    """Delete a test execution."""
    try:
        # TODO: Implement test deletion
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Test {test_id} not found"
        )
        
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error deleting test: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(exc)}"
        )


@router.post(
    "/import",
    response_model=Dict[str, Any],
    summary="Import Test Cases",
    description="Import test cases from uploaded files"
)
async def import_test_cases(
    file: UploadFile = File(..., description="Test cases file (JSON, CSV, or TXT)")
) -> Dict[str, Any]:
    """Import test cases from uploaded files."""
    try:
        # Validate file type
        allowed_extensions = ['.json', '.csv', '.txt']
        file_ext = '.' + file.filename.split('.')[-1].lower() if '.' in file.filename else ''
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Read file content
        content = await file.read()
        
        if len(content) > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Max size: {settings.MAX_FILE_SIZE} bytes"
            )
        
        # Parse content based on file type
        test_cases = []
        
        if file_ext == '.json':
            try:
                data = json.loads(content.decode('utf-8'))
                if isinstance(data, list):
                    test_cases = data
                elif isinstance(data, dict) and 'test_cases' in data:
                    test_cases = data['test_cases']
                else:
                    test_cases = [data]
            except json.JSONDecodeError as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid JSON format: {str(e)}"
                )
        
        elif file_ext == '.csv':
            try:
                csv_reader = csv.DictReader(io.StringIO(content.decode('utf-8')))
                test_cases = list(csv_reader)
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid CSV format: {str(e)}"
                )
        
        elif file_ext == '.txt':
            # Treat each line as a prompt
            lines = content.decode('utf-8').split('\\n')
            test_cases = [{"prompt": line.strip()} for line in lines if line.strip()]
        
        # TODO: Validate and store test cases in database
        
        return {
            "imported_count": len(test_cases),
            "file_name": file.filename,
            "file_size": len(content),
            "message": f"Successfully imported {len(test_cases)} test cases"
        }
        
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error importing test cases: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(exc)}"
        )


@router.get(
    "/export",
    summary="Export Test Cases",
    description="Export test cases in various formats"
)
async def export_test_cases(
    format: str = Query(default="json", regex="^(json|csv|txt)$", description="Export format"),
    category: Optional[TestCaseCategory] = Query(None, description="Filter by category"),
    difficulty: Optional[DifficultyLevel] = Query(None, description="Filter by difficulty"),
    limit: int = Query(default=1000, ge=1, le=10000, description="Maximum number of test cases")
) -> StreamingResponse:
    """Export test cases in the specified format."""
    try:
        # TODO: Fetch test cases from database with filters
        test_cases = []  # Placeholder
        
        # Generate export content
        if format == "json":
            content = json.dumps(test_cases, indent=2)
            media_type = "application/json"
            filename = f"test_cases_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        elif format == "csv":
            output = io.StringIO()
            if test_cases:
                fieldnames = test_cases[0].keys()
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(test_cases)
            content = output.getvalue()
            media_type = "text/csv"
            filename = f"test_cases_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        elif format == "txt":
            content = "\\n".join([tc.get("prompt", "") for tc in test_cases])
            media_type = "text/plain"
            filename = f"test_cases_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        # Create streaming response
        def generate():
            yield content.encode('utf-8')
        
        return StreamingResponse(
            generate(),
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as exc:
        logger.error(f"Error exporting test cases: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(exc)}"
        )


@router.post(
    "/schedule",
    response_model=Dict[str, Any],
    summary="Schedule Recurring Tests",
    description="Schedule automated recurring test execution"
)
async def schedule_recurring_tests(
    schedule_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Schedule recurring automated tests."""
    try:
        # TODO: Implement test scheduling functionality
        # This would integrate with a task scheduler like Celery
        
        return {
            "schedule_id": f"schedule_{datetime.now().timestamp()}",
            "message": "Scheduled test execution created",
            "next_execution": datetime.now().isoformat(),
            "config": schedule_config
        }
        
    except Exception as exc:
        logger.error(f"Error scheduling tests: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(exc)}"
        )


# WebSocket endpoint for real-time updates
from fastapi import WebSocket, WebSocketDisconnect

@router.websocket("/ws/{session_id}")
async def websocket_test_progress(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time test progress updates."""
    await websocket.accept()
    
    try:
        if test_runner is None:
            await websocket.send_json({"error": "Test runner not initialized"})
            return
        
        # Send initial status
        progress = await test_runner.get_session_progress(session_id)
        await websocket.send_json({
            "type": "progress",
            "data": progress
        })
        
        # Monitor progress
        while True:
            # Check if client is still connected
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
            except asyncio.TimeoutError:
                pass
            except WebSocketDisconnect:
                break
            
            # Send progress update
            progress = await test_runner.get_session_progress(session_id)
            await websocket.send_json({
                "type": "progress",
                "data": progress,
                "timestamp": datetime.now().isoformat()
            })
            
            # Stop if session is completed
            if progress.get("status") in ["completed", "failed", "cancelled"]:
                break
            
            await asyncio.sleep(2)  # Update every 2 seconds
    
    except WebSocketDisconnect:
        logger.info(f"Client disconnected from session {session_id}")
    except Exception as exc:
        logger.error(f"WebSocket error for session {session_id}: {exc}")
        await websocket.send_json({"error": str(exc)})
    finally:
        await websocket.close()
