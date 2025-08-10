"""
Test Execution Engine for Breakpoint LLM Stress Testing Platform.
Orchestrates test execution with real-time monitoring and comprehensive result tracking.
"""

import asyncio
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, AsyncGenerator, Callable
from enum import Enum
import json
import psutil

from app.services.llm_service import UniversalLLMService, LLMRequest, LLMResponse, LLMProvider
from app.services.test_generator import TestCase, TestCaseCategory, DifficultyLevel
from app.core.config import settings


logger = logging.getLogger(__name__)


class TestStatus(str, Enum):
    """Test execution status."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    RETRY = "retry"


class TestPriority(str, Enum):
    """Test execution priority."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class TestExecution:
    """Individual test execution record."""
    id: str
    test_case: TestCase
    llm_request: LLMRequest
    status: TestStatus = TestStatus.PENDING
    priority: TestPriority = TestPriority.NORMAL
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_ms: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    result: Optional[LLMResponse] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestSession:
    """Test session containing multiple executions."""
    id: str
    name: str
    description: str
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: TestStatus = TestStatus.PENDING
    executions: List[TestExecution] = field(default_factory=list)
    progress: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def total_tests(self) -> int:
        """Get total number of tests."""
        return len(self.executions)
    
    @property
    def completed_tests(self) -> int:
        """Get number of completed tests."""
        return len([e for e in self.executions if e.status in [TestStatus.COMPLETED, TestStatus.FAILED]])
    
    @property
    def failed_tests(self) -> int:
        """Get number of failed tests."""
        return len([e for e in self.executions if e.status == TestStatus.FAILED])
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_tests == 0:
            return 0.0
        successful = len([e for e in self.executions if e.status == TestStatus.COMPLETED])
        return successful / self.total_tests


@dataclass
class ResourceMetrics:
    """System resource usage metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    network_bytes_sent: int
    network_bytes_recv: int
    active_connections: int


@dataclass
class TestRunnerConfig:
    """Configuration for test runner."""
    max_concurrent_tests: int = 10
    test_timeout_seconds: int = 60
    retry_attempts: int = 3
    retry_delay_seconds: float = 1.0
    retry_backoff_multiplier: float = 2.0
    enable_metrics: bool = True
    metrics_interval_seconds: int = 5
    enable_cost_tracking: bool = True
    max_queue_size: int = 1000


class TestRunnerEvents:
    """Event handlers for test runner."""
    
    def __init__(self):
        self._handlers: Dict[str, List[Callable]] = {}
    
    def on(self, event: str, handler: Callable):
        """Register event handler."""
        if event not in self._handlers:
            self._handlers[event] = []
        self._handlers[event].append(handler)
    
    async def emit(self, event: str, *args, **kwargs):
        """Emit event to all handlers."""
        if event in self._handlers:
            for handler in self._handlers[event]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(*args, **kwargs)
                    else:
                        handler(*args, **kwargs)
                except Exception as exc:
                    logger.error(f"Error in event handler for {event}: {exc}")


class TestRunner:
    """Main test execution engine."""
    
    def __init__(self, llm_service: UniversalLLMService, config: TestRunnerConfig = None):
        self.llm_service = llm_service
        self.config = config or TestRunnerConfig()
        self.events = TestRunnerEvents()
        
        # Execution state
        self.sessions: Dict[str, TestSession] = {}
        self.active_executions: Dict[str, TestExecution] = {}
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=self.config.max_queue_size)
        
        # Control
        self.is_running = False
        self.is_paused = False
        self._shutdown_event = asyncio.Event()
        self._worker_tasks: List[asyncio.Task] = []
        
        # Metrics
        self.metrics_history: List[ResourceMetrics] = []
        self.total_cost: float = 0.0
        
        # Thread pool for blocking operations
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
    
    async def start(self):
        """Start the test runner."""
        if self.is_running:
            logger.warning("Test runner is already running")
            return
        
        self.is_running = True
        self._shutdown_event.clear()
        
        # Start worker tasks
        for i in range(self.config.max_concurrent_tests):
            task = asyncio.create_task(self._worker(f"worker-{i}"))
            self._worker_tasks.append(task)
        
        # Start metrics collection
        if self.config.enable_metrics:
            metrics_task = asyncio.create_task(self._collect_metrics())
            self._worker_tasks.append(metrics_task)
        
        logger.info(f"Test runner started with {self.config.max_concurrent_tests} workers")
        await self.events.emit("runner_started")
    
    async def stop(self):
        """Stop the test runner."""
        if not self.is_running:
            return
        
        logger.info("Stopping test runner...")
        self.is_running = False
        self._shutdown_event.set()
        
        # Cancel all worker tasks
        for task in self._worker_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        self._worker_tasks.clear()
        
        # Cleanup
        self.thread_pool.shutdown(wait=True)
        
        logger.info("Test runner stopped")
        await self.events.emit("runner_stopped")
    
    async def create_session(self, name: str, description: str = "", 
                           test_cases: List[TestCase] = None,
                           llm_configs: List[Dict[str, Any]] = None) -> str:
        """Create a new test session."""
        session_id = str(uuid.uuid4())
        
        session = TestSession(
            id=session_id,
            name=name,
            description=description,
            metadata={"total_test_cases": len(test_cases) if test_cases else 0}
        )
        
        # Create executions for test cases
        if test_cases and llm_configs:
            for test_case in test_cases:
                for llm_config in llm_configs:
                    execution = self._create_execution(test_case, llm_config)
                    session.executions.append(execution)
        
        self.sessions[session_id] = session
        
        logger.info(f"Created test session {session_id} with {len(session.executions)} executions")
        await self.events.emit("session_created", session)
        
        return session_id
    
    async def run_session(self, session_id: str) -> TestSession:
        """Run all tests in a session."""
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        if session.status != TestStatus.PENDING:
            raise ValueError(f"Session {session_id} is not in pending status")
        
        session.status = TestStatus.RUNNING
        session.started_at = datetime.now()
        
        logger.info(f"Starting session {session_id} with {len(session.executions)} tests")
        await self.events.emit("session_started", session)
        
        # Queue all executions
        for execution in session.executions:
            execution.status = TestStatus.QUEUED
            await self.queue.put(execution)
        
        # Wait for completion
        await self._wait_for_session_completion(session)
        
        session.completed_at = datetime.now()
        session.status = TestStatus.COMPLETED if session.failed_tests == 0 else TestStatus.FAILED
        
        logger.info(f"Session {session_id} completed. Success rate: {session.success_rate:.2%}")
        await self.events.emit("session_completed", session)
        
        return session
    
    async def run_single_test(self, test_case: TestCase, llm_config: Dict[str, Any]) -> TestExecution:
        """Run a single test case."""
        execution = self._create_execution(test_case, llm_config)
        execution.priority = TestPriority.HIGH  # Single tests get high priority
        
        # Create temporary session
        session_id = str(uuid.uuid4())
        session = TestSession(
            id=session_id,
            name=f"Single Test: {test_case.id}",
            description="Single test execution",
            executions=[execution]
        )
        self.sessions[session_id] = session
        
        # Queue and wait for execution
        await self.queue.put(execution)
        
        while execution.status in [TestStatus.PENDING, TestStatus.QUEUED, TestStatus.RUNNING]:
            await asyncio.sleep(0.1)
        
        return execution
    
    async def pause_session(self, session_id: str):
        """Pause a running session."""
        session = self.sessions.get(session_id)
        if session and session.status == TestStatus.RUNNING:
            self.is_paused = True
            logger.info(f"Paused session {session_id}")
            await self.events.emit("session_paused", session)
    
    async def resume_session(self, session_id: str):
        """Resume a paused session."""
        session = self.sessions.get(session_id)
        if session and self.is_paused:
            self.is_paused = False
            logger.info(f"Resumed session {session_id}")
            await self.events.emit("session_resumed", session)
    
    async def cancel_session(self, session_id: str):
        """Cancel a running session."""
        session = self.sessions.get(session_id)
        if not session:
            return
        
        session.status = TestStatus.CANCELLED
        
        # Cancel queued executions
        for execution in session.executions:
            if execution.status in [TestStatus.PENDING, TestStatus.QUEUED]:
                execution.status = TestStatus.CANCELLED
        
        logger.info(f"Cancelled session {session_id}")
        await self.events.emit("session_cancelled", session)
    
    async def get_session_progress(self, session_id: str) -> Dict[str, Any]:
        """Get real-time progress for a session."""
        session = self.sessions.get(session_id)
        if not session:
            return {}
        
        session.progress = session.completed_tests / session.total_tests if session.total_tests > 0 else 0
        
        return {
            "session_id": session_id,
            "status": session.status.value,
            "progress": session.progress,
            "total_tests": session.total_tests,
            "completed_tests": session.completed_tests,
            "failed_tests": session.failed_tests,
            "success_rate": session.success_rate,
            "estimated_completion": self._estimate_completion_time(session),
            "current_cost": self._calculate_session_cost(session)
        }
    
    def _create_execution(self, test_case: TestCase, llm_config: Dict[str, Any]) -> TestExecution:
        """Create a test execution from test case and LLM config."""
        execution_id = str(uuid.uuid4())
        
        llm_request = LLMRequest(
            prompt=test_case.prompt,
            model=llm_config.get("model", "gpt-3.5-turbo"),
            provider=LLMProvider(llm_config.get("provider", "openai")),
            max_tokens=llm_config.get("max_tokens"),
            temperature=llm_config.get("temperature"),
            top_p=llm_config.get("top_p"),
            metadata={"test_case_id": test_case.id, "execution_id": execution_id}
        )
        
        return TestExecution(
            id=execution_id,
            test_case=test_case,
            llm_request=llm_request,
            max_retries=self.config.retry_attempts,
            metadata={
                "test_category": test_case.category.value,
                "test_difficulty": test_case.difficulty.value,
                "llm_config": llm_config
            }
        )
    
    async def _worker(self, worker_id: str):
        """Worker coroutine for processing test executions."""
        logger.info(f"Worker {worker_id} started")
        
        while self.is_running:
            try:
                # Wait for shutdown or new execution
                try:
                    execution = await asyncio.wait_for(
                        self.queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Check if paused
                while self.is_paused and self.is_running:
                    await asyncio.sleep(0.5)
                
                if not self.is_running:
                    # Put execution back in queue
                    await self.queue.put(execution)
                    break
                
                # Execute test
                await self._execute_test(execution, worker_id)
                
            except Exception as exc:
                logger.error(f"Worker {worker_id} error: {exc}", exc_info=True)
        
        logger.info(f"Worker {worker_id} stopped")
    
    async def _execute_test(self, execution: TestExecution, worker_id: str):
        """Execute a single test."""
        execution.status = TestStatus.RUNNING
        execution.started_at = datetime.now()
        self.active_executions[execution.id] = execution
        
        logger.debug(f"Worker {worker_id} executing test {execution.id}")
        await self.events.emit("test_started", execution)
        
        try:
            # Execute with timeout
            response = await asyncio.wait_for(
                self.llm_service.generate(execution.llm_request),
                timeout=self.config.test_timeout_seconds
            )
            
            execution.result = response
            execution.status = TestStatus.COMPLETED if not response.error else TestStatus.FAILED
            execution.error = response.error
            
            # Track cost
            if self.config.enable_cost_tracking and response.cost_estimate:
                self.total_cost += response.cost_estimate
            
        except asyncio.TimeoutError:
            execution.status = TestStatus.TIMEOUT
            execution.error = f"Test timed out after {self.config.test_timeout_seconds} seconds"
            
        except Exception as exc:
            execution.status = TestStatus.FAILED
            execution.error = str(exc)
            logger.error(f"Test execution {execution.id} failed: {exc}")
        
        finally:
            execution.completed_at = datetime.now()
            execution.duration_ms = (
                (execution.completed_at - execution.started_at).total_seconds() * 1000
                if execution.started_at else None
            )
            
            # Remove from active executions
            self.active_executions.pop(execution.id, None)
            
            # Handle retries for failed tests
            if (execution.status == TestStatus.FAILED and 
                execution.retry_count < execution.max_retries):
                
                await self._schedule_retry(execution)
            else:
                await self.events.emit("test_completed", execution)
    
    async def _schedule_retry(self, execution: TestExecution):
        """Schedule a retry for a failed execution."""
        execution.retry_count += 1
        execution.status = TestStatus.RETRY
        
        # Calculate delay with exponential backoff
        delay = (self.config.retry_delay_seconds * 
                (self.config.retry_backoff_multiplier ** (execution.retry_count - 1)))
        
        logger.info(f"Scheduling retry {execution.retry_count} for test {execution.id} in {delay}s")
        
        await asyncio.sleep(delay)
        
        # Reset execution state for retry
        execution.status = TestStatus.QUEUED
        execution.started_at = None
        execution.completed_at = None
        execution.result = None
        execution.error = None
        
        # Re-queue
        await self.queue.put(execution)
        await self.events.emit("test_retried", execution)
    
    async def _wait_for_session_completion(self, session: TestSession):
        """Wait for all executions in a session to complete."""
        while True:
            pending_count = len([
                e for e in session.executions 
                if e.status in [TestStatus.PENDING, TestStatus.QUEUED, TestStatus.RUNNING, TestStatus.RETRY]
            ])
            
            if pending_count == 0:
                break
            
            await asyncio.sleep(1.0)
    
    async def _collect_metrics(self):
        """Collect system resource metrics."""
        while self.is_running:
            try:
                # Get system metrics
                cpu_percent = psutil.cpu_percent()
                memory = psutil.virtual_memory()
                network = psutil.net_io_counters()
                
                metrics = ResourceMetrics(
                    timestamp=datetime.now(),
                    cpu_percent=cpu_percent,
                    memory_percent=memory.percent,
                    memory_mb=memory.used / 1024 / 1024,
                    network_bytes_sent=network.bytes_sent,
                    network_bytes_recv=network.bytes_recv,
                    active_connections=len(self.active_executions)
                )
                
                self.metrics_history.append(metrics)
                
                # Keep only last hour of metrics
                cutoff_time = datetime.now() - timedelta(hours=1)
                self.metrics_history = [
                    m for m in self.metrics_history 
                    if m.timestamp > cutoff_time
                ]
                
                await self.events.emit("metrics_collected", metrics)
                
            except Exception as exc:
                logger.error(f"Error collecting metrics: {exc}")
            
            await asyncio.sleep(self.config.metrics_interval_seconds)
    
    def _estimate_completion_time(self, session: TestSession) -> Optional[datetime]:
        """Estimate session completion time."""
        if session.total_tests == 0 or session.completed_tests == 0:
            return None
        
        elapsed = datetime.now() - session.started_at if session.started_at else timedelta(0)
        remaining_tests = session.total_tests - session.completed_tests
        avg_time_per_test = elapsed / session.completed_tests
        estimated_remaining = avg_time_per_test * remaining_tests
        
        return datetime.now() + estimated_remaining
    
    def _calculate_session_cost(self, session: TestSession) -> float:
        """Calculate total cost for session."""
        total_cost = 0.0
        for execution in session.executions:
            if execution.result and execution.result.cost_estimate:
                total_cost += execution.result.cost_estimate
        return total_cost
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        latest_metrics = self.metrics_history[-1] if self.metrics_history else None
        
        return {
            "is_running": self.is_running,
            "is_paused": self.is_paused,
            "active_sessions": len([s for s in self.sessions.values() if s.status == TestStatus.RUNNING]),
            "total_sessions": len(self.sessions),
            "active_executions": len(self.active_executions),
            "queue_size": self.queue.qsize(),
            "total_cost": self.total_cost,
            "worker_count": len(self._worker_tasks),
            "latest_metrics": {
                "cpu_percent": latest_metrics.cpu_percent if latest_metrics else 0,
                "memory_percent": latest_metrics.memory_percent if latest_metrics else 0,
                "memory_mb": latest_metrics.memory_mb if latest_metrics else 0,
                "active_connections": latest_metrics.active_connections if latest_metrics else 0,
            } if latest_metrics else None
        }
    
    async def export_session_results(self, session_id: str, format_type: str = "json") -> str:
        """Export session results to various formats."""
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        if format_type == "json":
            return json.dumps({
                "session": {
                    "id": session.id,
                    "name": session.name,
                    "description": session.description,
                    "status": session.status.value,
                    "created_at": session.created_at.isoformat(),
                    "started_at": session.started_at.isoformat() if session.started_at else None,
                    "completed_at": session.completed_at.isoformat() if session.completed_at else None,
                    "total_tests": session.total_tests,
                    "success_rate": session.success_rate,
                    "total_cost": self._calculate_session_cost(session)
                },
                "executions": [
                    {
                        "id": exec.id,
                        "test_case_id": exec.test_case.id,
                        "status": exec.status.value,
                        "duration_ms": exec.duration_ms,
                        "retry_count": exec.retry_count,
                        "result": {
                            "text": exec.result.text if exec.result else None,
                            "latency_ms": exec.result.latency_ms if exec.result else None,
                            "cost_estimate": exec.result.cost_estimate if exec.result else None,
                            "error": exec.result.error if exec.result else None
                        } if exec.result else None,
                        "error": exec.error
                    }
                    for exec in session.executions
                ]
            }, indent=2)
        
        else:
            raise ValueError(f"Unsupported export format: {format_type}")


# Global test runner instance (to be initialized with LLM service)
test_runner: Optional[TestRunner] = None
