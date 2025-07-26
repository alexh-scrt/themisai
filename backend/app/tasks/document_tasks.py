"""
Document Processing Background Tasks

This module provides background task orchestration for legal document processing
in the Patexia Legal AI Chatbot. It manages async task queues, job scheduling,
retry mechanisms, and progress tracking for long-running document operations.

Key Features:
- Async task queue management with priority handling
- Document processing pipeline coordination
- Background job scheduling and execution
- Task retry mechanisms with exponential backoff
- Progress tracking and status updates via WebSocket
- Resource monitoring and capacity management
- Batch processing optimization for multiple documents
- Error handling and recovery workflows

Task Types:
- Document upload and initial processing
- Text extraction and content analysis
- Semantic chunking and structure preservation
- Embedding generation and vector indexing
- Document reprocessing and retry operations
- Batch document operations and cleanup
- Analytics and reporting generation
- System maintenance and optimization

Architecture Integration:
- Coordinates with DocumentProcessor for processing pipeline
- Uses EmbeddingService for vector generation
- Integrates with NotificationService for progress updates
- Manages VectorRepository operations for indexing
- Implements DocumentRepository updates for status tracking
- Provides WebSocket progress tracking for real-time updates
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import traceback

from config.settings import get_settings
from ..models.domain.document import (
    LegalDocument, ProcessingStatus, DocumentPriority, DocumentType
)
from ..models.api.document_schemas import (
    DocumentProgressUpdate, BatchDocumentOperation, BatchOperationResponse
)
from ..processors.document_processor import DocumentProcessor
from ..services.document_service import DocumentService
from ..services.embedding_service import EmbeddingService
from ..services.notification_service import NotificationService
from ..repositories.mongodb.document_repository import DocumentRepository
from ..repositories.weaviate.vector_repository import VectorRepository
from ..core.exceptions import (
    DocumentProcessingError, TaskError, ResourceError,
    ErrorCode, raise_task_error, raise_resource_error
)
from ..utils.logging import get_logger, performance_context

logger = get_logger(__name__)


class TaskType(str, Enum):
    """Types of background tasks."""
    DOCUMENT_UPLOAD = "document_upload"
    DOCUMENT_PROCESS = "document_process"
    DOCUMENT_RETRY = "document_retry"
    DOCUMENT_CLEANUP = "document_cleanup"
    BATCH_OPERATION = "batch_operation"
    EMBEDDING_GENERATION = "embedding_generation"
    INDEX_OPTIMIZATION = "index_optimization"
    ANALYTICS_GENERATION = "analytics_generation"
    SYSTEM_MAINTENANCE = "system_maintenance"


class TaskStatus(str, Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class TaskPriority(int, Enum):
    """Task priority levels (higher number = higher priority)."""
    LOW = 1
    NORMAL = 3
    HIGH = 5
    URGENT = 8
    CRITICAL = 10


@dataclass
class TaskContext:
    """Context information for task execution."""
    task_id: str
    user_id: str
    case_id: Optional[str]
    document_id: Optional[str]
    session_id: Optional[str] = None
    client_ip: Optional[str] = None
    correlation_id: Optional[str] = None


@dataclass
class TaskProgress:
    """Task progress tracking information."""
    current_step: str
    total_steps: int
    completed_steps: int
    progress_percentage: float
    estimated_time_remaining: Optional[int] = None
    current_operation: Optional[str] = None
    
    def update(self, step: str, completed: int, total: int, operation: str = None):
        """Update progress information."""
        self.current_step = step
        self.completed_steps = completed
        self.total_steps = total
        self.progress_percentage = (completed / total) * 100 if total > 0 else 0
        if operation:
            self.current_operation = operation


@dataclass
class TaskResult:
    """Task execution result."""
    success: bool
    result_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    execution_time_ms: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BackgroundTask:
    """Background task definition."""
    task_id: str
    task_type: TaskType
    priority: TaskPriority
    context: TaskContext
    parameters: Dict[str, Any]
    created_at: datetime
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: TaskStatus = TaskStatus.PENDING
    retry_count: int = 0
    max_retries: int = 3
    timeout_seconds: int = 300
    progress: Optional[TaskProgress] = None
    result: Optional[TaskResult] = None
    
    def __post_init__(self):
        """Initialize task after creation."""
        if self.scheduled_at is None:
            self.scheduled_at = self.created_at
    
    def is_expired(self) -> bool:
        """Check if task has expired."""
        if self.status in [TaskStatus.COMPLETED, TaskStatus.CANCELLED]:
            return False
        
        # Check if task has been running too long
        if self.started_at:
            runtime = datetime.now(timezone.utc) - self.started_at
            return runtime.total_seconds() > self.timeout_seconds
        
        # Check if task is too old to start
        age = datetime.now(timezone.utc) - self.created_at
        return age.total_seconds() > (self.timeout_seconds * 2)
    
    def can_retry(self) -> bool:
        """Check if task can be retried."""
        return (
            self.status == TaskStatus.FAILED and
            self.retry_count < self.max_retries and
            not self.is_expired()
        )
    
    def prepare_retry(self) -> None:
        """Prepare task for retry."""
        self.retry_count += 1
        self.status = TaskStatus.RETRYING
        self.started_at = None
        self.completed_at = None
        self.result = None
        
        # Exponential backoff for scheduling
        delay_seconds = min(300, 2 ** self.retry_count * 5)  # Max 5 minute delay
        self.scheduled_at = datetime.now(timezone.utc) + timedelta(seconds=delay_seconds)


@dataclass
class TaskStats:
    """Task execution statistics."""
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    cancelled_tasks: int = 0
    pending_tasks: int = 0
    running_tasks: int = 0
    retrying_tasks: int = 0
    average_execution_time_ms: float = 0.0
    success_rate: float = 0.0
    task_type_distribution: Dict[TaskType, int] = field(default_factory=dict)
    priority_distribution: Dict[TaskPriority, int] = field(default_factory=dict)


class DocumentTaskManager:
    """
    Background task manager for document processing operations.
    
    Orchestrates async task queues, job scheduling, retry mechanisms,
    and progress tracking for legal document processing workflows.
    """
    
    def __init__(
        self,
        document_service: DocumentService,
        document_processor: DocumentProcessor,
        embedding_service: EmbeddingService,
        notification_service: NotificationService,
        document_repository: DocumentRepository,
        vector_repository: VectorRepository,
        max_concurrent_tasks: int = 5,
        max_queue_size: int = 1000
    ):
        """
        Initialize document task manager.
        
        Args:
            document_service: Document service for business logic
            document_processor: Document processor for pipeline operations
            embedding_service: Embedding service for vector generation
            notification_service: Notification service for progress updates
            document_repository: MongoDB repository for document data
            vector_repository: Weaviate repository for vector operations
            max_concurrent_tasks: Maximum concurrent task execution
            max_queue_size: Maximum task queue size
        """
        self.document_service = document_service
        self.document_processor = document_processor
        self.embedding_service = embedding_service
        self.notification_service = notification_service
        self.document_repository = document_repository
        self.vector_repository = vector_repository
        
        # Configuration
        self.settings = get_settings()
        self.max_concurrent_tasks = max_concurrent_tasks
        self.max_queue_size = max_queue_size
        
        # Task management
        self.task_queue = asyncio.PriorityQueue(maxsize=max_queue_size)
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.task_registry: Dict[str, BackgroundTask] = {}
        self.task_history: deque = deque(maxlen=1000)  # Keep last 1000 tasks
        
        # Statistics and monitoring
        self.stats = TaskStats()
        self.task_handlers: Dict[TaskType, Callable] = {}
        
        # Background processing
        self._processing_semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self._worker_tasks: List[asyncio.Task] = []
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Initialize task handlers
        self._register_task_handlers()
        
        logger.info(
            "DocumentTaskManager initialized",
            max_concurrent_tasks=max_concurrent_tasks,
            max_queue_size=max_queue_size
        )
    
    async def start(self) -> None:
        """Start the task manager and background workers."""
        if self._running:
            return
        
        self._running = True
        
        # Start worker tasks
        for i in range(self.max_concurrent_tasks):
            worker_task = asyncio.create_task(self._worker_loop(f"worker-{i}"))
            self._worker_tasks.append(worker_task)
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info(
            "DocumentTaskManager started",
            worker_count=len(self._worker_tasks)
        )
    
    async def stop(self) -> None:
        """Stop the task manager and cancel running tasks."""
        self._running = False
        
        # Cancel all active tasks
        for task_id, task in self.active_tasks.items():
            if not task.done():
                task.cancel()
                
                # Update task status
                if task_id in self.task_registry:
                    self.task_registry[task_id].status = TaskStatus.CANCELLED
                    self.task_registry[task_id].completed_at = datetime.now(timezone.utc)
        
        # Wait for active tasks to complete or timeout
        if self.active_tasks:
            await asyncio.gather(*self.active_tasks.values(), return_exceptions=True)
        
        # Cancel worker tasks
        for worker_task in self._worker_tasks:
            worker_task.cancel()
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
        
        # Wait for workers to finish
        if self._worker_tasks:
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        
        if self._cleanup_task:
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("DocumentTaskManager stopped")
    
    async def submit_task(
        self,
        task_type: TaskType,
        context: TaskContext,
        parameters: Dict[str, Any],
        priority: TaskPriority = TaskPriority.NORMAL,
        scheduled_at: Optional[datetime] = None,
        timeout_seconds: int = 300,
        max_retries: int = 3
    ) -> str:
        """
        Submit a task for background execution.
        
        Args:
            task_type: Type of task to execute
            context: Task execution context
            parameters: Task-specific parameters
            priority: Task priority level
            scheduled_at: Optional scheduled execution time
            timeout_seconds: Task timeout in seconds
            max_retries: Maximum retry attempts
            
        Returns:
            Task ID for tracking
            
        Raises:
            ResourceError: If task queue is full
            TaskError: If task submission fails
        """
        try:
            # Check queue capacity
            if self.task_queue.qsize() >= self.max_queue_size:
                raise_resource_error(
                    "Task queue is full",
                    ErrorCode.TASK_QUEUE_FULL,
                    {"queue_size": self.task_queue.qsize(), "max_size": self.max_queue_size}
                )
            
            # Create task
            task = BackgroundTask(
                task_id=context.task_id,
                task_type=task_type,
                priority=priority,
                context=context,
                parameters=parameters,
                created_at=datetime.now(timezone.utc),
                scheduled_at=scheduled_at,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries
            )
            
            # Register task
            self.task_registry[task.task_id] = task
            
            # Queue task (priority queue uses negative priority for max-heap behavior)
            await self.task_queue.put((-priority.value, task.created_at, task))
            
            # Update statistics
            self.stats.total_tasks += 1
            self.stats.pending_tasks += 1
            
            if task_type not in self.stats.task_type_distribution:
                self.stats.task_type_distribution[task_type] = 0
            self.stats.task_type_distribution[task_type] += 1
            
            if priority not in self.stats.priority_distribution:
                self.stats.priority_distribution[priority] = 0
            self.stats.priority_distribution[priority] += 1
            
            logger.info(
                "Task submitted",
                task_id=task.task_id,
                task_type=task_type.value,
                priority=priority.value,
                queue_size=self.task_queue.qsize()
            )
            
            return task.task_id
            
        except Exception as e:
            logger.error(
                "Failed to submit task",
                task_type=task_type.value,
                error=str(e)
            )
            raise_task_error(
                f"Task submission failed: {str(e)}",
                ErrorCode.TASK_SUBMISSION_FAILED,
                {"task_type": task_type.value}
            )
    
    async def submit_document_processing_task(
        self,
        document_id: str,
        case_id: str,
        user_id: str,
        priority: DocumentPriority = DocumentPriority.NORMAL,
        force_retry: bool = False
    ) -> str:
        """
        Submit document processing task.
        
        Args:
            document_id: Document identifier
            case_id: Case identifier
            user_id: User identifier
            priority: Processing priority
            force_retry: Force retry if document failed
            
        Returns:
            Task ID for tracking
        """
        # Map document priority to task priority
        priority_mapping = {
            DocumentPriority.LOW: TaskPriority.LOW,
            DocumentPriority.NORMAL: TaskPriority.NORMAL,
            DocumentPriority.HIGH: TaskPriority.HIGH,
            DocumentPriority.URGENT: TaskPriority.URGENT
        }
        
        task_priority = priority_mapping.get(priority, TaskPriority.NORMAL)
        
        context = TaskContext(
            task_id=f"doc_process_{document_id}_{int(time.time())}",
            user_id=user_id,
            case_id=case_id,
            document_id=document_id,
            correlation_id=str(uuid.uuid4())
        )
        
        parameters = {
            "document_id": document_id,
            "case_id": case_id,
            "force_retry": force_retry,
            "priority": priority.value
        }
        
        return await self.submit_task(
            task_type=TaskType.DOCUMENT_PROCESS,
            context=context,
            parameters=parameters,
            priority=task_priority,
            timeout_seconds=600,  # 10 minutes for document processing
            max_retries=3
        )
    
    async def submit_batch_operation_task(
        self,
        operation: BatchDocumentOperation,
        user_id: str,
        case_id: Optional[str] = None
    ) -> str:
        """
        Submit batch document operation task.
        
        Args:
            operation: Batch operation definition
            user_id: User identifier
            case_id: Optional case identifier
            
        Returns:
            Task ID for tracking
        """
        context = TaskContext(
            task_id=f"batch_{operation.operation}_{int(time.time())}",
            user_id=user_id,
            case_id=case_id,
            correlation_id=str(uuid.uuid4())
        )
        
        parameters = {
            "operation": operation.operation,
            "document_ids": operation.document_ids,
            "operation_parameters": operation.parameters or {}
        }
        
        return await self.submit_task(
            task_type=TaskType.BATCH_OPERATION,
            context=context,
            parameters=parameters,
            priority=TaskPriority.HIGH,  # Batch operations get high priority
            timeout_seconds=1800,  # 30 minutes for batch operations
            max_retries=2
        )
    
    async def get_task_status(self, task_id: str) -> Optional[BackgroundTask]:
        """
        Get current status of a task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Task status information or None if not found
        """
        return self.task_registry.get(task_id)
    
    async def cancel_task(self, task_id: str, user_id: str) -> bool:
        """
        Cancel a pending or running task.
        
        Args:
            task_id: Task identifier
            user_id: User identifier for authorization
            
        Returns:
            True if task was cancelled, False otherwise
        """
        task = self.task_registry.get(task_id)
        if not task:
            return False
        
        # Check authorization
        if task.context.user_id != user_id:
            logger.warning(
                "Unauthorized task cancellation attempt",
                task_id=task_id,
                task_owner=task.context.user_id,
                requester=user_id
            )
            return False
        
        # Cancel if running
        if task_id in self.active_tasks:
            async_task = self.active_tasks[task_id]
            if not async_task.done():
                async_task.cancel()
        
        # Update task status
        task.status = TaskStatus.CANCELLED
        task.completed_at = datetime.now(timezone.utc)
        
        # Update statistics
        self.stats.cancelled_tasks += 1
        if task.status == TaskStatus.PENDING:
            self.stats.pending_tasks -= 1
        elif task.status == TaskStatus.RUNNING:
            self.stats.running_tasks -= 1
        
        logger.info(
            "Task cancelled",
            task_id=task_id,
            user_id=user_id
        )
        
        return True
    
    def get_task_statistics(self) -> TaskStats:
        """Get current task execution statistics."""
        # Update success rate
        completed_and_failed = self.stats.completed_tasks + self.stats.failed_tasks
        if completed_and_failed > 0:
            self.stats.success_rate = self.stats.completed_tasks / completed_and_failed
        
        return self.stats
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status information."""
        return {
            "queue_size": self.task_queue.qsize(),
            "max_queue_size": self.max_queue_size,
            "active_tasks": len(self.active_tasks),
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "pending_tasks": self.stats.pending_tasks,
            "running_tasks": self.stats.running_tasks,
            "is_running": self._running
        }
    
    # Private methods
    
    def _register_task_handlers(self) -> None:
        """Register task type handlers."""
        self.task_handlers = {
            TaskType.DOCUMENT_PROCESS: self._handle_document_processing,
            TaskType.DOCUMENT_RETRY: self._handle_document_retry,
            TaskType.BATCH_OPERATION: self._handle_batch_operation,
            TaskType.DOCUMENT_CLEANUP: self._handle_document_cleanup,
            TaskType.EMBEDDING_GENERATION: self._handle_embedding_generation,
            TaskType.INDEX_OPTIMIZATION: self._handle_index_optimization,
            TaskType.ANALYTICS_GENERATION: self._handle_analytics_generation,
            TaskType.SYSTEM_MAINTENANCE: self._handle_system_maintenance
        }
    
    async def _worker_loop(self, worker_id: str) -> None:
        """Main worker loop for processing tasks."""
        logger.info(f"Task worker {worker_id} started")
        
        while self._running:
            try:
                # Get next task with timeout
                try:
                    _, _, task = await asyncio.wait_for(
                        self.task_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Check if task should be executed now
                if task.scheduled_at and task.scheduled_at > datetime.now(timezone.utc):
                    # Re-queue for later
                    await self.task_queue.put((-task.priority.value, task.created_at, task))
                    await asyncio.sleep(1)
                    continue
                
                # Check if task is expired
                if task.is_expired():
                    task.status = TaskStatus.FAILED
                    task.completed_at = datetime.now(timezone.utc)
                    task.result = TaskResult(
                        success=False,
                        error_message="Task expired before execution",
                        error_code=ErrorCode.TASK_EXPIRED.value
                    )
                    self._update_task_stats(task)
                    continue
                
                # Execute task
                await self._execute_task(task)
                
            except Exception as e:
                logger.error(
                    f"Error in worker {worker_id}",
                    error=str(e),
                    exc_info=True
                )
                await asyncio.sleep(1)
        
        logger.info(f"Task worker {worker_id} stopped")
    
    async def _execute_task(self, task: BackgroundTask) -> None:
        """Execute a single task."""
        async with self._processing_semaphore:
            task_execution = asyncio.create_task(self._run_task(task))
            self.active_tasks[task.task_id] = task_execution
            
            try:
                await task_execution
            except asyncio.CancelledError:
                task.status = TaskStatus.CANCELLED
                task.completed_at = datetime.now(timezone.utc)
            except Exception as e:
                logger.error(
                    "Task execution failed",
                    task_id=task.task_id,
                    error=str(e),
                    exc_info=True
                )
            finally:
                if task.task_id in self.active_tasks:
                    del self.active_tasks[task.task_id]
                
                self._update_task_stats(task)
                
                # Move to history
                self.task_history.append(task)
    
    async def _run_task(self, task: BackgroundTask) -> None:
        """Run a task with timeout and error handling."""
        start_time = time.time()
        
        # Update task status
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now(timezone.utc)
        
        # Update statistics
        self.stats.pending_tasks -= 1
        self.stats.running_tasks += 1
        
        try:
            # Send start notification
            if self.notification_service:
                await self.notification_service.send_notification(
                    user_id=task.context.user_id,
                    notification_type="task_started",
                    context={
                        "task_id": task.task_id,
                        "task_type": task.task_type.value,
                        "document_id": task.context.document_id or "",
                        "case_id": task.context.case_id or ""
                    }
                )
            
            # Get task handler
            handler = self.task_handlers.get(task.task_type)
            if not handler:
                raise TaskError(f"No handler for task type: {task.task_type.value}")
            
            # Execute task with timeout
            try:
                result = await asyncio.wait_for(
                    handler(task),
                    timeout=task.timeout_seconds
                )
                
                task.status = TaskStatus.COMPLETED
                task.result = result
                
            except asyncio.TimeoutError:
                raise TaskError(f"Task timed out after {task.timeout_seconds} seconds")
            
            task.completed_at = datetime.now(timezone.utc)
            execution_time_ms = (time.time() - start_time) * 1000
            
            if task.result:
                task.result.execution_time_ms = execution_time_ms
            
            logger.info(
                "Task completed successfully",
                task_id=task.task_id,
                task_type=task.task_type.value,
                execution_time_ms=execution_time_ms
            )
            
            # Send completion notification
            if self.notification_service:
                await self.notification_service.send_notification(
                    user_id=task.context.user_id,
                    notification_type="task_completed",
                    context={
                        "task_id": task.task_id,
                        "task_type": task.task_type.value,
                        "execution_time_ms": execution_time_ms,
                        "success": task.result.success if task.result else False
                    }
                )
            
        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now(timezone.utc)
            task.result = TaskResult(
                success=False,
                error_message=str(e),
                error_code=getattr(e, 'error_code', 'UNKNOWN_ERROR'),
                execution_time_ms=execution_time_ms
            )
            
            logger.error(
                "Task failed",
                task_id=task.task_id,
                task_type=task.task_type.value,
                error=str(e),
                execution_time_ms=execution_time_ms
            )
            
            # Check if task should be retried
            if task.can_retry():
                task.prepare_retry()
                logger.info(
                    "Task queued for retry",
                    task_id=task.task_id,
                    retry_count=task.retry_count,
                    scheduled_at=task.scheduled_at.isoformat()
                )
                
                # Re-queue for retry
                await self.task_queue.put((-task.priority.value, task.created_at, task))
                
                # Don't update stats yet, wait for final outcome
                return
            
            # Send error notification
            if self.notification_service:
                await self.notification_service.send_notification(
                    user_id=task.context.user_id,
                    notification_type="task_failed",
                    context={
                        "task_id": task.task_id,
                        "task_type": task.task_type.value,
                        "error_message": str(e),
                        "retry_count": task.retry_count
                    }
                )
    
    # Task handlers
    
    async def _handle_document_processing(self, task: BackgroundTask) -> TaskResult:
        """Handle document processing task."""
        document_id = task.parameters["document_id"]
        case_id = task.parameters["case_id"]
        force_retry = task.parameters.get("force_retry", False)
        
        try:
            # Initialize progress tracking
            progress = TaskProgress(
                current_step="starting",
                total_steps=5,
                completed_steps=0,
                progress_percentage=0.0
            )
            task.progress = progress
            
            # Get document
            document = await self.document_repository.get_document(document_id)
            if not document:
                raise DocumentProcessingError(
                    f"Document {document_id} not found",
                    error_code=ErrorCode.DOCUMENT_NOT_FOUND
                )
            
            # Progress update
            progress.update("validation", 1, 5, "Validating document")
            await self._send_task_progress_update(task)
            
            # Process document using processor
            success = await self.document_processor.process_document(
                document_id=document_id,
                case_id=case_id,
                user_id=task.context.user_id
            )
            
            if success:
                progress.update("completed", 5, 5, "Processing completed")
                await self._send_task_progress_update(task)
                
                return TaskResult(
                    success=True,
                    result_data={
                        "document_id": document_id,
                        "case_id": case_id,
                        "processing_method": "LlamaIndex"
                    }
                )
            else:
                raise DocumentProcessingError(
                    "Document processing failed",
                    error_code=ErrorCode.DOCUMENT_PROCESSING_FAILED
                )
                
        except Exception as e:
            logger.error(
                "Document processing task failed",
                task_id=task.task_id,
                document_id=document_id,
                error=str(e)
            )
            raise
    
    async def _handle_document_retry(self, task: BackgroundTask) -> TaskResult:
        """Handle document retry task."""
        document_id = task.parameters["document_id"]
        case_id = task.parameters["case_id"]
        
        # Use document processing with force retry
        task.parameters["force_retry"] = True
        return await self._handle_document_processing(task)
    
    async def _handle_batch_operation(self, task: BackgroundTask) -> TaskResult:
        """Handle batch document operation task."""
        operation = task.parameters["operation"]
        document_ids = task.parameters["document_ids"]
        operation_params = task.parameters.get("operation_parameters", {})
        
        # Initialize progress tracking
        progress = TaskProgress(
            current_step="starting",
            total_steps=len(document_ids),
            completed_steps=0,
            progress_percentage=0.0
        )
        task.progress = progress
        
        results = {
            "operation": operation,
            "total_documents": len(document_ids),
            "successful": [],
            "failed": [],
            "errors": []
        }
        
        for i, document_id in enumerate(document_ids):
            try:
                progress.update(
                    f"processing_{i+1}",
                    i,
                    len(document_ids),
                    f"Processing document {i+1}/{len(document_ids)}"
                )
                await self._send_task_progress_update(task)
                
                # Execute operation based on type
                if operation == "retry":
                    success = await self._retry_document(document_id, task.context.case_id)
                elif operation == "delete":
                    success = await self._delete_document(document_id, task.context.user_id)
                elif operation == "reprocess":
                    success = await self._reprocess_document(document_id, task.context.case_id)
                else:
                    raise TaskError(f"Unsupported batch operation: {operation}")
                
                if success:
                    results["successful"].append(document_id)
                else:
                    results["failed"].append(document_id)
                    
            except Exception as e:
                results["failed"].append(document_id)
                results["errors"].append({
                    "document_id": document_id,
                    "error": str(e)
                })
                
                logger.error(
                    "Batch operation failed for document",
                    task_id=task.task_id,
                    operation=operation,
                    document_id=document_id,
                    error=str(e)
                )
        
        # Final progress update
        progress.update("completed", len(document_ids), len(document_ids), "Batch operation completed")
        await self._send_task_progress_update(task)
        
        return TaskResult(
            success=len(results["failed"]) == 0,
            result_data=results
        )
    
    async def _handle_document_cleanup(self, task: BackgroundTask) -> TaskResult:
        """Handle document cleanup task."""
        # Implement cleanup logic for orphaned documents, expired processing tasks, etc.
        return TaskResult(success=True, result_data={"cleanup_type": "document"})
    
    async def _handle_embedding_generation(self, task: BackgroundTask) -> TaskResult:
        """Handle standalone embedding generation task."""
        # Implement embedding generation for specific chunks or documents
        return TaskResult(success=True, result_data={"embeddings_generated": 0})
    
    async def _handle_index_optimization(self, task: BackgroundTask) -> TaskResult:
        """Handle vector index optimization task."""
        # Implement Weaviate index optimization
        return TaskResult(success=True, result_data={"optimization_type": "vector_index"})
    
    async def _handle_analytics_generation(self, task: BackgroundTask) -> TaskResult:
        """Handle analytics and reporting generation task."""
        # Implement analytics generation
        return TaskResult(success=True, result_data={"analytics_type": "document_processing"})
    
    async def _handle_system_maintenance(self, task: BackgroundTask) -> TaskResult:
        """Handle system maintenance task."""
        # Implement system maintenance operations
        return TaskResult(success=True, result_data={"maintenance_type": "general"})
    
    # Helper methods
    
    async def _retry_document(self, document_id: str, case_id: Optional[str]) -> bool:
        """Retry processing for a specific document."""
        try:
            return await self.document_processor.retry_document_processing(
                document_id, case_id or "", force_retry=True
            )
        except Exception as e:
            logger.error(f"Document retry failed: {e}")
            return False
    
    async def _delete_document(self, document_id: str, user_id: str) -> bool:
        """Delete a specific document."""
        try:
            return await self.document_service.delete_document(
                document_id, user_id, force=True
            )
        except Exception as e:
            logger.error(f"Document deletion failed: {e}")
            return False
    
    async def _reprocess_document(self, document_id: str, case_id: Optional[str]) -> bool:
        """Reprocess a specific document."""
        try:
            # Reset document status and reprocess
            document = await self.document_repository.get_document(document_id)
            if document:
                document.reset_for_reprocessing()
                await self.document_repository.update_document(document)
                
                return await self.document_processor.process_document(
                    document_id, case_id or "", None
                )
            return False
        except Exception as e:
            logger.error(f"Document reprocessing failed: {e}")
            return False
    
    async def _send_task_progress_update(self, task: BackgroundTask) -> None:
        """Send task progress update via notification service."""
        if not self.notification_service or not task.progress:
            return
        
        try:
            await self.notification_service.send_notification(
                user_id=task.context.user_id,
                notification_type="task_progress",
                context={
                    "task_id": task.task_id,
                    "task_type": task.task_type.value,
                    "current_step": task.progress.current_step,
                    "progress_percentage": task.progress.progress_percentage,
                    "current_operation": task.progress.current_operation,
                    "document_id": task.context.document_id or "",
                    "case_id": task.context.case_id or ""
                }
            )
        except Exception as e:
            logger.warning(
                "Failed to send task progress update",
                task_id=task.task_id,
                error=str(e)
            )
    
    def _update_task_stats(self, task: BackgroundTask) -> None:
        """Update task execution statistics."""
        if task.status == TaskStatus.COMPLETED:
            self.stats.completed_tasks += 1
            self.stats.running_tasks -= 1
            
            # Update average execution time
            if task.result and task.result.execution_time_ms > 0:
                current_avg = self.stats.average_execution_time_ms
                completed = self.stats.completed_tasks
                
                self.stats.average_execution_time_ms = (
                    (current_avg * (completed - 1) + task.result.execution_time_ms) / completed
                )
                
        elif task.status == TaskStatus.FAILED:
            self.stats.failed_tasks += 1
            self.stats.running_tasks -= 1
            
        elif task.status == TaskStatus.CANCELLED:
            self.stats.cancelled_tasks += 1
            
            if task.task_id in self.active_tasks:
                self.stats.running_tasks -= 1
            else:
                self.stats.pending_tasks -= 1
    
    async def _cleanup_loop(self) -> None:
        """Background cleanup loop for expired tasks and maintenance."""
        while self._running:
            try:
                current_time = datetime.now(timezone.utc)
                cleanup_count = 0
                
                # Clean up expired tasks from registry
                expired_task_ids = [
                    task_id for task_id, task in self.task_registry.items()
                    if task.is_expired() and task.status not in [
                        TaskStatus.RUNNING, TaskStatus.RETRYING
                    ]
                ]
                
                for task_id in expired_task_ids:
                    task = self.task_registry[task_id]
                    if task.status == TaskStatus.PENDING:
                        task.status = TaskStatus.FAILED
                        task.result = TaskResult(
                            success=False,
                            error_message="Task expired",
                            error_code="TASK_EXPIRED"
                        )
                        self._update_task_stats(task)
                    
                    # Move to history
                    self.task_history.append(task)
                    del self.task_registry[task_id]
                    cleanup_count += 1
                
                # Limit task history size
                if len(self.task_history) > 1000:
                    # Remove oldest entries
                    while len(self.task_history) > 1000:
                        self.task_history.popleft()
                
                if cleanup_count > 0:
                    logger.debug(f"Cleaned up {cleanup_count} expired tasks")
                
                # Sleep before next cleanup cycle
                await asyncio.sleep(60)  # Run every minute
                
            except Exception as e:
                logger.error(
                    "Error in cleanup loop",
                    error=str(e),
                    exc_info=True
                )
                await asyncio.sleep(60)
    
    async def cleanup(self) -> None:
        """Cleanup task manager resources."""
        await self.stop()
        
        # Clear state
        self.task_registry.clear()
        self.task_history.clear()
        self.active_tasks.clear()
        
        # Drain queue
        while not self.task_queue.empty():
            try:
                self.task_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        logger.info("DocumentTaskManager cleanup completed")