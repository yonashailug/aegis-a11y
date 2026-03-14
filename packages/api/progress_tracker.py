"""
Enhanced Progress Tracking System for Long-Running Operations

This module provides comprehensive progress tracking capabilities including:
- Real-time progress updates
- WebSocket notifications  
- Detailed operation breakdowns
- Performance metrics
- Persistent progress storage
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
from pathlib import Path

from config import get_settings


class OperationType(str, Enum):
    """Types of operations that can be tracked."""
    BATCH_PROCESSING = "batch_processing"
    DOCUMENT_ANALYSIS = "document_analysis"  
    PDF_DECOMPOSITION = "pdf_decomposition"
    REASONING_ANALYSIS = "reasoning_analysis"
    DOCUMENT_RECONSTRUCTION = "document_reconstruction"
    MODEL_LOADING = "model_loading"


class OperationStatus(str, Enum):
    """Status of tracked operations."""
    PENDING = "pending"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProgressStep:
    """Individual step in an operation's progress."""
    step_id: str
    name: str
    description: str
    status: OperationStatus = OperationStatus.PENDING
    progress_percentage: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


@dataclass
class PerformanceMetrics:
    """Performance metrics for operations."""
    processing_rate: float = 0.0  # items/second
    estimated_completion_time: Optional[datetime] = None
    memory_usage_mb: float = 0.0
    cpu_usage_percentage: float = 0.0
    api_calls_made: int = 0
    api_calls_cached: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'processing_rate': self.processing_rate,
            'estimated_completion_time': self.estimated_completion_time.isoformat() if self.estimated_completion_time else None,
            'memory_usage_mb': self.memory_usage_mb,
            'cpu_usage_percentage': self.cpu_usage_percentage,
            'api_calls_made': self.api_calls_made,
            'api_calls_cached': self.api_calls_cached,
        }


@dataclass
class OperationProgress:
    """Complete progress tracking for an operation."""
    operation_id: str
    operation_type: OperationType
    name: str
    description: str
    status: OperationStatus = OperationStatus.PENDING
    overall_progress_percentage: float = 0.0
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_steps: int = 0
    completed_steps: int = 0
    current_step: Optional[str] = None
    steps: List[ProgressStep] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None
    performance_metrics: PerformanceMetrics = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.steps is None:
            self.steps = []
        if self.metadata is None:
            self.metadata = {}
        if self.performance_metrics is None:
            self.performance_metrics = PerformanceMetrics()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'operation_id': self.operation_id,
            'operation_type': self.operation_type.value,
            'name': self.name,
            'description': self.description,
            'status': self.status.value,
            'overall_progress_percentage': self.overall_progress_percentage,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'total_steps': self.total_steps,
            'completed_steps': self.completed_steps,
            'current_step': self.current_step,
            'steps': [asdict(step) for step in self.steps],
            'error_message': self.error_message,
            'metadata': self.metadata,
            'performance_metrics': self.performance_metrics.to_dict(),
        }


class ProgressTracker:
    """
    Enhanced progress tracking system for long-running operations.
    
    Provides real-time progress updates, performance metrics, and
    detailed operation breakdowns.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = logging.getLogger(__name__)
        self.active_operations: Dict[str, OperationProgress] = {}
        self.subscribers: Dict[str, List[Callable]] = {}  # operation_id -> callbacks
        self.performance_history: Dict[str, List[PerformanceMetrics]] = {}
        
        # Create progress storage directory
        self.progress_dir = self.settings.output.output_dir / "progress"
        self.progress_dir.mkdir(parents=True, exist_ok=True)
    
    def create_operation(
        self,
        name: str,
        description: str,
        operation_type: OperationType,
        total_steps: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new tracked operation.
        
        Args:
            name: Display name for the operation
            description: Detailed description
            operation_type: Type of operation being tracked
            total_steps: Expected number of steps (if known)
            metadata: Additional operation metadata
            
        Returns:
            Operation ID for tracking
        """
        operation_id = str(uuid.uuid4())
        
        operation = OperationProgress(
            operation_id=operation_id,
            operation_type=operation_type,
            name=name,
            description=description,
            total_steps=total_steps,
            metadata=metadata or {}
        )
        
        self.active_operations[operation_id] = operation
        self.subscribers[operation_id] = []
        self.performance_history[operation_id] = []
        
        self.logger.info(f"Created operation {operation_id}: {name}")
        self._save_progress(operation_id)
        self._notify_subscribers(operation_id)
        
        return operation_id
    
    def add_step(
        self,
        operation_id: str,
        step_name: str,
        step_description: str,
        step_id: Optional[str] = None
    ) -> str:
        """
        Add a new step to an operation.
        
        Args:
            operation_id: ID of the operation
            step_name: Name of the step
            step_description: Description of the step
            step_id: Optional custom step ID
            
        Returns:
            Step ID
        """
        if operation_id not in self.active_operations:
            raise ValueError(f"Operation {operation_id} not found")
        
        if step_id is None:
            step_id = str(uuid.uuid4())
        
        step = ProgressStep(
            step_id=step_id,
            name=step_name,
            description=step_description
        )
        
        operation = self.active_operations[operation_id]
        operation.steps.append(step)
        operation.total_steps = len(operation.steps)
        
        self.logger.debug(f"Added step {step_id} to operation {operation_id}: {step_name}")
        self._save_progress(operation_id)
        self._notify_subscribers(operation_id)
        
        return step_id
    
    def start_operation(self, operation_id: str):
        """Start an operation."""
        if operation_id not in self.active_operations:
            raise ValueError(f"Operation {operation_id} not found")
        
        operation = self.active_operations[operation_id]
        operation.status = OperationStatus.RUNNING
        operation.started_at = datetime.now()
        
        self.logger.info(f"Started operation {operation_id}: {operation.name}")
        self._save_progress(operation_id)
        self._notify_subscribers(operation_id)
    
    def start_step(self, operation_id: str, step_id: str):
        """Start a specific step."""
        if operation_id not in self.active_operations:
            raise ValueError(f"Operation {operation_id} not found")
        
        operation = self.active_operations[operation_id]
        step = self._find_step(operation, step_id)
        
        if step:
            step.status = OperationStatus.RUNNING
            step.start_time = datetime.now()
            operation.current_step = step_id
            
            self.logger.debug(f"Started step {step_id} in operation {operation_id}: {step.name}")
            self._update_overall_progress(operation_id)
            self._save_progress(operation_id)
            self._notify_subscribers(operation_id)
    
    def update_step_progress(
        self,
        operation_id: str,
        step_id: str,
        progress_percentage: float,
        details: Optional[Dict[str, Any]] = None
    ):
        """Update progress for a specific step."""
        if operation_id not in self.active_operations:
            raise ValueError(f"Operation {operation_id} not found")
        
        operation = self.active_operations[operation_id]
        step = self._find_step(operation, step_id)
        
        if step:
            step.progress_percentage = max(0.0, min(100.0, progress_percentage))
            if details:
                step.details.update(details)
            
            self._update_overall_progress(operation_id)
            self._update_performance_metrics(operation_id)
            self._save_progress(operation_id)
            self._notify_subscribers(operation_id)
    
    def complete_step(
        self,
        operation_id: str,
        step_id: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """Mark a step as completed."""
        if operation_id not in self.active_operations:
            raise ValueError(f"Operation {operation_id} not found")
        
        operation = self.active_operations[operation_id]
        step = self._find_step(operation, step_id)
        
        if step:
            step.status = OperationStatus.COMPLETED
            step.progress_percentage = 100.0
            step.end_time = datetime.now()
            if details:
                step.details.update(details)
            
            operation.completed_steps = sum(
                1 for s in operation.steps if s.status == OperationStatus.COMPLETED
            )
            
            self.logger.debug(f"Completed step {step_id} in operation {operation_id}: {step.name}")
            self._update_overall_progress(operation_id)
            self._check_operation_completion(operation_id)
            self._save_progress(operation_id)
            self._notify_subscribers(operation_id)
    
    def fail_step(
        self,
        operation_id: str,
        step_id: str,
        error_message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """Mark a step as failed."""
        if operation_id not in self.active_operations:
            raise ValueError(f"Operation {operation_id} not found")
        
        operation = self.active_operations[operation_id]
        step = self._find_step(operation, step_id)
        
        if step:
            step.status = OperationStatus.FAILED
            step.error_message = error_message
            step.end_time = datetime.now()
            if details:
                step.details.update(details)
            
            self.logger.warning(f"Failed step {step_id} in operation {operation_id}: {error_message}")
            self._save_progress(operation_id)
            self._notify_subscribers(operation_id)
    
    def complete_operation(
        self,
        operation_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Mark an operation as completed."""
        if operation_id not in self.active_operations:
            raise ValueError(f"Operation {operation_id} not found")
        
        operation = self.active_operations[operation_id]
        operation.status = OperationStatus.COMPLETED
        operation.completed_at = datetime.now()
        operation.overall_progress_percentage = 100.0
        
        if metadata:
            operation.metadata.update(metadata)
        
        self.logger.info(f"Completed operation {operation_id}: {operation.name}")
        self._save_progress(operation_id)
        self._notify_subscribers(operation_id)
        
        # Archive the operation after a delay
        asyncio.create_task(self._archive_operation_later(operation_id))
    
    def fail_operation(
        self,
        operation_id: str,
        error_message: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Mark an operation as failed."""
        if operation_id not in self.active_operations:
            raise ValueError(f"Operation {operation_id} not found")
        
        operation = self.active_operations[operation_id]
        operation.status = OperationStatus.FAILED
        operation.error_message = error_message
        operation.completed_at = datetime.now()
        
        if metadata:
            operation.metadata.update(metadata)
        
        self.logger.error(f"Failed operation {operation_id}: {error_message}")
        self._save_progress(operation_id)
        self._notify_subscribers(operation_id)
    
    def get_operation_progress(self, operation_id: str) -> Optional[OperationProgress]:
        """Get current progress for an operation."""
        return self.active_operations.get(operation_id)
    
    def list_active_operations(self) -> List[OperationProgress]:
        """List all active operations."""
        return list(self.active_operations.values())
    
    def subscribe_to_operation(self, operation_id: str, callback: Callable[[OperationProgress], None]):
        """Subscribe to progress updates for an operation."""
        if operation_id not in self.subscribers:
            self.subscribers[operation_id] = []
        self.subscribers[operation_id].append(callback)
    
    def unsubscribe_from_operation(self, operation_id: str, callback: Callable[[OperationProgress], None]):
        """Unsubscribe from progress updates."""
        if operation_id in self.subscribers:
            try:
                self.subscribers[operation_id].remove(callback)
            except ValueError:
                pass
    
    def cancel_operation(self, operation_id: str) -> bool:
        """Cancel a running operation."""
        if operation_id not in self.active_operations:
            return False
        
        operation = self.active_operations[operation_id]
        if operation.status in [OperationStatus.RUNNING, OperationStatus.PENDING]:
            operation.status = OperationStatus.CANCELLED
            operation.completed_at = datetime.now()
            
            self.logger.info(f"Cancelled operation {operation_id}: {operation.name}")
            self._save_progress(operation_id)
            self._notify_subscribers(operation_id)
            return True
        
        return False
    
    def _find_step(self, operation: OperationProgress, step_id: str) -> Optional[ProgressStep]:
        """Find a step by ID within an operation."""
        return next((step for step in operation.steps if step.step_id == step_id), None)
    
    def _update_overall_progress(self, operation_id: str):
        """Update overall progress percentage based on step progress."""
        operation = self.active_operations[operation_id]
        
        if not operation.steps:
            return
        
        total_progress = sum(step.progress_percentage for step in operation.steps)
        operation.overall_progress_percentage = total_progress / len(operation.steps)
    
    def _update_performance_metrics(self, operation_id: str):
        """Update performance metrics for an operation."""
        operation = self.active_operations[operation_id]
        
        if not operation.started_at:
            return
        
        elapsed_time = (datetime.now() - operation.started_at).total_seconds()
        if elapsed_time > 0:
            # Calculate processing rate based on completed steps
            operation.performance_metrics.processing_rate = operation.completed_steps / elapsed_time
            
            # Estimate completion time
            if operation.overall_progress_percentage > 0:
                total_estimated_time = elapsed_time * (100 / operation.overall_progress_percentage)
                remaining_time = total_estimated_time - elapsed_time
                operation.performance_metrics.estimated_completion_time = (
                    datetime.now() + timedelta(seconds=remaining_time)
                )
        
        # Store performance history
        self.performance_history[operation_id].append(PerformanceMetrics(
            processing_rate=operation.performance_metrics.processing_rate,
            estimated_completion_time=operation.performance_metrics.estimated_completion_time,
            memory_usage_mb=operation.performance_metrics.memory_usage_mb,
            cpu_usage_percentage=operation.performance_metrics.cpu_usage_percentage,
            api_calls_made=operation.performance_metrics.api_calls_made,
            api_calls_cached=operation.performance_metrics.api_calls_cached,
        ))
    
    def _check_operation_completion(self, operation_id: str):
        """Check if an operation should be marked as completed."""
        operation = self.active_operations[operation_id]
        
        if operation.status != OperationStatus.RUNNING:
            return
        
        # Check if all steps are completed
        if operation.steps and all(
            step.status in [OperationStatus.COMPLETED, OperationStatus.FAILED]
            for step in operation.steps
        ):
            # Check if any steps failed
            failed_steps = [step for step in operation.steps if step.status == OperationStatus.FAILED]
            
            if failed_steps:
                # Mark operation as failed if any critical steps failed
                self.fail_operation(operation_id, f"Operation failed due to {len(failed_steps)} failed steps")
            else:
                # Mark as completed if all steps succeeded
                self.complete_operation(operation_id)
    
    def _save_progress(self, operation_id: str):
        """Save operation progress to file."""
        try:
            operation = self.active_operations[operation_id]
            progress_file = self.progress_dir / f"{operation_id}.json"
            
            with open(progress_file, 'w') as f:
                json.dump(operation.to_dict(), f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Failed to save progress for operation {operation_id}: {e}")
    
    def _notify_subscribers(self, operation_id: str):
        """Notify all subscribers about progress updates."""
        if operation_id not in self.subscribers:
            return
        
        operation = self.active_operations[operation_id]
        
        for callback in self.subscribers[operation_id]:
            try:
                callback(operation)
            except Exception as e:
                self.logger.error(f"Error in progress callback: {e}")
    
    async def _archive_operation_later(self, operation_id: str, delay_minutes: int = 60):
        """Archive an operation after a delay."""
        await asyncio.sleep(delay_minutes * 60)
        
        if operation_id in self.active_operations:
            operation = self.active_operations[operation_id]
            
            # Move to archived operations (could be implemented later)
            self.logger.info(f"Archiving operation {operation_id}: {operation.name}")
            
            # Clean up
            del self.active_operations[operation_id]
            if operation_id in self.subscribers:
                del self.subscribers[operation_id]
            if operation_id in self.performance_history:
                del self.performance_history[operation_id]
    
    def get_performance_history(self, operation_id: str) -> List[PerformanceMetrics]:
        """Get performance history for an operation."""
        return self.performance_history.get(operation_id, [])


# Global progress tracker instance
_progress_tracker = None

def get_progress_tracker() -> ProgressTracker:
    """Get the global progress tracker instance."""
    global _progress_tracker
    if _progress_tracker is None:
        _progress_tracker = ProgressTracker()
    return _progress_tracker
