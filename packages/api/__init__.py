"""
Aegis-A11y API Package

Fast API application for document accessibility analysis and batch processing.
"""

from .batch_processor import BatchProcessor, BatchStatus, DocumentStatus
from .progress_tracker import get_progress_tracker, OperationType, OperationStatus


__version__ = "0.1.0"

__all__ = [
    "BatchProcessor",
    "BatchStatus",
    "DocumentStatus",
    "get_progress_tracker",
    "OperationType",
    "OperationStatus"
]
