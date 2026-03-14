"""
Batch processing service for handling multiple PDF documents.

This module provides functionality to process multiple PDFs concurrently
while respecting configuration limits and providing progress tracking.
"""

import asyncio
import logging
import os
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

from config import get_settings
from .progress_tracker import get_progress_tracker, OperationType, OperationStatus
from .cache_manager import get_cache_manager, CacheType
from cv_layer import LayoutDecomposer, convert_pdf_to_images, extract_ocr_data
from reasoning_agent import (
    AltTextGenerator,
    ContextProcessor, 
    DeterministicVerifier,
    ReasoningInput,
    SemanticReasoner,
)
from reasoning_agent.element_filter import ElementFilter
from reconstruction import (
    DocumentReconstructionEngine,
    OutputFormat,
    ReconstructionInput,
)


class BatchStatus(str, Enum):
    """Status of batch processing."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DocumentStatus(str, Enum):
    """Status of individual document processing."""
    PENDING = "pending"
    PROCESSING = "processing" 
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class BatchDocument:
    """Represents a document in a batch."""
    id: str
    file_path: str
    original_filename: str
    status: DocumentStatus = DocumentStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    elements_extracted: int = 0
    elements_analyzed: int = 0
    output_files: Dict[str, str] = None
    processing_duration: float = 0.0
    
    def __post_init__(self):
        if self.output_files is None:
            self.output_files = {}


@dataclass
class BatchJob:
    """Represents a batch processing job."""
    id: str
    status: BatchStatus = BatchStatus.PENDING
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_documents: int = 0
    processed_documents: int = 0
    failed_documents: int = 0
    documents: List[BatchDocument] = None
    output_directory: Optional[Path] = None
    progress_percentage: float = 0.0
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.documents is None:
            self.documents = []


class BatchProcessor:
    """
    Service for batch processing of multiple PDF documents.
    
    Handles concurrent processing while respecting configuration limits
    and providing real-time progress tracking.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = logging.getLogger(__name__)
        self.active_batches: Dict[str, BatchJob] = {}
        self.progress_tracker = get_progress_tracker()
        self.cache_manager = get_cache_manager()
        
        # Initialize components (similar to main.py but with error handling)
        self._init_components()
    
    def _init_components(self):
        """Initialize processing components with error handling."""
        try:
            self.decomposer = LayoutDecomposer(self.settings.models.layoutlm_model)
            self.logger.info("LayoutDecomposer initialized for batch processing")
        except Exception as e:
            self.logger.error(f"Failed to initialize LayoutDecomposer: {e}")
            self.decomposer = None
            
        try:
            self.context_processor = ContextProcessor()
            self.alt_text_generator = AltTextGenerator()
            
            # Get API key for the configured model
            reasoning_model = self.settings.get_model_config("reasoning")
            api_key = self.settings.get_api_key(reasoning_model.provider.value)
            
            if api_key:
                self.semantic_reasoner = SemanticReasoner(
                    api_key=api_key,
                    model=reasoning_model.model_name,
                    max_tokens=reasoning_model.max_tokens,
                    temperature=reasoning_model.temperature,
                )
            else:
                self.semantic_reasoner = SemanticReasoner()
                
            self.verifier = DeterministicVerifier()
            self.element_filter = ElementFilter()
            self.logger.info("Reasoning components initialized for batch processing")
        except Exception as e:
            self.logger.error(f"Failed to initialize reasoning components: {e}")
            self.context_processor = None
            self.alt_text_generator = None
            self.semantic_reasoner = None
            self.verifier = None
            self.element_filter = None
            
        try:
            self.reconstruction_engine = DocumentReconstructionEngine()
            self.logger.info("Reconstruction engine initialized for batch processing")
        except Exception as e:
            self.logger.error(f"Failed to initialize reconstruction engine: {e}")
            self.reconstruction_engine = None
    
    def create_batch(self, file_paths: List[str], output_dir: Optional[str] = None) -> str:
        """
        Create a new batch processing job.
        
        Args:
            file_paths: List of PDF file paths to process
            output_dir: Optional custom output directory
            
        Returns:
            Batch job ID
            
        Raises:
            ValueError: If batch size exceeds limits or files don't exist
        """
        if not self.settings.processing.enable_batch_processing:
            raise ValueError("Batch processing is disabled in configuration")
            
        if len(file_paths) > self.settings.processing.max_batch_size:
            raise ValueError(f"Batch size {len(file_paths)} exceeds maximum of {self.settings.processing.max_batch_size}")
        
        # Validate all files exist
        missing_files = [path for path in file_paths if not os.path.exists(path)]
        if missing_files:
            raise ValueError(f"Files not found: {missing_files}")
            
        # Create batch job
        batch_id = str(uuid.uuid4())
        batch_output_dir = Path(output_dir) if output_dir else self.settings.output.output_dir / f"batch_{batch_id}"
        batch_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create documents
        documents = []
        for file_path in file_paths:
            doc_id = str(uuid.uuid4())
            documents.append(BatchDocument(
                id=doc_id,
                file_path=file_path,
                original_filename=os.path.basename(file_path)
            ))
        
        batch_job = BatchJob(
            id=batch_id,
            total_documents=len(documents),
            documents=documents,
            output_directory=batch_output_dir
        )
        
        self.active_batches[batch_id] = batch_job
        self.logger.info(f"Created batch {batch_id} with {len(file_paths)} documents")
        
        return batch_id
    
    async def process_batch(self, batch_id: str) -> BatchJob:
        """
        Process a batch of documents asynchronously.
        
        Args:
            batch_id: ID of the batch to process
            
        Returns:
            Completed batch job
            
        Raises:
            ValueError: If batch not found or components not available
        """
        if batch_id not in self.active_batches:
            raise ValueError(f"Batch {batch_id} not found")
            
        batch_job = self.active_batches[batch_id]
        
        # Check if components are available
        if not self.decomposer:
            raise ValueError("LayoutDecomposer not available")
        if not all([self.context_processor, self.semantic_reasoner, self.verifier]):
            raise ValueError("Reasoning components not available")
        if not self.reconstruction_engine:
            raise ValueError("Reconstruction engine not available")
            
        batch_job.status = BatchStatus.RUNNING
        batch_job.started_at = datetime.now()
        
        # Create progress tracking for this batch
        progress_id = self.progress_tracker.create_operation(
            name=f"Batch Processing ({len(batch_job.documents)} documents)",
            description=f"Processing batch {batch_id} with {len(batch_job.documents)} PDF documents",
            operation_type=OperationType.BATCH_PROCESSING,
            metadata={
                "batch_id": batch_id,
                "total_documents": len(batch_job.documents),
                "output_directory": str(batch_job.output_directory)
            }
        )
        
        # Add steps for each document
        for doc in batch_job.documents:
            self.progress_tracker.add_step(
                progress_id,
                f"Process {doc.original_filename}",
                f"Complete DRR pipeline for {doc.original_filename}",
                step_id=doc.id
            )
        
        self.progress_tracker.start_operation(progress_id)
        
        try:
            # Create semaphore to limit concurrent PDF processing
            semaphore = asyncio.Semaphore(self.settings.processing.max_concurrent_pdfs)
            
            # Process documents concurrently
            tasks = [
                self._process_document(batch_job, doc, semaphore, progress_id)
                for doc in batch_job.documents
            ]
            
            # Wait for all documents to complete with timeout
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.settings.processing.batch_timeout
            )
            
            # Update batch status
            batch_job.completed_at = datetime.now()
            batch_job.processed_documents = sum(
                1 for doc in batch_job.documents 
                if doc.status == DocumentStatus.COMPLETED
            )
            batch_job.failed_documents = sum(
                1 for doc in batch_job.documents
                if doc.status == DocumentStatus.FAILED
            )
            batch_job.progress_percentage = 100.0
            
            if batch_job.failed_documents == 0:
                batch_job.status = BatchStatus.COMPLETED
            elif batch_job.processed_documents > 0:
                batch_job.status = BatchStatus.COMPLETED  # Partial success
            else:
                batch_job.status = BatchStatus.FAILED
                
            # Complete progress tracking
            self.progress_tracker.complete_operation(
                progress_id,
                metadata={
                    "final_status": batch_job.status.value,
                    "processed_documents": batch_job.processed_documents,
                    "failed_documents": batch_job.failed_documents,
                }
            )
                
            self.logger.info(
                f"Batch {batch_id} completed: {batch_job.processed_documents}/{batch_job.total_documents} successful"
            )
            
        except asyncio.TimeoutError:
            batch_job.status = BatchStatus.FAILED
            batch_job.error_message = f"Batch processing timed out after {self.settings.processing.batch_timeout} seconds"
            batch_job.completed_at = datetime.now()
            
            self.progress_tracker.fail_operation(
                progress_id,
                f"Batch processing timed out after {self.settings.processing.batch_timeout} seconds"
            )
            
            self.logger.error(f"Batch {batch_id} timed out")
            
        except Exception as e:
            batch_job.status = BatchStatus.FAILED 
            batch_job.error_message = str(e)
            batch_job.completed_at = datetime.now()
            
            self.progress_tracker.fail_operation(progress_id, str(e))
            
            self.logger.error(f"Batch {batch_id} failed: {e}")
            
        return batch_job
    
    async def _process_document(self, batch_job: BatchJob, document: BatchDocument, semaphore: asyncio.Semaphore, progress_id: str):
        """Process a single document within a batch."""
        async with semaphore:
            document.status = DocumentStatus.PROCESSING
            document.start_time = datetime.now()
            
            # Start progress tracking for this document
            self.progress_tracker.start_step(progress_id, document.id)
            
            try:
                # Update batch progress
                self._update_batch_progress(batch_job)
                
                # Update progress - starting document pipeline
                self.progress_tracker.update_step_progress(
                    progress_id, 
                    document.id, 
                    10.0,
                    {"stage": "Starting document pipeline", "filename": document.original_filename}
                )
                
                result = await self._run_document_pipeline(document, batch_job.output_directory, progress_id)
                
                document.status = DocumentStatus.COMPLETED
                document.elements_extracted = result.get('elements_extracted', 0)
                document.elements_analyzed = result.get('elements_analyzed', 0)
                document.output_files = result.get('output_files', {})
                
                # Complete progress tracking for this document
                self.progress_tracker.complete_step(
                    progress_id,
                    document.id,
                    {
                        "elements_extracted": document.elements_extracted,
                        "elements_analyzed": document.elements_analyzed,
                        "output_files": document.output_files,
                        "processing_duration": (datetime.now() - document.start_time).total_seconds()
                    }
                )
                
                self.logger.info(f"Document {document.original_filename} processed successfully")
                
            except Exception as e:
                document.status = DocumentStatus.FAILED
                document.error_message = str(e)
                
                # Mark step as failed in progress tracking
                self.progress_tracker.fail_step(
                    progress_id,
                    document.id,
                    str(e),
                    {"filename": document.original_filename}
                )
                
                self.logger.error(f"Failed to process document {document.original_filename}: {e}")
                
            finally:
                document.end_time = datetime.now()
                if document.start_time:
                    document.processing_duration = (document.end_time - document.start_time).total_seconds()
                self._update_batch_progress(batch_job)
    
    async def _run_document_pipeline(self, document: BatchDocument, output_directory: Path, progress_id: str) -> Dict:
        """Run the complete DRR pipeline for a single document."""
        # Convert to async (run in thread pool to avoid blocking)
        loop = asyncio.get_event_loop()
        
        def run_pipeline():
            # Step 1: PDF Decomposition
            asyncio.run_coroutine_threadsafe(
                self._update_progress_async(progress_id, document.id, 20.0, {"stage": "PDF Decomposition"}),
                loop
            )
            
            images = convert_pdf_to_images(document.file_path)
            
            extracted_elements = []
            verified_reasoning_outputs = []
            
            total_pages = len(images)
            for page_num, image in enumerate(images):
                # Update progress for page processing
                page_progress = 20.0 + (page_num / total_pages) * 30.0  # 20-50% for decomposition
                asyncio.run_coroutine_threadsafe(
                    self._update_progress_async(
                        progress_id, 
                        document.id, 
                        page_progress,
                        {"stage": f"Decomposing page {page_num + 1}/{total_pages}"}
                    ),
                    loop
                )
                
                # Extract OCR data
                ocr_data = extract_ocr_data(image)
                
                # Decompose image layout
                page_elements = self.decomposer.decompose_image(image, ocr_data)
                
                # Convert to list and add page info
                for element in page_elements:
                    element_dict = (
                        element.model_dump() if hasattr(element, "model_dump") else element
                    )
                    element_dict["page_number"] = page_num + 1
                    extracted_elements.append(element_dict)
            
            # Step 2: Apply filtering
            asyncio.run_coroutine_threadsafe(
                self._update_progress_async(progress_id, document.id, 50.0, {"stage": "Filtering elements"}),
                loop
            )
            
            if self.element_filter:
                filtered_elements_data = self.element_filter.filter_elements(extracted_elements)
                filtered_elements_for_processing = [
                    {
                        "element": fe.element,
                        "aggregated_text": fe.aggregated_text,
                        "processing_priority": fe.processing_priority,
                        "filter_reason": fe.filter_reason,
                    }
                    for fe in filtered_elements_data
                ]
            else:
                filtered_elements_for_processing = [
                    {
                        "element": elem,
                        "aggregated_text": elem.get("ocr_text", ""),
                        "processing_priority": 2,
                        "filter_reason": "No filtering",
                    }
                    for elem in extracted_elements
                ]
            
            # Step 3: Reasoning Analysis
            asyncio.run_coroutine_threadsafe(
                self._update_progress_async(
                    progress_id, 
                    document.id, 
                    55.0,
                    {"stage": f"Reasoning analysis ({len(filtered_elements_for_processing)} elements)"}
                ),
                loop
            )
            
            total_elements = len(filtered_elements_for_processing)
            for idx, filtered_data in enumerate(filtered_elements_for_processing):
                # Update progress for element processing
                element_progress = 55.0 + (idx / total_elements) * 30.0  # 55-85% for reasoning
                asyncio.run_coroutine_threadsafe(
                    self._update_progress_async(
                        progress_id, 
                        document.id, 
                        element_progress,
                        {"stage": f"Analyzing element {idx + 1}/{total_elements}"}
                    ),
                    loop
                )
                element = filtered_data["element"]
                try:
                    # Generate cache key for this element
                    cache_key = self.cache_manager._generate_cache_key(
                        {
                            "element_text": element.get("ocr_text", ""),
                            "element_type": element.get("element_type", ""),
                            "bbox": element.get("bbox", []),
                            "page_number": element.get("page_number", 1)
                        },
                        "reasoning_"
                    )
                    
                    # Try to get from cache first
                    cached_result = self.cache_manager.get(cache_key, CacheType.REASONING_RESULT)
                    
                    if cached_result:
                        # Use cached result
                        reasoning_output = cached_result
                        self.logger.debug(f"Using cached reasoning result for element")
                    else:
                        # Process element normally
                        reasoning_input = ReasoningInput.from_cv_output(
                            extracted_element=element,
                            full_page_image=None,
                            page_metadata={
                                "page_number": element.get("page_number", 1),
                                "total_pages": len(images),
                                "document_path": document.file_path,
                            },
                        )
                        
                        reasoning_output = self.semantic_reasoner.process_element(reasoning_input)
                        
                        # Cache the result
                        self.cache_manager.put(
                            cache_key,
                            reasoning_output,
                            CacheType.REASONING_RESULT,
                            ttl=12 * 3600  # 12 hours
                        )
                    
                    verification_result = self.verifier.verify_reasoning_output(
                        reasoning_output,
                        element,
                        {"spatial_context": {}},
                    )
                    
                    if verification_result.overall_status.value == "pass":
                        verified_reasoning_outputs.append(reasoning_output)
                        
                except Exception as e:
                    self.logger.debug(f"Failed to process element: {e}")
                    continue
            
            # Step 4: Document Reconstruction
            asyncio.run_coroutine_threadsafe(
                self._update_progress_async(progress_id, document.id, 85.0, {"stage": "Document reconstruction"}),
                loop
            )
            
            output_files = {}
            if verified_reasoning_outputs:
                reconstruction_input = ReconstructionInput(
                    verified_elements=verified_reasoning_outputs,
                    original_layout=extracted_elements,
                    document_title=f"Accessible Document from {document.original_filename}",
                    document_language="en",
                    subject_area="general",
                    educational_level="general",
                    target_formats=[OutputFormat.HTML5, OutputFormat.PDF_UA],
                    preserve_layout=True,
                    include_metadata=True,
                    generate_navigation=True,
                )
                
                reconstruction_result = self.reconstruction_engine.reconstruct_document(reconstruction_input)
                
                # Save documents
                doc_name = Path(document.file_path).stem
                
                for format_type, doc_content in reconstruction_result.documents.items():
                    if format_type.value == "html5":
                        output_path = output_directory / f"{doc_name}_accessible.html"
                        with open(output_path, "w", encoding="utf-8") as f:
                            f.write(doc_content)
                        output_files["html5"] = str(output_path)
                    elif format_type.value == "pdf_ua":
                        output_path = output_directory / f"{doc_name}_accessible.pdf"
                        with open(output_path, "wb") as f:
                            f.write(doc_content)
                        output_files["pdf_ua"] = str(output_path)
            
            # Final progress update
            asyncio.run_coroutine_threadsafe(
                self._update_progress_async(
                    progress_id, 
                    document.id, 
                    95.0,
                    {"stage": "Finalizing", "output_files": list(output_files.keys())}
                ),
                loop
            )
            
            return {
                'elements_extracted': len(extracted_elements),
                'elements_analyzed': len(verified_reasoning_outputs),
                'output_files': output_files,
            }
        
        return await loop.run_in_executor(None, run_pipeline)
    
    async def _update_progress_async(self, progress_id: str, step_id: str, progress: float, details: dict):
        """Helper method to update progress from sync context."""
        self.progress_tracker.update_step_progress(progress_id, step_id, progress, details)
    
    def _update_batch_progress(self, batch_job: BatchJob):
        """Update batch progress based on document statuses."""
        completed = sum(
            1 for doc in batch_job.documents 
            if doc.status in [DocumentStatus.COMPLETED, DocumentStatus.FAILED]
        )
        batch_job.progress_percentage = (completed / batch_job.total_documents) * 100
    
    def get_batch_status(self, batch_id: str) -> Optional[BatchJob]:
        """Get current status of a batch job."""
        return self.active_batches.get(batch_id)
    
    def list_active_batches(self) -> List[BatchJob]:
        """List all active batch jobs."""
        return list(self.active_batches.values())
    
    def cancel_batch(self, batch_id: str) -> bool:
        """Cancel a running batch job."""
        if batch_id in self.active_batches:
            batch_job = self.active_batches[batch_id]
            if batch_job.status == BatchStatus.RUNNING:
                batch_job.status = BatchStatus.CANCELLED
                batch_job.completed_at = datetime.now()
                return True
        return False
    
    def cleanup_completed_batches(self, max_age_hours: int = 24):
        """Clean up completed batch jobs older than specified hours."""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        to_remove = [
            batch_id for batch_id, batch in self.active_batches.items()
            if batch.status in [BatchStatus.COMPLETED, BatchStatus.FAILED, BatchStatus.CANCELLED]
            and batch.completed_at and batch.completed_at < cutoff_time
        ]
        
        for batch_id in to_remove:
            del self.active_batches[batch_id]
            
        return len(to_remove)
