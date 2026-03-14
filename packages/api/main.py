from datetime import datetime
import logging
import os
from pathlib import Path

from config import get_settings
from fastapi import FastAPI, HTTPException, status, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Optional

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
from batch_processor import BatchProcessor, BatchStatus, DocumentStatus
from progress_tracker import get_progress_tracker, OperationType, OperationStatus
from cache_manager import get_cache_manager, CacheType, CacheStats


# Pydantic models for API requests/responses
class BatchCreateRequest(BaseModel):
    """Request model for creating a batch processing job."""
    file_paths: List[str]
    output_directory: Optional[str] = None


class BatchStatusResponse(BaseModel):
    """Response model for batch status."""
    id: str
    status: str
    total_documents: int
    processed_documents: int
    failed_documents: int
    progress_percentage: float
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None


class DocumentStatusResponse(BaseModel):
    """Response model for individual document status."""
    id: str
    original_filename: str
    status: str
    elements_extracted: int = 0
    elements_analyzed: int = 0
    processing_duration: float = 0.0
    error_message: Optional[str] = None
    output_files: dict = {}


class BatchDetailResponse(BaseModel):
    """Detailed response model for batch information."""
    id: str
    status: str
    total_documents: int
    processed_documents: int
    failed_documents: int
    progress_percentage: float
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    documents: List[DocumentStatusResponse]


# Progress tracking models
class ProgressStepResponse(BaseModel):
    """Response model for progress step."""
    step_id: str
    name: str
    description: str
    status: str
    progress_percentage: float
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    error_message: Optional[str] = None
    details: dict = {}


class PerformanceMetricsResponse(BaseModel):
    """Response model for performance metrics."""
    processing_rate: float
    estimated_completion_time: Optional[str] = None
    memory_usage_mb: float
    cpu_usage_percentage: float
    api_calls_made: int
    api_calls_cached: int


class OperationProgressResponse(BaseModel):
    """Response model for operation progress."""
    operation_id: str
    operation_type: str
    name: str
    description: str
    status: str
    overall_progress_percentage: float
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    total_steps: int
    completed_steps: int
    current_step: Optional[str] = None
    steps: List[ProgressStepResponse] = []
    error_message: Optional[str] = None
    metadata: dict = {}
    performance_metrics: PerformanceMetricsResponse


# Cache management models
class CacheStatsResponse(BaseModel):
    """Response model for cache statistics."""
    hits: int
    misses: int
    hit_rate_percent: float
    evictions: int
    total_size_bytes: int
    api_calls_saved: int
    estimated_cost_saved_usd: float
    uptime_hours: float
    memory_cache_entries: int
    disk_cache_entries: int
    memory_usage_mb: float
    memory_limit_mb: float
    by_type: dict = {}


class CacheEntryResponse(BaseModel):
    """Response model for cache entry information."""
    key: str
    cache_type: str
    size_bytes: int
    created_at: str
    last_accessed: str
    access_count: int
    ttl_seconds: int
    expired: bool
    metadata: dict = {}


class CacheClearRequest(BaseModel):
    """Request model for clearing cache."""
    cache_type: Optional[str] = None

# Get centralized settings
settings = get_settings()

# Configure logging based on settings
logging.basicConfig(
    level=getattr(logging, settings.logging.level), format=settings.logging.format
)
logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, settings.get_log_level_for_component("api")))


def save_generated_documents(
    reconstruction_result, sample_pdf_path: str
) -> dict[str, str]:
    """Save generated documents to files and return file paths."""

    # Use configured output directory
    output_dir = settings.output.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create timestamped subdirectory if enabled
    pdf_name = Path(sample_pdf_path).stem
    if settings.output.create_timestamped_dirs:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = output_dir / f"{pdf_name}_{timestamp}"
    else:
        session_dir = output_dir / pdf_name
    session_dir.mkdir(exist_ok=True)

    project_root = Path(__file__).parent.parent.parent
    saved_files = {}

    for format_type, document in reconstruction_result.documents.items():
        try:
            if format_type.value == "html5":
                # Save HTML5 document
                html_file = session_dir / f"{pdf_name}_accessible.html"
                with open(html_file, "w", encoding="utf-8") as f:
                    f.write(document)
                # Store relative path from project root
                relative_html_path = html_file.relative_to(project_root)
                saved_files["html5"] = str(relative_html_path)
                logger.info(f"Saved HTML5 document: {relative_html_path}")

            elif format_type.value == "pdf_ua":
                # Save PDF/UA document
                pdf_file = session_dir / f"{pdf_name}_accessible.pdf"
                with open(pdf_file, "wb") as f:
                    f.write(document)
                # Store relative path from project root
                relative_pdf_path = pdf_file.relative_to(project_root)
                saved_files["pdf_ua"] = str(relative_pdf_path)
                logger.info(f"Saved PDF/UA document: {pdf_file}")

        except Exception as e:
            logger.error(f"Failed to save {format_type.value} document: {e}")

    # Create a summary file with metadata
    summary_file = session_dir / "generation_summary.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("Aegis-A11y Document Generation Summary\n")
        f.write("=====================================\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Source PDF: {sample_pdf_path}\n")
        f.write(
            f"Processing Duration: {reconstruction_result.processing_duration:.2f}s\n\n"
        )
        f.write("Documents Generated:\n")
        for format_name, file_path in saved_files.items():
            f.write(f"  - {format_name.upper()}: {file_path}\n")
        f.write("\nQuality Metrics:\n")
        f.write(
            f"  - Accessibility Score: {reconstruction_result.accessibility_score:.2f}\n"
        )
        f.write(
            f"  - Reconstruction Quality: {reconstruction_result.reconstruction_quality:.2f}\n"
        )
        f.write(f"  - WCAG Compliance: {reconstruction_result.wcag_compliance}\n")
        f.write(
            f"  - Manual Review Required: {reconstruction_result.manual_review_required}\n"
        )

    # Store relative path from project root for summary file
    relative_summary_path = summary_file.relative_to(project_root)
    saved_files["summary"] = str(relative_summary_path)
    logger.info(f"Saved generation summary: {relative_summary_path}")

    return saved_files


app = FastAPI(
    title=settings.project_name + " API",
    description="Document accessibility analysis and decomposition API",
    version=settings.version,
    debug=settings.api.debug,
)

# Initialize components with error handling using configuration
try:
    decomposer = LayoutDecomposer(settings.models.layoutlm_model)
    logger.info(
        f"LayoutDecomposer initialized successfully with model: {settings.models.layoutlm_model}"
    )
except Exception as e:
    logger.error(f"Failed to initialize LayoutDecomposer: {e}")
    decomposer = None

try:
    # Initialize reasoning agent components with configuration
    context_processor = ContextProcessor()
    alt_text_generator = AltTextGenerator()

    # Initialize semantic reasoner with configured OpenAI settings
    openai_api_key = settings.get_openai_api_key()
    if openai_api_key:
        semantic_reasoner = SemanticReasoner(
            api_key=openai_api_key,
            model=settings.models.openai_model,
            max_tokens=settings.models.openai_max_tokens,
            temperature=settings.models.openai_temperature,
        )
    else:
        semantic_reasoner = SemanticReasoner()

    verifier = DeterministicVerifier()
    element_filter = ElementFilter()
    logger.info("Reasoning agent components initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize reasoning agent: {e}")
    context_processor = None
    alt_text_generator = None
    semantic_reasoner = None
    verifier = None
    element_filter = None

try:
    # Initialize reconstruction engine
    reconstruction_engine = DocumentReconstructionEngine()
    logger.info("Document reconstruction engine initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize reconstruction engine: {e}")
    reconstruction_engine = None

try:
    # Initialize batch processor
    batch_processor = BatchProcessor()
    logger.info("Batch processor initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize batch processor: {e}")
    batch_processor = None

try:
    # Initialize progress tracker
    progress_tracker = get_progress_tracker()
    logger.info("Progress tracker initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize progress tracker: {e}")
    progress_tracker = None

try:
    # Initialize cache manager
    cache_manager = get_cache_manager()
    logger.info("Cache manager initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize cache manager: {e}")
    cache_manager = None


@app.get("/")
async def health_check():
    """Health check endpoint to verify API is running."""
    return {
        "status": "ok",
        "message": f"{settings.project_name} API is running",
        "version": settings.version,
        "environment": settings.environment.value,
        "components": {
            "decomposer_loaded": decomposer is not None,
            "reasoning_agent_loaded": all(
                [
                    context_processor is not None,
                    alt_text_generator is not None,
                    semantic_reasoner is not None,
                    verifier is not None,
                ]
            ),
            "reconstruction_engine_loaded": reconstruction_engine is not None,
            "batch_processor_loaded": batch_processor is not None,
            "progress_tracker_loaded": progress_tracker is not None,
            "cache_manager_loaded": cache_manager is not None,
        },
        "configuration": {
            "filtering_enabled": settings.processing.enable_filtering,
            "max_pages": settings.processing.max_pages,
            "output_formats": {
                "html5": settings.output.enable_html5,
                "pdf_ua": settings.output.enable_pdf_ua,
            },
            "batch_processing": {
                "enabled": settings.processing.enable_batch_processing,
                "max_batch_size": settings.processing.max_batch_size,
                "max_concurrent_pdfs": settings.processing.max_concurrent_pdfs,
            },
        },
    }


@app.get("/api/v1/config")
async def get_configuration():
    """Get current API configuration (non-sensitive values only)."""
    return {
        "environment": settings.environment.value,
        "version": settings.version,
        "api": {
            "host": settings.api.host,
            "port": settings.api.port,
            "debug": settings.api.debug,
            "request_timeout": settings.api.request_timeout,
        },
        "processing": {
            "max_pdf_size": settings.processing.max_pdf_size,
            "max_pages": settings.processing.max_pages,
            "pdf_dpi": settings.processing.pdf_dpi,
            "enable_filtering": settings.processing.enable_filtering,
            "min_confidence_threshold": settings.processing.min_confidence_threshold,
            "enable_batch_processing": settings.processing.enable_batch_processing,
            "max_batch_size": settings.processing.max_batch_size,
            "max_concurrent_pdfs": settings.processing.max_concurrent_pdfs,
            "batch_timeout": settings.processing.batch_timeout,
        },
        "models": {
            "layoutlm_model": settings.models.layoutlm_model,
            "openai_model": settings.models.openai_model,
            "openai_max_tokens": settings.models.openai_max_tokens,
            "openai_temperature": settings.models.openai_temperature,
            "cache_models": settings.models.cache_models,
        },
        "output": {
            "enable_html5": settings.output.enable_html5,
            "enable_pdf_ua": settings.output.enable_pdf_ua,
            "create_timestamped_dirs": settings.output.create_timestamped_dirs,
        },
        "performance": {
            "max_worker_threads": settings.performance.max_worker_threads,
            "enable_result_cache": settings.performance.enable_result_cache,
            "cache_ttl": settings.performance.cache_ttl,
        },
    }


@app.post("/api/v1/decompose")
async def decompose_document():
    """
    Decompose a sample PDF document into layout elements with accessibility analysis.
    """
    # Use relative path from project root
    project_root = Path(__file__).parent.parent.parent
    sample_pdf_path = str(project_root / "docs" / "pdfs" / "trigonometry.pdf")

    try:
        # Check if decomposer is available
        if decomposer is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="LayoutDecomposer not available. Check model loading.",
            )

        # Validate PDF file exists
        if not os.path.exists(sample_pdf_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Sample PDF not found at {sample_pdf_path}",
            )

        logger.info(f"Processing PDF: {sample_pdf_path}")

        # Convert PDF to images
        images = convert_pdf_to_images(sample_pdf_path)
        logger.info(f"Converted PDF to {len(images)} images")

        document_elements = []
        ocr_results = []

        for page_num, image in enumerate(images):
            logger.info(f"Processing page {page_num + 1}")

            # Extract OCR data
            ocr_data = extract_ocr_data(image)
            ocr_results.append(
                {
                    "page": page_num + 1,
                    "words_count": len(ocr_data["words"]),
                    "boxes_count": len(ocr_data["boxes"]),
                }
            )

            # Decompose image layout
            page_elements = decomposer.decompose_image(image, ocr_data)

            # Add page information to elements
            for element in page_elements:
                element_dict = (
                    element.model_dump() if hasattr(element, "model_dump") else element
                )
                element_dict["page_number"] = page_num + 1
                document_elements.append(element_dict)

        logger.info(f"Successfully processed {len(document_elements)} elements")

        return {
            "status": "success",
            "document_path": sample_pdf_path,
            "pages_processed": len(images),
            "total_elements": len(document_elements),
            "ocr_summary": ocr_results,
            "elements": document_elements,
        }

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Required file not found: {e!s}",
        ) from e
    except Exception as e:
        logger.error(f"Decomposition failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document decomposition failed: {e!s}",
        ) from e


@app.post("/api/v1/analyze")
async def analyze_document():
    """
    Full pipeline: Decompose PDF + Reasoning Agent analysis for accessibility.
    """
    # Use relative path from project root
    project_root = Path(__file__).parent.parent.parent
    sample_pdf_path = str(project_root / "docs" / "pdfs" / "trigonometry.pdf")

    try:
        # Check if all components are available
        if decomposer is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="LayoutDecomposer not available. Check model loading.",
            )

        if not all(
            [context_processor, alt_text_generator, semantic_reasoner, verifier]
        ):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Reasoning agent components not available. Check initialization.",
            )

        # Validate PDF file exists
        if not os.path.exists(sample_pdf_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Sample PDF not found at {sample_pdf_path}",
            )

        logger.info(f"Starting full analysis pipeline for: {sample_pdf_path}")

        # Step 1: PDF Decomposition
        images = convert_pdf_to_images(sample_pdf_path)
        logger.info(f"Converted PDF to {len(images)} images")

        extracted_elements = []
        reasoning_outputs = []

        for page_num, image in enumerate(images):
            logger.info(f"Processing page {page_num + 1}")

            # Extract OCR data
            ocr_data = extract_ocr_data(image)

            # Decompose image layout
            page_elements = decomposer.decompose_image(image, ocr_data)

            # Convert to list and add page info
            for element in page_elements:
                element_dict = (
                    element.model_dump() if hasattr(element, "model_dump") else element
                )
                element_dict["page_number"] = page_num + 1
                extracted_elements.append(element_dict)

        # Step 2: Apply intelligent filtering to reduce API calls
        logger.info(f"Applying element filtering to {len(extracted_elements)} elements")

        if element_filter is None:
            logger.warning("Element filter not available, processing all elements")
            filtered_elements_for_processing = [
                {
                    "element": elem,
                    "aggregated_text": elem.get("ocr_text", ""),
                    "processing_priority": 2,
                    "filter_reason": "No filtering",
                }
                for elem in extracted_elements
            ]
        else:
            # Apply filtering to dramatically reduce API calls
            filtered_elements_data = element_filter.filter_elements(extracted_elements)
            filtered_elements_for_processing = [
                {
                    "element": fe.element,
                    "aggregated_text": fe.aggregated_text,
                    "processing_priority": fe.processing_priority,
                    "filter_reason": fe.filter_reason,
                }
                for fe in filtered_elements_data
            ]

            # Log filtering results
            filter_summary = element_filter.get_filter_summary(
                extracted_elements, filtered_elements_data
            )
            logger.info(
                f"Element filtering: {filter_summary['original_count']} → {filter_summary['filtered_count']} elements "
                f"({filter_summary['reduction_percentage']:.1f}% reduction)"
            )
            logger.info(
                f"Priority distribution: High={filter_summary['priority_breakdown']['high']}, "
                f"Medium={filter_summary['priority_breakdown']['medium']}, "
                f"Low={filter_summary['priority_breakdown']['low']}"
            )

        # Step 3: Reasoning Agent Analysis (now on filtered elements only)
        for filtered_data in filtered_elements_for_processing:
            element = filtered_data["element"]
            try:
                logger.debug(
                    f"Processing filtered element {element.get('element_id', 'unknown')}: {filtered_data['filter_reason']}"
                )

                # Convert ExtractedElement to ReasoningInput
                reasoning_input = ReasoningInput.from_cv_output(
                    extracted_element=element,
                    full_page_image=None,  # Could add full image bytes here
                    page_metadata={
                        "page_number": element.get("page_number", 1),
                        "total_pages": len(images),
                        "document_path": sample_pdf_path,
                    },
                )

                # Process through complete reasoning pipeline
                # The semantic_reasoner.process_element() method handles all steps internally:
                # 1. Context analysis, 2. Subject detection, 3. LLM processing, 4. Alt-text generation
                reasoning_output = semantic_reasoner.process_element(reasoning_input)

                # Verification
                verification_result = verifier.verify_reasoning_output(
                    reasoning_output,
                    element,
                    {"spatial_context": {}},  # Minimal context for verification
                )

                # Store results
                output_dict = reasoning_output.model_dump()
                output_dict["verification_passed"] = (
                    verification_result.overall_status.value == "pass"
                )
                output_dict["verification_issues"] = [
                    issue.model_dump() for issue in verification_result.issues
                ]
                output_dict["page_number"] = element.get("page_number", 1)
                output_dict["filter_reason"] = filtered_data["filter_reason"]
                output_dict["processing_priority"] = filtered_data[
                    "processing_priority"
                ]

                reasoning_outputs.append(output_dict)

            except Exception as e:
                logger.error(
                    f"Failed to process element {element.get('element_id', 'unknown')}: {e}"
                )
                # Continue with other elements
                continue

        logger.info(
            f"Analysis complete: {len(extracted_elements)} elements, {len(reasoning_outputs)} analyzed"
        )

        return {
            "status": "success",
            "document_path": sample_pdf_path,
            "pages_processed": len(images),
            "pipeline": {
                "decomposition": {
                    "total_elements": len(extracted_elements),
                    "elements": extracted_elements,
                },
                "reasoning": {
                    "total_analyzed": len(reasoning_outputs),
                    "verified_outputs": [
                        output
                        for output in reasoning_outputs
                        if output.get("verification_passed", False)
                    ],
                    "outputs": reasoning_outputs,
                },
            },
            "summary": {
                "elements_extracted": len(extracted_elements),
                "elements_analyzed": len(reasoning_outputs),
                "verification_pass_rate": (
                    sum(
                        1
                        for output in reasoning_outputs
                        if output.get("verification_passed", False)
                    )
                    / len(reasoning_outputs)
                    if reasoning_outputs
                    else 0
                ),
                "subject_areas_detected": list(
                    {
                        output.get("detected_subject_area")
                        for output in reasoning_outputs
                        if output.get("detected_subject_area")
                    }
                ),
            },
        }

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Full analysis pipeline failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document analysis pipeline failed: {e!s}",
        ) from e


@app.post("/api/v1/reconstruct")
async def reconstruct_document():
    """
    Complete DRR Pipeline: Decompose → Reasoning → Reconstruction
    Generates accessible documents (HTML5, PDF/UA) from PDF input.
    """
    # Use relative path from project root
    project_root = Path(__file__).parent.parent.parent
    sample_pdf_path = str(project_root / "docs" / "pdfs" / "trigonometry.pdf")

    try:
        # Check if all components are available
        if decomposer is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="LayoutDecomposer not available. Check model loading.",
            )

        if not all(
            [context_processor, alt_text_generator, semantic_reasoner, verifier]
        ):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Reasoning agent components not available. Check initialization.",
            )

        if reconstruction_engine is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Reconstruction engine not available. Check initialization.",
            )

        # Validate PDF file exists
        if not os.path.exists(sample_pdf_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Sample PDF not found at {sample_pdf_path}",
            )

        logger.info(f"Starting complete DRR pipeline for: {sample_pdf_path}")

        # Step 1: PDF Decomposition
        images = convert_pdf_to_images(sample_pdf_path)
        logger.info(f"Converted PDF to {len(images)} images")

        extracted_elements = []
        verified_reasoning_outputs = []

        for page_num, image in enumerate(images):
            logger.info(f"Processing page {page_num + 1}")

            # Extract OCR data
            ocr_data = extract_ocr_data(image)

            # Decompose image layout
            page_elements = decomposer.decompose_image(image, ocr_data)

            # Convert to list and add page info
            for element in page_elements:
                element_dict = (
                    element.model_dump() if hasattr(element, "model_dump") else element
                )
                element_dict["page_number"] = page_num + 1
                extracted_elements.append(element_dict)

        # Step 2: Apply intelligent filtering to reduce API calls
        logger.info(f"Applying element filtering to {len(extracted_elements)} elements")

        if element_filter is None:
            logger.warning("Element filter not available, processing all elements")
            filtered_elements_for_processing = [
                {
                    "element": elem,
                    "aggregated_text": elem.get("ocr_text", ""),
                    "processing_priority": 2,
                    "filter_reason": "No filtering",
                }
                for elem in extracted_elements
            ]
        else:
            # Apply filtering to dramatically reduce API calls
            filtered_elements_data = element_filter.filter_elements(extracted_elements)
            filtered_elements_for_processing = [
                {
                    "element": fe.element,
                    "aggregated_text": fe.aggregated_text,
                    "processing_priority": fe.processing_priority,
                    "filter_reason": fe.filter_reason,
                }
                for fe in filtered_elements_data
            ]

            # Log filtering results
            filter_summary = element_filter.get_filter_summary(
                extracted_elements, filtered_elements_data
            )
            logger.info(
                f"Element filtering: {filter_summary['original_count']} → {filter_summary['filtered_count']} elements "
                f"({filter_summary['reduction_percentage']:.1f}% reduction)"
            )

        # Step 3: Reasoning Agent Analysis (now on filtered elements only)
        for filtered_data in filtered_elements_for_processing:
            element = filtered_data["element"]
            try:
                # Convert ExtractedElement to ReasoningInput
                reasoning_input = ReasoningInput.from_cv_output(
                    extracted_element=element,
                    full_page_image=None,  # Could add full image bytes here
                    page_metadata={
                        "page_number": element.get("page_number", 1),
                        "total_pages": len(images),
                        "document_path": sample_pdf_path,
                    },
                )

                # Process through complete reasoning pipeline
                # The semantic_reasoner.process_element() method handles all steps internally:
                # 1. Context analysis, 2. Subject detection, 3. LLM processing, 4. Alt-text generation
                reasoning_output = semantic_reasoner.process_element(reasoning_input)

                # Verification
                verification_result = verifier.verify_reasoning_output(
                    reasoning_output,
                    element,
                    {"spatial_context": {}},  # Minimal context for verification
                )

                # Only include verified outputs for reconstruction
                if verification_result.overall_status.value == "pass":
                    verified_reasoning_outputs.append(reasoning_output)

            except Exception as e:
                logger.error(
                    f"Failed to process element {element.get('element_id', 'unknown')}: {e}"
                )
                # Continue with other elements
                continue

        logger.info(
            f"Reasoning complete: {len(verified_reasoning_outputs)} verified outputs for reconstruction"
        )

        # Step 3: Document Reconstruction
        if verified_reasoning_outputs:
            # Create reconstruction input
            reconstruction_input = ReconstructionInput(
                verified_elements=verified_reasoning_outputs,
                original_layout=extracted_elements,
                document_title=f"Accessible Document from {os.path.basename(sample_pdf_path)}",
                document_language="en",
                subject_area="mathematics",  # Could detect from reasoning outputs
                educational_level="high_school",
                target_formats=[OutputFormat.HTML5, OutputFormat.PDF_UA],
                preserve_layout=True,
                include_metadata=True,
                generate_navigation=True,
            )

            # Generate accessible documents
            reconstruction_result = reconstruction_engine.reconstruct_document(
                reconstruction_input
            )

            logger.info(
                f"Reconstruction complete: {len(reconstruction_result.documents)} documents generated"
            )

            # Save generated documents to files
            saved_file_paths = save_generated_documents(
                reconstruction_result, sample_pdf_path
            )
            logger.info(f"Documents saved to: {saved_file_paths}")

            # Convert binary data to base64 for JSON response
            documents_for_response = {}
            for format_type, document in reconstruction_result.documents.items():
                if isinstance(document, bytes):
                    # For PDF, encode as base64
                    import base64

                    documents_for_response[format_type.value] = {
                        "type": "binary",
                        "data": base64.b64encode(document).decode("utf-8"),
                        "size_bytes": len(document),
                    }
                else:
                    # For HTML, include as text
                    documents_for_response[format_type.value] = {
                        "type": "text",
                        "data": document,
                        "size_chars": len(document),
                    }

            return {
                "status": "success",
                "document_path": sample_pdf_path,
                "pages_processed": len(images),
                "pipeline": {
                    "decomposition": {"total_elements": len(extracted_elements)},
                    "reasoning": {
                        "total_analyzed": len(verified_reasoning_outputs),
                        "verification_pass_rate": (
                            len(verified_reasoning_outputs) / len(extracted_elements)
                            if extracted_elements
                            else 0
                        ),
                    },
                    "reconstruction": {
                        "documents_generated": len(reconstruction_result.documents),
                        "accessibility_score": reconstruction_result.accessibility_score,
                        "reconstruction_quality": reconstruction_result.reconstruction_quality,
                        "wcag_compliance": reconstruction_result.wcag_compliance,
                        "verifier_passed": reconstruction_result.verifier_passed,
                        "manual_review_required": reconstruction_result.manual_review_required,
                    },
                },
                "generated_documents": documents_for_response,
                "summary": {
                    "elements_extracted": len(extracted_elements),
                    "elements_verified": len(verified_reasoning_outputs),
                    "documents_created": len(reconstruction_result.documents),
                    "overall_quality": reconstruction_result.reconstruction_quality,
                    "accessibility_compliance": reconstruction_result.accessibility_score,
                    "processing_duration": reconstruction_result.processing_duration,
                },
                "saved_files": saved_file_paths,
            }
        else:
            logger.warning("No verified reasoning outputs available for reconstruction")
            return {
                "status": "partial_success",
                "message": "Document decomposition and reasoning completed, but no verified outputs for reconstruction",
                "elements_extracted": len(extracted_elements),
                "elements_verified": 0,
            }

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Complete DRR pipeline failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document reconstruction pipeline failed: {e!s}",
        ) from e


@app.post("/api/v1/batch/create", response_model=BatchStatusResponse)
async def create_batch(request: BatchCreateRequest):
    """
    Create a new batch processing job for multiple PDF documents.
    """
    try:
        if not batch_processor:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Batch processor not available"
            )
        
        # Create the batch
        batch_id = batch_processor.create_batch(
            file_paths=request.file_paths,
            output_dir=request.output_directory
        )
        
        # Get initial status
        batch_job = batch_processor.get_batch_status(batch_id)
        
        return BatchStatusResponse(
            id=batch_job.id,
            status=batch_job.status.value,
            total_documents=batch_job.total_documents,
            processed_documents=batch_job.processed_documents,
            failed_documents=batch_job.failed_documents,
            progress_percentage=batch_job.progress_percentage,
            created_at=batch_job.created_at.isoformat(),
            started_at=batch_job.started_at.isoformat() if batch_job.started_at else None,
            completed_at=batch_job.completed_at.isoformat() if batch_job.completed_at else None,
            error_message=batch_job.error_message
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to create batch: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create batch: {str(e)}"
        )


@app.post("/api/v1/batch/{batch_id}/start")
async def start_batch_processing(batch_id: str):
    """
    Start processing a created batch job.
    """
    try:
        if not batch_processor:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Batch processor not available"
            )
        
        # Start processing asynchronously
        import asyncio
        task = asyncio.create_task(batch_processor.process_batch(batch_id))
        
        return {"message": f"Batch {batch_id} processing started", "batch_id": batch_id}
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to start batch processing: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start batch processing: {str(e)}"
        )


@app.get("/api/v1/batch/{batch_id}/status", response_model=BatchStatusResponse)
async def get_batch_status(batch_id: str):
    """
    Get current status of a batch processing job.
    """
    try:
        if not batch_processor:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Batch processor not available"
            )
        
        batch_job = batch_processor.get_batch_status(batch_id)
        
        if not batch_job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Batch {batch_id} not found"
            )
        
        return BatchStatusResponse(
            id=batch_job.id,
            status=batch_job.status.value,
            total_documents=batch_job.total_documents,
            processed_documents=batch_job.processed_documents,
            failed_documents=batch_job.failed_documents,
            progress_percentage=batch_job.progress_percentage,
            created_at=batch_job.created_at.isoformat(),
            started_at=batch_job.started_at.isoformat() if batch_job.started_at else None,
            completed_at=batch_job.completed_at.isoformat() if batch_job.completed_at else None,
            error_message=batch_job.error_message
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get batch status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get batch status: {str(e)}"
        )


@app.get("/api/v1/batch/{batch_id}/details", response_model=BatchDetailResponse)
async def get_batch_details(batch_id: str):
    """
    Get detailed information about a batch processing job including individual document statuses.
    """
    try:
        if not batch_processor:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Batch processor not available"
            )
        
        batch_job = batch_processor.get_batch_status(batch_id)
        
        if not batch_job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Batch {batch_id} not found"
            )
        
        # Convert documents to response format
        document_responses = []
        for doc in batch_job.documents:
            document_responses.append(DocumentStatusResponse(
                id=doc.id,
                original_filename=doc.original_filename,
                status=doc.status.value,
                elements_extracted=doc.elements_extracted,
                elements_analyzed=doc.elements_analyzed,
                processing_duration=doc.processing_duration,
                error_message=doc.error_message,
                output_files=doc.output_files
            ))
        
        return BatchDetailResponse(
            id=batch_job.id,
            status=batch_job.status.value,
            total_documents=batch_job.total_documents,
            processed_documents=batch_job.processed_documents,
            failed_documents=batch_job.failed_documents,
            progress_percentage=batch_job.progress_percentage,
            created_at=batch_job.created_at.isoformat(),
            started_at=batch_job.started_at.isoformat() if batch_job.started_at else None,
            completed_at=batch_job.completed_at.isoformat() if batch_job.completed_at else None,
            error_message=batch_job.error_message,
            documents=document_responses
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get batch details: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get batch details: {str(e)}"
        )


@app.get("/api/v1/batch", response_model=List[BatchStatusResponse])
async def list_batches():
    """
    List all active batch processing jobs.
    """
    try:
        if not batch_processor:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Batch processor not available"
            )
        
        batches = batch_processor.list_active_batches()
        
        return [
            BatchStatusResponse(
                id=batch.id,
                status=batch.status.value,
                total_documents=batch.total_documents,
                processed_documents=batch.processed_documents,
                failed_documents=batch.failed_documents,
                progress_percentage=batch.progress_percentage,
                created_at=batch.created_at.isoformat(),
                started_at=batch.started_at.isoformat() if batch.started_at else None,
                completed_at=batch.completed_at.isoformat() if batch.completed_at else None,
                error_message=batch.error_message
            )
            for batch in batches
        ]
        
    except Exception as e:
        logger.error(f"Failed to list batches: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list batches: {str(e)}"
        )


@app.delete("/api/v1/batch/{batch_id}")
async def cancel_batch(batch_id: str):
    """
    Cancel a running batch processing job.
    """
    try:
        if not batch_processor:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Batch processor not available"
            )
        
        success = batch_processor.cancel_batch(batch_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Batch {batch_id} not found or not cancellable"
            )
        
        return {"message": f"Batch {batch_id} cancelled successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel batch: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel batch: {str(e)}"
        )


@app.get("/api/v1/progress/{operation_id}", response_model=OperationProgressResponse)
async def get_operation_progress(operation_id: str):
    """
    Get detailed progress information for a long-running operation.
    """
    try:
        if not progress_tracker:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Progress tracker not available"
            )
        
        operation = progress_tracker.get_operation_progress(operation_id)
        
        if not operation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Operation {operation_id} not found"
            )
        
        # Convert steps to response format
        steps_response = []
        for step in operation.steps:
            steps_response.append(ProgressStepResponse(
                step_id=step.step_id,
                name=step.name,
                description=step.description,
                status=step.status.value,
                progress_percentage=step.progress_percentage,
                start_time=step.start_time.isoformat() if step.start_time else None,
                end_time=step.end_time.isoformat() if step.end_time else None,
                error_message=step.error_message,
                details=step.details
            ))
        
        # Convert performance metrics
        perf_metrics = PerformanceMetricsResponse(
            processing_rate=operation.performance_metrics.processing_rate,
            estimated_completion_time=operation.performance_metrics.estimated_completion_time.isoformat() 
                if operation.performance_metrics.estimated_completion_time else None,
            memory_usage_mb=operation.performance_metrics.memory_usage_mb,
            cpu_usage_percentage=operation.performance_metrics.cpu_usage_percentage,
            api_calls_made=operation.performance_metrics.api_calls_made,
            api_calls_cached=operation.performance_metrics.api_calls_cached,
        )
        
        return OperationProgressResponse(
            operation_id=operation.operation_id,
            operation_type=operation.operation_type.value,
            name=operation.name,
            description=operation.description,
            status=operation.status.value,
            overall_progress_percentage=operation.overall_progress_percentage,
            created_at=operation.created_at.isoformat(),
            started_at=operation.started_at.isoformat() if operation.started_at else None,
            completed_at=operation.completed_at.isoformat() if operation.completed_at else None,
            total_steps=operation.total_steps,
            completed_steps=operation.completed_steps,
            current_step=operation.current_step,
            steps=steps_response,
            error_message=operation.error_message,
            metadata=operation.metadata,
            performance_metrics=perf_metrics
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get operation progress: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get operation progress: {str(e)}"
        )


@app.get("/api/v1/progress", response_model=List[OperationProgressResponse])
async def list_active_operations():
    """
    List all active operations being tracked.
    """
    try:
        if not progress_tracker:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Progress tracker not available"
            )
        
        operations = progress_tracker.list_active_operations()
        
        responses = []
        for operation in operations:
            # Convert steps to response format
            steps_response = []
            for step in operation.steps:
                steps_response.append(ProgressStepResponse(
                    step_id=step.step_id,
                    name=step.name,
                    description=step.description,
                    status=step.status.value,
                    progress_percentage=step.progress_percentage,
                    start_time=step.start_time.isoformat() if step.start_time else None,
                    end_time=step.end_time.isoformat() if step.end_time else None,
                    error_message=step.error_message,
                    details=step.details
                ))
            
            # Convert performance metrics
            perf_metrics = PerformanceMetricsResponse(
                processing_rate=operation.performance_metrics.processing_rate,
                estimated_completion_time=operation.performance_metrics.estimated_completion_time.isoformat() 
                    if operation.performance_metrics.estimated_completion_time else None,
                memory_usage_mb=operation.performance_metrics.memory_usage_mb,
                cpu_usage_percentage=operation.performance_metrics.cpu_usage_percentage,
                api_calls_made=operation.performance_metrics.api_calls_made,
                api_calls_cached=operation.performance_metrics.api_calls_cached,
            )
            
            responses.append(OperationProgressResponse(
                operation_id=operation.operation_id,
                operation_type=operation.operation_type.value,
                name=operation.name,
                description=operation.description,
                status=operation.status.value,
                overall_progress_percentage=operation.overall_progress_percentage,
                created_at=operation.created_at.isoformat(),
                started_at=operation.started_at.isoformat() if operation.started_at else None,
                completed_at=operation.completed_at.isoformat() if operation.completed_at else None,
                total_steps=operation.total_steps,
                completed_steps=operation.completed_steps,
                current_step=operation.current_step,
                steps=steps_response,
                error_message=operation.error_message,
                metadata=operation.metadata,
                performance_metrics=perf_metrics
            ))
        
        return responses
        
    except Exception as e:
        logger.error(f"Failed to list active operations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list active operations: {str(e)}"
        )


@app.delete("/api/v1/progress/{operation_id}")
async def cancel_operation(operation_id: str):
    """
    Cancel a running operation.
    """
    try:
        if not progress_tracker:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Progress tracker not available"
            )
        
        success = progress_tracker.cancel_operation(operation_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Operation {operation_id} not found or not cancellable"
            )
        
        return {"message": f"Operation {operation_id} cancelled successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel operation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel operation: {str(e)}"
        )


@app.get("/api/v1/cache/stats", response_model=CacheStatsResponse)
async def get_cache_stats():
    """
    Get comprehensive cache performance statistics.
    """
    try:
        if not cache_manager:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Cache manager not available"
            )
        
        stats = cache_manager.get_stats()
        
        return CacheStatsResponse(**stats)
        
    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get cache stats: {str(e)}"
        )


@app.get("/api/v1/cache/entries", response_model=List[CacheEntryResponse])
async def get_cache_entries(limit: int = 50):
    """
    Get information about cached entries.
    """
    try:
        if not cache_manager:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Cache manager not available"
            )
        
        entries_info = cache_manager.get_cache_info(limit)
        
        return [CacheEntryResponse(**entry) for entry in entries_info]
        
    except Exception as e:
        logger.error(f"Failed to get cache entries: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get cache entries: {str(e)}"
        )


@app.post("/api/v1/cache/clear")
async def clear_cache(request: CacheClearRequest):
    """
    Clear cache entries by type or all entries.
    """
    try:
        if not cache_manager:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Cache manager not available"
            )
        
        cache_type = None
        if request.cache_type:
            try:
                cache_type = CacheType(request.cache_type)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid cache type: {request.cache_type}"
                )
        
        cache_manager.clear(cache_type)
        
        message = f"Cleared {request.cache_type} cache" if request.cache_type else "Cleared all cache"
        return {"message": message}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear cache: {str(e)}"
        )


@app.post("/api/v1/cache/cleanup")
async def cleanup_cache():
    """
    Manually trigger cache cleanup (remove expired entries).
    """
    try:
        if not cache_manager:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Cache manager not available"
            )
        
        cache_manager.cleanup_expired()
        
        return {"message": "Cache cleanup completed"}
        
    except Exception as e:
        logger.error(f"Failed to cleanup cache: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cleanup cache: {str(e)}"
        )


def main():
    print("Hello from api!")


if __name__ == "__main__":
    main()
