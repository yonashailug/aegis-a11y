from datetime import datetime
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status

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

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_generated_documents(
    reconstruction_result, sample_pdf_path: str
) -> dict[str, str]:
    """Save generated documents to files and return file paths."""

    # Create output directory relative to project root
    project_root = Path(
        __file__
    ).parent.parent.parent  # Go up from api/main.py to project root
    output_dir = project_root / "generated_documents"
    output_dir.mkdir(exist_ok=True)

    # Create timestamped subdirectory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_name = Path(sample_pdf_path).stem
    session_dir = output_dir / f"{pdf_name}_{timestamp}"
    session_dir.mkdir(exist_ok=True)

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
    title="Aegis-A11y API",
    description="Document accessibility analysis and decomposition API",
    version="1.0.0",
)

# Initialize components with error handling
try:
    decomposer = LayoutDecomposer("microsoft/layoutlmv3-base")
    logger.info("LayoutDecomposer initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize LayoutDecomposer: {e}")
    decomposer = None

try:
    # Initialize reasoning agent components
    context_processor = ContextProcessor()
    alt_text_generator = AltTextGenerator()
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


@app.get("/")
async def health_check():
    """Health check endpoint to verify API is running."""
    return {
        "status": "ok",
        "message": "Aegis-A11y API is running",
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
        )
    except Exception as e:
        logger.error(f"Decomposition failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document decomposition failed: {e!s}",
        )


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
                    set(
                        output.get("detected_subject_area")
                        for output in reasoning_outputs
                        if output.get("detected_subject_area")
                    )
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
        )


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
        )


def main():
    print("Hello from api!")


if __name__ == "__main__":
    main()
