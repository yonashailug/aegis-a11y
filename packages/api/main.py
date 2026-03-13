import os
import logging
from pathlib import Path
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, status
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
from cv_layer import LayoutDecomposer, convert_pdf_to_images, extract_ocr_data
from reasoning_agent import (
    ReasoningInput, 
    ContextProcessor,
    AltTextGenerator,
    SemanticReasoner,
    DeterministicVerifier
)
from reconstruction import (
    DocumentReconstructionEngine,
    ReconstructionInput,
    ReconstructionOutput,
    OutputFormat
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Aegis-A11y API",
    description="Document accessibility analysis and decomposition API",
    version="1.0.0"
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
    logger.info("Reasoning agent components initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize reasoning agent: {e}")
    context_processor = None
    alt_text_generator = None
    semantic_reasoner = None
    verifier = None

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
        "reasoning_agent_loaded": all([
            context_processor is not None,
            alt_text_generator is not None,
            semantic_reasoner is not None,
            verifier is not None
        ]),
        "reconstruction_engine_loaded": reconstruction_engine is not None
    }

@app.post("/api/v1/decompose")
async def decompose_document():
    """
    Decompose a sample PDF document into layout elements with accessibility analysis.
    """
    # Use absolute path to avoid relative path issues
    sample_pdf_path = "/Users/yonashailug/workspace/aegis-a11y/docs/pdfs/trigonometry.pdf"
    
    try:
        # Check if decomposer is available
        if decomposer is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="LayoutDecomposer not available. Check model loading."
            )
        
        # Validate PDF file exists
        if not os.path.exists(sample_pdf_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Sample PDF not found at {sample_pdf_path}"
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
            ocr_results.append({
                "page": page_num + 1,
                "words_count": len(ocr_data["words"]),
                "boxes_count": len(ocr_data["boxes"])
            })
            
            # Decompose image layout
            page_elements = decomposer.decompose_image(image, ocr_data)
            
            # Add page information to elements
            for element in page_elements:
                element_dict = element.model_dump() if hasattr(element, 'model_dump') else element
                element_dict["page_number"] = page_num + 1
                document_elements.append(element_dict)
        
        logger.info(f"Successfully processed {len(document_elements)} elements")
        
        return {
            "status": "success",
            "document_path": sample_pdf_path,
            "pages_processed": len(images),
            "total_elements": len(document_elements),
            "ocr_summary": ocr_results,
            "elements": document_elements
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Required file not found: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Decomposition failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document decomposition failed: {str(e)}"
        )

@app.post("/api/v1/analyze")
async def analyze_document():
    """
    Full pipeline: Decompose PDF + Reasoning Agent analysis for accessibility.
    """
    # Use absolute path to avoid relative path issues
    sample_pdf_path = "/Users/yonashailug/workspace/aegis-a11y/docs/pdfs/trigonometry.pdf"
    
    try:
        # Check if all components are available
        if decomposer is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="LayoutDecomposer not available. Check model loading."
            )
        
        if not all([context_processor, alt_text_generator, semantic_reasoner, verifier]):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Reasoning agent components not available. Check initialization."
            )
        
        # Validate PDF file exists
        if not os.path.exists(sample_pdf_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Sample PDF not found at {sample_pdf_path}"
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
                element_dict = element.model_dump() if hasattr(element, 'model_dump') else element
                element_dict["page_number"] = page_num + 1
                extracted_elements.append(element_dict)
            
            # Step 2: Reasoning Agent Analysis
            for element in page_elements:
                try:
                    # Convert ExtractedElement to ReasoningInput
                    reasoning_input = ReasoningInput.from_cv_output(
                        extracted_element=element,
                        full_page_image=None,  # Could add full image bytes here
                        page_metadata={
                            "page_number": page_num + 1,
                            "total_pages": len(images),
                            "document_path": sample_pdf_path
                        }
                    )
                    
                    # Process through complete reasoning pipeline
                    # The semantic_reasoner.process_element() method handles all steps internally:
                    # 1. Context analysis, 2. Subject detection, 3. LLM processing, 4. Alt-text generation
                    reasoning_output = semantic_reasoner.process_element(reasoning_input)
                    
                    # Verification
                    verification_result = verifier.verify_reasoning_output(
                        reasoning_output,
                        element,
                        {"spatial_context": {}}  # Minimal context for verification
                    )
                    
                    # Store results
                    output_dict = reasoning_output.model_dump()
                    output_dict["verification_passed"] = verification_result.overall_status.value == "pass"
                    output_dict["verification_issues"] = [
                        issue.model_dump() for issue in verification_result.issues
                    ]
                    output_dict["page_number"] = page_num + 1
                    
                    reasoning_outputs.append(output_dict)
                    
                except Exception as e:
                    logger.error(f"Failed to process element {element.element_id}: {e}")
                    # Continue with other elements
                    continue
        
        logger.info(f"Analysis complete: {len(extracted_elements)} elements, {len(reasoning_outputs)} analyzed")
        
        return {
            "status": "success",
            "document_path": sample_pdf_path,
            "pages_processed": len(images),
            "pipeline": {
                "decomposition": {
                    "total_elements": len(extracted_elements),
                    "elements": extracted_elements
                },
                "reasoning": {
                    "total_analyzed": len(reasoning_outputs),
                    "verified_outputs": [
                        output for output in reasoning_outputs 
                        if output.get("verification_passed", False)
                    ],
                    "outputs": reasoning_outputs
                }
            },
            "summary": {
                "elements_extracted": len(extracted_elements),
                "elements_analyzed": len(reasoning_outputs),
                "verification_pass_rate": (
                    sum(1 for output in reasoning_outputs if output.get("verification_passed", False)) 
                    / len(reasoning_outputs) if reasoning_outputs else 0
                ),
                "subject_areas_detected": list(set(
                    output.get("detected_subject_area") 
                    for output in reasoning_outputs 
                    if output.get("detected_subject_area")
                ))
            }
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Full analysis pipeline failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document analysis pipeline failed: {str(e)}"
        )

@app.post("/api/v1/reconstruct")
async def reconstruct_document():
    """
    Complete DRR Pipeline: Decompose → Reasoning → Reconstruction
    Generates accessible documents (HTML5, PDF/UA) from PDF input.
    """
    # Use absolute path to avoid relative path issues
    sample_pdf_path = "/Users/yonashailug/workspace/aegis-a11y/docs/pdfs/trigonometry.pdf"
    
    try:
        # Check if all components are available
        if decomposer is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="LayoutDecomposer not available. Check model loading."
            )
        
        if not all([context_processor, alt_text_generator, semantic_reasoner, verifier]):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Reasoning agent components not available. Check initialization."
            )
            
        if reconstruction_engine is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Reconstruction engine not available. Check initialization."
            )
        
        # Validate PDF file exists
        if not os.path.exists(sample_pdf_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Sample PDF not found at {sample_pdf_path}"
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
                element_dict = element.model_dump() if hasattr(element, 'model_dump') else element
                element_dict["page_number"] = page_num + 1
                extracted_elements.append(element_dict)
            
            # Step 2: Reasoning Agent Analysis
            for element in page_elements:
                try:
                    # Convert ExtractedElement to ReasoningInput
                    reasoning_input = ReasoningInput.from_cv_output(
                        extracted_element=element,
                        full_page_image=None,  # Could add full image bytes here
                        page_metadata={
                            "page_number": page_num + 1,
                            "total_pages": len(images),
                            "document_path": sample_pdf_path
                        }
                    )
                    
                    # Process through complete reasoning pipeline
                    # The semantic_reasoner.process_element() method handles all steps internally:
                    # 1. Context analysis, 2. Subject detection, 3. LLM processing, 4. Alt-text generation
                    reasoning_output = semantic_reasoner.process_element(reasoning_input)
                    
                    # Verification
                    verification_result = verifier.verify_reasoning_output(
                        reasoning_output,
                        element,
                        {"spatial_context": {}}  # Minimal context for verification
                    )
                    
                    # Only include verified outputs for reconstruction
                    if verification_result.overall_status.value == "pass":
                        verified_reasoning_outputs.append(reasoning_output)
                    
                except Exception as e:
                    logger.error(f"Failed to process element {element.element_id}: {e}")
                    # Continue with other elements
                    continue
        
        logger.info(f"Reasoning complete: {len(verified_reasoning_outputs)} verified outputs for reconstruction")
        
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
                generate_navigation=True
            )
            
            # Generate accessible documents
            reconstruction_result = reconstruction_engine.reconstruct_document(reconstruction_input)
            
            logger.info(f"Reconstruction complete: {len(reconstruction_result.documents)} documents generated")
            
            # Convert binary data to base64 for JSON response
            documents_for_response = {}
            for format_type, document in reconstruction_result.documents.items():
                if isinstance(document, bytes):
                    # For PDF, encode as base64
                    import base64
                    documents_for_response[format_type.value] = {
                        "type": "binary",
                        "data": base64.b64encode(document).decode('utf-8'),
                        "size_bytes": len(document)
                    }
                else:
                    # For HTML, include as text
                    documents_for_response[format_type.value] = {
                        "type": "text",
                        "data": document,
                        "size_chars": len(document)
                    }
            
            return {
                "status": "success",
                "document_path": sample_pdf_path,
                "pages_processed": len(images),
                "pipeline": {
                    "decomposition": {
                        "total_elements": len(extracted_elements)
                    },
                    "reasoning": {
                        "total_analyzed": len(verified_reasoning_outputs),
                        "verification_pass_rate": len(verified_reasoning_outputs) / len(extracted_elements) if extracted_elements else 0
                    },
                    "reconstruction": {
                        "documents_generated": len(reconstruction_result.documents),
                        "accessibility_score": reconstruction_result.accessibility_score,
                        "reconstruction_quality": reconstruction_result.reconstruction_quality,
                        "wcag_compliance": reconstruction_result.wcag_compliance,
                        "verifier_passed": reconstruction_result.verifier_passed,
                        "manual_review_required": reconstruction_result.manual_review_required
                    }
                },
                "generated_documents": documents_for_response,
                "summary": {
                    "elements_extracted": len(extracted_elements),
                    "elements_verified": len(verified_reasoning_outputs),
                    "documents_created": len(reconstruction_result.documents),
                    "overall_quality": reconstruction_result.reconstruction_quality,
                    "accessibility_compliance": reconstruction_result.accessibility_score,
                    "processing_duration": reconstruction_result.processing_duration
                }
            }
        else:
            logger.warning("No verified reasoning outputs available for reconstruction")
            return {
                "status": "partial_success",
                "message": "Document decomposition and reasoning completed, but no verified outputs for reconstruction",
                "elements_extracted": len(extracted_elements),
                "elements_verified": 0
            }
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Complete DRR pipeline failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document reconstruction pipeline failed: {str(e)}"
        )

def main():
    print("Hello from api!")


if __name__ == "__main__":
    main()
