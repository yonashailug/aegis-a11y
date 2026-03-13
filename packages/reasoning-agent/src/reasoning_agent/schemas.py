"""Pydantic schemas for reasoning agent input and output."""

from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum
import datetime

class SubjectArea(str, Enum):
    """Subject area classifications for educational content."""
    PHYSICS = "physics"
    CHEMISTRY = "chemistry"
    BIOLOGY = "biology"
    MATHEMATICS = "mathematics"
    HISTORY = "history"
    LITERATURE = "literature"
    SOCIAL_STUDIES = "social_studies"
    GENERAL = "general"
    UNKNOWN = "unknown"

class ConfidenceLevel(str, Enum):
    """Confidence levels for AI-generated content."""
    HIGH = "high" # >0.8
    MEDIUM = "medium" # 0.5-0.8
    LOW = "low" # <0.5


class ReasoningInput(BaseModel):
    """Input schema for reasoning agent processing.
    
    This will be populated with ExtractedElement from cv-layer and
    additional context information.
    """
    extracted_element: Dict[str, Any] = Field(
        ...,
        description="ExtractedElement from cv-layer decomposition"
    )
    surrounding_elements: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Nearby elements for spatial context"
    )
    image_segment: Optional[bytes] = Field(
        None,
        description="Image bytes of the specific element region"
    )
    full_page_image: Optional[bytes] = Field(
        None,
        description="Full page image for broader context"
    )
    page_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Page-level information (document type, subject hints, etc.)"
    )
    processing_timestamp: datetime.datetime = Field(
        default_factory=datetime.datetime.now,
        description="When this element was submitted for processing"
    )

    @classmethod
    def from_cv_output(cls, extracted_element, **kwargs):
        """Create ReasoningInput from cv-layer ExtractedElement."""
        element_dict = extracted_element.model_dump() if hasattr(extracted_element, 'model_dump') else extracted_element
        return cls(
            extracted_element=element_dict,
            **kwargs
        )

    def get_element_classification(self) -> str:
        """Get the classification from the extracted element."""
        return self.extracted_element.get("classification", "unknown")

    def get_element_text(self) -> str:
        """Get the OCR text from the extracted element."""
        return self.extracted_element.get("ocr_text", "")


class ReasoningOutput(BaseModel):
    """Output schema for reasoning agent results.
    
    This will contain the semantic analysis and pedagogical alt-text
    for use by the verifier component.
    """
    # Element identification
    element_id: str = Field(..., description="Unique identifier from input")
    
    # Semantic analysis
    detected_subject_area: SubjectArea = Field(
        ...,
        description="Detected or assigned subject area"
    )
    subject_confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="Confidence in subject area detection"
    )
    
    # Learning context
    learning_objective: Optional[str] = Field(
        None,
        description="Identified pedagogical purpose/objective"
    )
    contextual_importance: str = Field(
        ...,
        description="Why this element is important for learning"
    )
    
    # Pedagogical alt-text (core output)
    pedagogical_alt_text: str = Field(
        ...,
        min_length=10,
        description="UDL-compliant educational description"
    )
    alt_text_rationale: str = Field(
        ...,
        description="Explanation of why this alt-text supports learning"
    )
    
    # Quality metrics
    pedagogical_quality_score: float = Field(
        ...,
        ge=1.0,
        le=5.0, 
        description="Self-assessed quality (1-5 scale from paper)"
    )
    confidence_level: ConfidenceLevel = Field(
        ...,
        description="Overall confidence in the output"
    )
    
    # Technical metadata
    processing_duration: float = Field(
        ...,
        description="Processing time in seconds"
    )
    llm_model_used: str = Field(
        default="gpt-4o",
        description="LLM model version used"
    )
    prompt_template_used: str = Field(
        ...,
        description="Which prompt template was applied"
    )
    
    # Error handling
    processing_warnings: List[str] = Field(
        default_factory=list,
        description="Non-fatal issues during processing"
    )
    fallback_used: bool = Field(
        default=False,
        description="Whether fallback processing was used"
    )
    
    # Raw LLM response (for debugging)
    raw_llm_response: Optional[str] = Field(
        None,
        description="Complete LLM response for debugging"
    )
    
    created_at: datetime.datetime = Field(
        default_factory=datetime.datetime.now,
        description="When this output was generated"
    )


class SpatialContext(BaseModel):
    """Spatial context extracted from surrounding elements."""
    
    preceding_text: str = Field(default="", description="Text before the element")
    following_text: str = Field(default="", description="Text after the element") 
    containing_section: Optional[str] = Field(None, description="Parent section title")
    nearby_headings: List[str] = Field(default_factory=list, description="Relevant headings")
    page_position: str = Field(..., description="top|middle|bottom of page")


class LLMRequest(BaseModel):
    """Schema for LLM API requests."""
    
    prompt: str = Field(..., description="Complete prompt for LLM")
    image_data: Optional[bytes] = Field(None, description="Image for multimodal processing")
    temperature: float = Field(0.3, ge=0.0, le=1.0, description="LLM temperature")
    max_tokens: int = Field(1000, gt=0, description="Maximum response tokens")
    subject_hint: Optional[SubjectArea] = Field(None, description="Subject area hint")


class LLMResponse(BaseModel):
    """Schema for LLM API responses."""
    
    content: str = Field(..., description="LLM response content")
    usage_tokens: int = Field(..., description="Tokens consumed")
    response_time: float = Field(..., description="Response time in seconds")
    model_version: str = Field(..., description="Model version used")
