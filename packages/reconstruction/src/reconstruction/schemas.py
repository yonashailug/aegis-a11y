"""Schemas for structural reconstruction and document generation."""

from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime
from reasoning_agent.schemas import ReasoningOutput


class OutputFormat(str, Enum):
    """Supported output formats for document generation."""
    HTML5 = "html5"
    PDF_UA = "pdf_ua"
    EPUB = "epub"
    WORD = "docx"


class AccessibilityStandard(str, Enum):
    """Accessibility standards compliance levels."""
    WCAG_2_1_AA = "wcag_2_1_aa"
    WCAG_2_2_AA = "wcag_2_2_aa"
    PDF_UA_1 = "pdf_ua_1"
    EPUB_A11Y = "epub_accessibility"


class DocumentStructure(BaseModel):
    """Hierarchical document structure representation."""
    
    element_type: str = Field(..., description="HTML tag or structure type")
    element_id: Optional[str] = Field(None, description="Unique element identifier")
    attributes: Dict[str, str] = Field(default_factory=dict, description="Element attributes")
    content: Optional[str] = Field(None, description="Text content")
    alt_text: Optional[str] = Field(None, description="Accessibility description")
    children: List['DocumentStructure'] = Field(default_factory=list, description="Child elements")
    
    # Positioning and layout
    bounding_box: Optional[List[float]] = Field(None, description="Element position [x,y,w,h]")
    z_index: int = Field(0, description="Layer order for complex layouts")
    
    # Accessibility metadata
    aria_label: Optional[str] = Field(None, description="ARIA label")
    aria_describedby: Optional[str] = Field(None, description="ARIA description reference")
    role: Optional[str] = Field(None, description="ARIA role")
    
    # Educational metadata
    subject_area: Optional[str] = Field(None, description="Educational subject")
    learning_objective: Optional[str] = Field(None, description="Pedagogical purpose")
    importance_level: Optional[str] = Field(None, description="Educational importance")


class ReconstructionInput(BaseModel):
    """Input for document reconstruction process."""
    
    # Source data
    verified_elements: List[ReasoningOutput] = Field(
        ..., description="Verified reasoning outputs from verifier"
    )
    original_layout: List[Dict[str, Any]] = Field(
        default_factory=list, description="Original CV layout decomposition"
    )
    
    # Document metadata
    document_title: str = Field("Reconstructed Document", description="Document title")
    document_language: str = Field("en", description="Document language (ISO 639-1)")
    subject_area: Optional[str] = Field(None, description="Primary subject area")
    educational_level: Optional[str] = Field(None, description="Grade/education level")
    
    # Output preferences
    target_formats: List[OutputFormat] = Field(
        [OutputFormat.HTML5], description="Desired output formats"
    )
    accessibility_standard: AccessibilityStandard = Field(
        AccessibilityStandard.WCAG_2_1_AA, description="Target accessibility standard"
    )
    
    # Processing options
    preserve_layout: bool = Field(True, description="Maintain original layout structure")
    include_metadata: bool = Field(True, description="Include educational metadata")
    generate_navigation: bool = Field(True, description="Create navigation aids")
    
    created_at: datetime = Field(default_factory=datetime.now)


class ReconstructionOutput(BaseModel):
    """Output from document reconstruction process."""
    
    # Generated documents
    documents: Dict[OutputFormat, Union[str, bytes]] = Field(
        default_factory=dict, description="Generated documents by format"
    )
    
    # Document structure
    structure_tree: DocumentStructure = Field(..., description="Hierarchical document structure")
    navigation_tree: Optional[Dict[str, Any]] = Field(None, description="Navigation structure")
    
    # Compliance validation
    accessibility_report: Dict[str, Any] = Field(
        default_factory=dict, description="Accessibility compliance report"
    )
    wcag_compliance: Dict[str, bool] = Field(
        default_factory=dict, description="WCAG criteria compliance status"
    )
    
    # Quality metrics
    reconstruction_quality: float = Field(..., ge=0.0, le=1.0, description="Overall reconstruction quality")
    structure_accuracy: float = Field(..., ge=0.0, le=1.0, description="Structural fidelity to original")
    accessibility_score: float = Field(..., ge=0.0, le=1.0, description="Accessibility compliance score")
    
    # Processing metadata
    processing_duration: float = Field(..., description="Processing time in seconds")
    elements_processed: int = Field(..., description="Number of elements reconstructed")
    warnings: List[str] = Field(default_factory=list, description="Processing warnings")
    errors: List[str] = Field(default_factory=list, description="Processing errors")
    
    # Verification info
    verifier_passed: bool = Field(True, description="Whether output passed final verification")
    manual_review_required: bool = Field(False, description="Whether human review is needed")
    
    created_at: datetime = Field(default_factory=datetime.now)


class TagMapping(BaseModel):
    """Mapping configuration for JSON-to-tag conversion."""
    
    classification_to_tag: Dict[str, str] = Field(
        default_factory=lambda: {
            "heading": "h2",
            "paragraph": "p",
            "list": "ul",
            "list_item": "li", 
            "table": "table",
            "table_row": "tr",
            "table_cell": "td",
            "table_header": "th",
            "equation": "math",
            "functional_diagram": "figure",
            "decorative_image": "img",
            "code": "pre",
            "quote": "blockquote"
        },
        description="Mapping from CV classifications to HTML tags"
    )
    
    aria_role_mapping: Dict[str, str] = Field(
        default_factory=lambda: {
            "functional_diagram": "img",
            "equation": "math", 
            "table": "table",
            "list": "list",
            "navigation": "navigation",
            "main_content": "main",
            "complementary": "complementary"
        },
        description="ARIA role assignments"
    )
    
    heading_hierarchy: Dict[str, str] = Field(
        default_factory=lambda: {
            "title": "h1",
            "chapter": "h2", 
            "section": "h3",
            "subsection": "h4",
            "subsubsection": "h5",
            "paragraph_heading": "h6"
        },
        description="Heading level assignments"
    )


class ValidationRule(BaseModel):
    """Individual accessibility validation rule."""
    
    rule_id: str = Field(..., description="WCAG criterion identifier")
    rule_name: str = Field(..., description="Human-readable rule name")
    description: str = Field(..., description="Rule description")
    validation_function: str = Field(..., description="Validation function name")
    severity: str = Field("error", description="Rule severity level")
    
    
class ComplianceReport(BaseModel):
    """Detailed accessibility compliance report."""
    
    standard: AccessibilityStandard = Field(..., description="Evaluated standard")
    overall_compliance: bool = Field(..., description="Overall compliance status")
    compliance_percentage: float = Field(..., ge=0.0, le=100.0, description="Compliance percentage")
    
    passed_rules: List[str] = Field(default_factory=list, description="Passed validation rules")
    failed_rules: List[str] = Field(default_factory=list, description="Failed validation rules")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    
    detailed_results: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, description="Detailed rule-by-rule results"
    )
    
    recommendations: List[str] = Field(
        default_factory=list, description="Improvement recommendations"
    )
    
    generated_at: datetime = Field(default_factory=datetime.now)