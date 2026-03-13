"""Aegis-A11y Reasoning Agent Package
Multi-modal semantic reasoning layer for educational content accessibility.
"""

from .alt_text_generator import AltTextGenerator
from .context_processor import ContextProcessor
from .element_filter import ElementFilter, FilteredElement, FilterStats
from .human_validator import HumanValidator, ReviewFeedback, ReviewSession
from .prompt_templates import FEW_SHOT_EXAMPLES, get_template_for_subject
from .quality_assessor import PedagogicalLevel, QualityAssessor, QualityMetrics
from .schemas import ConfidenceLevel, ReasoningInput, ReasoningOutput, SubjectArea
from .semantic_reasoner import SemanticReasoner
from .verifier import DeterministicVerifier, ValidationIssue, VerificationResult

__version__ = "0.1.0"

__all__ = [
    "FEW_SHOT_EXAMPLES",
    "AltTextGenerator",
    "ConfidenceLevel",
    "ContextProcessor",
    "DeterministicVerifier",
    "ElementFilter",
    "FilterStats",
    "FilteredElement",
    "HumanValidator",
    "PedagogicalLevel",
    "QualityAssessor",
    "QualityMetrics",
    "ReasoningInput",
    "ReasoningOutput",
    "ReviewFeedback",
    "ReviewSession",
    "SemanticReasoner",
    "SubjectArea",
    "ValidationIssue",
    "VerificationResult",
    "get_template_for_subject",
]
