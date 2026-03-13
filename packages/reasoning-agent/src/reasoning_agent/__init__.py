"""Aegis-A11y Reasoning Agent Package
    Multi-modal semantic reasoning layer for educational content accessibility.
"""

from .schemas import ReasoningInput, ReasoningOutput, SubjectArea, ConfidenceLevel
from .context_processor import ContextProcessor
from .alt_text_generator import AltTextGenerator
from .semantic_reasoner import SemanticReasoner
from .prompt_templates import get_template_for_subject, FEW_SHOT_EXAMPLES
from .verifier import DeterministicVerifier, VerificationResult, ValidationIssue
from .quality_assessor import QualityAssessor, QualityMetrics, PedagogicalLevel
from .human_validator import HumanValidator, ReviewFeedback, ReviewSession

__version__ = "0.1.0"

__all__ = [
    "ReasoningInput",
    "ReasoningOutput", 
    "SubjectArea",
    "ConfidenceLevel",
    "ContextProcessor",
    "AltTextGenerator",
    "SemanticReasoner",
    "get_template_for_subject",
    "FEW_SHOT_EXAMPLES",
    "DeterministicVerifier",
    "VerificationResult",
    "ValidationIssue",
    "QualityAssessor",
    "QualityMetrics",
    "PedagogicalLevel",
    "HumanValidator",
    "ReviewFeedback",
    "ReviewSession"
]
