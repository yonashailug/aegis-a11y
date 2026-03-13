"""Aegis-A11y Structural Reconstruction Package

Document generation and accessibility compliance layer.
Based on technical paper Section 3.3: Structural Reconstruction & Validation
"""

from .document_engine import DocumentReconstructionEngine
from .html_generator import HTML5Generator
from .pdf_generator import PDFUAGenerator
from .schemas import OutputFormat, ReconstructionInput, ReconstructionOutput

__version__ = "0.1.0"

__all__ = [
    "DocumentReconstructionEngine",
    "HTML5Generator",
    "OutputFormat",
    "PDFUAGenerator",
    "ReconstructionInput",
    "ReconstructionOutput",
]
