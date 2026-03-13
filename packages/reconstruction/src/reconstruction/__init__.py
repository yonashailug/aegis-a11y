"""Aegis-A11y Structural Reconstruction Package
    
Document generation and accessibility compliance layer.
Based on technical paper Section 3.3: Structural Reconstruction & Validation
"""

from .document_engine import DocumentReconstructionEngine
from .schemas import ReconstructionInput, ReconstructionOutput, OutputFormat
from .html_generator import HTML5Generator
from .pdf_generator import PDFUAGenerator

__version__ = "0.1.0"

__all__ = [
    "DocumentReconstructionEngine",
    "ReconstructionInput", 
    "ReconstructionOutput",
    "OutputFormat",
    "HTML5Generator",
    "PDFUAGenerator"
]