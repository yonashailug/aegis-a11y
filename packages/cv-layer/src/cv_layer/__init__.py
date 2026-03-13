from importlib.metadata import PackageNotFoundError, version

__all__ = ["LayoutDecomposer", "convert_pdf_to_images", "extract_ocr_data"]

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "0.0.0"

from .decomposer import LayoutDecomposer
from .main import convert_pdf_to_images, extract_ocr_data
