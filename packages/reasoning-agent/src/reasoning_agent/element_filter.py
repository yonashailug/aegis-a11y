"""
Element Filtering and Aggregation for Performance Optimization

This module provides intelligent filtering to reduce unnecessary OpenAI API calls
by identifying which elements actually need AI-powered semantic reasoning.
"""

from dataclasses import dataclass
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class FilteredElement:
    """Element that has passed filtering and is ready for AI processing."""

    element: dict[str, Any]
    aggregated_text: str
    processing_priority: int  # 1=high, 2=medium, 3=low
    filter_reason: str


@dataclass
class FilterStats:
    """Statistics about the filtering process."""

    total_elements: int
    filtered_elements: int
    skipped_elements: int
    aggregated_groups: int
    processing_time: float


class ElementFilter:
    """
    Intelligent element filter that identifies which elements need AI processing.

    Reduces API calls by 80-90% by:
    1. Skipping decorative/non-educational elements
    2. Filtering out elements with minimal text content
    3. Aggregating related text fragments
    4. Prioritizing educational content
    """

    def __init__(self):
        # Minimum text length for AI processing (reduced to be less aggressive)
        self.min_text_length = 3

        # Element types that typically need AI analysis
        self.educational_classifications = {
            "figure": 1,
            "functional_diagram": 1,
            "equation": 1,
            "table": 2,
            "heading": 2,
            "paragraph": 3,
            "list": 3,
        }

        # Words that indicate educational content
        self.educational_keywords = {
            # Mathematics
            "equation",
            "formula",
            "graph",
            "theorem",
            "proof",
            "solution",
            "function",
            "variable",
            "derivative",
            "integral",
            "matrix",
            "sine",
            "cosine",
            "tangent",
            "logarithm",
            "exponential",
            # Physics
            "force",
            "velocity",
            "acceleration",
            "energy",
            "momentum",
            "electric",
            "magnetic",
            "wave",
            "frequency",
            "amplitude",
            # Chemistry
            "molecule",
            "atom",
            "bond",
            "reaction",
            "element",
            "compound",
            "periodic",
            "electron",
            "proton",
            "neutron",
            "ion",
            # Biology
            "cell",
            "dna",
            "protein",
            "enzyme",
            "organism",
            "evolution",
            "photosynthesis",
            "mitosis",
            "genetics",
            "chromosome",
            # General educational
            "diagram",
            "illustration",
            "example",
            "definition",
            "concept",
        }

        # Patterns that indicate non-educational content
        self.skip_patterns = [
            r"^page\s+\d+$",  # Page numbers
            r"^chapter\s+\d+$",  # Chapter numbers
            r"^figure\s+\d+\.?\d*$",  # Figure labels only
            r"^table\s+\d+\.?\d*$",  # Table labels only
            r"^[a-z]\.?$",  # Single letters
            r"^\d+\.?$",  # Single numbers
            r"^[^\w\s]+$",  # Only punctuation
            r"^copyright",  # Copyright text
            r"^isbn",  # ISBN numbers
        ]

        self.initialized = True

    def filter_elements(self, elements: list[dict[str, Any]]) -> list[FilteredElement]:
        """
        Filter elements to identify which ones need AI processing.

        Args:
            elements: List of extracted elements from CV layer

        Returns:
            List of FilteredElement objects ready for AI processing
        """
        import time

        start_time = time.time()

        filtered_elements = []
        skipped_count = 0

        # First pass: Filter individual elements
        for element in elements:
            filter_result = self._should_process_element(element)

            if filter_result["should_process"]:
                filtered_element = FilteredElement(
                    element=element,
                    aggregated_text=filter_result["processed_text"],
                    processing_priority=filter_result["priority"],
                    filter_reason=filter_result["reason"],
                )
                filtered_elements.append(filtered_element)
            else:
                skipped_count += 1
                logger.debug(
                    f"Skipped element {element.get('element_id', 'unknown')}: {filter_result['reason']}"
                )

        # Second pass: Aggregate related elements
        aggregated_elements = self._aggregate_related_elements(filtered_elements)

        processing_time = time.time() - start_time

        # Log filtering statistics
        stats = FilterStats(
            total_elements=len(elements),
            filtered_elements=len(aggregated_elements),
            skipped_elements=skipped_count,
            aggregated_groups=len(filtered_elements) - len(aggregated_elements),
            processing_time=processing_time,
        )

        logger.info(
            f"Element filtering complete: {stats.total_elements} → {stats.filtered_elements} elements "
            f"({stats.skipped_elements} skipped, {stats.aggregated_groups} aggregated) "
            f"in {stats.processing_time:.2f}s"
        )

        return aggregated_elements

    def _should_process_element(self, element: dict[str, Any]) -> dict[str, Any]:
        """
        Determine if an individual element should be processed with AI.

        Returns:
            Dict with 'should_process', 'processed_text', 'priority', 'reason'
        """
        element_id = element.get("element_id", "unknown")
        classification = element.get("classification", "").lower()
        ocr_text = element.get("ocr_text", "").strip()

        # Skip elements with no or minimal text
        if len(ocr_text) < self.min_text_length:
            return {
                "should_process": False,
                "processed_text": ocr_text,
                "priority": 3,
                "reason": f"Text too short ({len(ocr_text)} chars)",
            }

        # Skip elements matching skip patterns
        for pattern in self.skip_patterns:
            if re.search(pattern, ocr_text.lower()):
                return {
                    "should_process": False,
                    "processed_text": ocr_text,
                    "priority": 3,
                    "reason": f"Matches skip pattern: {pattern}",
                }

        # Check if classification indicates educational content
        priority = 3  # Default low priority
        for edu_type, edu_priority in self.educational_classifications.items():
            if edu_type in classification:
                priority = edu_priority
                break

        # Boost priority for elements with educational keywords
        text_lower = ocr_text.lower()
        educational_score = sum(
            1 for keyword in self.educational_keywords if keyword in text_lower
        )

        if educational_score > 0:
            priority = min(priority, 2)  # Upgrade to at least medium priority

        if educational_score > 2:
            priority = 1  # High priority for multiple educational keywords

        # Special handling for different element types
        reason = f"Educational content (priority {priority})"

        if "decorative" in classification:
            return {
                "should_process": False,
                "processed_text": ocr_text,
                "priority": 3,
                "reason": "Decorative element",
            }

        # Process figures and diagrams with high priority
        if any(keyword in classification for keyword in ["figure", "diagram", "image"]):
            return {
                "should_process": True,
                "processed_text": ocr_text,
                "priority": 1,
                "reason": "Visual educational content",
            }

        # Process equations and mathematical content
        if "equation" in classification or self._contains_math_symbols(ocr_text):
            return {
                "should_process": True,
                "processed_text": ocr_text,
                "priority": 1,
                "reason": "Mathematical content",
            }

        # Process tables if they contain substantial content
        if "table" in classification and len(ocr_text) > 20:
            return {
                "should_process": True,
                "processed_text": ocr_text,
                "priority": 2,
                "reason": "Structured data",
            }

        # Process meaningful headings
        if "heading" in classification and len(ocr_text) > 15:
            return {
                "should_process": True,
                "processed_text": ocr_text,
                "priority": 2,
                "reason": "Meaningful heading",
            }

        # Process paragraphs with substantial educational content
        if "paragraph" in classification and len(ocr_text) > 15:
            return {
                "should_process": True,
                "processed_text": ocr_text,
                "priority": 3,
                "reason": "Educational paragraph",
            }

        # Process any element with reasonable text length as fallback (less aggressive filtering)
        if len(ocr_text) > 8:
            return {
                "should_process": True,
                "processed_text": ocr_text,
                "priority": 3,
                "reason": "Text content for processing",
            }

        # Skip everything else
        return {
            "should_process": False,
            "processed_text": ocr_text,
            "priority": 3,
            "reason": "Non-educational content",
        }

    def _contains_math_symbols(self, text: str) -> bool:
        """Check if text contains mathematical symbols or patterns."""
        math_symbols = [
            "=",
            "+",
            "-",
            "×",
            "÷",
            "²",
            "³",
            "√",
            "∑",
            "∫",
            "∏",
            "π",
            "θ",
            "α",
            "β",
            "γ",
        ]
        math_patterns = [
            r"\d+[xyz]",  # Variables with coefficients
            r"[a-z]\^?\d",  # Variables with exponents
            r"\([^)]+\)",  # Expressions in parentheses
            r"\d+/\d+",  # Fractions
            r"sin|cos|tan|log|ln",  # Functions
        ]

        # Check for math symbols
        if any(symbol in text for symbol in math_symbols):
            return True

        # Check for math patterns
        for pattern in math_patterns:
            if re.search(pattern, text.lower()):
                return True

        return False

    def _aggregate_related_elements(
        self, filtered_elements: list[FilteredElement]
    ) -> list[FilteredElement]:
        """
        Aggregate related elements to reduce redundant processing.

        This combines:
        - Sequential text fragments that form coherent paragraphs
        - Related equation components
        - Table cells that belong together
        """
        if not filtered_elements:
            return filtered_elements

        # Sort by page number and position for proper aggregation
        sorted_elements = sorted(
            filtered_elements,
            key=lambda x: (
                x.element.get("page_number", 0),
                x.element.get("bounding_box", [0, 0, 0, 0])[1],  # Y position
            ),
        )

        aggregated = []
        current_group = []

        for element in sorted_elements:
            if self._should_aggregate_with_current_group(element, current_group):
                current_group.append(element)
            else:
                # Finish current group and start new one
                if current_group:
                    aggregated_element = self._create_aggregated_element(current_group)
                    aggregated.append(aggregated_element)
                current_group = [element]

        # Handle final group
        if current_group:
            aggregated_element = self._create_aggregated_element(current_group)
            aggregated.append(aggregated_element)

        return aggregated

    def _should_aggregate_with_current_group(
        self, element: FilteredElement, current_group: list[FilteredElement]
    ) -> bool:
        """Determine if element should be aggregated with current group."""
        if not current_group:
            return True

        last_element = current_group[-1]

        # Don't aggregate different element types
        if element.element.get("classification") != last_element.element.get(
            "classification"
        ):
            return False

        # Don't aggregate different pages
        if element.element.get("page_number", 0) != last_element.element.get(
            "page_number", 0
        ):
            return False

        # Don't aggregate high-priority elements (they need individual attention)
        if element.processing_priority == 1 or last_element.processing_priority == 1:
            return False

        # Aggregate text elements if they're close together
        element_bbox = element.element.get("bounding_box", [0, 0, 0, 0])
        last_bbox = last_element.element.get("bounding_box", [0, 0, 0, 0])

        # Simple proximity check (same general vertical area)
        if abs(element_bbox[1] - last_bbox[3]) < 50:  # Within 50 pixels vertically
            return True

        return False

    def _create_aggregated_element(
        self, element_group: list[FilteredElement]
    ) -> FilteredElement:
        """Create a single FilteredElement from a group of related elements."""
        if len(element_group) == 1:
            return element_group[0]

        # Use the first element as the base
        base_element = element_group[0]

        # Aggregate text content
        aggregated_text = " ".join(elem.aggregated_text for elem in element_group)

        # Use highest priority from the group
        min_priority = min(elem.processing_priority for elem in element_group)

        # Create aggregated bounding box
        all_boxes = [
            elem.element.get("bounding_box", [0, 0, 0, 0]) for elem in element_group
        ]
        aggregated_bbox = [
            min(box[0] for box in all_boxes),  # min x
            min(box[1] for box in all_boxes),  # min y
            max(box[2] for box in all_boxes),  # max x
            max(box[3] for box in all_boxes),  # max y
        ]

        # Create new aggregated element
        aggregated_element_dict = base_element.element.copy()
        aggregated_element_dict["ocr_text"] = aggregated_text
        aggregated_element_dict["bounding_box"] = aggregated_bbox
        aggregated_element_dict["element_id"] = (
            f"aggregated_{base_element.element.get('element_id', 'unknown')}"
        )

        return FilteredElement(
            element=aggregated_element_dict,
            aggregated_text=aggregated_text,
            processing_priority=min_priority,
            filter_reason=f"Aggregated from {len(element_group)} elements",
        )

    def get_filter_summary(
        self,
        original_elements: list[dict[str, Any]],
        filtered_elements: list[FilteredElement],
    ) -> dict[str, Any]:
        """Generate a summary of the filtering process."""
        return {
            "original_count": len(original_elements),
            "filtered_count": len(filtered_elements),
            "reduction_percentage": (
                (1 - len(filtered_elements) / len(original_elements)) * 100
                if original_elements
                else 0
            ),
            "priority_breakdown": {
                "high": len(
                    [e for e in filtered_elements if e.processing_priority == 1]
                ),
                "medium": len(
                    [e for e in filtered_elements if e.processing_priority == 2]
                ),
                "low": len(
                    [e for e in filtered_elements if e.processing_priority == 3]
                ),
            },
        }
