"""Context processing for spatial and semantic analysis."""

from dataclasses import dataclass
import re
from typing import Any

from .schemas import SpatialContext, SubjectArea


@dataclass
class SubjectDetectionResult:
    """Result of subject area detection."""

    area: SubjectArea
    confidence: float
    indicators: list[str]


class ContextProcessor:
    """Processes spatial and semantic context for reasoning agent.

    This class extracts surrounding text context, detects subject areas,
    and prepares contextual information for LLM processing.

    Based on paper Section 3.2: Contextual Injection requirements.
    """

    def __init__(self):
        """Initialize the context processor."""
        # Subject detection keywords (could be expanded with ML models)
        self.subject_keywords = {
            SubjectArea.PHYSICS: [
                "force",
                "velocity",
                "acceleration",
                "energy",
                "momentum",
                "gravity",
                "friction",
                "pressure",
                "wave",
                "frequency",
                "amplitude",
                "newton",
                "kinematics",
                "dynamics",
                "thermodynamics",
                "electricity",
                "magnetism",
                "optics",
                "quantum",
                "relativity",
                "mass",
                "weight",
                "motion",
            ],
            SubjectArea.CHEMISTRY: [
                "molecule",
                "atom",
                "bond",
                "reaction",
                "element",
                "compound",
                "periodic",
                "ionic",
                "covalent",
                "oxidation",
                "reduction",
                "catalyst",
                "acid",
                "base",
                "ph",
                "mole",
                "molarity",
                "stoichiometry",
                "organic",
                "inorganic",
                "polymer",
                "isomer",
                "electron",
            ],
            SubjectArea.BIOLOGY: [
                "cell",
                "organism",
                "evolution",
                "genetics",
                "protein",
                "dna",
                "rna",
                "enzyme",
                "photosynthesis",
                "respiration",
                "mitosis",
                "meiosis",
                "ecosystem",
                "species",
                "habitat",
                "adaptation",
                "mutation",
                "chromosome",
                "gene",
                "allele",
                "phenotype",
            ],
            SubjectArea.MATHEMATICS: [
                "equation",
                "function",
                "variable",
                "graph",
                "solution",
                "theorem",
                "proof",
                "derivative",
                "integral",
                "limit",
                "matrix",
                "vector",
                "polynomial",
                "logarithm",
                "trigonometry",
                "geometry",
                "algebra",
                "calculus",
                "statistics",
                "probability",
                "coefficient",
            ],
            SubjectArea.HISTORY: [
                "century",
                "civilization",
                "empire",
                "revolution",
                "war",
                "treaty",
                "democracy",
                "monarchy",
                "republic",
                "constitution",
                "independence",
                "colonial",
                "ancient",
                "medieval",
                "renaissance",
                "industrial",
                "primary source",
                "artifact",
                "chronology",
                "era",
                "dynasty",
            ],
            SubjectArea.LITERATURE: [
                "metaphor",
                "symbolism",
                "theme",
                "character",
                "plot",
                "narrative",
                "poetry",
                "prose",
                "stanza",
                "rhyme",
                "meter",
                "alliteration",
                "protagonist",
                "antagonist",
                "irony",
                "foreshadowing",
                "imagery",
                "allegory",
                "satire",
                "tragedy",
                "comedy",
                "genre",
            ],
            SubjectArea.SOCIAL_STUDIES: [
                "government",
                "democracy",
                "citizenship",
                "economics",
                "supply",
                "demand",
                "market",
                "capitalism",
                "socialism",
                "culture",
                "society",
                "geography",
                "climate",
                "population",
                "urban",
                "rural",
                "globalization",
                "migration",
                "infrastructure",
                "policy",
            ],
        }

        self.initialized = True

    def extract_spatial_context(
        self, target_element: dict[str, Any], surrounding_elements: list[dict[str, Any]]
    ) -> SpatialContext:
        """Extract spatial context from surrounding elements.

        Args:
            target_element: The main element being processed
            surrounding_elements: List of nearby elements for context

        Returns:
            SpatialContext with extracted information
        """
        target_bbox = target_element.get("bounding_box", [0, 0, 0, 0])
        target_x, target_y = target_bbox[0], target_bbox[1]

        preceding_text = ""
        following_text = ""
        nearby_headings = []
        containing_section = None

        # Sort elements by position (top to bottom, left to right)
        sorted_elements = sorted(
            surrounding_elements,
            key=lambda el: (
                el.get("bounding_box", [0, 0, 0, 0])[1],
                el.get("bounding_box", [0, 0, 0, 0])[0],
            ),
        )

        for element in sorted_elements:
            el_bbox = element.get("bounding_box", [0, 0, 0, 0])
            el_x, el_y = el_bbox[0], el_bbox[1]
            el_text = element.get("ocr_text", "").strip()
            el_classification = element.get("classification", "")

            if not el_text:
                continue

            # Collect headings
            if (
                "heading" in el_classification.lower()
                or "h1" in element.get("html_tag", "").lower()
            ):
                nearby_headings.append(el_text)

                # Find containing section (closest preceding heading)
                if el_y < target_y and (
                    not containing_section or el_y > target_y - 200
                ):
                    containing_section = el_text

            # Collect surrounding text
            distance = abs(el_x - target_x) + abs(el_y - target_y)
            if distance < 300:  # Within reasonable proximity
                if el_y < target_y:  # Above/before target
                    preceding_text += " " + el_text
                elif el_y > target_y:  # Below/after target
                    following_text += " " + el_text

        # Determine page position
        page_position = "middle"
        if target_y < 200:
            page_position = "top"
        elif target_y > 600:  # Assuming ~800px page height
            page_position = "bottom"

        return SpatialContext(
            preceding_text=preceding_text.strip()[:500],  # Limit length
            following_text=following_text.strip()[:500],
            containing_section=containing_section,
            nearby_headings=nearby_headings[:5],  # Top 5 headings
            page_position=page_position,
        )

    def detect_subject_area(
        self, context_text: str, page_metadata: dict[str, Any]
    ) -> SubjectDetectionResult:
        """Detect the subject area of the content.

        Args:
            context_text: Combined text from surrounding context
            page_metadata: Additional hints from page-level information

        Returns:
            SubjectDetectionResult with detected area and confidence
        """
        # Check for explicit subject hint in metadata
        if "subject_hint" in page_metadata:
            hint = page_metadata["subject_hint"].lower()
            for subject in SubjectArea:
                if subject.value in hint:
                    return SubjectDetectionResult(
                        area=subject,
                        confidence=0.9,
                        indicators=[f"metadata hint: {hint}"],
                    )

        # Analyze context text for subject keywords
        text_lower = context_text.lower()
        subject_scores = {}

        for subject, keywords in self.subject_keywords.items():
            score = 0
            found_keywords = []

            for keyword in keywords:
                # Count keyword occurrences with word boundaries
                pattern = r"\b" + re.escape(keyword.lower()) + r"\b"
                matches = len(re.findall(pattern, text_lower))
                if matches > 0:
                    score += matches
                    found_keywords.append(keyword)

            if score > 0:
                subject_scores[subject] = {"score": score, "keywords": found_keywords}

        # Determine best match
        if not subject_scores:
            return SubjectDetectionResult(
                area=SubjectArea.GENERAL,
                confidence=0.3,
                indicators=["no specific subject keywords found"],
            )

        best_subject = max(
            subject_scores.keys(), key=lambda s: subject_scores[s]["score"]
        )
        best_score = subject_scores[best_subject]["score"]
        best_keywords = subject_scores[best_subject]["keywords"]

        # Calculate confidence based on score and keyword diversity
        total_words = len(text_lower.split())
        keyword_density = best_score / max(total_words, 1)
        keyword_diversity = len(best_keywords) / len(
            self.subject_keywords[best_subject]
        )

        confidence = min(0.95, 0.4 + (keyword_density * 2) + (keyword_diversity * 0.3))

        return SubjectDetectionResult(
            area=best_subject,
            confidence=confidence,
            indicators=best_keywords[:5],  # Top 5 keywords
        )

    def get_learning_context(
        self, spatial_context: SpatialContext, detected_subject: SubjectDetectionResult
    ) -> dict[str, Any]:
        """Extract additional learning context information.

        Args:
            spatial_context: Spatial context information
            detected_subject: Subject detection results

        Returns:
            Dictionary with learning context metadata
        """
        context = {
            "subject_area": detected_subject.area.value,
            "confidence": detected_subject.confidence,
            "section_title": spatial_context.containing_section,
            "page_position": spatial_context.page_position,
            "context_indicators": detected_subject.indicators,
        }

        # Add subject-specific context hints
        if detected_subject.area in [
            SubjectArea.PHYSICS,
            SubjectArea.CHEMISTRY,
            SubjectArea.BIOLOGY,
        ]:
            context["content_type"] = "STEM"
            context["description_focus"] = "data and relationships"
        elif detected_subject.area in [
            SubjectArea.HISTORY,
            SubjectArea.LITERATURE,
            SubjectArea.SOCIAL_STUDIES,
        ]:
            context["content_type"] = "Humanities"
            context["description_focus"] = "context and significance"
        else:
            context["content_type"] = "General"
            context["description_focus"] = "clear explanation"

        return context
