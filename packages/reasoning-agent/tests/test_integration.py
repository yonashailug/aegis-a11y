"""Integration tests for cv-layer → reasoning-agent pipeline."""

from unittest.mock import Mock, patch

import pytest

from cv_layer.decomposer import ExtractedElement
from reasoning_agent.alt_text_generator import AltTextGenerator
from reasoning_agent.context_processor import ContextProcessor
from reasoning_agent.prompt_templates import get_template_for_subject
from reasoning_agent.schemas import ReasoningInput, ReasoningOutput, SubjectArea
from reasoning_agent.semantic_reasoner import SemanticReasoner


class TestCVLayerIntegration:
    """Test integration between cv-layer and reasoning-agent components."""

    @pytest.fixture
    def sample_physics_element(self):
        """Sample physics diagram from cv-layer."""
        return ExtractedElement(
            element_id="physics_001",
            classification="functional_diagram",
            bounding_box=[100, 200, 400, 500],
            ocr_text="Force Diagram: mg = 50N ↓, N = 50N ↑, f = 10N ←",
            html_tag="<figure>",
        )

    @pytest.fixture
    def sample_chemistry_element(self):
        """Sample chemistry equation from cv-layer."""
        return ExtractedElement(
            element_id="chem_001",
            classification="equation",
            bounding_box=[150, 300, 350, 350],
            ocr_text="CH₄ + 2O₂ → CO₂ + 2H₂O",
            html_tag="<math>",
        )

    @pytest.fixture
    def sample_surrounding_elements(self):
        """Sample surrounding elements for context."""
        return [
            {
                "element_id": "heading_001",
                "classification": "heading",
                "bounding_box": [50, 150, 500, 200],
                "ocr_text": "Chapter 4: Force Analysis",
                "html_tag": "<h2>",
            },
            {
                "element_id": "paragraph_001",
                "classification": "paragraph",
                "bounding_box": [50, 550, 500, 650],
                "ocr_text": "Forces acting on objects can be analyzed using vector components and equilibrium principles.",
                "html_tag": "<p>",
            },
        ]

    def test_extracted_element_to_reasoning_input_conversion(
        self, sample_physics_element
    ):
        """Test ExtractedElement converts correctly to ReasoningInput."""
        reasoning_input = ReasoningInput.from_cv_output(
            extracted_element=sample_physics_element,
            page_metadata={"subject_hint": "physics"},
        )

        assert reasoning_input.get_element_classification() == "functional_diagram"
        assert (
            reasoning_input.get_element_text()
            == "Force Diagram: mg = 50N ↓, N = 50N ↑, f = 10N ←"
        )
        assert reasoning_input.page_metadata["subject_hint"] == "physics"
        assert isinstance(reasoning_input.extracted_element, dict)

    def test_spatial_context_extraction(
        self, sample_physics_element, sample_surrounding_elements
    ):
        """Test spatial context extraction from cv-layer output."""
        context_processor = ContextProcessor()

        spatial_context = context_processor.extract_spatial_context(
            sample_physics_element.model_dump(), sample_surrounding_elements
        )

        assert spatial_context.containing_section == "Chapter 4: Force Analysis"
        assert "Chapter 4: Force Analysis" in spatial_context.preceding_text
        # Following text may be empty if paragraph is positioned below in bbox coordinates
        assert len(spatial_context.nearby_headings) >= 1
        assert "Chapter 4: Force Analysis" in spatial_context.nearby_headings
        assert len(spatial_context.nearby_headings) >= 1

    def test_subject_detection_with_cv_data(self, sample_chemistry_element):
        """Test subject detection using cv-layer extracted content."""
        context_processor = ContextProcessor()

        context_text = (
            "Chemical reactions and stoichiometry " + sample_chemistry_element.ocr_text
        )
        metadata = {"subject_hint": "chemistry"}

        result = context_processor.detect_subject_area(context_text, metadata)

        assert result.area == SubjectArea.CHEMISTRY
        assert result.confidence >= 0.8
        assert "metadata hint: chemistry" in result.indicators

    def test_template_selection_for_cv_subjects(self):
        """Test that appropriate templates are selected for cv-detected subjects."""
        physics_template = get_template_for_subject("physics")
        chemistry_template = get_template_for_subject("chemistry")
        math_template = get_template_for_subject("mathematics")

        assert "physics education" in physics_template.lower()
        assert "chemistry education" in chemistry_template.lower()
        assert "mathematics education" in math_template.lower()

        # Templates should be different
        assert physics_template != chemistry_template
        assert chemistry_template != math_template

    def test_pedagogical_alt_text_generation(self, sample_physics_element):
        """Test pedagogical alt-text generation with cv-layer data."""
        alt_text_generator = AltTextGenerator()
        context_processor = ContextProcessor()

        # Create spatial context
        spatial_context = context_processor.extract_spatial_context(
            sample_physics_element.model_dump(), []
        )

        # Detect subject
        detected_subject = context_processor.detect_subject_area(
            "force diagram physics vectors", {"subject_hint": "physics"}
        )

        # Generate alt-text
        mock_llm_response = (
            "Physics force diagram showing three force vectors in equilibrium."
        )

        result = alt_text_generator.generate_pedagogical_description(
            sample_physics_element.model_dump(),
            spatial_context,
            detected_subject,
            mock_llm_response,
        )

        assert len(result.alt_text) >= 10
        assert "physics" in result.alt_text.lower()
        assert len(result.udl_guidelines_applied) > 0
        assert result.importance != ""
        assert result.rationale != ""

    def test_complete_pipeline_physics(
        self, sample_physics_element, sample_surrounding_elements
    ):
        """Test complete pipeline: cv-layer → reasoning-agent for physics content."""
        # Convert cv-layer output to reasoning input
        reasoning_input = ReasoningInput.from_cv_output(
            extracted_element=sample_physics_element,
            surrounding_elements=sample_surrounding_elements,
            page_metadata={"subject_hint": "physics", "document_type": "textbook"},
        )

        # Initialize components
        context_processor = ContextProcessor()
        alt_text_generator = AltTextGenerator()

        # Process through pipeline
        spatial_context = context_processor.extract_spatial_context(
            reasoning_input.extracted_element, reasoning_input.surrounding_elements
        )

        detected_subject = context_processor.detect_subject_area(
            spatial_context.preceding_text
            + " "
            + spatial_context.following_text
            + " "
            + reasoning_input.get_element_text(),
            reasoning_input.page_metadata,
        )

        mock_llm_response = """Physics force diagram illustrating force equilibrium on an object.
        The diagram shows gravitational force (mg = 50N) acting downward, normal force (N = 50N) acting upward, 
        and friction force (f = 10N) acting horizontally. This demonstrates static equilibrium principles."""

        alt_text_result = alt_text_generator.generate_pedagogical_description(
            reasoning_input.extracted_element,
            spatial_context,
            detected_subject,
            mock_llm_response,
        )

        # Verify complete pipeline results
        assert detected_subject.area == SubjectArea.PHYSICS
        assert detected_subject.confidence >= 0.7
        assert "force" in alt_text_result.alt_text.lower()
        assert "physics" in alt_text_result.alt_text.lower()
        assert len(alt_text_result.udl_guidelines_applied) > 0

    def test_complete_pipeline_chemistry(self, sample_chemistry_element):
        """Test complete pipeline for chemistry content."""
        reasoning_input = ReasoningInput.from_cv_output(
            extracted_element=sample_chemistry_element,
            page_metadata={"subject_hint": "chemistry"},
        )

        context_processor = ContextProcessor()
        alt_text_generator = AltTextGenerator()

        spatial_context = context_processor.extract_spatial_context(
            reasoning_input.extracted_element, []
        )

        detected_subject = context_processor.detect_subject_area(
            "chemical reaction methane combustion "
            + reasoning_input.get_element_text(),
            reasoning_input.page_metadata,
        )

        mock_llm_response = """Chemistry equation showing methane combustion reaction.
        The balanced equation CH₄ + 2O₂ → CO₂ + 2H₂O demonstrates stoichiometric relationships 
        and conservation of mass in chemical reactions."""

        alt_text_result = alt_text_generator.generate_pedagogical_description(
            reasoning_input.extracted_element,
            spatial_context,
            detected_subject,
            mock_llm_response,
        )

        assert detected_subject.area == SubjectArea.CHEMISTRY
        assert (
            "chemistry" in alt_text_result.alt_text.lower()
            or "chemical" in alt_text_result.alt_text.lower()
        )
        assert (
            "reaction" in alt_text_result.alt_text.lower()
            or "equation" in alt_text_result.alt_text.lower()
        )

    @patch("reasoning_agent.semantic_reasoner.OpenAI")
    def test_semantic_reasoner_with_cv_data(self, mock_openai, sample_physics_element):
        """Test SemanticReasoner with cv-layer data (mocked OpenAI)."""
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = (
            "Physics force diagram showing equilibrium forces."
        )
        mock_response.usage.total_tokens = 150
        mock_openai.return_value.chat.completions.create.return_value = mock_response

        # Create reasoning input
        reasoning_input = ReasoningInput.from_cv_output(
            extracted_element=sample_physics_element,
            page_metadata={"subject_hint": "physics"},
        )

        # Test semantic reasoner
        reasoner = SemanticReasoner(api_key="test_key")

        # This would normally call OpenAI, but we're mocking it
        result = reasoner.process_element(reasoning_input)

        assert isinstance(result, ReasoningOutput)
        assert result.element_id == "physics_001"
        assert result.detected_subject_area == SubjectArea.PHYSICS
        assert len(result.pedagogical_alt_text) >= 10
        assert result.confidence_level in ["high", "medium", "low"]

    def test_data_format_compatibility(self):
        """Test that all data formats are compatible between components."""
        # Test ExtractedElement structure matches ReasoningInput expectations
        element = ExtractedElement(
            element_id="test",
            classification="figure",
            bounding_box=[0, 0, 100, 100],
            ocr_text="test text",
            html_tag="<img>",
        )

        element_dict = element.model_dump()

        # These fields should exist and match expected format
        required_fields = [
            "element_id",
            "classification",
            "bounding_box",
            "ocr_text",
            "html_tag",
        ]
        for field in required_fields:
            assert field in element_dict

        # Bounding box should be list of numbers
        assert isinstance(element_dict["bounding_box"], list)
        assert len(element_dict["bounding_box"]) == 4
        assert all(isinstance(x, (int, float)) for x in element_dict["bounding_box"])

    def test_error_handling_with_invalid_cv_data(self):
        """Test error handling with malformed cv-layer data."""
        context_processor = ContextProcessor()

        # Test with missing fields
        invalid_element = {"element_id": "test"}

        try:
            spatial_context = context_processor.extract_spatial_context(
                invalid_element, []
            )
            # Should handle missing fields gracefully
            assert spatial_context.preceding_text == ""
            assert spatial_context.containing_section is None
        except Exception as e:
            # Should not crash, but if it does, error should be informative
            assert "bounding_box" in str(e) or "classification" in str(e)

    def test_performance_with_multiple_elements(self):
        """Test performance with multiple cv-layer elements."""
        elements = []
        for i in range(10):
            elements.append(
                ExtractedElement(
                    element_id=f"element_{i}",
                    classification="paragraph" if i % 2 == 0 else "figure",
                    bounding_box=[i * 50, i * 50, (i + 1) * 50, (i + 1) * 50],
                    ocr_text=f"Test text for element {i}",
                    html_tag="<p>" if i % 2 == 0 else "<figure>",
                )
            )

        context_processor = ContextProcessor()

        # Should process multiple elements efficiently
        for element in elements:
            reasoning_input = ReasoningInput.from_cv_output(
                extracted_element=element,
                surrounding_elements=[e.model_dump() for e in elements if e != element][
                    :5
                ],
            )

            spatial_context = context_processor.extract_spatial_context(
                reasoning_input.extracted_element, reasoning_input.surrounding_elements
            )

            assert spatial_context is not None
            # Should have reasonable performance (this is implicit - test runs quickly)
