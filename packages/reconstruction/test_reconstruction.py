"""End-to-end test for the document reconstruction pipeline.

Tests the complete DRR (Decomposition-Reasoning-Reconstruction) flow with real data.
"""

from pathlib import Path
import sys

# Add the src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from reasoning_agent.schemas import ConfidenceLevel, ReasoningOutput, SubjectArea
from reconstruction import (
    DocumentReconstructionEngine,
    OutputFormat,
    ReconstructionInput,
)


def create_test_reasoning_outputs():
    """Create sample verified reasoning outputs for testing."""
    return [
        ReasoningOutput(
            element_id="title_1",
            detected_subject_area=SubjectArea.PHYSICS,
            subject_confidence=0.95,
            contextual_importance="Main heading establishes the learning context for force and motion concepts",
            pedagogical_alt_text="Physics lesson title: Forces and Motion - introduces fundamental concepts of how pushes and pulls affect object movement",
            alt_text_rationale="Provides clear context for students about the educational focus on fundamental physics concepts",
            pedagogical_quality_score=4.5,
            confidence_level=ConfidenceLevel.HIGH,
            processing_duration=0.5,
            prompt_template_used="physics_heading_template",
        ),
        ReasoningOutput(
            element_id="paragraph_1",
            detected_subject_area=SubjectArea.PHYSICS,
            subject_confidence=0.92,
            contextual_importance="Foundational explanation of force concept and Newton's First Law of Motion",
            pedagogical_alt_text="Introductory paragraph defining force as a push or pull that changes object motion, introducing Newton's First Law about objects at rest and in motion",
            alt_text_rationale="Essential conceptual foundation that students need before advancing to more complex physics topics",
            pedagogical_quality_score=4.2,
            confidence_level=ConfidenceLevel.HIGH,
            processing_duration=0.7,
            prompt_template_used="physics_paragraph_template",
        ),
        ReasoningOutput(
            element_id="diagram_1",
            detected_subject_area=SubjectArea.PHYSICS,
            subject_confidence=0.98,
            contextual_importance="Visual representation critical for understanding force vector analysis and equilibrium concepts",
            pedagogical_alt_text="Free body diagram showing a rectangular block with four force vectors: weight pointing downward (gravity), normal force pointing upward (surface reaction), friction force pointing left (surface resistance), and applied force pointing right (external push). The block sits on a horizontal surface demonstrating force balance principles.",
            alt_text_rationale="Diagram provides essential visual scaffolding for students to understand abstract force concepts through concrete vector representation",
            pedagogical_quality_score=4.8,
            confidence_level=ConfidenceLevel.HIGH,
            processing_duration=1.2,
            prompt_template_used="physics_diagram_template",
        ),
        ReasoningOutput(
            element_id="equation_1",
            detected_subject_area=SubjectArea.PHYSICS,
            subject_confidence=0.99,
            contextual_importance="Fundamental equation connecting force, mass, and acceleration - cornerstone of classical mechanics",
            pedagogical_alt_text="Newton's Second Law equation: F equals m times a, where F represents force in Newtons, m represents mass in kilograms, and a represents acceleration in meters per second squared",
            alt_text_rationale="Mathematical expression essential for quantitative understanding of force relationships in physics problem-solving",
            pedagogical_quality_score=4.9,
            confidence_level=ConfidenceLevel.HIGH,
            processing_duration=0.4,
            prompt_template_used="physics_equation_template",
        ),
        ReasoningOutput(
            element_id="list_1",
            detected_subject_area=SubjectArea.PHYSICS,
            subject_confidence=0.90,
            contextual_importance="Categorization helps students organize different types of forces for systematic understanding",
            pedagogical_alt_text="Classification of force types: Contact forces including friction (surface resistance), normal force (surface support), and tension (string/rope pulling); Non-contact forces including gravity (mass attraction), magnetic forces (magnet interactions), and electric forces (charge interactions)",
            alt_text_rationale="Organizational framework helps students categorize and remember different force types encountered in physics problems",
            pedagogical_quality_score=4.1,
            confidence_level=ConfidenceLevel.HIGH,
            processing_duration=0.6,
            prompt_template_used="physics_list_template",
        ),
    ]


def create_test_reconstruction_input():
    """Create test input for reconstruction engine."""
    verified_elements = create_test_reasoning_outputs()

    original_layout = [
        {
            "type": "title",
            "bbox": [50, 50, 500, 100],
            "text": "Physics: Forces and Motion",
        },
        {
            "type": "paragraph",
            "bbox": [50, 120, 500, 200],
            "text": "Force is a push...",
        },
        {"type": "diagram", "bbox": [150, 220, 300, 350], "text": "Free body diagram"},
        {"type": "equation", "bbox": [200, 370, 200, 50], "text": "F = ma"},
        {"type": "list", "bbox": [50, 450, 500, 150], "text": "Types of forces..."},
    ]

    return ReconstructionInput(
        verified_elements=verified_elements,
        original_layout=original_layout,
        document_title="Physics Lesson: Forces and Motion",
        document_language="en",
        subject_area="physics",
        educational_level="high_school",
        target_formats=[OutputFormat.HTML5, OutputFormat.PDF_UA],
        preserve_layout=True,
        include_metadata=True,
        generate_navigation=True,
    )


def test_reconstruction_pipeline():
    """Test the complete reconstruction pipeline."""
    print("🚀 Starting end-to-end reconstruction pipeline test...")

    # Step 1: Create test data
    print("📝 Creating test data...")
    reconstruction_input = create_test_reconstruction_input()
    print(
        f"   ✅ Created input with {len(reconstruction_input.verified_elements)} verified elements"
    )

    # Step 2: Initialize engine
    print("🔧 Initializing reconstruction engine...")
    engine = DocumentReconstructionEngine()
    print("   ✅ Engine initialized")

    # Step 3: Run reconstruction
    print("⚙️  Running reconstruction pipeline...")
    try:
        result = engine.reconstruct_document(reconstruction_input)
        print(f"   ✅ Reconstruction completed in {result.processing_duration:.2f}s")

        # Step 4: Validate results
        print("🔍 Validating results...")

        # Check documents were generated
        print(f"   📄 Generated {len(result.documents)} document(s):")
        for format_type in result.documents:
            size = len(result.documents[format_type])
            print(f"      - {format_type.value}: {size:,} characters/bytes")

        # Check structure
        element_count = count_structure_elements(result.structure_tree)
        print(f"   🏗️  Document structure: {element_count} elements")

        # Check accessibility
        print(f"   ♿ Accessibility score: {result.accessibility_score:.1%}")
        passed_rules = sum(1 for passed in result.wcag_compliance.values() if passed)
        total_rules = len(result.wcag_compliance)
        print(f"   📋 WCAG compliance: {passed_rules}/{total_rules} rules passed")

        # Check quality metrics
        print(f"   📊 Reconstruction quality: {result.reconstruction_quality:.1%}")
        print(f"   📏 Structure accuracy: {result.structure_accuracy:.1%}")

        # Check navigation
        if result.navigation_tree:
            headings = len(result.navigation_tree.get("headings", []))
            landmarks = len(result.navigation_tree.get("landmarks", []))
            print(f"   🧭 Navigation: {headings} headings, {landmarks} landmarks")

        # Check warnings/errors
        if result.warnings:
            print(f"   ⚠️  {len(result.warnings)} warnings:")
            for warning in result.warnings[:3]:  # Show first 3
                print(f"      - {warning}")

        if result.errors:
            print(f"   ❌ {len(result.errors)} errors:")
            for error in result.errors:
                print(f"      - {error}")

        # Final verification status
        print(f"   ✅ Verification: {'PASSED' if result.verifier_passed else 'FAILED'}")
        if result.manual_review_required:
            print("   👁️  Manual review required")

        print("\n📈 Test Summary:")
        print(f"   Elements processed: {result.elements_processed}")
        print(f"   Processing time: {result.processing_duration:.2f}s")
        print(f"   Overall quality: {result.reconstruction_quality:.1%}")
        print(
            f"   Test status: {'✅ PASSED' if result.verifier_passed else '❌ FAILED'}"
        )

        # Write sample outputs for inspection
        if OutputFormat.HTML5 in result.documents:
            output_file = Path("test_output.html")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(result.documents[OutputFormat.HTML5])
            print(f"   💾 HTML5 output saved to: {output_file}")

        if OutputFormat.PDF_UA in result.documents:
            output_file = Path("test_output.pdf")
            with open(output_file, "wb") as f:
                f.write(result.documents[OutputFormat.PDF_UA])
            print(f"   💾 PDF/UA output saved to: {output_file}")

        return True

    except Exception as e:
        print(f"   ❌ Reconstruction failed: {e!s}")
        import traceback

        print("   🔍 Full error trace:")
        print(traceback.format_exc())
        return False


def count_structure_elements(structure):
    """Count total elements in document structure tree."""
    count = 1
    for child in structure.children:
        count += count_structure_elements(child)
    return count


if __name__ == "__main__":
    print("Aegis-A11y Reconstruction Pipeline Test")
    print("=" * 50)

    success = test_reconstruction_pipeline()

    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests completed successfully!")
        print("The reconstruction pipeline is working correctly.")
    else:
        print("💥 Tests failed!")
        print("Check the error messages above for debugging information.")

    print("\nNext steps:")
    print("1. Review generated output files (test_output.html, test_output.pdf)")
    print("2. Validate accessibility compliance manually")
    print("3. Test with additional document types and subjects")
