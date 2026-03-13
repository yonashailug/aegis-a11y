#!/usr/bin/env python3
"""
Isolated test for the reconstruction pipeline to verify it's working correctly.
This bypasses the API server and tests reconstruction components directly.
"""

import sys
import os
from pathlib import Path

# Add package directories to Python path
sys.path.insert(0, str(Path(__file__).parent / "packages/reconstruction/src"))
sys.path.insert(0, str(Path(__file__).parent / "packages/reasoning-agent/src"))

def test_reconstruction_pipeline():
    """Test the reconstruction pipeline with mock data to verify it works."""
    
    try:
        # Import reconstruction components
        from reconstruction import DocumentReconstructionEngine, ReconstructionInput, OutputFormat
        from reasoning_agent import ReasoningOutput, SubjectArea, ConfidenceLevel
        
        print("✅ All reconstruction imports successful")
        
        # Create mock verified reasoning output data
        mock_reasoning_outputs = [
            ReasoningOutput(
                element_id="test_element_1",
                detected_subject_area=SubjectArea.MATHEMATICS,
                subject_confidence=0.95,
                learning_objective="Understanding trigonometric functions",
                contextual_importance="Essential for understanding wave behavior and periodic functions",
                pedagogical_alt_text="Mathematics diagram showing the unit circle with trigonometric function relationships. The circle demonstrates how sine and cosine values relate to angles, with key angles marked at 0, π/2, π, and 3π/2 radians.",
                alt_text_rationale="Focuses on mathematical relationships and problem-solving context rather than visual appearance",
                pedagogical_quality_score=4.2,
                confidence_level=ConfidenceLevel.HIGH,
                processing_duration=2.1,
                prompt_template_used="mathematics_template",
                processing_warnings=[],
                fallback_used=False,
                raw_llm_response="This diagram illustrates the fundamental relationships in trigonometry using the unit circle..."
            ),
            ReasoningOutput(
                element_id="test_element_2", 
                detected_subject_area=SubjectArea.MATHEMATICS,
                subject_confidence=0.88,
                learning_objective="Understanding trigonometric identities",
                contextual_importance="Critical for advanced trigonometric problem solving",
                pedagogical_alt_text="Mathematics equation showing the Pythagorean identity: sin²(θ) + cos²(θ) = 1. This fundamental identity demonstrates the relationship between sine and cosine functions.",
                alt_text_rationale="Emphasizes mathematical relationships and problem-solving context",
                pedagogical_quality_score=4.5,
                confidence_level=ConfidenceLevel.HIGH,
                processing_duration=1.8,
                prompt_template_used="mathematics_template",
                processing_warnings=[],
                fallback_used=False,
                raw_llm_response="This equation represents one of the most important trigonometric identities..."
            )
        ]
        
        print(f"✅ Created {len(mock_reasoning_outputs)} mock reasoning outputs")
        
        # Create mock original layout data
        mock_original_layout = [
            {
                "element_id": "test_element_1",
                "classification": "functional_diagram",
                "bounding_box": [100, 100, 400, 400],
                "ocr_text": "Unit Circle θ sin cos",
                "html_tag": "<figure>",
                "page_number": 1
            },
            {
                "element_id": "test_element_2",
                "classification": "equation", 
                "bounding_box": [100, 450, 350, 500],
                "ocr_text": "sin²(θ) + cos²(θ) = 1",
                "html_tag": "<math>",
                "page_number": 1
            }
        ]
        
        print(f"✅ Created {len(mock_original_layout)} mock layout elements")
        
        # Initialize reconstruction engine
        reconstruction_engine = DocumentReconstructionEngine()
        print("✅ Reconstruction engine initialized")
        
        # Create reconstruction input
        reconstruction_input = ReconstructionInput(
            verified_elements=mock_reasoning_outputs,
            original_layout=mock_original_layout,
            document_title="Test Trigonometry Document",
            document_language="en",
            subject_area="mathematics",
            educational_level="high_school", 
            target_formats=[OutputFormat.HTML5],  # Test just HTML5 first
            preserve_layout=True,
            include_metadata=True,
            generate_navigation=True
        )
        
        print("✅ Reconstruction input created")
        
        # Test reconstruction
        print("🔄 Running document reconstruction...")
        reconstruction_result = reconstruction_engine.reconstruct_document(reconstruction_input)
        
        print("✅ Reconstruction completed successfully!")
        
        # Analyze results
        print(f"📊 Reconstruction Results:")
        print(f"   - Documents generated: {len(reconstruction_result.documents)}")
        print(f"   - Accessibility score: {reconstruction_result.accessibility_score:.2f}")
        print(f"   - Reconstruction quality: {reconstruction_result.reconstruction_quality:.2f}")
        print(f"   - WCAG compliance: {reconstruction_result.wcag_compliance}")
        print(f"   - Verifier passed: {reconstruction_result.verifier_passed}")
        print(f"   - Manual review required: {reconstruction_result.manual_review_required}")
        print(f"   - Processing duration: {reconstruction_result.processing_duration:.2f}s")
        
        # Check generated documents
        for format_type, document in reconstruction_result.documents.items():
            print(f"📄 Generated {format_type.value}:")
            if isinstance(document, str):
                print(f"   - Content length: {len(document)} characters")
                if len(document) > 200:
                    print(f"   - Preview: {document[:200]}...")
                else:
                    print(f"   - Content: {document}")
            else:
                print(f"   - Binary content length: {len(document)} bytes")
        
        # Test successful
        print("\n🎉 Reconstruction pipeline test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ Reconstruction pipeline test FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_reconstruction_pipeline()
    sys.exit(0 if success else 1)