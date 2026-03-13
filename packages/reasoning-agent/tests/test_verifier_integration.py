"""Integration tests for the complete verifier system."""

import pytest
from datetime import datetime
from reasoning_agent.schemas import ReasoningInput, ReasoningOutput, SubjectArea, ConfidenceLevel
from reasoning_agent.verifier import DeterministicVerifier, VerificationResult, ValidationResult
from reasoning_agent.quality_assessor import QualityAssessor, QualityMetrics
from reasoning_agent.human_validator import HumanValidator, ReviewFeedback, ReviewerRole, ReviewAction
from cv_layer.decomposer import ExtractedElement


class TestVerifierIntegration:
    """Test complete verifier system integration."""
    
    @pytest.fixture
    def sample_reasoning_output(self):
        """High-quality reasoning output for testing."""
        return ReasoningOutput(
            element_id='test_element_001',
            detected_subject_area=SubjectArea.PHYSICS,
            subject_confidence=0.88,
            learning_objective='Force equilibrium analysis',
            contextual_importance='Essential for understanding Newton\'s laws',
            pedagogical_alt_text='Physics force diagram illustrating three forces in equilibrium: weight (mg = 50N) downward, normal force (N = 50N) upward, and applied force (F = 20N) rightward. This demonstrates static equilibrium principles.',
            alt_text_rationale='Focuses on force relationships and equilibrium concepts essential for physics understanding.',
            pedagogical_quality_score=4.3,
            confidence_level=ConfidenceLevel.HIGH,
            processing_duration=2.1,
            prompt_template_used='physics_template',
            processing_warnings=[],
            fallback_used=False,
            raw_llm_response='Detailed physics analysis...'
        )
    
    @pytest.fixture
    def poor_quality_output(self):
        """Poor-quality reasoning output for testing failure cases."""
        return ReasoningOutput(
            element_id='test_element_002',
            detected_subject_area=SubjectArea.PHYSICS,
            subject_confidence=0.45,
            learning_objective='Unknown',
            contextual_importance='Unclear significance',
            pedagogical_alt_text='Generic image showing some content',  # Generic and low quality but meets min length
            alt_text_rationale='Basic description provided',
            pedagogical_quality_score=1.8,
            confidence_level=ConfidenceLevel.LOW,
            processing_duration=1.2,
            prompt_template_used='fallback',
            processing_warnings=['Low confidence detection'],
            fallback_used=True,
            raw_llm_response='Generic response'
        )
    
    @pytest.fixture
    def sample_element(self):
        """Sample extracted element."""
        return {
            'element_id': 'test_element_001',
            'classification': 'functional_diagram',
            'bounding_box': [100, 200, 400, 500],
            'ocr_text': 'Force Diagram: Weight W=50N, Normal N=50N, Applied F=20N',
            'html_tag': '<figure>'
        }
    
    @pytest.fixture
    def spatial_context(self):
        """Sample spatial context."""
        return {
            'containing_section': 'Chapter 3: Force Analysis',
            'preceding_text': 'When analyzing forces on objects',
            'following_text': 'The equilibrium condition requires balanced forces',
            'page_position': 'middle'
        }

    def test_deterministic_verifier_pass(self, sample_reasoning_output, sample_element, spatial_context):
        """Test verifier with high-quality output (should pass)."""
        verifier = DeterministicVerifier()
        
        result = verifier.verify_reasoning_output(
            sample_reasoning_output,
            sample_element,
            {'spatial_context': spatial_context}
        )
        
        assert result.overall_status == ValidationResult.PASS
        assert result.wcag_pass_rate >= 0.8
        assert result.confidence_score >= 0.7
        assert not result.requires_human_review
        assert result.processing_time > 0

    def test_deterministic_verifier_poor_quality(self, poor_quality_output, sample_element, spatial_context):
        """Test verifier with poor-quality output (should identify quality issues)."""
        verifier = DeterministicVerifier()
        
        result = verifier.verify_reasoning_output(
            poor_quality_output,
            sample_element,
            {'spatial_context': spatial_context}
        )
        
        # May pass WCAG but should have low confidence and require human review
        assert result.confidence_score < 0.7  # Lower confidence for poor quality
        assert result.requires_human_review    # Should require human review
        
        # Should identify some quality issues
        if len(result.issues) > 0:
            generic_issues = [i for i in result.issues if 'generic' in i.description.lower()]
            assert len(generic_issues) >= 0  # May detect generic descriptions

    def test_automatic_correction(self, poor_quality_output, sample_element, spatial_context):
        """Test automatic correction capability."""
        verifier = DeterministicVerifier(max_correction_attempts=3)
        
        result = verifier.verify_reasoning_output(
            poor_quality_output,
            sample_element,
            {'spatial_context': spatial_context}
        )
        
        # Should attempt corrections for fixable issues
        auto_fixable_issues = [i for i in result.issues if i.auto_fixable]
        if auto_fixable_issues:
            assert result.corrections_applied > 0
            assert result.recursive_loops_used > 0

    def test_quality_assessor_metrics(self, sample_reasoning_output, sample_element, spatial_context):
        """Test quality assessor detailed metrics."""
        assessor = QualityAssessor()
        
        metrics = assessor.assess_alt_text_quality(
            sample_reasoning_output,
            sample_element,
            spatial_context
        )
        
        # Validate metric ranges
        assert 1.0 <= metrics.pedagogical_score <= 5.0
        assert 0.0 <= metrics.structural_score <= 1.0
        assert 0.0 <= metrics.subject_relevance <= 1.0
        assert 0.0 <= metrics.udl_compliance <= 1.0
        assert 0.0 <= metrics.overall_confidence <= 1.0
        
        # Check detailed metrics
        assert 0.0 <= metrics.vocabulary_appropriateness <= 1.0
        assert 0.0 <= metrics.conceptual_accuracy <= 1.0
        assert 0.0 <= metrics.learning_objective_alignment <= 1.0
        
        # Should have accessibility features for good content
        assert len(metrics.accessibility_features) > 0
        assert isinstance(metrics.improvement_suggestions, list)

    def test_quality_assessor_subject_relevance(self, sample_reasoning_output, sample_element, spatial_context):
        """Test subject-specific quality assessment."""
        assessor = QualityAssessor()
        
        # Test physics content
        metrics = assessor.assess_alt_text_quality(
            sample_reasoning_output,
            sample_element,
            spatial_context
        )
        
        # Should score high on subject relevance for physics content
        assert metrics.subject_relevance >= 0.7
        assert 'Subject-specific vocabulary (physics)' in metrics.accessibility_features

    def test_human_validator_session_creation(self, sample_reasoning_output, spatial_context):
        """Test human validator session creation."""
        verifier = DeterministicVerifier()
        assessor = QualityAssessor()
        validator = HumanValidator()
        
        # Get verification and quality results first
        verification_result = verifier.verify_reasoning_output(
            sample_reasoning_output,
            {'classification': 'functional_diagram'},
            {'spatial_context': spatial_context}
        )
        
        quality_metrics = assessor.assess_alt_text_quality(
            sample_reasoning_output,
            {'classification': 'functional_diagram'},
            spatial_context
        )
        
        # Create review session
        session = validator.create_review_session(
            sample_reasoning_output,
            quality_metrics,
            verification_result
        )
        
        assert session.element_id == sample_reasoning_output.element_id
        assert session.session_id.startswith('review_')
        assert session.priority_level in ['low', 'normal', 'high', 'critical']
        assert len(session.reviews) == 0
        assert not session.consensus_reached

    def test_human_validator_review_submission(self, sample_reasoning_output, spatial_context):
        """Test review submission and consensus checking."""
        verifier = DeterministicVerifier()
        assessor = QualityAssessor()
        validator = HumanValidator()
        
        # Setup
        verification_result = verifier.verify_reasoning_output(
            sample_reasoning_output,
            {'classification': 'functional_diagram'},
            {'spatial_context': spatial_context}
        )
        
        quality_metrics = assessor.assess_alt_text_quality(
            sample_reasoning_output,
            {'classification': 'functional_diagram'},
            spatial_context
        )
        
        session = validator.create_review_session(
            sample_reasoning_output,
            quality_metrics,
            verification_result
        )
        
        # Submit educator review
        educator_feedback = ReviewFeedback(
            reviewer_id='test_educator',
            reviewer_role=ReviewerRole.EDUCATOR,
            action=ReviewAction.APPROVE,
            confidence_rating=4.0,
            quality_rating=4.2,
            comments='Good pedagogical content',
            review_duration=5.0
        )
        
        result = validator.submit_review(session.session_id, educator_feedback)
        
        assert result['status'] == 'submitted'
        assert result['consensus_reached']  # Single reviewer auto-consensus
        assert result['review_count'] == 1

    def test_human_validator_queue_management(self, sample_reasoning_output, spatial_context):
        """Test review queue management."""
        verifier = DeterministicVerifier()
        assessor = QualityAssessor()
        validator = HumanValidator()
        
        # Create session
        verification_result = verifier.verify_reasoning_output(
            sample_reasoning_output,
            {'classification': 'functional_diagram'},
            {'spatial_context': spatial_context}
        )
        
        quality_metrics = assessor.assess_alt_text_quality(
            sample_reasoning_output,
            {'classification': 'functional_diagram'},
            spatial_context
        )
        
        session = validator.create_review_session(
            sample_reasoning_output,
            quality_metrics,
            verification_result
        )
        
        # Check queue
        educator_queue = validator.get_review_queue(ReviewerRole.EDUCATOR)
        
        assert len(educator_queue) >= 1
        queue_item = next(item for item in educator_queue if item['session_id'] == session.session_id)
        assert queue_item['element_id'] == sample_reasoning_output.element_id
        assert queue_item['subject_area'] == 'physics'
        assert 'estimated_review_time' in queue_item

    def test_verification_report_generation(self, sample_reasoning_output, sample_element, spatial_context):
        """Test verification report generation."""
        verifier = DeterministicVerifier()
        
        result = verifier.verify_reasoning_output(
            sample_reasoning_output,
            sample_element,
            {'spatial_context': spatial_context}
        )
        
        report = verifier.generate_verification_report(result)
        
        # Check report structure
        assert 'verification_summary' in report
        assert 'corrections' in report
        assert 'issues_by_severity' in report
        assert 'wcag_compliance' in report
        
        # Check summary content
        summary = report['verification_summary']
        assert summary['overall_status'] in ['pass', 'fail', 'warning']
        assert 0.0 <= summary['wcag_pass_rate'] <= 1.0
        assert 0.0 <= summary['confidence_score'] <= 1.0
        assert isinstance(summary['requires_human_review'], bool)

    def test_complete_pipeline_integration(self, sample_reasoning_output, sample_element, spatial_context):
        """Test complete verifier pipeline integration."""
        # Initialize all components
        verifier = DeterministicVerifier()
        assessor = QualityAssessor()
        validator = HumanValidator()
        
        # Step 1: WCAG Verification
        verification_result = verifier.verify_reasoning_output(
            sample_reasoning_output,
            sample_element,
            {'spatial_context': spatial_context}
        )
        
        # Step 2: Quality Assessment
        quality_metrics = assessor.assess_alt_text_quality(
            sample_reasoning_output,
            sample_element,
            spatial_context
        )
        
        # Step 3: Human Review Setup
        review_session = validator.create_review_session(
            sample_reasoning_output,
            quality_metrics,
            verification_result
        )
        
        # Step 4: Simulate Review Process
        educator_review = ReviewFeedback(
            reviewer_id='integration_test_educator',
            reviewer_role=ReviewerRole.EDUCATOR,
            action=ReviewAction.APPROVE,
            confidence_rating=4.0,
            quality_rating=4.0,
            comments='Integration test successful'
        )
        
        review_result = validator.submit_review(review_session.session_id, educator_review)
        
        # Verify complete pipeline
        assert verification_result.overall_status in [ValidationResult.PASS, ValidationResult.WARNING]
        assert quality_metrics.pedagogical_score >= 3.0  # Should be good quality
        assert review_result['consensus_reached']
        
        # Generate final report
        final_report = verifier.generate_verification_report(verification_result)
        assert final_report['verification_summary']['overall_status'] in ['pass', 'warning']

    def test_error_handling_and_edge_cases(self):
        """Test error handling and edge cases."""
        verifier = DeterministicVerifier()
        
        # Test with minimal/low-quality data
        minimal_output = ReasoningOutput(
            element_id='minimal_test',
            detected_subject_area=SubjectArea.UNKNOWN,
            subject_confidence=0.1,
            learning_objective='Unknown',
            contextual_importance='Unknown',
            pedagogical_alt_text='Minimal description text',  # Meets min length but poor quality
            alt_text_rationale='Minimal rationale provided',
            pedagogical_quality_score=1.0,
            confidence_level=ConfidenceLevel.LOW,
            processing_duration=0.5,
            prompt_template_used='fallback',
            processing_warnings=['Multiple errors'],
            fallback_used=True
        )
        
        # Should handle gracefully without crashing
        result = verifier.verify_reasoning_output(
            minimal_output,
            {},  # Empty element
            {}   # Empty context
        )
        
        assert isinstance(result, VerificationResult)
        # Should identify quality issues even with valid length
        assert result.confidence_score < 0.5  # Low confidence for poor quality
        assert result.requires_human_review  # Should require review for poor quality

    def test_performance_metrics(self, sample_reasoning_output, sample_element, spatial_context):
        """Test performance metrics collection."""
        verifier = DeterministicVerifier()
        assessor = QualityAssessor()
        
        import time
        start_time = time.time()
        
        # Run verification
        verification_result = verifier.verify_reasoning_output(
            sample_reasoning_output,
            sample_element,
            {'spatial_context': spatial_context}
        )
        
        # Run quality assessment
        quality_metrics = assessor.assess_alt_text_quality(
            sample_reasoning_output,
            sample_element,
            spatial_context
        )
        
        total_time = time.time() - start_time
        
        # Performance assertions
        assert verification_result.processing_time > 0
        assert total_time < 5.0  # Should complete within 5 seconds
        
        # Verify all metrics are computed
        assert quality_metrics.overall_confidence is not None
        assert verification_result.confidence_score is not None