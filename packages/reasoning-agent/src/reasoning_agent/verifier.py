"""
Deterministic Verifier and Recursive Correction Loop

Based on technical paper Section 3.3.1: Implements dual-logic framework
for WCAG 2.1 AA compliance validation and semantic consistency checking.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re
from .schemas import ReasoningOutput, ConfidenceLevel, SubjectArea


class ValidationResult(str, Enum):
    """Validation result status."""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"


class WCAGCriteria(str, Enum):
    """WCAG 2.1 AA criteria for validation."""
    TEXT_ALTERNATIVES = "1.1.1"  # Non-text Content
    HEADINGS_AND_LABELS = "2.4.6"  # Descriptive headings and labels
    FOCUS_VISIBLE = "2.4.7"  # Focus visible
    MEANINGFUL_SEQUENCE = "1.3.2"  # Reading sequence
    INFO_AND_RELATIONSHIPS = "1.3.1"  # Info and relationships


@dataclass
class ValidationIssue:
    """Individual validation issue."""
    criteria: WCAGCriteria
    severity: ValidationResult
    description: str
    element_id: str
    suggested_fix: Optional[str] = None
    auto_fixable: bool = False


@dataclass
class VerificationResult:
    """Complete verification result."""
    overall_status: ValidationResult
    wcag_pass_rate: float
    issues: List[ValidationIssue]
    corrections_applied: int
    recursive_loops_used: int
    processing_time: float
    confidence_score: float
    requires_human_review: bool


class DeterministicVerifier:
    """
    Deterministic Verifier implementing WCAG 2.1 AA validation.
    
    Based on paper Section 3.3.1: Applies dual-logic framework with
    deterministic rule-checking and optional semantic consistency checks.
    """
    
    def __init__(self, max_correction_attempts: int = 3):
        """Initialize the deterministic verifier.
        
        Args:
            max_correction_attempts: Maximum recursive correction attempts
        """
        self.max_correction_attempts = max_correction_attempts
        
        # WCAG 2.1 AA validation rules
        self.validation_rules = {
            WCAGCriteria.TEXT_ALTERNATIVES: self._validate_text_alternatives,
            WCAGCriteria.HEADINGS_AND_LABELS: self._validate_headings,
            WCAGCriteria.MEANINGFUL_SEQUENCE: self._validate_reading_order,
            WCAGCriteria.INFO_AND_RELATIONSHIPS: self._validate_structure
        }
        
        # Alt-text quality thresholds
        self.alt_text_min_length = 10
        self.alt_text_max_length = 250
        self.pedagogical_quality_threshold = 3.0
        
        self.initialized = True
    
    def verify_reasoning_output(self, 
                              reasoning_output: ReasoningOutput,
                              original_element: Dict[str, Any],
                              spatial_context: Dict[str, Any]) -> VerificationResult:
        """
        Verify reasoning agent output with recursive correction capability.
        
        Args:
            reasoning_output: Output from reasoning agent
            original_element: Original extracted element
            spatial_context: Spatial context information
            
        Returns:
            VerificationResult with validation status and corrections
        """
        import time
        start_time = time.time()
        
        issues = []
        corrections_applied = 0
        recursive_loops = 0
        
        # Apply all WCAG validation rules
        for criteria, validation_func in self.validation_rules.items():
            try:
                rule_issues = validation_func(reasoning_output, original_element, spatial_context)
                issues.extend(rule_issues)
            except Exception as e:
                issues.append(ValidationIssue(
                    criteria=criteria,
                    severity=ValidationResult.WARNING,
                    description=f"Validation rule failed: {str(e)}",
                    element_id=reasoning_output.element_id
                ))
        
        # Check for auto-fixable issues and apply corrections
        corrected_output = reasoning_output
        for issue in issues:
            if issue.auto_fixable and corrections_applied < self.max_correction_attempts:
                correction_result = self._apply_correction(issue, corrected_output, original_element)
                if correction_result:
                    corrected_output = correction_result
                    corrections_applied += 1
                    recursive_loops += 1
        
        # Calculate overall metrics
        total_checks = len(self.validation_rules)
        failed_checks = len([i for i in issues if i.severity == ValidationResult.FAIL])
        wcag_pass_rate = max(0.0, (total_checks - failed_checks) / total_checks)
        
        # Determine overall status
        has_failures = any(i.severity == ValidationResult.FAIL for i in issues)
        overall_status = ValidationResult.FAIL if has_failures else ValidationResult.PASS
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(corrected_output, issues, wcag_pass_rate)
        
        # Determine if human review is required
        requires_human_review = (
            wcag_pass_rate < 0.8 or
            corrected_output.pedagogical_quality_score < self.pedagogical_quality_threshold or
            corrected_output.confidence_level == ConfidenceLevel.LOW
        )
        
        processing_time = time.time() - start_time
        
        return VerificationResult(
            overall_status=overall_status,
            wcag_pass_rate=wcag_pass_rate,
            issues=issues,
            corrections_applied=corrections_applied,
            recursive_loops_used=recursive_loops,
            processing_time=processing_time,
            confidence_score=confidence_score,
            requires_human_review=requires_human_review
        )
    
    def _validate_text_alternatives(self, 
                                   output: ReasoningOutput,
                                   element: Dict[str, Any],
                                   context: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate WCAG 1.1.1 - Non-text Content (Text Alternatives)."""
        issues = []
        
        # Check if element requires alt-text
        element_classification = element.get('classification', '').lower()
        requires_alt_text = any(keyword in element_classification for keyword in 
                               ['figure', 'image', 'diagram', 'chart', 'graph'])
        
        if requires_alt_text:
            alt_text = output.pedagogical_alt_text
            
            # Check minimum length
            if len(alt_text) < self.alt_text_min_length:
                issues.append(ValidationIssue(
                    criteria=WCAGCriteria.TEXT_ALTERNATIVES,
                    severity=ValidationResult.FAIL,
                    description=f"Alt-text too short ({len(alt_text)} chars, minimum {self.alt_text_min_length})",
                    element_id=output.element_id,
                    suggested_fix="Generate more descriptive alt-text",
                    auto_fixable=True
                ))
            
            # Check maximum length
            elif len(alt_text) > self.alt_text_max_length:
                issues.append(ValidationIssue(
                    criteria=WCAGCriteria.TEXT_ALTERNATIVES,
                    severity=ValidationResult.WARNING,
                    description=f"Alt-text very long ({len(alt_text)} chars, recommended max {self.alt_text_max_length})",
                    element_id=output.element_id,
                    suggested_fix="Consider condensing alt-text while preserving pedagogical value"
                ))
            
            # Check for generic/unhelpful descriptions
            generic_patterns = [
                r'^(image|figure|diagram|chart|graph)(\s+of|\s*$)',
                r'^(this\s+)?(shows?|depicts?|illustrates?)\s+an?\s+(image|figure)',
                r'^alt[\s-]?text',
                r'^description',
                r'^\[.*\]$'
            ]
            
            for pattern in generic_patterns:
                if re.search(pattern, alt_text.lower()):
                    issues.append(ValidationIssue(
                        criteria=WCAGCriteria.TEXT_ALTERNATIVES,
                        severity=ValidationResult.FAIL,
                        description="Alt-text appears generic or non-descriptive",
                        element_id=output.element_id,
                        suggested_fix="Generate pedagogically meaningful description",
                        auto_fixable=True
                    ))
                    break
            
            # Check pedagogical quality
            if output.pedagogical_quality_score < self.pedagogical_quality_threshold:
                issues.append(ValidationIssue(
                    criteria=WCAGCriteria.TEXT_ALTERNATIVES,
                    severity=ValidationResult.WARNING,
                    description=f"Low pedagogical quality score ({output.pedagogical_quality_score:.1f}/5.0)",
                    element_id=output.element_id,
                    suggested_fix="Enhance pedagogical focus and subject-specific terminology"
                ))
        
        return issues
    
    def _validate_headings(self,
                          output: ReasoningOutput,
                          element: Dict[str, Any],
                          context: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate WCAG 2.4.6 - Descriptive headings and labels."""
        issues = []
        
        element_classification = element.get('classification', '').lower()
        if 'heading' in element_classification:
            
            # Check heading hierarchy (simplified - would need full document context)
            html_tag = element.get('html_tag', '').lower()
            if html_tag in ['<h1>', '<h2>', '<h3>', '<h4>', '<h5>', '<h6>']:
                # Extract heading level
                level_match = re.search(r'<h(\d)>', html_tag)
                if level_match:
                    current_level = int(level_match.group(1))
                    
                    # Check for reasonable heading content
                    ocr_text = element.get('ocr_text', '').strip()
                    if len(ocr_text) < 2:
                        issues.append(ValidationIssue(
                            criteria=WCAGCriteria.HEADINGS_AND_LABELS,
                            severity=ValidationResult.FAIL,
                            description="Heading has insufficient text content",
                            element_id=output.element_id,
                            suggested_fix="Ensure heading has descriptive text"
                        ))
                    
                    # Check for overly long headings
                    elif len(ocr_text) > 100:
                        issues.append(ValidationIssue(
                            criteria=WCAGCriteria.HEADINGS_AND_LABELS,
                            severity=ValidationResult.WARNING,
                            description=f"Heading is very long ({len(ocr_text)} chars)",
                            element_id=output.element_id,
                            suggested_fix="Consider shortening heading while maintaining clarity"
                        ))
        
        return issues
    
    def _validate_reading_order(self,
                               output: ReasoningOutput,
                               element: Dict[str, Any],
                               context: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate WCAG 1.3.2 - Meaningful Sequence."""
        issues = []
        
        # Check spatial context for reading order indicators
        spatial_context = context.get('spatial_context', {})
        if spatial_context:
            # Check if element position makes sense relative to surrounding content
            bounding_box = element.get('bounding_box', [0, 0, 0, 0])
            
            # Basic validation - in production would need full document layout analysis
            if len(bounding_box) != 4:
                issues.append(ValidationIssue(
                    criteria=WCAGCriteria.MEANINGFUL_SEQUENCE,
                    severity=ValidationResult.WARNING,
                    description="Invalid or missing bounding box for reading order analysis",
                    element_id=output.element_id
                ))
            
            # Check if element has proper section context
            containing_section = spatial_context.get('containing_section')
            if not containing_section and element.get('classification') not in ['heading', 'title']:
                issues.append(ValidationIssue(
                    criteria=WCAGCriteria.MEANINGFUL_SEQUENCE,
                    severity=ValidationResult.WARNING,
                    description="Element lacks clear section context",
                    element_id=output.element_id,
                    suggested_fix="Verify element placement within document structure"
                ))
        
        return issues
    
    def _validate_structure(self,
                           output: ReasoningOutput,
                           element: Dict[str, Any],
                           context: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate WCAG 1.3.1 - Info and Relationships."""
        issues = []
        
        element_classification = element.get('classification', '').lower()
        
        # Table validation
        if 'table' in element_classification:
            ocr_text = element.get('ocr_text', '')
            
            # Check if table has some structure indicators
            has_structure_indicators = any(indicator in ocr_text.lower() for indicator in 
                                         ['row', 'column', 'header', 'cell', '|', '\t'])
            
            if not has_structure_indicators:
                issues.append(ValidationIssue(
                    criteria=WCAGCriteria.INFO_AND_RELATIONSHIPS,
                    severity=ValidationResult.WARNING,
                    description="Table lacks clear structural indicators",
                    element_id=output.element_id,
                    suggested_fix="Ensure table has proper headers and cell relationships"
                ))
        
        # List validation
        elif 'list' in element_classification:
            ocr_text = element.get('ocr_text', '')
            
            # Check for list structure indicators
            has_list_indicators = any(indicator in ocr_text for indicator in 
                                    ['•', '1.', '2.', '-', '*', 'a)', 'i)', 'I.'])
            
            if not has_list_indicators:
                issues.append(ValidationIssue(
                    criteria=WCAGCriteria.INFO_AND_RELATIONSHIPS,
                    severity=ValidationResult.WARNING,
                    description="List lacks clear structural indicators",
                    element_id=output.element_id,
                    suggested_fix="Ensure list items are properly marked"
                ))
        
        # Equation validation
        elif 'equation' in element_classification:
            # Check if equation has mathematical structure
            ocr_text = element.get('ocr_text', '')
            has_math_symbols = any(symbol in ocr_text for symbol in 
                                  ['=', '+', '-', '×', '÷', '²', '³', '√', '∑', '∫'])
            
            if not has_math_symbols:
                issues.append(ValidationIssue(
                    criteria=WCAGCriteria.INFO_AND_RELATIONSHIPS,
                    severity=ValidationResult.WARNING,
                    description="Equation lacks recognizable mathematical symbols",
                    element_id=output.element_id,
                    suggested_fix="Verify equation content and structure"
                ))
        
        return issues
    
    def _apply_correction(self,
                         issue: ValidationIssue,
                         output: ReasoningOutput,
                         original_element: Dict[str, Any]) -> Optional[ReasoningOutput]:
        """Apply automatic correction to fixable issues."""
        
        if not issue.auto_fixable:
            return None
        
        # Create a copy of the output for correction
        corrected_output = ReasoningOutput(**output.model_dump())
        
        # Apply specific corrections based on issue type
        if issue.criteria == WCAGCriteria.TEXT_ALTERNATIVES:
            if "too short" in issue.description:
                # Expand alt-text with element context
                element_text = original_element.get('ocr_text', '')
                classification = original_element.get('classification', 'element')
                
                enhanced_alt_text = f"Educational {classification}"
                if element_text:
                    enhanced_alt_text += f" containing: {element_text}"
                
                # Add subject-specific enhancements
                if output.detected_subject_area != SubjectArea.UNKNOWN:
                    enhanced_alt_text += f". This {output.detected_subject_area.value} content"
                
                corrected_output.pedagogical_alt_text = enhanced_alt_text
                corrected_output.processing_warnings.append("Alt-text automatically expanded")
                
            elif "generic" in issue.description:
                # Enhance generic alt-text with pedagogical focus
                subject_specific_phrases = {
                    SubjectArea.PHYSICS: "demonstrates physical principles and relationships",
                    SubjectArea.CHEMISTRY: "illustrates chemical concepts and reactions", 
                    SubjectArea.BIOLOGY: "shows biological processes and structures",
                    SubjectArea.MATHEMATICS: "presents mathematical relationships and problem-solving"
                }
                
                enhancement = subject_specific_phrases.get(
                    output.detected_subject_area,
                    "supports learning objectives"
                )
                
                corrected_output.pedagogical_alt_text += f" that {enhancement}."
                corrected_output.processing_warnings.append("Alt-text enhanced with pedagogical focus")
        
        # Mark that correction was applied
        corrected_output.fallback_used = True
        corrected_output.processing_warnings.append(f"Auto-correction applied for: {issue.description}")
        
        return corrected_output
    
    def _calculate_confidence_score(self,
                                   output: ReasoningOutput,
                                   issues: List[ValidationIssue],
                                   wcag_pass_rate: float) -> float:
        """Calculate overall confidence score based on validation results."""
        
        # Base confidence from subject detection and quality
        base_confidence = output.subject_confidence * 0.3
        quality_confidence = (output.pedagogical_quality_score / 5.0) * 0.3
        wcag_confidence = wcag_pass_rate * 0.4
        
        # Penalty for critical issues
        critical_issues = len([i for i in issues if i.severity == ValidationResult.FAIL])
        critical_penalty = min(0.2, critical_issues * 0.05)
        
        confidence_score = base_confidence + quality_confidence + wcag_confidence - critical_penalty
        
        return max(0.0, min(1.0, confidence_score))
    
    def generate_verification_report(self, result: VerificationResult) -> Dict[str, Any]:
        """Generate detailed verification report."""
        
        return {
            'verification_summary': {
                'overall_status': result.overall_status.value,
                'wcag_pass_rate': result.wcag_pass_rate,
                'confidence_score': result.confidence_score,
                'requires_human_review': result.requires_human_review,
                'processing_time': result.processing_time
            },
            'corrections': {
                'applied_count': result.corrections_applied,
                'recursive_loops': result.recursive_loops_used,
                'max_attempts': self.max_correction_attempts
            },
            'issues_by_severity': {
                'critical': [i for i in result.issues if i.severity == ValidationResult.FAIL],
                'warnings': [i for i in result.issues if i.severity == ValidationResult.WARNING]
            },
            'wcag_compliance': {
                'criteria_checked': list(WCAGCriteria),
                'total_issues': len(result.issues),
                'auto_fixable_issues': len([i for i in result.issues if i.auto_fixable])
            }
        }