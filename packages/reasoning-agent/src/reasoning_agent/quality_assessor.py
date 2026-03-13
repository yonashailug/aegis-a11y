"""
Quality Assessment Algorithms for Pedagogical Alt-Text

Based on technical paper Section 4.6.1: Implements 1-5 ordinal rubric
for Alt-Text quality assessment and pedagogical alignment scoring.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import re
from .schemas import ReasoningOutput, SubjectArea, ConfidenceLevel


class QualityDimension(str, Enum):
    """Quality assessment dimensions from paper Table 1."""
    STRUCTURAL_CORRECTNESS = "structural_correctness"
    READING_ORDER = "reading_order"  
    ALT_TEXT_QUALITY = "alt_text_quality"
    WCAG_CHECKLIST = "wcag_checklist"


class PedagogicalLevel(int, Enum):
    """Pedagogical quality levels (1-5 scale from paper)."""
    FAILURE = 1  # "functional failure or hallucination"
    BASIC = 2    # "minimal educational value"
    ADEQUATE = 3  # "meets basic requirements"
    GOOD = 4     # "strong pedagogical focus"
    EXPERT = 5   # "expert-level pedagogical alignment"


@dataclass
class QualityMetrics:
    """Comprehensive quality metrics for alt-text assessment."""
    pedagogical_score: float  # 1-5 scale
    structural_score: float   # 0-1 F1 score
    readability_score: float  # 0-1 scale
    subject_relevance: float  # 0-1 scale
    udl_compliance: float     # 0-1 scale
    overall_confidence: float # 0-1 scale
    
    # Detailed breakdowns
    vocabulary_appropriateness: float
    conceptual_accuracy: float
    learning_objective_alignment: float
    accessibility_features: List[str]
    improvement_suggestions: List[str]


class QualityAssessor:
    """
    Quality assessment for pedagogical alt-text based on paper evaluation protocol.
    
    Implements the 1-5 ordinal rubric from Section 4.6.1 and provides detailed
    quality metrics for human review and system optimization.
    """
    
    def __init__(self):
        """Initialize quality assessment algorithms."""
        
        # Subject-specific pedagogical vocabulary
        self.subject_vocabularies = {
            SubjectArea.PHYSICS: {
                'concepts': ['force', 'velocity', 'acceleration', 'energy', 'momentum', 'gravity', 
                           'friction', 'wave', 'frequency', 'amplitude', 'motion', 'dynamics'],
                'relationships': ['proportional', 'inverse', 'equilibrium', 'conservation', 
                               'transformation', 'interaction', 'correlation', 'causation'],
                'measurements': ['newton', 'joule', 'meter', 'second', 'hertz', 'watt', 'pascal']
            },
            SubjectArea.CHEMISTRY: {
                'concepts': ['molecule', 'atom', 'bond', 'reaction', 'element', 'compound',
                           'ion', 'electron', 'proton', 'neutron', 'orbital', 'catalyst'],
                'relationships': ['synthesis', 'decomposition', 'oxidation', 'reduction',
                               'equilibrium', 'stoichiometry', 'concentration', 'solubility'],
                'measurements': ['mole', 'molarity', 'ph', 'gram', 'liter', 'temperature']
            },
            SubjectArea.BIOLOGY: {
                'concepts': ['cell', 'organism', 'evolution', 'genetics', 'protein', 'dna',
                           'enzyme', 'photosynthesis', 'respiration', 'mitosis', 'ecosystem'],
                'relationships': ['adaptation', 'mutation', 'inheritance', 'selection',
                               'symbiosis', 'predation', 'competition', 'diversity'],
                'measurements': ['population', 'generation', 'species', 'habitat', 'biomass']
            },
            SubjectArea.MATHEMATICS: {
                'concepts': ['function', 'equation', 'variable', 'coefficient', 'theorem',
                           'proof', 'derivative', 'integral', 'matrix', 'vector', 'limit'],
                'relationships': ['proportional', 'linear', 'quadratic', 'exponential',
                               'logarithmic', 'periodic', 'asymptotic', 'convergent'],
                'measurements': ['degree', 'slope', 'intercept', 'domain', 'range', 'maximum']
            }
        }
        
        # UDL guidelines indicators
        self.udl_indicators = {
            'multiple_representations': ['shows', 'illustrates', 'demonstrates', 'depicts', 'represents'],
            'clear_language': ['explains', 'describes', 'clarifies', 'defines', 'outlines'],
            'contextual_support': ['section', 'chapter', 'lesson', 'objective', 'purpose'],
            'engagement': ['important', 'essential', 'fundamental', 'key', 'critical'],
            'comprehension': ['relationship', 'pattern', 'concept', 'principle', 'process']
        }
        
        # Quality assessment weights
        self.assessment_weights = {
            'pedagogical_focus': 0.3,
            'subject_vocabulary': 0.25,
            'learning_alignment': 0.2,
            'clarity': 0.15,
            'accessibility': 0.1
        }
        
        self.initialized = True
    
    def assess_alt_text_quality(self,
                               output: ReasoningOutput,
                               original_element: Dict[str, Any],
                               spatial_context: Dict[str, Any]) -> QualityMetrics:
        """
        Comprehensive quality assessment of pedagogical alt-text.
        
        Args:
            output: ReasoningOutput from reasoning agent
            original_element: Original extracted element
            spatial_context: Spatial context information
            
        Returns:
            QualityMetrics with detailed assessment scores
        """
        alt_text = output.pedagogical_alt_text.lower()
        subject_area = output.detected_subject_area
        
        # Core quality assessments
        pedagogical_score = self._assess_pedagogical_alignment(alt_text, subject_area, spatial_context)
        structural_score = self._assess_structural_correctness(output, original_element)
        readability_score = self._assess_readability(alt_text)
        subject_relevance = self._assess_subject_relevance(alt_text, subject_area)
        udl_compliance = self._assess_udl_compliance(alt_text, output)
        
        # Detailed sub-assessments
        vocab_score = self._assess_vocabulary_appropriateness(alt_text, subject_area)
        conceptual_score = self._assess_conceptual_accuracy(alt_text, subject_area, original_element)
        learning_alignment = self._assess_learning_objective_alignment(alt_text, spatial_context)
        
        # Accessibility features
        accessibility_features = self._identify_accessibility_features(alt_text, output)
        
        # Improvement suggestions
        suggestions = self._generate_improvement_suggestions(
            alt_text, subject_area, pedagogical_score, vocab_score
        )
        
        # Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(
            pedagogical_score, structural_score, subject_relevance, output.subject_confidence
        )
        
        return QualityMetrics(
            pedagogical_score=pedagogical_score,
            structural_score=structural_score,
            readability_score=readability_score,
            subject_relevance=subject_relevance,
            udl_compliance=udl_compliance,
            overall_confidence=overall_confidence,
            vocabulary_appropriateness=vocab_score,
            conceptual_accuracy=conceptual_score,
            learning_objective_alignment=learning_alignment,
            accessibility_features=accessibility_features,
            improvement_suggestions=suggestions
        )
    
    def _assess_pedagogical_alignment(self,
                                    alt_text: str,
                                    subject_area: SubjectArea,
                                    spatial_context: Dict[str, Any]) -> float:
        """Assess pedagogical alignment using 1-5 scale from paper."""
        
        score = 3.0  # Start with baseline "adequate"
        
        # Check for pedagogical language indicators
        pedagogical_indicators = [
            'demonstrates', 'illustrates', 'shows', 'explains', 'represents',
            'depicts', 'reveals', 'clarifies', 'exemplifies', 'highlights'
        ]
        
        has_pedagogical_language = any(indicator in alt_text for indicator in pedagogical_indicators)
        if has_pedagogical_language:
            score += 0.5
        
        # Check for learning-focused descriptions
        learning_indicators = [
            'learning', 'understanding', 'concept', 'principle', 'objective',
            'essential', 'fundamental', 'key', 'important', 'critical'
        ]
        
        learning_focus = sum(1 for indicator in learning_indicators if indicator in alt_text)
        score += min(0.5, learning_focus * 0.1)
        
        # Check for contextual integration
        if spatial_context.get('containing_section'):
            section_context = spatial_context['containing_section'].lower()
            if any(word in alt_text for word in section_context.split()[:3]):
                score += 0.3  # Bonus for section integration
        
        # Check for subject-specific pedagogical patterns
        if subject_area in self.subject_vocabularies:
            vocab = self.subject_vocabularies[subject_area]
            concept_mentions = sum(1 for concept in vocab['concepts'] if concept in alt_text)
            relationship_mentions = sum(1 for rel in vocab['relationships'] if rel in alt_text)
            
            if concept_mentions > 0:
                score += min(0.4, concept_mentions * 0.1)
            if relationship_mentions > 0:
                score += min(0.3, relationship_mentions * 0.15)
        
        # Penalty for generic descriptions
        generic_patterns = [
            'image of', 'picture of', 'shows an image', 'figure containing',
            'graphic with', 'visual representation'
        ]
        
        if any(pattern in alt_text for pattern in generic_patterns):
            score -= 0.5
        
        return max(1.0, min(5.0, score))
    
    def _assess_structural_correctness(self,
                                     output: ReasoningOutput,
                                     original_element: Dict[str, Any]) -> float:
        """Assess structural correctness (element-level accuracy)."""
        
        score = 1.0
        
        # Check element classification accuracy
        detected_class = original_element.get('classification', '').lower()
        alt_text = output.pedagogical_alt_text.lower()
        
        # Expected terms for different element types
        classification_terms = {
            'figure': ['diagram', 'figure', 'illustration', 'chart', 'graph'],
            'equation': ['equation', 'formula', 'expression', 'calculation'],
            'table': ['table', 'data', 'row', 'column', 'cell'],
            'heading': ['heading', 'title', 'section', 'chapter']
        }
        
        expected_terms = []
        for class_type, terms in classification_terms.items():
            if class_type in detected_class:
                expected_terms.extend(terms)
        
        if expected_terms and any(term in alt_text for term in expected_terms):
            score *= 1.0  # Maintain full score for correct classification
        elif expected_terms:
            score *= 0.7  # Reduce score for classification mismatch
        
        # Check for appropriate detail level
        element_text = original_element.get('ocr_text', '')
        if element_text:
            # Good balance: alt-text should be longer but not excessively so
            text_ratio = len(output.pedagogical_alt_text) / len(element_text)
            if 1.5 <= text_ratio <= 5.0:
                score *= 1.0
            elif text_ratio < 1.0:
                score *= 0.8  # Too brief
            elif text_ratio > 10.0:
                score *= 0.9  # Potentially verbose
        
        return score
    
    def _assess_readability(self, alt_text: str) -> float:
        """Assess readability and clarity of alt-text."""
        
        # Basic readability metrics
        sentences = alt_text.split('.')
        words = alt_text.split()
        
        if len(sentences) == 0 or len(words) == 0:
            return 0.1
        
        avg_sentence_length = len(words) / max(1, len(sentences))
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Optimal ranges for educational content
        readability_score = 1.0
        
        # Sentence length (10-20 words optimal for accessibility)
        if 10 <= avg_sentence_length <= 20:
            readability_score *= 1.0
        elif avg_sentence_length < 5:
            readability_score *= 0.8  # Too choppy
        elif avg_sentence_length > 25:
            readability_score *= 0.7  # Too complex
        
        # Word complexity (balance academic vocabulary with accessibility)
        if 4 <= avg_word_length <= 7:
            readability_score *= 1.0
        elif avg_word_length > 9:
            readability_score *= 0.8  # May be too complex
        
        # Check for clear structure indicators
        structure_indicators = [',', ';', ':', 'and', 'which', 'that', 'showing', 'with']
        structure_score = sum(1 for indicator in structure_indicators if indicator in alt_text)
        
        if structure_score >= 2:
            readability_score *= 1.1  # Bonus for structured writing
        
        return min(1.0, readability_score)
    
    def _assess_subject_relevance(self, alt_text: str, subject_area: SubjectArea) -> float:
        """Assess relevance to detected subject area."""
        
        if subject_area == SubjectArea.UNKNOWN or subject_area not in self.subject_vocabularies:
            return 0.5  # Neutral score for unknown subjects
        
        vocab = self.subject_vocabularies[subject_area]
        total_terms = len(vocab['concepts']) + len(vocab['relationships']) + len(vocab['measurements'])
        
        # Count subject-specific terms
        concept_matches = sum(1 for term in vocab['concepts'] if term in alt_text)
        relationship_matches = sum(1 for term in vocab['relationships'] if term in alt_text)
        measurement_matches = sum(1 for term in vocab['measurements'] if term in alt_text)
        
        total_matches = concept_matches + relationship_matches + measurement_matches
        
        # Calculate relevance score
        relevance_score = min(1.0, (total_matches / max(1, total_terms * 0.1)))
        
        # Bonus for balanced vocabulary usage
        if concept_matches > 0 and relationship_matches > 0:
            relevance_score *= 1.1
        
        return min(1.0, relevance_score)
    
    def _assess_udl_compliance(self, alt_text: str, output: ReasoningOutput) -> float:
        """Assess Universal Design for Learning compliance."""
        
        udl_score = 0.0
        total_guidelines = len(self.udl_indicators)
        
        for guideline, indicators in self.udl_indicators.items():
            if any(indicator in alt_text for indicator in indicators):
                udl_score += 1.0
        
        # Normalize to 0-1 scale
        base_score = udl_score / total_guidelines
        
        # Bonus for UDL guidelines explicitly applied
        if hasattr(output, 'udl_guidelines_applied') and len(output.udl_guidelines_applied) > 0:
            base_score *= 1.2
        
        return min(1.0, base_score)
    
    def _assess_vocabulary_appropriateness(self, alt_text: str, subject_area: SubjectArea) -> float:
        """Assess appropriateness of vocabulary for educational level."""
        
        # Check for appropriate academic vocabulary
        academic_indicators = [
            'demonstrates', 'illustrates', 'represents', 'indicates', 'reveals',
            'suggests', 'implies', 'exemplifies', 'characterizes'
        ]
        
        has_academic_vocab = any(indicator in alt_text for indicator in academic_indicators)
        
        # Check for overly complex vocabulary
        complex_words = [word for word in alt_text.split() if len(word) > 12]
        complexity_ratio = len(complex_words) / len(alt_text.split())
        
        vocab_score = 0.8 if has_academic_vocab else 0.6
        
        # Penalty for excessive complexity
        if complexity_ratio > 0.3:
            vocab_score *= 0.7
        
        return vocab_score
    
    def _assess_conceptual_accuracy(self,
                                  alt_text: str,
                                  subject_area: SubjectArea,
                                  original_element: Dict[str, Any]) -> float:
        """Assess conceptual accuracy relative to element content."""
        
        element_text = original_element.get('ocr_text', '').lower()
        
        if not element_text:
            return 0.7  # Neutral score when no reference text available
        
        # Check for consistency between OCR and alt-text
        element_words = set(element_text.split())
        alt_words = set(alt_text.split())
        
        # Find overlap in key terms
        common_words = element_words.intersection(alt_words)
        overlap_ratio = len(common_words) / max(1, len(element_words))
        
        # Good overlap indicates conceptual consistency
        if overlap_ratio >= 0.3:
            accuracy_score = 0.9
        elif overlap_ratio >= 0.1:
            accuracy_score = 0.7
        else:
            accuracy_score = 0.5
        
        return accuracy_score
    
    def _assess_learning_objective_alignment(self,
                                           alt_text: str,
                                           spatial_context: Dict[str, Any]) -> float:
        """Assess alignment with learning objectives from context."""
        
        # Check for learning-focused language
        learning_terms = [
            'understand', 'learn', 'analyze', 'compare', 'evaluate',
            'identify', 'explain', 'describe', 'demonstrate', 'apply'
        ]
        
        learning_alignment = sum(1 for term in learning_terms if term in alt_text)
        
        # Check for context integration
        section = spatial_context.get('containing_section', '').lower()
        if section and any(word in alt_text for word in section.split()[:4]):
            learning_alignment += 1
        
        # Normalize score
        return min(1.0, learning_alignment * 0.3)
    
    def _identify_accessibility_features(self,
                                       alt_text: str,
                                       output: ReasoningOutput) -> List[str]:
        """Identify accessibility features present in the alt-text."""
        
        features = []
        
        # Check for descriptive language
        if any(word in alt_text for word in ['shows', 'depicts', 'illustrates']):
            features.append("Descriptive language")
        
        # Check for structural information
        if any(word in alt_text for word in ['section', 'chapter', 'part']):
            features.append("Contextual positioning")
        
        # Check for learning focus
        if any(word in alt_text for word in ['demonstrates', 'explains', 'concept']):
            features.append("Pedagogical focus")
        
        # Check for subject-specific terminology
        if output.detected_subject_area != SubjectArea.UNKNOWN:
            features.append(f"Subject-specific vocabulary ({output.detected_subject_area.value})")
        
        # Check for appropriate length
        if 50 <= len(alt_text) <= 200:
            features.append("Appropriate detail level")
        
        return features
    
    def _generate_improvement_suggestions(self,
                                        alt_text: str,
                                        subject_area: SubjectArea,
                                        pedagogical_score: float,
                                        vocab_score: float) -> List[str]:
        """Generate specific improvement suggestions."""
        
        suggestions = []
        
        if pedagogical_score < 3.5:
            suggestions.append("Enhance pedagogical focus with learning-oriented language")
        
        if vocab_score < 0.7:
            suggestions.append("Improve subject-specific vocabulary usage")
        
        if len(alt_text) < 30:
            suggestions.append("Expand description to provide more educational context")
        
        if len(alt_text) > 250:
            suggestions.append("Consider condensing while maintaining key educational points")
        
        # Subject-specific suggestions
        if subject_area in self.subject_vocabularies:
            vocab = self.subject_vocabularies[subject_area]
            concept_present = any(concept in alt_text for concept in vocab['concepts'])
            if not concept_present:
                suggestions.append(f"Include relevant {subject_area.value} concepts")
        
        return suggestions
    
    def _calculate_overall_confidence(self,
                                    pedagogical_score: float,
                                    structural_score: float,
                                    subject_relevance: float,
                                    subject_confidence: float) -> float:
        """Calculate overall confidence in the quality assessment."""
        
        # Weighted combination of quality metrics
        confidence = (
            pedagogical_score / 5.0 * 0.4 +  # Pedagogical quality (1-5 scale)
            structural_score * 0.3 +         # Structural correctness (0-1)
            subject_relevance * 0.2 +        # Subject relevance (0-1)
            subject_confidence * 0.1         # Subject detection confidence (0-1)
        )
        
        return min(1.0, max(0.0, confidence))