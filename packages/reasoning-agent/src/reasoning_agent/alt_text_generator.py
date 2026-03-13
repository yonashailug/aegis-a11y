"""UDL-compliant pedagogical alt-text generation."""

import re
from typing import Dict, Any, List
from dataclasses import dataclass

from .schemas import SubjectArea, SpatialContext
from .context_processor import SubjectDetectionResult


@dataclass
class AltTextResult:
    """Result of pedagogical alt-text generation."""
    alt_text: str
    rationale: str
    importance: str
    udl_guidelines_applied: List[str]


class AltTextGenerator:
    """Generates pedagogically-focused alt-text following UDL guidelines.
    
    This class creates educational descriptions that focus on learning
    objectives rather than pure visual description.
    
    Based on paper Section 3.2: Pedagogical alt-text generation requirements
    and Universal Design for Learning (UDL) principles.
    """
    
    def __init__(self):
        """Initialize the alt-text generator."""
        
        # UDL Guidelines for educational alt-text
        self.udl_principles = {
            'engagement': [
                'Connect to learner interests and goals',
                'Provide appropriate challenges',
                'Foster collaboration and learning community'
            ],
            'representation': [
                'Offer ways of customizing the display of information',
                'Offer alternatives for auditory information',
                'Offer alternatives for visual information',
                'Provide options for comprehension'
            ],
            'action_expression': [
                'Provide options for physical action',
                'Provide options for expression and communication',
                'Provide options for executive functions'
            ]
        }
        
        # Subject-specific description patterns
        self.subject_patterns = {
            SubjectArea.PHYSICS: {
                'focus': 'relationships, measurements, and physical phenomena',
                'key_elements': ['forces', 'directions', 'magnitudes', 'relationships', 'trends'],
                'avoid': ['colors without context', 'purely aesthetic descriptions']
            },
            SubjectArea.CHEMISTRY: {
                'focus': 'molecular structures, reactions, and chemical relationships',
                'key_elements': ['bonds', 'structures', 'reactions', 'states', 'transformations'],
                'avoid': ['decorative elements', 'non-functional visual details']
            },
            SubjectArea.BIOLOGY: {
                'focus': 'biological structures, processes, and relationships',
                'key_elements': ['structures', 'functions', 'processes', 'interactions', 'systems'],
                'avoid': ['superficial appearance', 'non-functional characteristics']
            },
            SubjectArea.MATHEMATICS: {
                'focus': 'mathematical relationships, patterns, and problem-solving',
                'key_elements': ['patterns', 'relationships', 'values', 'trends', 'solutions'],
                'avoid': ['purely visual arrangement', 'decorative mathematical symbols']
            },
            SubjectArea.HISTORY: {
                'focus': 'historical significance, context, and cause-effect relationships',
                'key_elements': ['significance', 'context', 'relationships', 'impact', 'chronology'],
                'avoid': ['purely descriptive appearance', 'decorative historical elements']
            },
            SubjectArea.LITERATURE: {
                'focus': 'literary devices, themes, and symbolic meaning',
                'key_elements': ['symbolism', 'themes', 'character development', 'literary devices'],
                'avoid': ['surface-level description', 'decorative text formatting']
            }
        }
        
        self.initialized = True
    
    def generate_pedagogical_description(self,
                                       element: Dict[str, Any],
                                       spatial_context: SpatialContext,
                                       detected_subject: SubjectDetectionResult,
                                       llm_response: str) -> AltTextResult:
        """Generate UDL-compliant pedagogical alt-text.
        
        Args:
            element: The extracted element being processed
            spatial_context: Spatial context information
            detected_subject: Subject detection results
            llm_response: Raw LLM response content
            
        Returns:
            AltTextResult with pedagogical alt-text and metadata
        """
        classification = element.get('classification', 'unknown')
        element_text = element.get('ocr_text', '')
        
        # Parse LLM response for key information
        parsed_content = self._parse_llm_response(llm_response)
        
        # Apply subject-specific pedagogical patterns
        base_description = self._apply_subject_patterns(
            parsed_content,
            detected_subject,
            classification,
            element_text
        )
        
        # Apply UDL guidelines for accessibility
        enhanced_description = self._apply_udl_guidelines(
            base_description,
            detected_subject,
            spatial_context
        )
        
        # Generate rationale and importance
        rationale = self._generate_rationale(
            enhanced_description,
            detected_subject,
            spatial_context
        )
        
        importance = self._determine_importance(
            classification,
            spatial_context,
            detected_subject
        )
        
        applied_guidelines = self._identify_applied_guidelines(enhanced_description)
        
        return AltTextResult(
            alt_text=enhanced_description,
            rationale=rationale,
            importance=importance,
            udl_guidelines_applied=applied_guidelines
        )
    
    def _parse_llm_response(self, llm_response: str) -> Dict[str, str]:
        """Parse structured information from LLM response."""
        parsed = {
            'description': '',
            'educational_purpose': '',
            'key_concepts': ''
        }
        
        # Simple parsing - in production, this could be more sophisticated
        lines = llm_response.split('\n')
        current_section = 'description'
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Look for section headers
            if 'purpose:' in line.lower() or 'educational' in line.lower():
                current_section = 'educational_purpose'
                parsed[current_section] = line.split(':', 1)[-1].strip()
            elif 'concept' in line.lower() or 'key' in line.lower():
                current_section = 'key_concepts'
                parsed[current_section] = line.split(':', 1)[-1].strip()
            else:
                parsed[current_section] += ' ' + line
        
        # Clean up
        for key in parsed:
            parsed[key] = parsed[key].strip()
            
        return parsed
    
    def _apply_subject_patterns(self,
                              parsed_content: Dict[str, str],
                              detected_subject: SubjectDetectionResult,
                              classification: str,
                              element_text: str) -> str:
        """Apply subject-specific pedagogical patterns."""
        
        subject_config = self.subject_patterns.get(
            detected_subject.area, 
            self.subject_patterns.get(SubjectArea.GENERAL, {})
        )
        
        description = parsed_content.get('description', '')
        
        # Start with base description from LLM
        if not description and element_text:
            # Fallback if LLM didn't provide good description
            description = f"{classification} containing: {element_text}"
        
        # Add subject-specific focus
        if detected_subject.area == SubjectArea.PHYSICS:
            description = self._enhance_physics_description(description, element_text)
        elif detected_subject.area == SubjectArea.CHEMISTRY:
            description = self._enhance_chemistry_description(description, element_text)
        elif detected_subject.area == SubjectArea.BIOLOGY:
            description = self._enhance_biology_description(description, element_text)
        elif detected_subject.area == SubjectArea.MATHEMATICS:
            description = self._enhance_math_description(description, element_text)
        
        return description
    
    def _enhance_physics_description(self, description: str, element_text: str) -> str:
        """Enhance description with physics-specific pedagogical focus."""
        
        # Look for physics concepts in element text
        physics_indicators = {
            'force': 'showing force interactions and directions',
            'velocity': 'illustrating motion and speed relationships',
            'energy': 'demonstrating energy transformations and conservation',
            'wave': 'depicting wave properties and behavior',
            'electric': 'showing electrical phenomena and field patterns'
        }
        
        for concept, enhancement in physics_indicators.items():
            if concept in element_text.lower():
                description += f" {enhancement}"
                break
        
        # Emphasize quantitative relationships
        if any(word in element_text.lower() for word in ['graph', 'chart', 'plot']):
            description += " Focus on the relationship between variables and trends in the data."
        
        return description
    
    def _enhance_chemistry_description(self, description: str, element_text: str) -> str:
        """Enhance description with chemistry-specific pedagogical focus."""
        
        chemistry_indicators = {
            'molecule': 'showing molecular structure and bonding patterns',
            'reaction': 'illustrating chemical transformation and reaction mechanisms',
            'periodic': 'demonstrating element properties and periodic trends',
            'bond': 'depicting atomic bonding and electron interactions',
            'equation': 'showing chemical equation balancing and stoichiometry'
        }
        
        for concept, enhancement in chemistry_indicators.items():
            if concept in element_text.lower():
                description += f" {enhancement}"
                break
        
        return description
    
    def _enhance_biology_description(self, description: str, element_text: str) -> str:
        """Enhance description with biology-specific pedagogical focus."""
        
        biology_indicators = {
            'cell': 'illustrating cellular structure and function relationships',
            'dna': 'showing genetic information flow and molecular structure',
            'ecosystem': 'depicting ecological relationships and energy flow',
            'evolution': 'demonstrating evolutionary processes and adaptations',
            'photosynthesis': 'illustrating metabolic pathways and energy conversion'
        }
        
        for concept, enhancement in biology_indicators.items():
            if concept in element_text.lower():
                description += f" {enhancement}"
                break
        
        return description
    
    def _enhance_math_description(self, description: str, element_text: str) -> str:
        """Enhance description with mathematics-specific pedagogical focus."""
        
        math_indicators = {
            'graph': 'showing mathematical relationships and function behavior',
            'equation': 'illustrating problem-solving steps and algebraic relationships',
            'geometry': 'demonstrating spatial relationships and geometric properties',
            'function': 'depicting input-output relationships and mathematical patterns',
            'data': 'showing statistical patterns and data interpretation'
        }
        
        for concept, enhancement in math_indicators.items():
            if concept in element_text.lower():
                description += f" {enhancement}"
                break
        
        return description
    
    def _apply_udl_guidelines(self,
                            description: str,
                            detected_subject: SubjectDetectionResult,
                            spatial_context: SpatialContext) -> str:
        """Apply UDL guidelines to enhance accessibility."""
        
        enhanced = description
        
        # UDL Principle: Multiple means of representation
        if spatial_context.containing_section:
            enhanced = f"In the '{spatial_context.containing_section}' section: {enhanced}"
        
        # UDL Principle: Clear learning purpose
        if detected_subject.area != SubjectArea.UNKNOWN:
            subject_purpose = f"This {detected_subject.area.value} element "
            if not enhanced.lower().startswith(subject_purpose.lower()):
                enhanced = subject_purpose + enhanced.lower()
        
        # UDL Principle: Multiple means of comprehension
        if len(enhanced.split()) > 20:  # Long descriptions need structure
            sentences = enhanced.split('.')
            if len(sentences) > 2:
                enhanced = '. '.join(sentences[:2]) + '. ' + ' '.join(sentences[2:])
        
        return enhanced.strip()
    
    def _generate_rationale(self,
                          description: str,
                          detected_subject: SubjectDetectionResult,
                          spatial_context: SpatialContext) -> str:
        """Generate rationale for why this alt-text supports learning."""
        
        rationale_parts = []
        
        # Subject-specific rationale
        if detected_subject.area in [SubjectArea.PHYSICS, SubjectArea.CHEMISTRY, SubjectArea.BIOLOGY]:
            rationale_parts.append("Focuses on scientific concepts and relationships rather than visual appearance")
        elif detected_subject.area == SubjectArea.MATHEMATICS:
            rationale_parts.append("Emphasizes mathematical relationships and problem-solving context")
        elif detected_subject.area in [SubjectArea.HISTORY, SubjectArea.LITERATURE]:
            rationale_parts.append("Highlights educational significance and contextual meaning")
        
        # Context-specific rationale
        if spatial_context.containing_section:
            rationale_parts.append(f"Connects to the learning objectives of the '{spatial_context.containing_section}' section")
        
        # UDL rationale
        rationale_parts.append("Provides equivalent access to visual information for screen reader users")
        
        return ". ".join(rationale_parts) + "."
    
    def _determine_importance(self,
                            classification: str,
                            spatial_context: SpatialContext,
                            detected_subject: SubjectDetectionResult) -> str:
        """Determine why this element is important for learning."""
        
        if 'figure' in classification.lower() or 'image' in classification.lower():
            if detected_subject.area in [SubjectArea.PHYSICS, SubjectArea.CHEMISTRY, SubjectArea.BIOLOGY]:
                return "Visual representation of scientific concepts essential for understanding"
            elif detected_subject.area == SubjectArea.MATHEMATICS:
                return "Mathematical visualization crucial for problem comprehension"
            else:
                return "Visual content supporting educational objectives"
        elif 'table' in classification.lower():
            return "Structured data presentation important for information analysis"
        elif 'equation' in classification.lower():
            return "Mathematical expression fundamental to problem solving"
        else:
            return "Educational content contributing to lesson understanding"
    
    def _identify_applied_guidelines(self, description: str) -> List[str]:
        """Identify which UDL guidelines were applied in the description."""
        
        applied = []
        
        if 'section' in description.lower():
            applied.append("Contextual orientation (UDL 3.1)")
        
        if any(word in description.lower() for word in ['shows', 'illustrates', 'demonstrates']):
            applied.append("Alternative format for visual information (UDL 1.3)")
        
        if any(word in description.lower() for word in ['relationship', 'pattern', 'concept']):
            applied.append("Background knowledge activation (UDL 3.2)")
        
        if len(description.split()) < 30:  # Concise description
            applied.append("Clear and simple language (UDL 3.1)")
        
        return applied
