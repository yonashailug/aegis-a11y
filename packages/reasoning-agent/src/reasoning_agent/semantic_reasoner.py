"""Main semantic reasoner class for LLM-based processing."""

import os
import time
import asyncio
import base64
from typing import Optional, Dict, Any, List
from openai import OpenAI
from dotenv import load_dotenv

from .schemas import (
    ReasoningInput, 
    ReasoningOutput, 
    SubjectArea, 
    ConfidenceLevel,
    LLMRequest,
    LLMResponse
)
from .context_processor import ContextProcessor
from .alt_text_generator import AltTextGenerator
from .prompt_templates import STEM_TEMPLATES, HUMANITIES_TEMPLATES, GENERAL_TEMPLATES

# Load environment variables
load_dotenv()


class SemanticReasoner:
    """Main class for multi-modal semantic reasoning using LLMs.
    
    This class integrates with OpenAI GPT-4o to process layout
    decomposition output and generate pedagogical accessibility metadata.
    
    Based on paper Section 3.2: Multi-modal Semantic Reasoning Layer
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the semantic reasoner.
        
        Args:
            api_key: OpenAI API key (if not provided, reads from OPENAI_API_KEY env var)
        """
        # Initialize OpenAI client
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.client = OpenAI(api_key=self.api_key)
        
        # Initialize processing components
        self.context_processor = ContextProcessor()
        self.alt_text_generator = AltTextGenerator()
        
        # Configuration
        self.model = "gpt-4o"  # As specified in paper Section 4.6
        self.max_retries = 3
        self.timeout_seconds = 30
        
        self.initialized = True
    
    def process_element(self, input_data: ReasoningInput) -> ReasoningOutput:
        """Process a single element through semantic reasoning.
        
        Implements the multi-modal semantic reasoning described in paper Section 3.2.
        
        Args:
            input_data: ReasoningInput containing element and context
            
        Returns:
            ReasoningOutput with semantic analysis and pedagogical alt-text
        """
        start_time = time.time()
        processing_warnings = []
        fallback_used = False
        
        try:
            # Step 1: Extract spatial context (Contextual Injection from paper)
            spatial_context = self.context_processor.extract_spatial_context(
                input_data.extracted_element,
                input_data.surrounding_elements
            )
            
            # Step 2: Detect subject area
            detected_subject = self.context_processor.detect_subject_area(
                spatial_context.preceding_text + " " + spatial_context.following_text,
                input_data.page_metadata
            )
            
            # Step 3: Build multi-modal prompt
            llm_request = self._build_llm_request(
                input_data,
                spatial_context,
                detected_subject
            )
            
            # Step 4: Call OpenAI GPT-4o (Multi-modal Semantic Reasoning)
            llm_response = self._call_openai(llm_request)
            
            # Step 5: Generate pedagogical alt-text (UDL compliance)
            alt_text_result = self.alt_text_generator.generate_pedagogical_description(
                input_data.extracted_element,
                spatial_context,
                detected_subject,
                llm_response.content
            )
            
            # Step 6: Assess quality
            quality_score = self._assess_quality(alt_text_result, detected_subject)
            confidence = self._determine_confidence(quality_score, llm_response)
            
            processing_duration = time.time() - start_time
            
            return ReasoningOutput(
                element_id=input_data.extracted_element.get("element_id", "unknown"),
                detected_subject_area=detected_subject.area,
                subject_confidence=detected_subject.confidence,
                learning_objective=spatial_context.containing_section,
                contextual_importance=alt_text_result.importance,
                pedagogical_alt_text=alt_text_result.alt_text,
                alt_text_rationale=alt_text_result.rationale,
                pedagogical_quality_score=quality_score,
                confidence_level=confidence,
                processing_duration=processing_duration,
                prompt_template_used=f"{detected_subject.area.value}_template",
                processing_warnings=processing_warnings,
                fallback_used=fallback_used,
                raw_llm_response=llm_response.content
            )
            
        except Exception as e:
            # Fallback processing for errors
            processing_warnings.append(f"Error in main processing: {str(e)}")
            fallback_used = True
            
            return self._fallback_processing(
                input_data, 
                start_time, 
                processing_warnings, 
                str(e)
            )
    
    def _build_llm_request(self, 
                          input_data: ReasoningInput, 
                          spatial_context, 
                          detected_subject) -> LLMRequest:
        """Build LLM request with contextual injection."""
        
        # Get appropriate prompt template
        template = self._get_prompt_template(detected_subject.area)
        
        # Build context-aware prompt
        context_text = f"""
        Element Classification: {input_data.extracted_element.get('classification', 'unknown')}
        OCR Text: {input_data.extracted_element.get('ocr_text', '')}
        
        Surrounding Context:
        Before: {spatial_context.preceding_text[:200]}
        After: {spatial_context.following_text[:200]}
        
        Subject Area: {detected_subject.area.value} (confidence: {detected_subject.confidence:.2f})
        Section: {spatial_context.containing_section or 'Unknown'}
        """
        
        prompt = template.format(
            context=context_text,
            element_text=input_data.extracted_element.get('ocr_text', ''),
            subject=detected_subject.area.value
        )
        
        return LLMRequest(
            prompt=prompt,
            image_data=input_data.image_segment,
            subject_hint=detected_subject.area,
            temperature=0.3,  # Low temperature for consistency
            max_tokens=1000
        )
    
    def _call_openai(self, request: LLMRequest) -> LLMResponse:
        """Call OpenAI GPT-4o with retry logic."""
        
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                
                # Prepare messages
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": request.prompt}
                        ]
                    }
                ]
                
                # Add image if provided
                if request.image_data:
                    image_b64 = base64.b64encode(request.image_data).decode('utf-8')
                    messages[0]["content"].append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        }
                    })
                
                # Call OpenAI
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens
                )
                
                response_time = time.time() - start_time
                
                return LLMResponse(
                    content=response.choices[0].message.content,
                    usage_tokens=response.usage.total_tokens,
                    response_time=response_time,
                    model_version=self.model
                )
                
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise e
                time.sleep(2 ** attempt)  # Exponential backoff
        
        raise Exception(f"Failed after {self.max_retries} attempts")
    
    def _get_prompt_template(self, subject_area: SubjectArea) -> str:
        """Get appropriate prompt template for subject area."""
        
        if subject_area in [SubjectArea.PHYSICS, SubjectArea.CHEMISTRY, SubjectArea.BIOLOGY, SubjectArea.MATHEMATICS]:
            return STEM_TEMPLATES.get(subject_area.value, GENERAL_TEMPLATES["default"])
        elif subject_area in [SubjectArea.HISTORY, SubjectArea.LITERATURE, SubjectArea.SOCIAL_STUDIES]:
            return HUMANITIES_TEMPLATES.get(subject_area.value, GENERAL_TEMPLATES["default"])
        else:
            return GENERAL_TEMPLATES["default"]
    
    def _assess_quality(self, alt_text_result, detected_subject) -> float:
        """Assess pedagogical quality on 1-5 scale (paper requirement)."""
        
        # Simple heuristic-based quality assessment
        # In production, this could be a fine-tuned model
        score = 3.0  # Base score
        
        # Length check
        if len(alt_text_result.alt_text) < 20:
            score -= 1.0
        elif len(alt_text_result.alt_text) > 50:
            score += 0.5
        
        # Subject-specific vocabulary check
        subject_keywords = {
            SubjectArea.PHYSICS: ['force', 'velocity', 'acceleration', 'energy', 'momentum'],
            SubjectArea.CHEMISTRY: ['molecule', 'atom', 'bond', 'reaction', 'element'],
            SubjectArea.BIOLOGY: ['cell', 'organism', 'evolution', 'genetics', 'protein'],
            SubjectArea.MATHEMATICS: ['equation', 'function', 'variable', 'graph', 'solution']
        }
        
        keywords = subject_keywords.get(detected_subject.area, [])
        keyword_matches = sum(1 for kw in keywords if kw.lower() in alt_text_result.alt_text.lower())
        if keyword_matches > 0:
            score += 0.5
        
        # Pedagogical language check
        pedagogical_indicators = ['shows', 'demonstrates', 'illustrates', 'represents', 'depicts']
        if any(indicator in alt_text_result.alt_text.lower() for indicator in pedagogical_indicators):
            score += 0.3
        
        return max(1.0, min(5.0, score))  # Clamp between 1-5
    
    def _determine_confidence(self, quality_score: float, llm_response: LLMResponse) -> ConfidenceLevel:
        """Determine overall confidence level."""
        
        if quality_score >= 4.0 and llm_response.response_time < 10.0:
            return ConfidenceLevel.HIGH
        elif quality_score >= 3.0:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
    
    def _fallback_processing(self, 
                            input_data: ReasoningInput, 
                            start_time: float, 
                            warnings: List[str],
                            error: str) -> ReasoningOutput:
        """Fallback processing when main pipeline fails."""
        
        processing_duration = time.time() - start_time
        element_text = input_data.extracted_element.get('ocr_text', '')
        
        # Simple fallback alt-text
        classification = input_data.extracted_element.get('classification', 'element')
        fallback_alt_text = f"Educational {classification}"
        
        if element_text:
            fallback_alt_text += f" containing: {element_text[:100]}"
        
        return ReasoningOutput(
            element_id=input_data.extracted_element.get("element_id", "unknown"),
            detected_subject_area=SubjectArea.UNKNOWN,
            subject_confidence=0.1,
            learning_objective="Unknown - processing failed",
            contextual_importance="Unable to determine due to processing error",
            pedagogical_alt_text=fallback_alt_text,
            alt_text_rationale="Fallback description due to processing failure",
            pedagogical_quality_score=2.0,
            confidence_level=ConfidenceLevel.LOW,
            processing_duration=processing_duration,
            prompt_template_used="fallback",
            processing_warnings=warnings + [f"Fallback used due to: {error}"],
            fallback_used=True,
            raw_llm_response=None
        )