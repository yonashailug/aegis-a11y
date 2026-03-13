"""
JSON-to-Tag Mapping System

Based on technical paper Section 3.3: Maps detected structure to hierarchical tree
and converts CV+LLM metadata into compliant HTML/PDF markup.
"""

from typing import Dict, List, Any, Optional, Tuple
import re
import uuid
from dataclasses import dataclass

from reasoning_agent.schemas import ReasoningOutput, SubjectArea
from .schemas import DocumentStructure, TagMapping, ReconstructionInput


@dataclass
class LayoutContext:
    """Context information for layout-aware tag mapping."""
    page_width: float
    page_height: float
    column_count: int
    reading_order: List[str]  # Element IDs in reading order
    section_boundaries: List[Dict[str, Any]]


class JSONToTagMapper:
    """
    Converts reasoning agent output to hierarchical document structure.
    
    Based on paper Section 3.3: "JSON-to-Tag mapping: The detected structure 
    is mapped to a hierarchical tree. If the CV model identifies a header 
    followed by three list items, the reconstruction engine ensures these 
    are wrapped in proper <h1> and <ul> tags."
    """
    
    def __init__(self, tag_mapping: Optional[TagMapping] = None):
        """Initialize the JSON-to-tag mapper."""
        self.tag_mapping = tag_mapping or TagMapping()
        self.structure_cache = {}
        self.reading_order_cache = {}
        
        # HTML5 semantic structure rules
        self.semantic_rules = {
            'main_content': ['article', 'main'],
            'navigation': ['nav'],
            'complementary': ['aside'],
            'contentinfo': ['footer'],
            'banner': ['header']
        }
        
        # Heading hierarchy tracking
        self.heading_levels = {}
        self.current_heading_level = 1
        
    def map_to_document_structure(self, 
                                 reconstruction_input: ReconstructionInput) -> DocumentStructure:
        """
        Convert reasoning outputs to hierarchical document structure.
        
        Args:
            reconstruction_input: Input containing verified reasoning outputs
            
        Returns:
            DocumentStructure representing the complete document hierarchy
        """
        verified_elements = reconstruction_input.verified_elements
        
        # Step 1: Analyze layout and create reading order
        layout_context = self._analyze_layout(verified_elements, reconstruction_input.original_layout)
        
        # Step 2: Build structure hierarchy
        root_structure = self._build_document_root(reconstruction_input)
        
        # Step 3: Process elements in reading order
        structured_elements = self._process_elements_in_order(
            verified_elements, layout_context
        )
        
        # Step 4: Create semantic document sections
        document_sections = self._create_semantic_sections(structured_elements, layout_context)
        
        # Step 5: Build final hierarchy
        root_structure.children = document_sections
        
        # Step 6: Apply accessibility enhancements
        self._apply_accessibility_enhancements(root_structure)
        
        return root_structure
    
    def _analyze_layout(self, 
                       elements: List[ReasoningOutput],
                       original_layout: List[Dict[str, Any]]) -> LayoutContext:
        """Analyze layout structure and determine reading order."""
        
        if not elements:
            return LayoutContext(
                page_width=800, page_height=1200, column_count=1,
                reading_order=[], section_boundaries=[]
            )
        
        # Extract bounding boxes and positions
        element_positions = {}
        for element in elements:
            # Try to get bounding box from extracted_element or use original_layout as fallback
            bbox = None
            if hasattr(element, 'extracted_element') and 'bounding_box' in element.extracted_element:
                bbox = element.extracted_element['bounding_box']
            
            # Fallback: find matching element in original layout
            if not bbox and original_layout:
                for layout_item in original_layout:
                    if layout_item.get('bbox') and (
                        element.element_id in str(layout_item) or 
                        element.detected_subject_area.value in str(layout_item)
                    ):
                        bbox = layout_item['bbox']
                        break
            
            # Use default position if no bbox found
            if not bbox:
                bbox = [0, len(element_positions) * 100, 500, 80]  # Stacked layout
                
            element_positions[element.element_id] = {
                'x': bbox[0], 'y': bbox[1], 'width': bbox[2], 'height': bbox[3],
                'element': element
            }
        
        # Determine page dimensions
        page_width = max((pos['x'] + pos['width'] for pos in element_positions.values()), default=800)
        page_height = max((pos['y'] + pos['height'] for pos in element_positions.values()), default=1200)
        
        # Detect column layout
        column_count = self._detect_column_count(list(element_positions.values()))
        
        # Create reading order
        reading_order = self._determine_reading_order(list(element_positions.values()), column_count)
        
        # Identify section boundaries
        section_boundaries = self._identify_sections(elements, element_positions)
        
        return LayoutContext(
            page_width=page_width,
            page_height=page_height,
            column_count=column_count,
            reading_order=reading_order,
            section_boundaries=section_boundaries
        )
    
    def _detect_column_count(self, positions: List[Dict[str, Any]]) -> int:
        """Detect number of columns in the layout."""
        if not positions:
            return 1
            
        # Group elements by approximate x-position
        x_positions = [pos['x'] for pos in positions]
        x_positions.sort()
        
        # Find significant gaps that indicate column breaks
        column_breaks = []
        for i in range(1, len(x_positions)):
            gap = x_positions[i] - x_positions[i-1]
            if gap > 50:  # Significant gap threshold
                column_breaks.append(x_positions[i])
        
        return len(column_breaks) + 1
    
    def _determine_reading_order(self, 
                                positions: List[Dict[str, Any]], 
                                column_count: int) -> List[str]:
        """Determine logical reading order for elements."""
        if not positions:
            return []
        
        if column_count == 1:
            # Single column: sort by Y position (top to bottom)
            sorted_positions = sorted(positions, key=lambda p: p['y'])
        else:
            # Multi-column: sort by column, then by Y within column
            sorted_positions = self._sort_multicolumn_reading_order(positions, column_count)
        
        return [pos['element'].element_id for pos in sorted_positions]
    
    def _sort_multicolumn_reading_order(self, 
                                       positions: List[Dict[str, Any]], 
                                       column_count: int) -> List[Dict[str, Any]]:
        """Sort elements for multi-column reading order."""
        # Find column boundaries
        x_positions = sorted(set(pos['x'] for pos in positions))
        column_width = max(x_positions) / column_count if column_count > 1 else float('inf')
        
        # Assign elements to columns
        columns = [[] for _ in range(column_count)]
        for pos in positions:
            column_index = min(int(pos['x'] / column_width), column_count - 1)
            columns[column_index].append(pos)
        
        # Sort within each column by Y position
        for column in columns:
            column.sort(key=lambda p: p['y'])
        
        # Merge columns in reading order
        result = []
        for column in columns:
            result.extend(column)
        
        return result
    
    def _identify_sections(self, 
                          elements: List[ReasoningOutput],
                          positions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify document sections based on headings and content."""
        sections = []
        current_section = None
        
        for element in elements:
            # Get classification from element_id or subject area as fallback
            classification = ''
            if hasattr(element, 'extracted_element') and 'classification' in element.extracted_element:
                classification = element.extracted_element['classification'].lower()
            elif 'title' in element.element_id:
                classification = 'heading'
            elif 'paragraph' in element.element_id:
                classification = 'paragraph'
            elif 'diagram' in element.element_id:
                classification = 'functional_diagram'
            elif 'equation' in element.element_id:
                classification = 'equation'
            elif 'list' in element.element_id:
                classification = 'list'
            else:
                classification = 'paragraph'  # Default fallback
            
            if 'heading' in classification:
                # Start new section
                if current_section:
                    sections.append(current_section)
                
                current_section = {
                    'heading_id': element.element_id,
                    'heading_text': element.pedagogical_alt_text or '',
                    'elements': [element.element_id],
                    'start_y': positions.get(element.element_id, {}).get('y', 0)
                }
            elif current_section:
                # Add to current section
                current_section['elements'].append(element.element_id)
        
        # Add final section
        if current_section:
            sections.append(current_section)
        
        return sections
    
    def _build_document_root(self, reconstruction_input: ReconstructionInput) -> DocumentStructure:
        """Build the root document structure."""
        return DocumentStructure(
            element_type="html",
            element_id="document_root",
            attributes={
                "lang": reconstruction_input.document_language,
                "class": "aegis-a11y-document"
            },
            content=None,
            children=[],
            subject_area=reconstruction_input.subject_area,
            learning_objective=f"Educational content in {reconstruction_input.subject_area or 'multiple subjects'}",
            importance_level="primary"
        )
    
    def _process_elements_in_order(self, 
                                  elements: List[ReasoningOutput],
                                  layout_context: LayoutContext) -> List[DocumentStructure]:
        """Process elements in logical reading order."""
        structured_elements = []
        element_lookup = {elem.element_id: elem for elem in elements}
        
        for element_id in layout_context.reading_order:
            if element_id in element_lookup:
                element = element_lookup[element_id]
                structured_element = self._convert_element_to_structure(element)
                structured_elements.append(structured_element)
        
        # Add any remaining elements not in reading order
        processed_ids = set(layout_context.reading_order)
        for element in elements:
            if element.element_id not in processed_ids:
                structured_element = self._convert_element_to_structure(element)
                structured_elements.append(structured_element)
        
        return structured_elements
    
    def _convert_element_to_structure(self, element: ReasoningOutput) -> DocumentStructure:
        """Convert a single reasoning output to document structure."""
        # Get classification from element_id as fallback
        classification = ''
        if hasattr(element, 'extracted_element') and 'classification' in element.extracted_element:
            classification = element.extracted_element['classification'].lower()
        elif 'title' in element.element_id:
            classification = 'heading'
        elif 'paragraph' in element.element_id:
            classification = 'paragraph'
        elif 'diagram' in element.element_id:
            classification = 'functional_diagram'
        elif 'equation' in element.element_id:
            classification = 'equation'
        elif 'list' in element.element_id:
            classification = 'list'
        else:
            classification = 'paragraph'  # Default fallback
            
        # Get OCR text from extracted_element or use pedagogical_alt_text as fallback
        ocr_text = ''
        if hasattr(element, 'extracted_element') and 'ocr_text' in element.extracted_element:
            ocr_text = element.extracted_element['ocr_text']
        else:
            # Use pedagogical_alt_text as content for display
            ocr_text = element.pedagogical_alt_text
        
        # Map classification to HTML tag
        html_tag = self.tag_mapping.classification_to_tag.get(classification, 'div')
        
        # Handle special cases
        if 'heading' in classification:
            html_tag = self._determine_heading_level(element, ocr_text)
        elif classification == 'list':
            return self._create_list_structure(element)
        elif classification == 'table':
            return self._create_table_structure(element)
        
        # Build attributes
        attributes = self._build_element_attributes(element, classification)
        
        # Get ARIA role
        aria_role = self.tag_mapping.aria_role_mapping.get(classification)
        if aria_role:
            attributes['role'] = aria_role
        
        return DocumentStructure(
            element_type=html_tag,
            element_id=element.element_id,
            attributes=attributes,
            content=ocr_text,
            alt_text=element.pedagogical_alt_text,
            bounding_box=None,  # Will be set if available in layout context
            aria_label=element.pedagogical_alt_text if classification in ['functional_diagram', 'equation'] else None,
            subject_area=element.detected_subject_area.value,
            learning_objective=element.learning_objective,
            importance_level=element.contextual_importance
        )
    
    def _determine_heading_level(self, element: ReasoningOutput, text: str) -> str:
        """Determine appropriate heading level (h1-h6)."""
        # Analyze text patterns for heading hierarchy
        text_lower = text.lower()
        
        # Title/main heading indicators
        if any(indicator in text_lower for indicator in ['chapter', 'lesson', 'unit']):
            level = 'h1'
        elif any(indicator in text_lower for indicator in ['section', 'part']):
            level = 'h2' 
        elif any(indicator in text_lower for indicator in ['subsection', 'topic']):
            level = 'h3'
        else:
            # Default based on context or length
            if len(text) > 50:
                level = 'h2'  # Long text likely major section
            else:
                level = 'h3'  # Shorter text likely subsection
        
        # Track heading hierarchy
        level_num = int(level[1])
        self.current_heading_level = level_num
        
        return level
    
    def _create_list_structure(self, element: ReasoningOutput) -> DocumentStructure:
        """Create list structure with proper list items."""
        ocr_text = element.pedagogical_alt_text or 'Content'
        
        # Parse list items from OCR text
        list_items = self._parse_list_items(ocr_text)
        
        # Determine list type (ordered vs unordered)
        list_type = 'ol' if self._is_ordered_list(ocr_text) else 'ul'
        
        # Create list item structures
        list_item_structures = []
        for i, item_text in enumerate(list_items):
            item_structure = DocumentStructure(
                element_type='li',
                element_id=f"{element.element_id}_item_{i}",
                attributes={},
                content=item_text.strip(),
                subject_area=element.detected_subject_area.value
            )
            list_item_structures.append(item_structure)
        
        return DocumentStructure(
            element_type=list_type,
            element_id=element.element_id,
            attributes=self._build_element_attributes(element, 'list'),
            children=list_item_structures,
            subject_area=element.detected_subject_area.value,
            learning_objective=element.learning_objective
        )
    
    def _parse_list_items(self, text: str) -> List[str]:
        """Parse individual list items from OCR text."""
        # Common list item patterns
        patterns = [
            r'^\s*[\d]+[\.\)]\s*(.+)$',  # 1. or 1) 
            r'^\s*[•\-\*]\s*(.+)$',     # • - *
            r'^\s*[a-zA-Z][\.\)]\s*(.+)$',  # a. a)
            r'^\s*[ivxl]+[\.\)]\s*(.+)$'    # i. ii. (roman numerals)
        ]
        
        items = []
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Try each pattern
            for pattern in patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    items.append(match.group(1))
                    break
            else:
                # No pattern matched, treat as regular text
                if line and len(items) == 0:
                    # If no items found yet, split by common separators
                    items.extend(re.split(r'[•\-\*]', text))
                    break
                elif line:
                    items.append(line)
        
        return [item for item in items if item.strip()]
    
    def _is_ordered_list(self, text: str) -> bool:
        """Determine if list should be ordered (ol) or unordered (ul)."""
        # Look for numbered list indicators
        numbered_patterns = [r'\d+[\.\)]', r'[a-zA-Z][\.\)]', r'[ivxl]+[\.\)]']
        
        for pattern in numbered_patterns:
            if re.search(pattern, text):
                return True
        
        return False
    
    def _create_table_structure(self, element: ReasoningOutput) -> DocumentStructure:
        """Create table structure with rows and cells."""
        ocr_text = element.pedagogical_alt_text or 'Content'
        
        # Parse table data (simplified - could be enhanced with more sophisticated parsing)
        table_data = self._parse_table_data(ocr_text)
        
        # Create table structure
        table_rows = []
        for i, row_data in enumerate(table_data):
            cell_structures = []
            for j, cell_text in enumerate(row_data):
                cell_tag = 'th' if i == 0 else 'td'  # First row as headers
                cell_structure = DocumentStructure(
                    element_type=cell_tag,
                    element_id=f"{element.element_id}_cell_{i}_{j}",
                    attributes={'scope': 'col' if i == 0 else None},
                    content=cell_text.strip(),
                    subject_area=element.detected_subject_area.value
                )
                cell_structures.append(cell_structure)
            
            row_structure = DocumentStructure(
                element_type='tr',
                element_id=f"{element.element_id}_row_{i}",
                attributes={},
                children=cell_structures,
                subject_area=element.detected_subject_area.value
            )
            table_rows.append(row_structure)
        
        return DocumentStructure(
            element_type='table',
            element_id=element.element_id,
            attributes={
                **self._build_element_attributes(element, 'table'),
                'role': 'table'
            },
            children=table_rows,
            alt_text=element.pedagogical_alt_text,
            subject_area=element.detected_subject_area.value,
            learning_objective=element.learning_objective
        )
    
    def _parse_table_data(self, text: str) -> List[List[str]]:
        """Parse table data from OCR text."""
        # Split by rows (newlines)
        rows = [row.strip() for row in text.split('\n') if row.strip()]
        
        # Split each row by columns (tabs, pipes, or multiple spaces)
        table_data = []
        for row in rows:
            # Try different separators
            if '\t' in row:
                cells = row.split('\t')
            elif '|' in row:
                cells = [cell.strip() for cell in row.split('|') if cell.strip()]
            else:
                # Split on multiple spaces
                cells = re.split(r'\s{2,}', row)
            
            if cells:
                table_data.append(cells)
        
        return table_data
    
    def _build_element_attributes(self, 
                                 element: ReasoningOutput, 
                                 classification: str) -> Dict[str, str]:
        """Build HTML attributes for an element."""
        attributes = {
            'class': f'aegis-element {classification}',
            'data-subject': element.detected_subject_area.value,
            'data-confidence': str(element.subject_confidence),
            'data-quality': str(element.pedagogical_quality_score)
        }
        
        # Add accessibility attributes
        if element.pedagogical_alt_text and classification in ['functional_diagram', 'equation']:
            attributes['aria-label'] = element.pedagogical_alt_text
        
        # Add learning context
        if element.learning_objective:
            attributes['data-learning-objective'] = element.learning_objective
        
        return attributes
    
    def _create_semantic_sections(self, 
                                 structured_elements: List[DocumentStructure],
                                 layout_context: LayoutContext) -> List[DocumentStructure]:
        """Create semantic document sections."""
        if not structured_elements:
            return []
        
        # Create main document sections
        sections = []
        current_section = None
        
        for element in structured_elements:
            # Check if this is a heading that starts a new section
            if element.element_type.startswith('h') and element.element_type[1:].isdigit():
                # Save previous section
                if current_section and current_section.children:
                    sections.append(current_section)
                
                # Start new section
                section_id = f"section_{len(sections)}"
                current_section = DocumentStructure(
                    element_type='section',
                    element_id=section_id,
                    attributes={
                        'class': 'document-section',
                        'aria-labelledby': element.element_id
                    },
                    children=[element],
                    subject_area=element.subject_area
                )
            elif current_section:
                # Add to current section
                current_section.children.append(element)
            else:
                # Create default section for content without heading
                if not current_section:
                    current_section = DocumentStructure(
                        element_type='section',
                        element_id='section_main',
                        attributes={'class': 'document-section'},
                        children=[element]
                    )
        
        # Add final section
        if current_section and current_section.children:
            sections.append(current_section)
        
        # Wrap in main content area
        if sections:
            main_content = DocumentStructure(
                element_type='main',
                element_id='main_content',
                attributes={
                    'role': 'main',
                    'class': 'document-main'
                },
                children=sections
            )
            return [main_content]
        
        return structured_elements
    
    def _apply_accessibility_enhancements(self, root_structure: DocumentStructure):
        """Apply accessibility enhancements to the document structure."""
        self._add_skip_navigation(root_structure)
        self._ensure_heading_hierarchy(root_structure)
        self._add_landmark_roles(root_structure)
        self._validate_alt_text_coverage(root_structure)
    
    def _add_skip_navigation(self, root_structure: DocumentStructure):
        """Add skip navigation link for screen readers."""
        skip_nav = DocumentStructure(
            element_type='a',
            element_id='skip_to_main',
            attributes={
                'href': '#main_content',
                'class': 'skip-link',
                'aria-label': 'Skip to main content'
            },
            content='Skip to main content'
        )
        
        # Insert at beginning
        root_structure.children.insert(0, skip_nav)
    
    def _ensure_heading_hierarchy(self, structure: DocumentStructure):
        """Ensure proper heading hierarchy (no skipped levels)."""
        headings = []
        self._collect_headings(structure, headings)
        
        # Adjust heading levels to ensure proper hierarchy
        for i, heading in enumerate(headings):
            if i == 0:
                # First heading should be h1
                heading.element_type = 'h1'
            else:
                prev_level = int(headings[i-1].element_type[1:])
                current_level = int(heading.element_type[1:])
                
                # Don't skip levels
                if current_level > prev_level + 1:
                    heading.element_type = f'h{prev_level + 1}'
    
    def _collect_headings(self, structure: DocumentStructure, headings: List[DocumentStructure]):
        """Recursively collect all heading elements."""
        if structure.element_type.startswith('h') and structure.element_type[1:].isdigit():
            headings.append(structure)
        
        for child in structure.children:
            self._collect_headings(child, headings)
    
    def _add_landmark_roles(self, structure: DocumentStructure):
        """Add ARIA landmark roles for navigation."""
        self._add_landmark_roles_recursive(structure)
    
    def _add_landmark_roles_recursive(self, structure: DocumentStructure):
        """Recursively add landmark roles."""
        # Add appropriate landmark roles
        landmark_mapping = {
            'header': 'banner',
            'nav': 'navigation', 
            'main': 'main',
            'aside': 'complementary',
            'footer': 'contentinfo'
        }
        
        if structure.element_type in landmark_mapping:
            if 'role' not in structure.attributes:
                structure.attributes['role'] = landmark_mapping[structure.element_type]
        
        # Process children
        for child in structure.children:
            self._add_landmark_roles_recursive(child)
    
    def _validate_alt_text_coverage(self, structure: DocumentStructure):
        """Ensure all images and figures have appropriate alt-text."""
        self._validate_alt_text_recursive(structure)
    
    def _validate_alt_text_recursive(self, structure: DocumentStructure):
        """Recursively validate alt-text coverage."""
        # Check if element needs alt-text
        needs_alt_text = structure.element_type in ['img', 'figure'] or 'diagram' in structure.element_type
        
        if needs_alt_text and not structure.alt_text and not structure.attributes.get('aria-label'):
            # Add warning or default alt-text
            structure.alt_text = f"Educational {structure.element_type} element"
            if structure.subject_area:
                structure.alt_text += f" for {structure.subject_area}"
        
        # Process children
        for child in structure.children:
            self._validate_alt_text_recursive(child)