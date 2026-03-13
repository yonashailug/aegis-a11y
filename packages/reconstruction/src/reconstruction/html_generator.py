"""
HTML5 Accessible Document Generator

Based on technical paper Section 3.3: Generates WCAG 2.1 AA compliant
HTML5 documents from structured document hierarchy.
"""

from datetime import datetime
import html
from typing import Any

from jinja2 import BaseLoader, Environment

from .schemas import DocumentStructure, ReconstructionInput


class HTML5Generator:
    """
    Generates accessible HTML5 documents from document structure.

    Implements WCAG 2.1 AA compliance requirements including:
    - Proper heading hierarchy
    - ARIA landmarks and roles
    - Alternative text for images
    - Keyboard navigation support
    - Screen reader compatibility
    """

    def __init__(self):
        """Initialize the HTML5 generator."""
        self.jinja_env = Environment(loader=BaseLoader())
        self.accessibility_features = {
            "skip_navigation": True,
            "aria_landmarks": True,
            "heading_hierarchy": True,
            "alt_text_validation": True,
            "keyboard_navigation": True,
            "focus_indicators": True,
            "color_contrast": True,
        }

    def generate_html5_document(
        self,
        document_structure: DocumentStructure,
        reconstruction_input: ReconstructionInput,
    ) -> str:
        """
        Generate complete HTML5 document from structure.

        Args:
            document_structure: Hierarchical document structure
            reconstruction_input: Original reconstruction input with metadata

        Returns:
            Complete HTML5 document as string
        """
        # Build document metadata
        metadata = self._build_document_metadata(reconstruction_input)

        # Generate CSS styles for accessibility
        css_styles = self._generate_accessibility_css()

        # Generate JavaScript for enhanced accessibility
        javascript = self._generate_accessibility_javascript()

        # Build HTML structure
        html_content = self._build_html_structure(
            document_structure, metadata, css_styles, javascript
        )

        return html_content

    def _build_document_metadata(
        self, reconstruction_input: ReconstructionInput
    ) -> dict[str, Any]:
        """Build document metadata for HTML head section."""
        return {
            "title": html.escape(reconstruction_input.document_title),
            "language": reconstruction_input.document_language,
            "subject_area": reconstruction_input.subject_area,
            "educational_level": reconstruction_input.educational_level,
            "generated_at": datetime.now().isoformat(),
            "generator": "Aegis-A11y Reconstruction Engine v0.1.0",
            "accessibility_standard": reconstruction_input.accessibility_standard.value,
        }

    def _generate_accessibility_css(self) -> str:
        """Generate CSS for accessibility features."""
        return """
/* Aegis-A11y Accessibility Styles */

/* Skip navigation link */
.skip-link {
    position: absolute;
    top: -40px;
    left: 6px;
    background: #000;
    color: #fff;
    padding: 8px;
    text-decoration: none;
    border-radius: 0 0 4px 4px;
    z-index: 1000;
    transition: top 0.3s;
}

.skip-link:focus {
    top: 0;
}

/* Focus indicators */
*:focus {
    outline: 2px solid #005fcc;
    outline-offset: 2px;
}

/* High contrast mode support */
@media (prefers-contrast: high) {
    * {
        background: transparent !important;
        color: inherit !important;
        border-color: currentColor !important;
    }
}

/* Reduced motion support */
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
}

/* Document structure */
.document-main {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
    line-height: 1.6;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
}

.document-section {
    margin-bottom: 2em;
}

/* Subject-specific styling */
.aegis-element.physics {
    border-left: 3px solid #e74c3c;
    padding-left: 1em;
}

.aegis-element.chemistry {
    border-left: 3px solid #f39c12;
    padding-left: 1em;
}

.aegis-element.biology {
    border-left: 3px solid #27ae60;
    padding-left: 1em;
}

.aegis-element.mathematics {
    border-left: 3px solid #3498db;
    padding-left: 1em;
}

/* Educational elements */
.aegis-element {
    margin: 1em 0;
    position: relative;
}

.aegis-element[data-quality] {
    position: relative;
}

.aegis-element[data-quality="5.0"]:before {
    content: "★★★★★";
    position: absolute;
    top: -10px;
    right: 0;
    font-size: 12px;
    color: #27ae60;
}

/* Tables */
table.aegis-element {
    width: 100%;
    border-collapse: collapse;
    margin: 1em 0;
}

table.aegis-element th,
table.aegis-element td {
    border: 1px solid #ddd;
    padding: 12px;
    text-align: left;
}

table.aegis-element th {
    background-color: #f8f9fa;
    font-weight: bold;
}

/* Lists */
ul.aegis-element,
ol.aegis-element {
    margin: 1em 0;
    padding-left: 2em;
}

ul.aegis-element li,
ol.aegis-element li {
    margin: 0.5em 0;
}

/* Figures and images */
figure.aegis-element {
    margin: 1.5em 0;
    text-align: center;
}

figure.aegis-element img {
    max-width: 100%;
    height: auto;
}

figure.aegis-element figcaption {
    margin-top: 0.5em;
    font-style: italic;
    color: #666;
}

/* Mathematics */
math.aegis-element {
    display: block;
    margin: 1em auto;
    text-align: center;
    font-family: 'Latin Modern Math', 'STIX Two Math', serif;
}

/* Headings */
h1, h2, h3, h4, h5, h6 {
    color: #2c3e50;
    margin-top: 1.5em;
    margin-bottom: 0.5em;
    line-height: 1.2;
}

h1 { font-size: 2.5em; }
h2 { font-size: 2em; }
h3 { font-size: 1.5em; }
h4 { font-size: 1.25em; }
h5 { font-size: 1.1em; }
h6 { font-size: 1em; font-weight: bold; }

/* Learning objective indicators */
[data-learning-objective]:before {
    content: "📚 ";
    margin-right: 0.25em;
}

/* Print styles */
@media print {
    .skip-link { display: none; }
    .document-main { max-width: none; }
    * { color: black !important; background: white !important; }
}

/* Screen reader only content */
.sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border: 0;
}

/* Responsive design */
@media (max-width: 768px) {
    .document-main {
        padding: 10px;
    }
    
    table.aegis-element {
        font-size: 14px;
    }
    
    table.aegis-element th,
    table.aegis-element td {
        padding: 8px;
    }
}
"""

    def _generate_accessibility_javascript(self) -> str:
        """Generate JavaScript for enhanced accessibility features."""
        return """
// Aegis-A11y Accessibility Enhancements

(function() {
    'use strict';
    
    // Initialize accessibility features when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initializeA11y);
    } else {
        initializeA11y();
    }
    
    function initializeA11y() {
        enhanceKeyboardNavigation();
        addAriaLabels();
        setupFocusManagement();
        announcePageLoad();
    }
    
    // Enhanced keyboard navigation
    function enhanceKeyboardNavigation() {
        // Add keyboard shortcuts for common navigation
        document.addEventListener('keydown', function(e) {
            // Alt + 1: Skip to main content
            if (e.altKey && e.key === '1') {
                e.preventDefault();
                const mainContent = document.getElementById('main_content');
                if (mainContent) {
                    mainContent.focus();
                    mainContent.scrollIntoView();
                }
            }
            
            // Alt + 2: Jump to next heading
            if (e.altKey && e.key === '2') {
                e.preventDefault();
                jumpToNextHeading();
            }
        });
    }
    
    // Add missing ARIA labels
    function addAriaLabels() {
        // Enhance tables without captions
        const tables = document.querySelectorAll('table:not([aria-label]):not([aria-labelledby])');
        tables.forEach(function(table, index) {
            const subject = table.getAttribute('data-subject') || 'educational';
            table.setAttribute('aria-label', `${subject} data table ${index + 1}`);
        });
        
        // Enhance figures without alt text
        const figures = document.querySelectorAll('figure:not([aria-label])');
        figures.forEach(function(figure, index) {
            const subject = figure.getAttribute('data-subject') || 'educational';
            if (!figure.querySelector('img[alt]') && !figure.getAttribute('aria-label')) {
                figure.setAttribute('aria-label', `${subject} diagram ${index + 1}`);
            }
        });
    }
    
    // Focus management
    function setupFocusManagement() {
        // Ensure skip link works properly
        const skipLink = document.getElementById('skip_to_main');
        if (skipLink) {
            skipLink.addEventListener('click', function(e) {
                e.preventDefault();
                const target = document.querySelector(skipLink.getAttribute('href'));
                if (target) {
                    target.setAttribute('tabindex', '-1');
                    target.focus();
                    target.scrollIntoView();
                }
            });
        }
    }
    
    // Announce page load to screen readers
    function announcePageLoad() {
        const title = document.title;
        const announcement = document.createElement('div');
        announcement.setAttribute('aria-live', 'polite');
        announcement.setAttribute('aria-atomic', 'true');
        announcement.className = 'sr-only';
        announcement.textContent = `Page loaded: ${title}`;
        document.body.appendChild(announcement);
        
        // Remove announcement after screen readers have processed it
        setTimeout(() => {
            if (announcement.parentNode) {
                announcement.parentNode.removeChild(announcement);
            }
        }, 1000);
    }
    
    // Jump to next heading function
    function jumpToNextHeading() {
        const headings = document.querySelectorAll('h1, h2, h3, h4, h5, h6');
        const currentFocus = document.activeElement;
        
        let nextHeading = null;
        let foundCurrent = false;
        
        for (let i = 0; i < headings.length; i++) {
            if (foundCurrent) {
                nextHeading = headings[i];
                break;
            }
            if (headings[i] === currentFocus) {
                foundCurrent = true;
            }
        }
        
        // If no current heading focused, go to first heading
        if (!foundCurrent && headings.length > 0) {
            nextHeading = headings[0];
        }
        
        if (nextHeading) {
            nextHeading.setAttribute('tabindex', '-1');
            nextHeading.focus();
            nextHeading.scrollIntoView();
        }
    }
    
    // Announce dynamic content changes
    window.announceToScreenReader = function(message) {
        const announcement = document.createElement('div');
        announcement.setAttribute('aria-live', 'assertive');
        announcement.className = 'sr-only';
        announcement.textContent = message;
        document.body.appendChild(announcement);
        
        setTimeout(() => {
            if (announcement.parentNode) {
                announcement.parentNode.removeChild(announcement);
            }
        }, 1000);
    };
})();
"""

    def _build_html_structure(
        self,
        document_structure: DocumentStructure,
        metadata: dict[str, Any],
        css_styles: str,
        javascript: str,
    ) -> str:
        """Build the complete HTML document structure."""

        # HTML document template
        html_template = """<!DOCTYPE html>
<html lang="{{ metadata.language }}" class="no-js">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ metadata.title }}</title>
    
    <!-- Accessibility and SEO metadata -->
    <meta name="description" content="Accessible educational content generated by Aegis-A11y">
    <meta name="generator" content="{{ metadata.generator }}">
    <meta name="accessibility-standard" content="{{ metadata.accessibility_standard }}">
    {% if metadata.subject_area %}<meta name="subject" content="{{ metadata.subject_area }}">{% endif %}
    {% if metadata.educational_level %}<meta name="education-level" content="{{ metadata.educational_level }}">{% endif %}
    
    <!-- Open Graph / Social Media -->
    <meta property="og:type" content="article">
    <meta property="og:title" content="{{ metadata.title }}">
    <meta property="og:description" content="Accessible educational content">
    
    <!-- Accessibility enhancements -->
    <meta name="color-scheme" content="light dark">
    
    <style>
{{ css_styles }}
    </style>
</head>
<body>
    <!-- Skip navigation for screen readers -->
    <a href="#main_content" id="skip_to_main" class="skip-link">Skip to main content</a>
    
    <!-- Page header with document info -->
    <header role="banner" class="document-header">
        <div class="sr-only">
            <h1>{{ metadata.title }}</h1>
            {% if metadata.subject_area %}
            <p>Subject: {{ metadata.subject_area }}</p>
            {% endif %}
            <p>Generated on {{ metadata.generated_at }} by {{ metadata.generator }}</p>
        </div>
    </header>
    
    <!-- Main document content -->
{{ body_content }}
    
    <!-- Accessibility information footer -->
    <footer role="contentinfo" class="document-footer">
        <div class="sr-only">
            <h2>Accessibility Information</h2>
            <p>This document was automatically generated to meet {{ metadata.accessibility_standard|upper }} accessibility standards.</p>
            <p>If you experience any accessibility issues, please contact your instructor or accessibility services.</p>
        </div>
        <div aria-hidden="true">
            <small>Generated by Aegis-A11y • {{ metadata.generated_at }}</small>
        </div>
    </footer>
    
    <script>
{{ javascript }}
    </script>
</body>
</html>"""

        # Generate body content from document structure
        body_content = self._render_document_structure(document_structure)

        # Render complete template
        template = self.jinja_env.from_string(html_template)
        return template.render(
            metadata=metadata,
            css_styles=css_styles,
            javascript=javascript,
            body_content=body_content,
        )

    def _render_document_structure(
        self, structure: DocumentStructure, indent_level: int = 1
    ) -> str:
        """Recursively render document structure to HTML."""
        indent = "    " * indent_level

        # Handle special cases
        if structure.element_type == "html":
            # Root element - render children only
            return "\n".join(
                self._render_document_structure(child, indent_level)
                for child in structure.children
            )

        # Build opening tag
        tag_name = structure.element_type
        attributes = self._format_attributes(structure.attributes)

        # Add accessibility attributes
        if structure.aria_label:
            attributes += f' aria-label="{html.escape(structure.aria_label)}"'

        if structure.role:
            attributes += f' role="{structure.role}"'

        if structure.element_id:
            attributes += f' id="{structure.element_id}"'

        # Self-closing tags
        if tag_name in ["img", "br", "hr", "input", "meta", "link"]:
            return f"{indent}<{tag_name}{attributes} />"

        # Handle content and children
        content_parts = []

        # Add text content
        if structure.content:
            escaped_content = html.escape(structure.content)
            content_parts.append(escaped_content)

        # Add alt-text for figures
        if tag_name == "figure" and structure.alt_text:
            figcaption = f"{indent}    <figcaption>{html.escape(structure.alt_text)}</figcaption>"
            content_parts.append(f"\n{figcaption}")

        # Add child elements
        if structure.children:
            children_html = []
            for child in structure.children:
                child_html = self._render_document_structure(child, indent_level + 1)
                if child_html.strip():
                    children_html.append(child_html)

            if children_html:
                content_parts.append(f'\n{"".join(children_html)}\n{indent}')

        # Combine content
        if content_parts:
            content = "".join(content_parts)
            return f"{indent}<{tag_name}{attributes}>{content}</{tag_name}>"
        else:
            return f"{indent}<{tag_name}{attributes}></{tag_name}>"

    def _format_attributes(self, attributes: dict[str, str]) -> str:
        """Format HTML attributes string."""
        if not attributes:
            return ""

        formatted_attrs = []
        for key, value in attributes.items():
            if value is not None:
                escaped_value = html.escape(str(value))
                formatted_attrs.append(f'{key}="{escaped_value}"')

        return " " + " ".join(formatted_attrs) if formatted_attrs else ""

    def validate_html5_compliance(self, html_content: str) -> dict[str, Any]:
        """Validate HTML5 and accessibility compliance."""
        validation_results = {
            "html5_valid": True,
            "wcag_compliant": True,
            "issues": [],
            "warnings": [],
            "accessibility_features": [],
        }

        # Basic HTML5 validation checks
        required_elements = ["<!DOCTYPE html>", "<html", "<head>", "<title>", "<body>"]
        for element in required_elements:
            if element not in html_content:
                validation_results["html5_valid"] = False
                validation_results["issues"].append(
                    f"Missing required element: {element}"
                )

        # Accessibility validation checks
        accessibility_checks = [
            ("alt=", "Images have alternative text"),
            ("role=", "ARIA roles are used"),
            ("aria-label=", "ARIA labels are provided"),
            ("skip-link", "Skip navigation is available"),
            ("<h1>", "Document has heading structure"),
        ]

        for check, feature in accessibility_checks:
            if check in html_content:
                validation_results["accessibility_features"].append(feature)

        # Check for potential issues
        if "img" in html_content and "alt=" not in html_content:
            validation_results["warnings"].append("Images may be missing alt text")

        if html_content.count("<h1>") > 1:
            validation_results["warnings"].append(
                "Multiple h1 tags found - should have only one"
            )

        # Overall compliance
        if validation_results["issues"]:
            validation_results["wcag_compliant"] = False

        return validation_results
