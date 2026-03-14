# Aegis-A11y: Educational Document Accessibility Framework

A hybrid LLM-Computer Vision framework for automated structural remediation of complex educational digital assets. This system transforms inaccessible PDFs into WCAG 2.1 AA compliant documents using advanced AI-powered semantic reasoning.

## Quick Start

```bash
# Clone and setup
git clone <repository-url>
cd aegis-a11y

# Install dependencies using UV
uv sync

# Configure environment
cp .env.example .env  # Add your OpenAI API key

# Start the API server
uv run uvicorn packages.api.main:app --reload

# Test the API
curl http://localhost:8000/
```

## Prerequisites

- **Python 3.11+**
- **UV package manager** - [Install UV](https://docs.astral.sh/uv/)
- **OpenAI API key** - Required for semantic reasoning

## Installation & Development Setup

### 1. Environment Setup

```bash
# Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone <repository-url>
cd aegis-a11y

# Create and activate virtual environment with dependencies
uv sync
```

### 2. Environment Configuration

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Verify Installation

```bash
# Check all components are properly installed
uv run python -c "
from cv_layer import LayoutDecomposer
from reasoning_agent import SemanticReasoner
from reconstruction import DocumentReconstructionEngine
print('✅ All components installed successfully')
"
```

## Running the Application

### Start the API Server

```bash
# Development mode with auto-reload
uv run uvicorn packages.api.main:app --reload

# Production mode
uv run uvicorn packages.api.main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### Health Check

```bash
curl http://localhost:8000/
```

Expected response:
```json
{
  "status": "ok",
  "message": "Aegis-A11y API is running",
  "decomposer_loaded": true,
  "reasoning_agent_loaded": true,
  "reconstruction_engine_loaded": true
}
```

## Project Structure

```
aegis-a11y/
├── packages/
│   ├── api/                     # FastAPI REST API
│   │   ├── main.py             # API endpoints and server
│   │   └── pyproject.toml      # API package configuration
│   │
│   ├── cv-layer/               # Phase 1: Document Decomposition
│   │   ├── src/cv_layer/
│   │   │   ├── decomposer.py   # LayoutLMv3 integration
│   │   │   └── main.py         # OCR and image processing
│   │   └── pyproject.toml
│   │
│   ├── reasoning-agent/        # Phase 2: Semantic Reasoning
│   │   ├── src/reasoning_agent/
│   │   │   ├── semantic_reasoner.py    # OpenAI GPT-4o integration
│   │   │   ├── alt_text_generator.py   # UDL-compliant alt-text
│   │   │   ├── context_processor.py    # Spatial context analysis
│   │   │   ├── verifier.py            # WCAG 2.1 AA validation
│   │   │   └── schemas.py             # Data models
│   │   └── pyproject.toml
│   │
│   └── reconstruction/         # Phase 3: Document Generation
│       ├── src/reconstruction/
│       │   ├── document_engine.py     # Main orchestration
│       │   ├── tag_mapper.py          # JSON-to-tag conversion
│       │   ├── html_generator.py      # HTML5 generation
│       │   ├── pdf_generator.py       # PDF/UA generation
│       │   └── schemas.py             # Data models
│       └── pyproject.toml
│
├── docs/                       # Documentation and samples
│   └── pdfs/                   # Sample PDF documents
├── pyproject.toml             # UV workspace configuration
├── uv.lock                    # Dependency lock file
└── README.md                  # This file
```

## API Reference

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check and component status |
| `/api/v1/decompose` | POST | PDF decomposition only |
| `/api/v1/analyze` | POST | Decomposition + AI reasoning |
| `/api/v1/reconstruct` | POST | Complete DRR pipeline |

### Complete DRR Pipeline

Process a PDF through the complete pipeline (Decomposition → Reasoning → Reconstruction):

```bash
curl -X POST "http://localhost:8000/api/v1/reconstruct" \
     -H "Content-Type: application/json"
```

**Response Structure:**
```json
{
  "status": "success",
  "pages_processed": 3,
  "pipeline": {
    "decomposition": {
      "total_elements": 45
    },
    "reasoning": {
      "total_analyzed": 43,
      "verification_pass_rate": 0.95
    },
    "reconstruction": {
      "documents_generated": 2,
      "accessibility_score": 4.2,
      "wcag_compliance": "AA"
    }
  },
  "generated_documents": {
    "html5": {
      "type": "text",
      "data": "<html>...</html>",
      "size_chars": 15420
    },
    "pdf_ua": {
      "type": "binary", 
      "data": "base64-encoded-pdf-data",
      "size_bytes": 245760
    }
  }
}
```

### Document Analysis Only

Get AI-powered accessibility analysis without document generation:

```bash
curl -X POST "http://localhost:8000/api/v1/analyze" \
     -H "Content-Type: application/json"
```

### Basic Decomposition

Extract document elements without AI processing:

```bash
curl -X POST "http://localhost:8000/api/v1/decompose" \
     -H "Content-Type: application/json"
```

## DRR Pipeline Architecture

The Aegis-A11y system implements a three-phase **DRR (Decomposition-Reasoning-Reconstruction)** pipeline:

### Phase 1: Document Decomposition
- **LayoutLMv3** model for layout understanding
- **OCR processing** with text extraction
- **Element classification** (headings, paragraphs, figures, equations, tables)
- **Spatial relationship** analysis

### Phase 2: Semantic Reasoning  
- **Multi-modal AI analysis** using OpenAI GPT-4o
- **Subject area detection** (mathematics, physics, chemistry, etc.)
- **Pedagogical alt-text generation** following UDL principles
- **WCAG 2.1 AA verification** with automatic corrections

### Phase 3: Document Reconstruction
- **JSON-to-tag mapping** for hierarchical structure
- **HTML5 generation** with accessibility features
- **PDF/UA creation** with tagged structure
- **Navigation generation** and metadata embedding

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key for GPT-4o |
| `LOG_LEVEL` | No | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `API_HOST` | No | API server host (default: localhost) |
| `API_PORT` | No | API server port (default: 8000) |

### Model Configuration

The system uses:
- **LayoutLMv3-base** for document decomposition
- **GPT-4o** for semantic reasoning and alt-text generation
- **Custom verification** rules for WCAG 2.1 AA compliance

## Development

### Adding New Components

1. Create new package in `packages/` directory
2. Add to `pyproject.toml` workspace members
3. Update API integration in `packages/api/main.py`

### Testing

```bash
# Run component tests
uv run python packages/reasoning-agent/tests/test_integration.py

# Test reconstruction pipeline
uv run python packages/reconstruction/test_reconstruction.py

# Test API endpoints
curl -X POST http://localhost:8000/api/v1/decompose
```

### Package Management

```bash
# Add new dependency to specific package
cd packages/reasoning-agent
uv add openai

# Update all dependencies
uv sync

# Add development dependency
uv add --dev pytest
```

## Troubleshooting

### Common Issues

**1. OpenAI API Key Error**
```
ValueError: OpenAI API key required
```
Solution: Ensure `OPENAI_API_KEY` is set in your `.env` file

**2. Model Loading Issues**
```
Failed to initialize LayoutDecomposer
```
Solution: Check internet connection and HuggingFace access

**3. API Timeout**
```
Command timed out after 2m 0.0s
```
Solution: The pipeline processes 600+ elements with AI calls. Increase timeout or process smaller documents.

### Performance Optimization

- **Concurrent processing**: The API processes elements in batches
- **Model caching**: Models are loaded once at startup
- **Memory management**: Large documents may require increased memory allocation

### Debug Logging

Enable detailed logging:

```bash
export LOG_LEVEL=DEBUG
uv run uvicorn packages.api.main:app --reload
```

## Sample Document Formats

The system processes educational PDFs and generates:

- **HTML5**: Fully accessible web documents with ARIA landmarks
- **PDF/UA**: Universally accessible PDF format
- **Structured metadata**: Subject-specific educational annotations

## Code Quality & Linting

The project uses modern Python tooling for consistent code quality:

### Available Commands
```bash
# Lint code (check for issues)
uv run --with ruff ruff check packages/

# Format code with black
uv run --with black black packages/

# Auto-fix linting issues
uv run --with ruff ruff check --fix packages/

# Run type checking
uv run --with mypy mypy packages/

# Complete workflow (format + lint + fix + typecheck)
uv run --with black black packages/ && \
uv run --with ruff ruff check --fix packages/ && \
uv run --with mypy mypy packages/
```

### Tools Used
- **Ruff**: Ultra-fast Python linter and formatter (replaces flake8, isort)
- **Black**: Uncompromising code formatter
- **MyPy**: Static type checking

### Configuration
All linting configuration is centralized in `pyproject.toml` at the project root:
- Line length: 88 characters
- Python target: 3.11+
- Import sorting with first-party package recognition
- Strict type checking with external library ignore patterns

### Pre-commit Workflow
Before committing changes, run:
```bash
uv run --with black black packages/ && \
uv run --with ruff ruff check --fix packages/ && \
uv run --with mypy mypy packages/
```

This ensures code quality and catches issues before they reach CI.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-component`)
3. **Run the complete linting workflow to ensure code quality**
4. Make changes and test thoroughly
5. Submit a pull request

## License



## Related Documentation

- [WCAG 2.1 AA Guidelines](https://www.w3.org/WAI/WCAG21/quickref/)
- [Universal Design for Learning (UDL)](https://www.cast.org/impact/universal-design-for-learning-udl)
- [LayoutLMv3 Model](https://huggingface.co/microsoft/layoutlmv3-base)
- [OpenAI GPT-4o API](https://platform.openai.com/docs/)

---

**Aegis-A11y** - Making educational content accessible for everyone through advanced AI technology.
