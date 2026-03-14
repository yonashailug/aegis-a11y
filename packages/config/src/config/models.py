"""
Configuration data models for all Aegis-A11y components.

These models define the structure and validation rules for different
configuration sections using Pydantic for type safety and validation.
"""

from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator


class LogLevel(str, Enum):
    """Available logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"  
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Environment(str, Enum):
    """Deployment environments."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class ModelProvider(str, Enum):
    """Supported AI model providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"


class APIConfig(BaseModel):
    """API server configuration."""
    host: str = Field(default="localhost", description="API server host")
    port: int = Field(default=8000, ge=1, le=65535, description="API server port")
    workers: int = Field(default=1, ge=1, le=16, description="Number of worker processes")
    reload: bool = Field(default=False, description="Auto-reload on code changes")
    debug: bool = Field(default=False, description="Enable debug mode")
    cors_origins: List[str] = Field(default=["http://localhost:3000"], description="CORS allowed origins")
    request_timeout: int = Field(default=300, ge=30, le=3600, description="Request timeout in seconds")
    max_request_size: int = Field(default=50 * 1024 * 1024, description="Max request size in bytes (50MB)")


class AIModelConfig(BaseModel):
    """Configuration for an AI model provider."""
    provider: ModelProvider = Field(description="AI model provider")
    model_name: str = Field(description="Model name/identifier")
    api_key: Optional[str] = Field(default=None, description="API key for the provider")
    
    # Generation parameters
    max_tokens: int = Field(default=1000, ge=100, le=4000, description="Max tokens per request")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Temperature setting for generation")
    timeout: int = Field(default=60, ge=10, le=300, description="API timeout in seconds")
    
    # Provider-specific settings (stored as flexible dict)
    provider_settings: Dict[str, Union[str, int, float, bool]] = Field(
        default_factory=dict,
        description="Provider-specific configuration options"
    )
    
    @validator('api_key')
    def validate_api_key(cls, v, values):
        """Validate API key format based on provider."""
        if not v:
            return v
            
        provider = values.get('provider')
        if provider == ModelProvider.OPENAI:
            if not v.startswith(('sk-', 'org-')):
                raise ValueError("Invalid OpenAI API key format")
        elif provider == ModelProvider.ANTHROPIC:
            if not v.startswith('sk-ant-'):
                raise ValueError("Invalid Anthropic API key format")
        elif provider == ModelProvider.AZURE_OPENAI:
            # Azure OpenAI uses different key format - just ensure it's not empty
            if len(v) < 10:
                raise ValueError("Azure OpenAI API key too short")
        
        return v


class ModelConfig(BaseModel):
    """ML model configuration."""
    # Document processing models
    layoutlm_model: str = Field(default="microsoft/layoutlmv3-base", description="LayoutLM model identifier")
    
    # AI models for different tasks
    text_generation: AIModelConfig = Field(
        default_factory=lambda: AIModelConfig(
            provider=ModelProvider.OPENAI,
            model_name="gpt-4o",
            max_tokens=1000,
            temperature=0.7
        ),
        description="AI model for text generation tasks"
    )
    
    reasoning: AIModelConfig = Field(
        default_factory=lambda: AIModelConfig(
            provider=ModelProvider.OPENAI,
            model_name="gpt-4o",
            max_tokens=2000,
            temperature=0.3  # Lower temperature for reasoning
        ),
        description="AI model for reasoning and analysis tasks"
    )
    
    # Model caching
    cache_models: bool = Field(default=True, description="Cache loaded models in memory")
    cache_dir: Optional[Path] = Field(default=None, description="Model cache directory")
    
    # Default provider (for backward compatibility)
    default_provider: ModelProvider = Field(default=ModelProvider.OPENAI, description="Default AI model provider")
    
    @validator('text_generation', 'reasoning', pre=True)
    def ensure_ai_model_config(cls, v):
        """Ensure AI model configs are properly instantiated."""
        if isinstance(v, dict):
            return AIModelConfig(**v)
        return v


class ProcessingConfig(BaseModel):
    """Document processing configuration."""
    max_pdf_size: int = Field(default=100 * 1024 * 1024, description="Max PDF size in bytes (100MB)")
    max_pages: int = Field(default=50, ge=1, le=500, description="Maximum pages to process per PDF")
    pdf_dpi: int = Field(default=200, ge=72, le=300, description="PDF to image conversion DPI")
    
    # Element filtering
    enable_filtering: bool = Field(default=True, description="Enable intelligent element filtering")
    min_text_length: int = Field(default=3, ge=1, le=20, description="Minimum text length for processing")
    filter_aggressiveness: float = Field(default=0.8, ge=0.0, le=1.0, description="Filtering aggressiveness (0=none, 1=max)")
    
    # Performance
    max_concurrent_elements: int = Field(default=10, ge=1, le=50, description="Max concurrent element processing")
    enable_parallel_pages: bool = Field(default=True, description="Enable parallel page processing")
    
    # Batch processing
    enable_batch_processing: bool = Field(default=True, description="Enable batch processing of multiple PDFs")
    max_batch_size: int = Field(default=5, ge=1, le=20, description="Maximum number of PDFs in a single batch")
    max_concurrent_pdfs: int = Field(default=2, ge=1, le=8, description="Max concurrent PDF processing in batch")
    batch_timeout: int = Field(default=1800, ge=300, le=7200, description="Batch processing timeout in seconds (30min)")
    
    # Quality thresholds
    min_confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0, description="Minimum confidence for output")
    require_verification: bool = Field(default=True, description="Require verification of outputs")


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: LogLevel = Field(default=LogLevel.INFO, description="Global log level")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format"
    )
    file_enabled: bool = Field(default=False, description="Enable file logging")
    file_path: Optional[Path] = Field(default=None, description="Log file path")
    file_max_size: int = Field(default=10 * 1024 * 1024, description="Max log file size in bytes (10MB)")
    file_backup_count: int = Field(default=5, ge=1, le=20, description="Number of backup log files")
    
    # Component-specific levels
    component_levels: Dict[str, LogLevel] = Field(
        default_factory=lambda: {
            "cv_layer": LogLevel.INFO,
            "reasoning_agent": LogLevel.INFO, 
            "reconstruction": LogLevel.INFO,
            "api": LogLevel.INFO,
        },
        description="Per-component log levels"
    )


class SecurityConfig(BaseModel):
    """Security configuration."""
    # API keys for different providers (backward compatibility)
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key (deprecated - use model configs)")
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key")
    huggingface_api_token: Optional[str] = Field(default=None, description="HuggingFace API token")
    
    # General API keys storage (flexible for any provider)
    api_keys: Dict[str, str] = Field(default_factory=dict, description="API keys for various providers")
    
    # Application security
    api_key_required: bool = Field(default=False, description="Require API key for requests")
    allowed_file_types: List[str] = Field(
        default=["pdf"], 
        description="Allowed file upload types"
    )
    max_upload_size: int = Field(default=50 * 1024 * 1024, description="Max upload size in bytes")
    
    @validator('openai_api_key')
    def validate_openai_key(cls, v):
        """Validate OpenAI API key format (backward compatibility)."""
        if v and not v.startswith(('sk-', 'org-')):
            raise ValueError("Invalid OpenAI API key format")
        return v
    
    @validator('anthropic_api_key')
    def validate_anthropic_key(cls, v):
        """Validate Anthropic API key format."""
        if v and not v.startswith('sk-ant-'):
            raise ValueError("Invalid Anthropic API key format")
        return v


class OutputConfig(BaseModel):
    """Output and storage configuration."""
    output_dir: Path = Field(default=Path("generated_documents"), description="Output directory for documents")
    create_timestamped_dirs: bool = Field(default=True, description="Create timestamped subdirectories")
    keep_intermediate_files: bool = Field(default=False, description="Keep intermediate processing files")
    
    # Output formats
    enable_html5: bool = Field(default=True, description="Generate HTML5 output")
    enable_pdf_ua: bool = Field(default=True, description="Generate PDF/UA output")
    enable_json_metadata: bool = Field(default=True, description="Generate JSON metadata")
    
    # File naming
    filename_template: str = Field(
        default="{basename}_accessible.{extension}",
        description="Output filename template"
    )
    
    @validator('output_dir')
    def ensure_absolute_path(cls, v):
        """Ensure output directory is absolute."""
        return v.resolve() if not v.is_absolute() else v


class PerformanceConfig(BaseModel):
    """Performance and resource configuration."""
    max_memory_usage: int = Field(default=2 * 1024 * 1024 * 1024, description="Max memory usage in bytes (2GB)")
    enable_gpu: bool = Field(default=False, description="Enable GPU acceleration if available")
    max_worker_threads: int = Field(default=4, ge=1, le=16, description="Max worker threads")
    
    # Caching
    enable_result_cache: bool = Field(default=True, description="Cache processing results")
    cache_ttl: int = Field(default=3600, ge=300, le=86400, description="Cache TTL in seconds")
    max_cache_size: int = Field(default=1000, ge=10, le=10000, description="Max cache entries")