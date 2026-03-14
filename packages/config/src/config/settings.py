"""
Main settings module for Aegis-A11y.

Provides centralized configuration management with environment variable support,
validation, and easy access patterns for all application components.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .models import (
    AIModelConfig,
    APIConfig,
    Environment,
    LoggingConfig,
    ModelConfig,
    ModelProvider,
    OutputConfig,
    PerformanceConfig,
    ProcessingConfig,
    SecurityConfig,
)


class Settings(BaseSettings):
    """
    Main application settings with environment variable support.
    
    This class combines all configuration sections and provides environment
    variable overrides using the AEGIS_ prefix.
    """
    
    model_config = SettingsConfigDict(
        env_prefix="AEGIS_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Allow extra fields that we'll handle manually
        env_nested_delimiter="__",
    )
    
    # Environment and basic settings
    environment: Environment = Field(default=Environment.DEVELOPMENT, description="Deployment environment")
    project_name: str = Field(default="Aegis-A11y", description="Project name")
    version: str = Field(default="0.1.0", description="Application version")
    debug: bool = Field(default=False, description="Global debug mode")
    
    # Configuration sections
    api: APIConfig = Field(default_factory=APIConfig, description="API configuration")
    models: ModelConfig = Field(default_factory=ModelConfig, description="ML model configuration") 
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig, description="Processing configuration")
    logging: LoggingConfig = Field(default_factory=LoggingConfig, description="Logging configuration")
    security: SecurityConfig = Field(default_factory=SecurityConfig, description="Security configuration")
    output: OutputConfig = Field(default_factory=OutputConfig, description="Output configuration")
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig, description="Performance configuration")
    
    def __init__(self, **kwargs):
        """Initialize settings with environment-specific defaults."""
        # Handle direct openai_api_key parameter by moving it to security section
        if 'openai_api_key' in kwargs:
            openai_key = kwargs.pop('openai_api_key')
            if 'security' not in kwargs:
                kwargs['security'] = {}
            if isinstance(kwargs['security'], dict):
                kwargs['security']['openai_api_key'] = openai_key
                
        super().__init__(**kwargs)
        self._apply_environment_defaults()
        self._validate_dependencies()
    
    def _apply_environment_defaults(self):
        """Apply environment-specific default values."""
        if self.environment == Environment.PRODUCTION:
            # Production defaults
            self.debug = False
            self.api.debug = False
            self.api.reload = False
            self.logging.level = "INFO"
            self.processing.enable_filtering = True
            self.performance.enable_result_cache = True
            
        elif self.environment == Environment.DEVELOPMENT:
            # Development defaults  
            self.debug = True
            self.api.debug = True
            self.api.reload = True
            self.logging.level = "DEBUG"
            self.output.keep_intermediate_files = True
            
        elif self.environment == Environment.TESTING:
            # Testing defaults
            self.debug = True
            self.logging.level = "WARNING"
            self.processing.max_pages = 5  # Limit for faster tests
            self.models.cache_models = False  # Don't cache in tests
            self.performance.enable_result_cache = False
    
    def _validate_dependencies(self):
        """Validate configuration dependencies and constraints."""
        self._setup_ai_model_keys()
        self._setup_backward_compatibility()
        
        # Ensure output directory exists
        self.output.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate performance constraints
        if self.processing.max_concurrent_elements > self.performance.max_worker_threads:
            self.processing.max_concurrent_elements = self.performance.max_worker_threads
    
    def _setup_ai_model_keys(self):
        """Setup API keys for AI models from environment variables."""
        # Get API keys from environment variables
        api_keys = {
            "openai": os.getenv("OPENAI_API_KEY") or os.getenv("AEGIS_SECURITY__OPENAI_API_KEY"),
            "anthropic": os.getenv("ANTHROPIC_API_KEY") or os.getenv("AEGIS_SECURITY__ANTHROPIC_API_KEY"),
            "azure_openai": os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AEGIS_SECURITY__AZURE_OPENAI_API_KEY"),
            "huggingface": os.getenv("HUGGINGFACE_API_TOKEN") or os.getenv("AEGIS_SECURITY__HUGGINGFACE_API_TOKEN"),
        }
        
        # Update AI model configurations with API keys
        for model_config in [self.models.text_generation, self.models.reasoning]:
            provider_key = api_keys.get(model_config.provider.value)
            if provider_key and not model_config.api_key:
                model_config.api_key = provider_key
        
        # Store keys in security config for backward compatibility and general access
        for provider, key in api_keys.items():
            if key:
                self.security.api_keys[provider] = key
                
        # Set backward compatibility fields
        if api_keys["openai"] and not self.security.openai_api_key:
            self.security.openai_api_key = api_keys["openai"]
        if api_keys["anthropic"] and not self.security.anthropic_api_key:
            self.security.anthropic_api_key = api_keys["anthropic"]
        if api_keys["huggingface"] and not self.security.huggingface_api_token:
            self.security.huggingface_api_token = api_keys["huggingface"]
    
    def _setup_backward_compatibility(self):
        """Ensure backward compatibility with old OpenAI-specific configuration."""
        # If using old-style configuration, migrate to new structure
        if self.security.openai_api_key:
            # Update model configurations if they don't have keys
            for model_config in [self.models.text_generation, self.models.reasoning]:
                if model_config.provider.value == "openai" and not model_config.api_key:
                    model_config.api_key = self.security.openai_api_key
        
        # Validate that required models have API keys in production
        if self.environment == Environment.PRODUCTION:
            missing_keys = []
            for name, model_config in [("text_generation", self.models.text_generation), ("reasoning", self.models.reasoning)]:
                if model_config.provider.value in ["openai", "anthropic", "azure_openai"] and not model_config.api_key:
                    missing_keys.append(f"{name} ({model_config.provider.value})")
            
            if missing_keys:
                raise ValueError(f"API keys required in production for models: {', '.join(missing_keys)}")
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == Environment.PRODUCTION
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == Environment.DEVELOPMENT
    
    @property  
    def is_testing(self) -> bool:
        """Check if running in testing environment."""
        return self.environment == Environment.TESTING
    
    def get_openai_api_key(self) -> Optional[str]:
        """Get OpenAI API key with fallback to environment variable (deprecated - use get_api_key)."""
        return self.get_api_key("openai")
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for a specific provider."""
        # Check model configurations first
        for model_config in [self.models.text_generation, self.models.reasoning]:
            if model_config.provider.value == provider and model_config.api_key:
                return model_config.api_key
        
        # Check general security storage
        if provider in self.security.api_keys:
            return self.security.api_keys[provider]
            
        # Check specific security fields for backward compatibility
        if provider == "openai":
            return self.security.openai_api_key
        elif provider == "anthropic":
            return self.security.anthropic_api_key
        elif provider == "huggingface":
            return self.security.huggingface_api_token
        
        return None
    
    def get_model_config(self, task: str = "text_generation") -> "AIModelConfig":
        """Get AI model configuration for a specific task."""
        if task == "text_generation":
            return self.models.text_generation
        elif task == "reasoning":
            return self.models.reasoning
        else:
            raise ValueError(f"Unknown task: {task}. Available: text_generation, reasoning")
    
    def create_model_config(self, provider: str, model_name: str, **kwargs) -> "AIModelConfig":
        """Create a new AI model configuration."""
        from .models import AIModelConfig, ModelProvider
        
        try:
            provider_enum = ModelProvider(provider)
        except ValueError:
            raise ValueError(f"Unsupported provider: {provider}. Available: {[p.value for p in ModelProvider]}")
        
        return AIModelConfig(
            provider=provider_enum,
            model_name=model_name,
            **kwargs
        )
    
    def get_log_level_for_component(self, component: str) -> str:
        """Get log level for specific component."""
        return self.logging.component_levels.get(component, self.logging.level).value
    
    def update_from_dict(self, config_dict: dict) -> None:
        """Update settings from dictionary (useful for runtime configuration)."""
        for key, value in config_dict.items():
            if hasattr(self, key):
                if isinstance(getattr(self, key), BaseSettings):
                    # Update nested configuration
                    getattr(self, key).update(value)
                else:
                    setattr(self, key, value)
    
    def to_dict(self) -> dict:
        """Export settings to dictionary."""
        return self.model_dump()
    
    def export_env_template(self, file_path: Optional[Path] = None) -> str:
        """Export environment variable template."""
        template_lines = [
            "# Aegis-A11y Environment Configuration",
            "# Copy this file to .env and update values as needed",
            "",
            "# Environment (development, staging, production, testing)",
            "AEGIS_ENVIRONMENT=development",
            "",
            "# OpenAI Configuration", 
            "OPENAI_API_KEY=your_openai_api_key_here",
            "",
            "# API Configuration",
            "AEGIS_API__HOST=localhost",
            "AEGIS_API__PORT=8000",
            "AEGIS_API__DEBUG=true",
            "",
            "# Logging Configuration",
            "AEGIS_LOGGING__LEVEL=INFO",
            "AEGIS_LOGGING__FILE_ENABLED=false",
            "",
            "# Processing Configuration", 
            "AEGIS_PROCESSING__MAX_PAGES=50",
            "AEGIS_PROCESSING__ENABLE_FILTERING=true",
            "AEGIS_PROCESSING__MIN_CONFIDENCE_THRESHOLD=0.6",
            "",
            "# Model Configuration",
            "AEGIS_MODELS__OPENAI_MODEL=gpt-4o",
            "AEGIS_MODELS__OPENAI_MAX_TOKENS=1000",
            "AEGIS_MODELS__CACHE_MODELS=true",
            "",
            "# Output Configuration",
            "AEGIS_OUTPUT__OUTPUT_DIR=./generated_documents",
            "AEGIS_OUTPUT__ENABLE_HTML5=true",
            "AEGIS_OUTPUT__ENABLE_PDF_UA=true",
            "",
            "# Performance Configuration",
            "AEGIS_PERFORMANCE__ENABLE_RESULT_CACHE=true",
            "AEGIS_PERFORMANCE__MAX_WORKER_THREADS=4",
        ]
        
        template_content = "\n".join(template_lines)
        
        if file_path:
            file_path.write_text(template_content)
        
        return template_content


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    This function provides a singleton pattern for settings access
    across the application with automatic caching.
    """
    return Settings()


def reload_settings() -> Settings:
    """
    Reload settings (useful for testing or runtime configuration changes).
    
    Clears the cache and creates a new settings instance.
    """
    get_settings.cache_clear()
    return get_settings()


def create_test_settings(**overrides) -> Settings:
    """
    Create settings instance for testing with optional overrides.
    
    Args:
        **overrides: Configuration overrides for testing
        
    Returns:
        Settings instance configured for testing
    """
    test_config = {
        "environment": Environment.TESTING,
        "debug": True,
        "api": {"debug": True, "reload": False},
        "logging": {"level": "WARNING"},
        "processing": {"max_pages": 5, "enable_filtering": False},
        "models": {"cache_models": False},
        "performance": {"enable_result_cache": False},
        **overrides
    }
    
    return Settings(**test_config)