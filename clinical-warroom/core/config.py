"""
Clinical War Room - Configuration Management

Centralized environment settings and configuration.
All configuration is loaded from environment variables with safe defaults.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass(frozen=True)
class MCPConfig:
    """MCP Tool configuration."""
    tool_timeout_seconds: int = 30
    max_retries: int = 3
    strict_schema_validation: bool = True


@dataclass(frozen=True)
class RAGConfig:
    """RAG retrieval configuration."""
    collection_name: str = "clinical_guidelines"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 5
    similarity_threshold: float = 0.7


@dataclass(frozen=True)
class AgentConfig:
    """LLM Agent configuration."""
    model_name: str = "llama-3.3-70b-versatile"
    temperature: float = 0.3
    max_tokens: int = 2048
    timeout_seconds: int = 60


@dataclass(frozen=True)
class SafetyConfig:
    """Hard safety thresholds - CANNOT be overridden."""
    min_confidence_threshold: float = 0.6
    max_risk_threshold: float = 0.7
    ethics_veto_enabled: bool = True
    require_unanimous_for_execute: bool = False
    min_agents_for_decision: int = 3


@dataclass(frozen=True)
class APIConfig:
    """API server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    cors_origins: list = field(default_factory=lambda: ["*"])


@dataclass
class Settings:
    """
    Master configuration container.
    
    All settings are immutable after initialization to prevent
    runtime configuration changes that could affect safety.
    """
    
    # Environment
    environment: str = field(default_factory=lambda: os.getenv("ENVIRONMENT", "development"))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    
    # API Keys
    groq_api_key: Optional[str] = field(default_factory=lambda: os.getenv("GROQ_API_KEY"))
    
    # Paths
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data")
    logs_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "logs")
    
    # Sub-configurations
    mcp: MCPConfig = field(default_factory=MCPConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    api: APIConfig = field(default_factory=APIConfig)
    
    def validate(self) -> None:
        """Validate critical configuration."""
        if self.environment == "production" and not self.groq_api_key:
            raise ValueError("GROQ_API_KEY is required in production")
        
        # Ensure directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def is_production(self) -> bool:
        return self.environment == "production"
    
    @property
    def is_development(self) -> bool:
        return self.environment == "development"


# Global settings instance
settings = Settings()
