"""
Configuration management for the Gemini agent.
Handles environment variables and LangSmith setup.
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class Config:
    """Configuration for the Gemini agent."""
    
    def __init__(self):
        # API Configuration
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.api_base_url = os.getenv("API_BASE_URL", "https://agents-course-unit4-scoring.hf.space")
        
        # LangSmith Configuration
        self.langsmith_tracing = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
        self.langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
        self.langsmith_endpoint = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
        self.langsmith_project = os.getenv("LANGSMITH_PROJECT", "hf-agent-cert")
        
        # Model Configuration
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.temperature = float(os.getenv("TEMPERATURE", "0.1"))
        self.max_output_tokens = int(os.getenv("MAX_OUTPUT_TOKENS", "4096"))

        # Execution Guards
        # Maximum number of agent-tool iterations before forcing finalization
        self.max_iterations = int(os.getenv("MAX_ITERATIONS", "6"))
        # Recursion/call-depth limit for the runnable/graph
        self.recursion_limit = int(os.getenv("RECURSION_LIMIT", "25"))
        
        # Logging Configuration
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        
    def setup_langsmith(self):
        """Configure LangSmith if enabled."""
        if self.langsmith_tracing and self.langsmith_api_key:
            # Set environment variables for LangChain
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = self.langsmith_api_key
            os.environ["LANGCHAIN_ENDPOINT"] = self.langsmith_endpoint
            os.environ["LANGCHAIN_PROJECT"] = self.langsmith_project
            
            logger.info(f"LangSmith tracing enabled for project: {self.langsmith_project}")
            return True
        else:
            if self.langsmith_tracing and not self.langsmith_api_key:
                logger.warning("LangSmith tracing requested but LANGSMITH_API_KEY not set")
            return False
    
    def validate(self) -> bool:
        """Validate required configuration."""
        if not self.gemini_api_key:
            logger.error("GEMINI_API_KEY is required but not set")
            return False
        return True
    
    def __repr__(self):
        """String representation for debugging."""
        return (
            f"Config(api_base_url={self.api_base_url}, "
            f"model={self.model_name}, "
            f"langsmith_enabled={self.langsmith_tracing}, "
            f"project={self.langsmith_project})"
        )
