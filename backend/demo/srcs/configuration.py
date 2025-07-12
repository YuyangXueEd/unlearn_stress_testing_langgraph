"""
Demo Configuration

Simple configuration for the demo chatbot.
"""

from pydantic import BaseModel, Field


class DemoConfiguration(BaseModel):
    """Configuration for the demo chatbot."""
    
    # Model settings
    model_name: str = Field(
        default="qwen3",
        description="The name of the language model to use"
    )
    
    # Response settings
    max_response_length: int = Field(
        default=2000,
        description="Maximum length of responses"
    )
