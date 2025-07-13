"""
Demo Configuration

Simple configuration for the demo chatbot.
"""

from pydantic import BaseModel, Field


class DemoConfiguration(BaseModel):
    """Configuration for the demo chatbot."""
    
    # Model settings
    model_name: str = Field(
        default="gemma3",
        description="The name of the language model to use"
    )
    
    # Response settings
    max_response_length: int = Field(
        default=2000,
        description="Maximum length of responses"
    )
    
    # Database search settings
    max_database_search_iterations: int = Field(
        default=3,
        description="Maximum number of database search iterations before finalizing answer"
    )
