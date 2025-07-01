import os
from typing import Any, Optional
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig


class Configuration(BaseModel):
    """The configuration for the agent."""

    query_generator_model: str = Field(
        default="qwen3",
        metadata={
            "description": "The name of the language model to use for the agent's query generation."
        },
    )

    reasoning_model: str = Field(
        default="qwen3",
        metadata={
            "description": "The name of the language model to use for the agent's reasoning."
        },
    )

    reflection_model: str = Field(
        default="qwen3",
        metadata={
            "description": "The name of the language model to use for the agent's reflection."
        },
    )

    answer_model: str = Field(
        default="qwen3",
        metadata={
            "description": "The name of the language model to use for the agent's answer."
        },
    )

    number_of_initial_queries: int = Field(
        default=3,
        metadata={"description": "The number of initial search queries to generate."},
    )

    max_research_loops: int = Field(
        default=2,
        metadata={"description": "The maximum number of research loops to perform."},
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None, base_model: Optional[str] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )

        # Get raw values from environment or config
        raw_values: dict[str, Any] = {
            name: os.environ.get(name.upper(), configurable.get(name))
            for name in cls.model_fields.keys()
        }

        # Filter out None values
        values = {k: v for k, v in raw_values.items() if v is not None}

        # If base_model is provided, set all fields to it unless already set
        model_fields = [name for name in cls.model_fields.keys() if name.endswith("_model")]
        if base_model:
            for field in model_fields:
                if field not in values:
                    values[field] = base_model

        return cls(**values)
