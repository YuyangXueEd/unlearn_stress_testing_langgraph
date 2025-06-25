import os
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

MODEL_DEFAULTS = {
    "qwen3": {
        "query_generator_model": "qwen3",
        "reasoning_model": "qwen3",
        "reflection_model": "qwen3",
        "answer_model": "qwen3",
    },
    "gemma3": {
        "query_generator_model": "gemma3",
        "reasoning_model": "gemma3",
        "reflection_model": "gemma3",
        "answer_model": "gemma3",
    },
}


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

        # Try to get reasoning_model from base_model, environment, or configurable
        reasoning_model = (
            base_model
            or os.environ.get("REASONING_MODEL")
            or (configurable.get("reasoning_model"))
        )

        # Use mapping if reasoning_model is present
        model_defaults = MODEL_DEFAULTS.get(reasoning_model, {})

        # Get raw values from environment, configurable, or model_defaults
        raw_values: dict[str, Any] = {
            name: (
                os.environ.get(name.upper())
                or configurable.get(name)
                or model_defaults.get(name)
            )
            for name in cls.model_fields.keys()
        }

        # Filter out None values
        values = {k: v for k, v in raw_values.items() if v is not None}

        return cls(**values)
