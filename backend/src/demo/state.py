"""
Demo State Management

Simplified state management for the demo chatbot.
"""

from typing import TypedDict, List, Dict
from typing_extensions import Annotated
from langgraph.graph import add_messages


class ChatState(TypedDict):
    """Simple state for chat conversations with memory."""
    messages: Annotated[List, add_messages]  # This maintains conversation history
    user_message: str
    response: str
    model_name: str
    max_response_length: int
