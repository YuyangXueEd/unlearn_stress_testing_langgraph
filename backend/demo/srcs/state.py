"""
Demo State Management

Simplified state management for the demo chatbot.
"""

from typing import TypedDict, List, Dict, Any, Optional
from typing_extensions import Annotated
from langgraph.graph import add_messages


class ChatState(TypedDict):
    """Enhanced state for chat conversations with memory and routing support."""
    messages: Annotated[List, add_messages]  # This maintains conversation history
    user_message: str
    response: str
    model_name: str
    max_response_length: int
    tool_result: Optional[Dict[str, Any]]  # Store results from tool execution
    task_type: Optional[str]  # Type of task: 'conversation' or 'image_generation'
