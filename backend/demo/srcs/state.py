"""
Demo State Management

Simplified state management for the demo chatbot.
"""

from typing import TypedDict, List, Dict, Any, Optional
from typing_extensions import Annotated
from langgraph.graph import add_messages


class ChatState(TypedDict):
    """Enhanced state for chat conversations with memory, routing, database support, and code generation."""
    messages: Annotated[List, add_messages]  # This maintains conversation history
    user_message: str
    response: str
    model_name: str
    max_response_length: int
    tool_result: Optional[Dict[str, Any]]  # Store results from tool execution (images, code, etc.)
    task_type: Optional[str]  # Type of task: 'conversation', 'image_generation', 'code_generation', or 'database_search'
    search_results: Optional[Dict[str, Any]]  # Store database search results
    search_iteration: int  # Track number of search iterations (max 3)
    previous_queries: List[str]  # Track previous search queries to avoid repetition
    reflection_result: Optional[Dict[str, Any]]  # Store reflection analysis results
    
    # Code generation state
    code_generation: Optional[Dict[str, Any]]  # Store code generation workflow state including:
                                             # - status: current workflow status
                                             # - attempt: current attempt number
                                             # - max_attempts: maximum attempts allowed
                                             # - requirements: extracted user requirements
                                             # - code: generated code
                                             # - explanation: code explanation
                                             # - execution_result: code execution results
                                             # - validation_result: code validation results
                                             # - error: any error messages
                                             # - error_context: context for iteration feedback
