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
    
    # Stress testing state
    stress_testing: Optional[Dict[str, Any]]  # Store stress testing workflow state including:
                                            # - status: current workflow status
                                            # - concept: concept to be tested (XXX)
                                            # - method: erasure method (YYY, optional)
                                            # - model: target model name (ZZZ)
                                            # - iteration: current iteration number
                                            # - max_iterations: maximum iterations allowed
                                            # - plan: generated stress testing plan
                                            # - rag_results: results from RAG search
                                            # - test_code: generated testing code
                                            # - execution_results: test execution results
                                            # - evaluation_results: concept detection results
                                            # - concept_resurgence_rate: percentage of concept presence
                                            # - report_content: final report content
                                            # - generated_images: list of generated test images
                                            # - refined_queries: refined search queries for RAG
                                                    # - refined_queries: RAG query results
                                                    # - rag_results: database search results
                                                    # - plan: generated testing plan
                                                    # - generated_code: stress testing code
                                                    # - execution_results: test execution results
                                                    # - evaluation_results: concept detection results
                                                    # - iteration: current iteration number
