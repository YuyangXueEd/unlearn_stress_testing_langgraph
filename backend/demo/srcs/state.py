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
                                            # - concept: concept to be tested (extracted from user input)
                                            # - method: erasure method (used in queries, not stored as state)
                                            # - model: target model name (extracted from user input)
                                            # - iteration: current iteration number
                                            # - max_iterations: maximum iterations allowed
                                            # - plan: generated stress testing plan
                                            # - plan_generated_at: timestamp when plan was generated
                                            # - refined_queries: refined search queries for RAG
                                            # - rag_results: aggregated results from RAG search
                                            # - generated_code: generated testing code
                                            # - code_filename: filename where code was saved
                                            # - code_generated_at: timestamp when code was generated  
                                            # - code_attempt: current code generation attempt
                                            # - previous_errors: list of previous error messages
                                            # - execution_result: test execution results object
                                            # - execution_output: execution output text
                                            # - execution_analysis: LLM analysis of execution results
                                            # - generated_images: list of generated test images with metadata
                                            # - image_count: number of generated images
                                            # - evaluation_code: generated evaluation code
                                            # - evaluation_result: raw evaluation execution result
                                            # - evaluation_output: evaluation execution output text
                                            # - evaluation_method: evaluation methodology from plan
                                            # - evaluation_assessment: LLM assessment of results
                                            # - evaluation_error: error message if evaluation fails
                                            # - concept_resurgence_rate: percentage of concept presence
                                            # - error: general error field for various error states
                                            # - report_content: final report content
