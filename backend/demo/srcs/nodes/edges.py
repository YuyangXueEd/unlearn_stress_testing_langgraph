"""
Graph Edges

Functions for defining edges and conditional routing in the LangGraph.
"""

from langgraph.graph import END, START
from langchain_core.messages import AIMessage


def setup_basic_edges(builder):
    """
    Set up basic edges for the simple demo graph (legacy - keeping for compatibility).
    
    This creates a linear flow: START -> chat -> END
    
    Args:
        builder: StateGraph builder instance
    """
    # Set up the linear flow
    builder.add_edge(START, "chat")
    builder.add_edge("chat", END)


def setup_conditional_edges(builder):
    """
    Set up conditional edges for routing between conversation, image generation, code generation, and database search.
    
    This implements the architecture:
    START -> router -> (conversation OR image_generation OR code_generation OR database_search) -> END
    
    Code generation follows 3-node pattern from PDF:
    code_generation -> generate -> execute_and_check_code -> decide_to_finish -> (generate OR END)
    
    Args:
        builder: StateGraph builder instance
    """
    # Start with the router
    builder.add_edge(START, "router")
    
    # Add conditional routing based on task type
    builder.add_conditional_edges(
        "router",
        determine_task_route,
        {
            "conversation": "conversation",
            "image_generation": "image_generation", 
            "code_generation": "code_generation",
            "database_search": "database_search"
        }
    )
    
    # Code generation workflow (3-node pattern from PDF)
    builder.add_edge("code_generation", "generate")  # Entry -> Generate
    builder.add_edge("generate", "execute_and_check_code")  # Generate -> Execute
    builder.add_conditional_edges(
        "execute_and_check_code",
        determine_execute_next,
        {
            "decide": "decide_to_finish"
        }
    )
    builder.add_conditional_edges(
        "decide_to_finish", 
        determine_code_workflow_next,
        {
            "generate": "generate",  # Iterate: go back to generate
            "end": END              # Finish: go to end
        }
    )
    
    # Database search flows to reflection for analysis
    builder.add_edge("database_search", "reflection")
    
    # Reflection can route to either more database search or final answer
    builder.add_conditional_edges(
        "reflection",
        determine_reflection_route,
        {
            "database_search": "database_search",
            "final_answer": "final_answer"
        }
    )
    
    # End nodes
    builder.add_edge("conversation", END)
    builder.add_edge("image_generation", END)
    builder.add_edge("final_answer", END)


def determine_task_route(state):
    """
    Determine whether to route to conversation, image generation, code generation, or database search based on task type.
    
    Args:
        state: Current chat state containing task_type
        
    Returns:
        String indicating the next node to visit: 'conversation', 'image_generation', 'code_generation', or 'database_search'
    """
    task_type = state.get("task_type", "conversation")
    
    # Default to conversation if task_type is not set or unknown
    if task_type not in ["conversation", "image_generation", "code_generation", "database_search"]:
        return "conversation"
    
    return task_type


def determine_reflection_route(state):
    """
    Determine whether to route to another database search or final answer based on reflection.
    
    Args:
        state: Current chat state containing reflection results
        
    Returns:
        String indicating the next node: 'database_search' or 'final_answer'
    """
    task_type = state.get("task_type", "final_answer")
    
    # Only allow database_search or final_answer
    if task_type in ["database_search", "final_answer"]:
        return task_type
    
    # Default to final_answer for safety
    return "final_answer"


def determine_execute_next(state):
    """
    Determine next step after code execution - always go to decide_to_finish.
    
    Args:
        state: Current chat state
        
    Returns:
        String indicating next node: 'decide'
    """
    return "decide"


def determine_code_workflow_next(state):
    """
    Determine whether to iterate (generate again) or finish based on decide_to_finish results.
    
    Args:
        state: Current chat state containing code generation results
        
    Returns:
        String indicating next node: 'generate' or 'end'
    """
    code_gen = state.get("code_generation", {})
    status = code_gen.get('status', '')
    
    # Continue if we need to refine
    if status == 'refining':
        return "generate"
    
    # Finish if completed or failed
    if status in ['completed', 'failed']:
        return "end"
    
    # Default: continue if status is unclear
    execution_result = code_gen.get('execution_result', {})
    max_attempts = code_gen.get('max_attempts', 3)
    attempt = code_gen.get('attempt', 1)
    
    # Continue if we have errors and haven't reached max attempts
    if (execution_result.get('error') or execution_result.get('status') != 'success') and attempt < max_attempts:
        return "generate"
    
    # Otherwise, we're done
    return "end"
