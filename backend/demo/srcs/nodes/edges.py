"""
Graph Edges

Functions for defining edges and conditional routing in the LangGraph.
"""

from langgraph.graph import END, START


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
    Set up conditional edges for routing between conversation, image generation, and database search with reflection.
    
    This implements the architecture:
    START -> router -> (conversation OR image_generation OR code_generation OR database_search) -> END
                                                                                        ↓
                                                                                 reflection -> (database_search OR final_answer) -> END
    
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
    builder.add_edge("code_generation", END)
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
