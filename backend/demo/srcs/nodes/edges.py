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
    Set up conditional edges for routing between conversation and image generation.
    
    This implements the new architecture:
    START -> router -> (conversation OR image_generation) -> END
    
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
            "image_generation": "image_generation"
        }
    )
    
    # Both conversation and image generation end the flow
    builder.add_edge("conversation", END)
    builder.add_edge("image_generation", END)


def determine_task_route(state):
    """
    Determine whether to route to conversation or image generation based on task type.
    
    Args:
        state: Current chat state containing task_type
        
    Returns:
        String indicating the next node to visit: 'conversation' or 'image_generation'
    """
    task_type = state.get("task_type", "conversation")
    
    # Default to conversation if task_type is not set or unknown
    if task_type not in ["conversation", "image_generation"]:
        return "conversation"
    
    return task_type
