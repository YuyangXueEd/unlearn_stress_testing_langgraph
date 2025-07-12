"""
Graph Builder

Functions for constructing and configuring the LangGraph.
"""

from langgraph.graph import StateGraph

from state import ChatState
from nodes.chat_nodes import chat_node
from nodes.image_nodes import image_generation_node
from nodes.routing_nodes import router_node
from nodes.edges import setup_conditional_edges


def create_demo_graph():
    """
    Create and configure the demo graph with conditional routing.
    
    This function:
    1. Creates a StateGraph with ChatState
    2. Adds router, conversation, and image generation nodes
    3. Sets up conditional edges for routing
    4. Compiles and returns the graph
    
    Architecture:
    START -> router -> (conversation OR image_generation) -> END
    
    Returns:
        Compiled LangGraph ready for execution
    """
    # Create the graph builder
    builder = StateGraph(ChatState)
    
    # Add nodes
    _add_nodes(builder)
    
    # Set up conditional edges
    _setup_edges(builder)
    
    # Compile and return the graph
    return builder.compile()


def _add_nodes(builder):
    """
    Add all nodes to the graph builder.
    
    Args:
        builder: StateGraph builder instance
    """
    # Add routing node to determine task type
    builder.add_node("router", router_node)
    
    # Add conversation processing node
    builder.add_node("conversation", chat_node)
    
    # Add image generation node
    builder.add_node("image_generation", image_generation_node)
    
    # Future nodes can be added here:
    # builder.add_node("preprocessor", preprocess_node)
    # builder.add_node("postprocessor", postprocess_node)
    # builder.add_node("error_handler", error_handling_node)


def _setup_edges(builder):
    """
    Set up all edges for the graph with conditional routing.
    
    Args:
        builder: StateGraph builder instance
    """
    # Set up conditional routing flow
    setup_conditional_edges(builder)
