"""
Graph Builder

Functions for constructing and configuring the LangGraph.
"""

from langgraph.graph import StateGraph

from demo.state import ChatState
from demo.nodes.chat_nodes import chat_node
from demo.nodes.edges import setup_basic_edges


def create_demo_graph():
    """
    Create and configure the demo graph.
    
    This function:
    1. Creates a StateGraph with ChatState
    2. Adds all necessary nodes
    3. Sets up edges and routing
    4. Compiles and returns the graph
    
    Returns:
        Compiled LangGraph ready for execution
    """
    # Create the graph builder
    builder = StateGraph(ChatState)
    
    # Add nodes
    _add_nodes(builder)
    
    # Set up edges
    _setup_edges(builder)
    
    # Compile and return the graph
    return builder.compile()


def _add_nodes(builder):
    """
    Add all nodes to the graph builder.
    
    Args:
        builder: StateGraph builder instance
    """
    # Add chat processing node
    builder.add_node("chat", chat_node)
    
    # Future nodes can be added here:
    # builder.add_node("preprocessor", preprocess_node)
    # builder.add_node("postprocessor", postprocess_node)
    # builder.add_node("error_handler", error_handling_node)


def _setup_edges(builder):
    """
    Set up all edges for the graph.
    
    Args:
        builder: StateGraph builder instance
    """
    # Set up basic linear flow
    setup_basic_edges(builder)
    
    # Future: Add conditional routing
    # setup_conditional_edges(builder)
