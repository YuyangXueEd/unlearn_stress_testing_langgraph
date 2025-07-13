"""
Graph Builder

Functions for constructing and configuring the LangGraph.
"""

from langgraph.graph import StateGraph

from state import ChatState
from nodes.chat_nodes import chat_node, finalise_answer
from nodes.image_nodes import image_generation_node
from nodes.code_nodes import code_generation_node, generate, execute_and_check_code, decide_to_finish
from nodes.database_nodes import database_search_node, reflection_node, final_answer_node
from nodes.routing_nodes import router_node
from nodes.edges import setup_conditional_edges


def create_demo_graph():
    """
    Create and configure the demo graph with conditional routing.
    
    This function:
    1. Creates a StateGraph with ChatState
    2. Adds router, conversation, image generation, and database nodes
    3. Sets up conditional edges for routing
    4. Compiles and returns the graph
    
    Architecture:
    START -> router -> (conversation OR image_generation OR code_generation OR database_search) -> END
    
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
    
    # Add finalise answer node for summary and concise responses
    builder.add_node("finalise_answer", finalise_answer)
    
    # Add image generation node
    builder.add_node("image_generation", image_generation_node)
    
    # Add code generation workflow nodes (3-node pattern from PDF)
    builder.add_node("code_generation", code_generation_node)  # Entry point
    builder.add_node("generate", generate)                    # Node 1: Generate code
    builder.add_node("execute_and_check_code", execute_and_check_code)  # Node 2: Execute
    builder.add_node("decide_to_finish", decide_to_finish)    # Node 3: Decide
    
    # Add database search node
    builder.add_node("database_search", database_search_node)
    
    # Add reflection and final answer nodes (only for database search flow)
    builder.add_node("reflection", reflection_node)
    builder.add_node("final_answer", final_answer_node)
    
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
