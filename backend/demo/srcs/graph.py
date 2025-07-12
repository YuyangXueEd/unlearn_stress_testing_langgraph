"""
Demo Graph

Simplified LangGraph implementation for demo purposes.
Uses organized node structure from the nodes package.
"""

try:
    from nodes.graph_builder import create_demo_graph
except ImportError:
    try:
        from srcs.nodes.graph_builder import create_demo_graph
    except ImportError:
        from .nodes.graph_builder import create_demo_graph

# Create the graph instance
demo_graph = create_demo_graph()
