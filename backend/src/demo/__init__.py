"""
Simple Chatbot Demo Package

A minimal demonstration of a conversational AI assistant using LangGraph.
"""

from .app import app
from .manager import ChatbotManager
from .graph import demo_graph
from . import nodes

__version__ = "1.0.0"
__all__ = ["app", "ChatbotManager", "demo_graph", "nodes"]
