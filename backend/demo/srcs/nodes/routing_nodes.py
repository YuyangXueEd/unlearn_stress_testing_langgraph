"""
Routing Nodes

Nodes responsible for determining the flow and routing messages to appropriate handlers.
"""

import logging
from state import ChatState

logger = logging.getLogger(__name__)


def router_node(state: ChatState) -> ChatState:
    """
    Router node that determines the appropriate task type for the user message.
    
    This node analyzes the user message and routes to:
    - image_generation: For image creation requests
    - database_search: For research paper queries and information retrieval
    - conversation: For general chat
    
    Args:
        state: Current chat state
        
    Returns:
        Updated state with routing decision
    """
    try:
        user_message = state.get("user_message", "")
        if not user_message and state.get("messages"):
            # Extract from messages if available
            last_message = state["messages"][-1]
            if hasattr(last_message, 'content'):
                user_message = last_message.content
        
        # Determine task type based on message content
        if _is_image_generation_request(user_message):
            task_type = "image_generation"
        elif _is_database_search_request(user_message):
            task_type = "database_search"
        else:
            task_type = "conversation"
        
        logger.info(f"Router decision: {task_type} for message: {user_message[:50]}...")
        
        # Add routing decision to state
        return {
            "task_type": task_type,
            "user_message": user_message  # Ensure user_message is in state
        }
        
    except Exception as e:
        logger.error(f"Error in router node: {e}")
        # Default to conversation on error
        return {
            "task_type": "conversation",
            "user_message": user_message
        }


def _is_database_search_request(message: str) -> bool:
    """
    Check if the user message is requesting database/paper search.
    
    Args:
        message: User message to analyze
        
    Returns:
        True if this appears to be a database search request
    """
    search_keywords = [
        # Direct search keywords
        "search", "find", "look for", "search for", "find information",
        "what does the paper say", "according to the paper", "in the research",
        "from the papers", "paper about", "research about", "study about",
        
        # Academic/research terms
        "literature", "publication", "research", "study", "analysis",
        "experiment", "methodology", "results", "findings", "conclusion",
        "abstract", "introduction", "related work", "evaluation",
        
        # Question patterns about research
        "what is", "how does", "explain", "describe", "tell me about",
        "what are the", "how to", "why does", "when was", "who proposed",
        
        # Technical terms that might be in papers
        "algorithm", "model", "method", "approach", "technique", "framework",
        "dataset", "benchmark", "evaluation", "performance", "accuracy",
        "neural network", "machine learning", "deep learning", "AI",
        "artificial intelligence", "computer vision", "NLP", "natural language"
    ]
    
    message_lower = message.lower()
    
    # First check for exact keyword matches
    if any(keyword in message_lower for keyword in search_keywords):
        return True
    
    # Check for question patterns that suggest academic inquiry
    import re
    
    academic_patterns = [
        r'\b(what|how|why|when|where|who)\s+(is|are|does|do|did|was|were)\s+.*(method|algorithm|model|approach|technique|research|study)',
        r'\b(explain|describe|tell me about|discuss)\s+.*(method|algorithm|model|approach|technique)',
        r'\b(paper|research|study|literature)\s+(says?|shows?|demonstrates?|proves?|suggests?)',
        r'\baccording to\s+(the\s+)?(paper|research|study|literature)',
        r'\bin\s+the\s+(paper|research|study|literature)',
        r'\bwhat\s+(is|are)\s+the\s+(results?|findings?|conclusions?)',
    ]
    
    for pattern in academic_patterns:
        if re.search(pattern, message_lower):
            return True
    
    return False


def _is_image_generation_request(message: str) -> bool:
    """
    Check if the user message is requesting image generation.
    
    Args:
        message: User message to analyze
        
    Returns:
        True if this appears to be an image generation request
    """
    image_keywords = [
        # Direct generation keywords
        "generate image", "create image", "generate picture", "create picture",
        "generate an image", "create an image", "generate a picture", "create a picture",
        "make image", "make picture", "make an image", "make a picture",
        
        # Photo-related keywords
        "generate photo", "create photo", "generate a photo", "create a photo",
        "make photo", "make a photo", "photo of", "picture of",
        "generate me a photo", "generate me an image", "generate me a picture",
        "create me a photo", "create me an image", "create me a picture",
        
        # Art and drawing keywords
        "draw", "paint", "sketch", "illustrate", "visualize",
        "draw me", "paint me", "sketch me", "illustrate me",
        "draw a", "paint a", "sketch a", "illustrate a",
        
        # Show/display keywords
        "show me", "show me a", "display", "render"
    ]
    
    message_lower = message.lower()
    
    # First check for exact keyword matches
    if any(keyword in message_lower for keyword in image_keywords):
        return True
    
    # Additional pattern-based detection
    import re
    
    # Check for patterns like "generate [something] of [description]"
    generate_patterns = [
        r'\b(generate|create|make|draw|paint)\s+(me\s+)?(a\s+)?(photo|image|picture|drawing|painting|sketch)\s+(of|showing|depicting)',
        r'\b(photo|image|picture|drawing|painting|sketch)\s+of\b',
        r'\bgenerate\s+me\s+(a\s+)?(photo|image|picture)\b',
        r'\bcreate\s+me\s+(a\s+)?(photo|image|picture)\b'
    ]
    
    for pattern in generate_patterns:
        if re.search(pattern, message_lower):
            return True
    
    return False
