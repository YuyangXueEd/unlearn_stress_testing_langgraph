"""
Routing Nodes

Nodes responsible for determining the flow and routing messages to appropriate handlers.
"""

import logging
from state import ChatState

logger = logging.getLogger(__name__)


def router_node(state: ChatState) -> ChatState:
    """
    Router node that determines whether the user wants image generation or conversation.
    
    This node analyzes the user message and adds routing information to the state
    without modifying the message flow.
    
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
        
        # Determine if this is an image generation request
        is_image_request = _is_image_generation_request(user_message)
        
        logger.info(f"Router decision: {'image_generation' if is_image_request else 'conversation'} for message: {user_message[:50]}...")
        
        # Add routing decision to state
        return {
            "task_type": "image_generation" if is_image_request else "conversation",
            "user_message": user_message  # Ensure user_message is in state
        }
        
    except Exception as e:
        logger.error(f"Error in router node: {e}")
        # Default to conversation on error
        return {
            "task_type": "conversation",
            "user_message": user_message
        }


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
