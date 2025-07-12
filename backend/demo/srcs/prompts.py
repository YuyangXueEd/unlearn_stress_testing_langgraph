"""
Demo Prompts

Prompt templates for the simple chatbot demo.
"""


def get_chat_prompt(user_message: str, conversation_context: str = "") -> str:
    """
    Get the main chat prompt for responding to user messages.
    
    Args:
        user_message: The user's input message
        conversation_context: Previous conversation context
        
    Returns:
        Formatted prompt for the LLM
    """
    base_prompt = """You are a helpful AI assistant. You can help with various topics including:
- General questions and conversations
- Research and information
- Technical topics
- Creative tasks

"""
    
    if conversation_context:
        base_prompt += f"""Here is our conversation so far:
{conversation_context}

Please continue this conversation by responding to the latest message."""
    else:
        base_prompt += f"""Please provide a helpful and informative response to the following question:

{user_message}"""
    
    base_prompt += "\n\nKeep your response clear, concise, and helpful."
    
    return base_prompt


def get_error_prompt(error_message: str) -> str:
    """
    Get a prompt for handling errors gracefully.
    
    Args:
        error_message: The error that occurred
        
    Returns:
        User-friendly error message
    """
    return f"Sorry, I encountered an error while processing your request: {error_message}. Please try rephrasing your question or try again later."


def get_fallback_prompt() -> str:
    """
    Get a fallback prompt when no user message is received.
    
    Returns:
        Fallback response message
    """
    return "I didn't receive a message. Please try again with your question or request."
