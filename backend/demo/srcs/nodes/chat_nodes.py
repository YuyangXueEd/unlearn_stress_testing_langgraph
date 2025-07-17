"""
Chat Nodes

Nodes responsible for handling conversation and chat interactions.
"""

import logging
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama

from state import ChatState
from configuration import DemoConfiguration
from prompts import get_chat_prompt, get_error_prompt, get_fallback_prompt

logger = logging.getLogger(__name__)

# Configuration instance for model settings
_config = DemoConfiguration()


def _get_model_name(state: ChatState) -> str:
    """Get the model name from state or configuration."""
    return state.get("model_name", _config.model_name)


def chat_node(state: ChatState) -> ChatState:
    """
    Simplified chat node that handles only conversational responses.
    
    This node:
    1. Extracts the user message from state
    2. Builds conversation context from message history
    3. Generates a conversational response using the configured LLM
    4. Returns the response with updated state
    
    Args:
        state: Current chat state containing messages and configuration
        
    Returns:
        Updated state with the assistant's response
    """
    try:
        # Get the user message
        user_message = state.get("user_message", "")
        if not user_message and state.get("messages"):
            # Extract from messages if available
            last_message = state["messages"][-1]
            if hasattr(last_message, 'content'):
                user_message = last_message.content
        
        if not user_message:
            return {"response": get_fallback_prompt()}
        
        # Initialize the chat model
        model_name = _get_model_name(state)
        chat = ChatOllama(
            model=model_name,
            temperature=0.7,
            base_url="http://localhost:11434"
        )
        
        # Create conversation context from message history
        conversation_context = _build_conversation_context(state.get("messages", []))
        
        # Create a prompt with conversation context
        prompt = get_chat_prompt(user_message, conversation_context)
        
        # Get response from the model
        response = chat.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Limit response length based on configuration  
        max_length = state.get("max_response_length", 2000)
        if len(response_text) > max_length:
            response_text = response_text[:max_length-3] + "..."
        
        logger.info(f"Generated conversation response for: {user_message[:50]}...")
        
        return {
            "response": response_text,
            "messages": [AIMessage(content=response_text)]
        }
        
    except Exception as e:
        logger.error(f"Error in chat_node: {e}")
        return {
            "response": get_error_prompt(str(e)),
            "messages": [AIMessage(content=get_error_prompt(str(e)))]
        }


def _build_conversation_context(messages):
    """
    Build conversation context from message history.
    
    Args:
        messages: List of messages from the conversation
        
    Returns:
        String containing formatted conversation context
    """
    if len(messages) <= 1:  # No previous context
        return ""
    
    context_parts = []
    for msg in messages[:-1]:  # Exclude the last message (current user input)
        if hasattr(msg, 'content'):
            if hasattr(msg, 'type'):
                role = "Human" if msg.type == "human" else "Assistant"
            else:
                role = "Human" if isinstance(msg, HumanMessage) else "Assistant"
            context_parts.append(f"{role}: {msg.content}")
    
    return "\n".join(context_parts)


def finalise_answer(state: ChatState) -> ChatState:
    """
    Finalise answer node that provides a summary and concise response.
    
    This node:
    1. Takes the previous response and conversation context
    2. Creates a concise, well-structured final answer
    3. Ensures the response is clear and actionable
    4. Maintains conversation flow while being comprehensive
    
    Args:
        state: Current chat state containing the previous response and context
        
    Returns:
        Updated state with the finalised, concise response
    """
    try:
        # Get the previous response and conversation context
        previous_response = state.get("response", "")
        user_message = state.get("user_message", "")
        conversation_context = _build_conversation_context(state.get("messages", []))
        
        if not previous_response:
            # If no previous response, fall back to regular chat
            return chat_node(state)
        
        # Initialize the chat model
        model_name = _get_model_name(state)
        chat = ChatOllama(
            model=model_name,
            temperature=0.3,  # Lower temperature for more focused responses
            base_url="http://localhost:11434"
        )
        
        # Create a prompt for finalizing the answer
        finalize_prompt = f"""You are an AI assistant tasked with providing a final, concise, and well-structured response.

Previous response: {previous_response}

Original user question: {user_message}

Conversation context: {conversation_context}

Your task is to:
1. Summarize the key points from the previous response
2. Provide a clear, concise, and actionable final answer
3. Ensure the response directly addresses the user's question
4. Remove any redundancy while maintaining important information
5. Structure the response in a clear, easy-to-read format

Generate a final response that is comprehensive yet concise:"""

        # Get the finalized response
        response = chat.invoke(finalize_prompt)
        final_response = response.content if hasattr(response, 'content') else str(response)
        
        # Apply length limits
        max_length = state.get("max_response_length", 1500)  # Slightly shorter for final answers
        if len(final_response) > max_length:
            final_response = final_response[:max_length-3] + "..."
        
        logger.info(f"Generated finalized response for: {user_message[:50]}...")
        
        return {
            "response": final_response,
            "messages": [AIMessage(content=final_response)]
        }
        
    except Exception as e:
        logger.error(f"Error in finalise_answer: {e}")
        # Fall back to the previous response if finalization fails
        fallback_response = state.get("response", get_error_prompt(str(e)))
        return {
            "response": fallback_response,
            "messages": [AIMessage(content=fallback_response)]
        }
