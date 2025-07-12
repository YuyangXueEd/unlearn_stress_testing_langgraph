"""
Demo Manager

Manages the chatbot state and interactions.
"""

import logging
from typing import Dict, Any
from datetime import datetime

from langchain_core.messages import HumanMessage
from demo.graph import demo_graph
from demo.configuration import DemoConfiguration

logger = logging.getLogger(__name__)


class ChatbotManager:
    """Simple chatbot manager for the demo with conversation memory."""
    
    def __init__(self):
        self.initialized = False
        self.config = DemoConfiguration()
        self.conversation_messages = []  # Store conversation history
    
    async def initialize(self):
        """Initialize the chatbot manager."""
        if self.initialized:
            return True
        
        try:
            logger.info("Initializing demo chatbot manager...")
            self.initialized = True
            logger.info("Demo chatbot manager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize chatbot manager: {e}")
            self.initialized = False
            return False
    
    async def process_message(self, user_message: str) -> Dict[str, Any]:
        """Process a user message and return a response."""
        if not self.initialized:
            await self.initialize()
            if not self.initialized:
                return {
                    "response": "Sorry, the chatbot is not properly initialized.",
                    "timestamp": datetime.now().isoformat()
                }
        
        try:
            # Add new user message to conversation history
            new_user_message = HumanMessage(content=user_message)
            self.conversation_messages.append(new_user_message)
            
            # Create input state with full conversation history
            input_state = {
                "messages": self.conversation_messages.copy(),  # Pass all messages
                "user_message": user_message,
                "model_name": self.config.model_name,
                "max_response_length": self.config.max_response_length
            }
            
            logger.info(f"Processing message: {user_message[:100]}...")
            
            # Run the graph
            result = await demo_graph.ainvoke(input_state)
            
            # Extract response
            response_text = result.get("response", "I couldn't generate a response.")
            
            # Add AI response to conversation history
            if "messages" in result and result["messages"]:
                # Use the AI message from the result
                self.conversation_messages.extend(result["messages"])
            
            return {
                "response": response_text,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return {
                "response": f"Sorry, I encountered an error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    def clear_conversation(self):
        """Clear the conversation history."""
        self.conversation_messages.clear()
        logger.info("Conversation history cleared")
    
    def get_conversation_count(self) -> int:
        """Get the number of messages in the conversation."""
        return len(self.conversation_messages)
