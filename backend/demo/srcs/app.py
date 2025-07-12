"""
Demo Application Entry Point

Simple FastAPI application for demonstrating the conversational agent.
"""

import os
import sys
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

# Add the parent directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from demo.graph import demo_graph
from demo.manager import ChatbotManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global chatbot manager
chatbot_manager = ChatbotManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events."""
    logger.info("Starting demo application...")
    await chatbot_manager.initialize()
    logger.info("Demo application startup complete")
    yield
    logger.info("Demo application shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Simple Conversational Chatbot Demo",
    description="A basic demo of the LangGraph-based conversational agent",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/", response_class=HTMLResponse)
async def get_chat_interface():
    """Serve the main chat interface."""
    from demo.interface import get_chat_html
    return get_chat_html()


@app.post("/chat")
async def chat_endpoint(request: dict):
    """Handle chat messages."""
    message = request.get("message", "")
    if not message:
        return {"error": "No message provided"}
    
    try:
        response = await chatbot_manager.process_message(message)
        return response
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return {"error": str(e)}


@app.post("/clear")
async def clear_conversation():
    """Clear the conversation history."""
    try:
        chatbot_manager.clear_conversation()
        return {"message": "Conversation cleared", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Clear conversation error: {e}")
        return {"error": str(e)}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "initialized": chatbot_manager.initialized,
        "conversation_length": chatbot_manager.get_conversation_count()
    }


if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting simple chatbot demo...")
    print("📍 Open your browser to: http://localhost:8000")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
