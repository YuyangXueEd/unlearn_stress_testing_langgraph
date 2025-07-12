"""
Demo Application Entry Point

Simple FastAPI application for demonstrating the conversational agent.
"""

import os
import sys
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# Add the current directory to Python path for LangGraph CLI compatibility
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Add the parent directory to the Python path
parent_dir = os.path.join(os.path.dirname(__file__), '..')
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from graph import demo_graph
    from manager import ChatbotManager
except ImportError:
    # Try with demo prefix for LangGraph CLI
    try:
        from srcs.graph import demo_graph
        from srcs.manager import ChatbotManager
    except ImportError:
        # Try with relative imports
        from .graph import demo_graph
        from .manager import ChatbotManager

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

# Mount static files for serving generated images
tmps_path = Path(__file__).parent.parent / "tmps"
tmps_path.mkdir(exist_ok=True)  # Ensure the directory exists
app.mount("/images", StaticFiles(directory=str(tmps_path)), name="images")


@app.get("/", response_class=HTMLResponse)
async def get_chat_interface():
    """Serve the main chat interface."""
    try:
        from interface import get_chat_html
    except ImportError:
        try:
            from srcs.interface import get_chat_html
        except ImportError:
            from .interface import get_chat_html
    return get_chat_html()


@app.post("/chat")
async def chat_endpoint(request: dict):
    """Handle chat messages."""
    message = request.get("message", "")
    if not message:
        return {"error": "No message provided"}
    
    try:
        response = await chatbot_manager.process_message(message)
        
        # Check if an image was generated and add the URL
        if "tool_result" in response and response["tool_result"] and response["tool_result"].get("success"):
            tool_result = response["tool_result"]
            if "image_path" in tool_result:
                # Extract filename from path
                image_path = Path(tool_result["image_path"])
                filename = image_path.name
                # Add image URL to response
                response["image_url"] = f"/images/{filename}"
                response["image_filename"] = filename
        
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


@app.get("/image/{filename}")
async def get_image(filename: str):
    """Serve generated images."""
    image_path = tmps_path / filename
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(image_path)


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
