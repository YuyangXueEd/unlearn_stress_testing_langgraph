# Simple Chatbot Demo with Image Generation

A minimal demonstration of a conversational AI assistant using LangGraph and Ollama, now with **Stable Diffusion image generation capabilities**! This demo showcases the basic structure and patterns used in the main agent, but in a simplified form for easy understanding and customization.

## 🤖 What is this?

A minimal, general-purpose chatbot demo that showcases:

- **LangGraph Integration**: Simple graph-based conversation flow
- **Conversation Memory**: Maintains context across multiple exchanges
- **Ollama Support**: Local LLM integration for privacy and control
- **🎨 Image Generation**: Stable Diffusion v1.4 integration for creating images from text prompts
- **Tool Integration**: Extensible tool framework for adding new capabilities
- **Clean Architecture**: Organized code structure following best practices
- **Web Interface**: Simple, responsive web UI with conversation history
- **Easy Setup**: Minimal dependencies and configuration

## 📁 Project Structure

### Core Files
- `app.py` - Main FastAPI application
- `graph.py` - LangGraph definition with enhanced chat node
- `state.py` - Enhanced state management with tool support
- `configuration.py` - Basic configuration
- `manager.py` - Conversation management
- `interface.py` - Web UI components with image generation examples
- `prompts.py` - Prompt templates
- `tools.py` - **NEW**: Tool definitions including Stable Diffusion

### Utility Files
- `run.py` - Simple demo runner
- `setup_sd.sh` - **NEW**: Setup script for Stable Diffusion dependencies
- `__init__.py` - Package initialization

## 🚀 Quick Start

### Prerequisites

1. **Ollama** running with a chat model:
   ```bash
   # Install and start Ollama
   ollama serve
   
   # Pull a recommended model (or any chat model you prefer)
   ollama pull qwen3
   ```

2. **Python dependencies**:
   ```bash
   pip install fastapi uvicorn langchain-ollama langgraph
   ```

3. **🎨 For Image Generation (Optional)**:
   ```bash
   # Run the setup script
   chmod +x setup_sd.sh
   ./setup_sd.sh
   
   # Or manually install dependencies:
   pip install torch torchvision torchaudio diffusers transformers accelerate safetensors pillow
   
   # Download Stable Diffusion v1.4 model
   git clone https://huggingface.co/CompVis/stable-diffusion-v1-4 demo/models/CompVis/stable-diffusion-v1-4
   ```

### Running the Demo

```bash
cd backend/src/demo
python run.py
# Or directly: python app.py
# Then open: http://localhost:8000
```

**Note**: Image generation requires the Stable Diffusion model to be downloaded. Without it, the chatbot will still work for conversations, but image generation requests will return an error message.

## 💡 Example Conversations

### General Questions
- "What is machine learning?"
- "Explain quantum computing in simple terms"
- "How does artificial intelligence work?"
- "Tell me about renewable energy"

### Technical Topics
- "What are neural networks?"
- "How do large language models work?"
- "Explain blockchain technology"
- "What is cloud computing?"

### 🎨 Image Generation Examples
- "Generate an image of a sunset over mountains"
- "Create a picture of a cute cat in a garden"
- "Draw a futuristic city skyline"
- "Make an image of a peaceful forest scene"
- "Visualize a steampunk robot"

### Multi-Round Conversations
- "Hi, my name is Alice"
- "What's my name?" (tests memory)
- "Tell me about Python programming"
- "Can you give me an example of what we just discussed?" (tests context)
- "Generate an image of what we just discussed" (combines conversation memory with image generation)

## 🔧 How It Works

### Architecture Overview

```
User Input → Enhanced Graph → Chat Node → Tool Detection → LLM/Tool Response → User
                                  ↓
                            Tool Execution (if needed)
                                  ↓
                         Stable Diffusion Pipeline
```

1. **Input Processing**: User message is received via web or CLI
2. **Tool Detection**: System checks if the request requires tool usage (e.g., image generation)
3. **Tool Execution**: If needed, appropriate tools are called (Stable Diffusion for images)
4. **Response Generation**: Either tool results or conversational LLM responses are returned
2. **Graph Execution**: Simple LangGraph processes the message
3. **LLM Generation**: Ollama model generates response
4. **Output Formatting**: Response is formatted and returned

### Key Components

- **State Management** (`state.py`): Simple state structure for conversations
- **Graph Definition** (`graph.py`): Single-node LangGraph for basic chat
- **Manager** (`manager.py`): Handles conversation flow and history
- **Configuration** (`configuration.py`): Simple settings management
- **Interfaces**: Web (FastAPI) and CLI options

## 🛠️ Customization

### Changing the Model
Edit `configuration.py`:
```python
model_name: str = Field(
    default="llama3",  # Change to your preferred model
    description="The name of the language model to use"
)
```

### Adjusting Response Length
In `configuration.py`:
```python
max_response_length: int = Field(
    default=1000,  # Shorter responses
    description="Maximum length of responses"
)
```

### Customizing the Prompt
Edit the prompt in `graph.py` in the `chat_node` function:
```python
prompt = f"""You are a helpful assistant specializing in [YOUR DOMAIN].
Please provide helpful responses about [YOUR TOPIC].

Question: {user_message}"""
```

## 🐛 Troubleshooting

### Common Issues

**"Failed to initialize chatbot"**
- Check that Ollama is running: `curl http://localhost:11434`
- Verify a model is available: `ollama list`
- Try pulling a model: `ollama pull qwen3`

**"Model not found" or connection errors**
- Ensure Ollama server is accessible at `localhost:11434`
- Check if the model name in config matches available models
- Try restarting Ollama: `ollama serve`

**Web interface not loading**
- Check that the app is running on the correct port (8000)
- Verify no other services are using the same port
- Look at console logs for error messages

**Slow responses**
- Smaller models respond faster (try `qwen2:0.5b`)
- Reduce `max_response_length` in configuration
- Check CPU/memory usage during inference

### Debug Mode

Run with detailed logging:
```bash
PYTHONPATH=.. python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from cli import main
import asyncio
asyncio.run(main())
"
```

## � Development Notes

### Adding New Features

1. **New Graph Nodes**: Add to `graph.py`
2. **State Changes**: Update `state.py` 
3. **Configuration**: Add settings to `configuration.py`
4. **UI Changes**: Modify `interface.py` for web or `cli.py` for terminal

### Code Structure

The demo follows the same patterns as the main agent:
- Separation of concerns (graph, state, config, interface)
- Async/await patterns for LLM calls
- Type hints and documentation
- Error handling and logging

This makes it easy to understand and extend the main agent codebase.

## 📄 License

This demo code follows the same license as the main project. See the LICENSE file in the project root.
