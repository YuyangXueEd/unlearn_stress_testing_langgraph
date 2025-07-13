"""
Image Generation Nodes

Nodes responsible for handling image generation tasks.
"""

import logging
import re
from langchain_core.messages import AIMessage

from state import ChatState
from tools import execute_tool_async

logger = logging.getLogger(__name__)


async def image_generation_node(state: ChatState) -> ChatState:
    """
    Dedicated node for handling image generation requests.
    
    This node:
    1. Extracts the image prompt from the user message
    2. Calls the Stable Diffusion tool
    3. Returns formatted response with image details
    
    Args:
        state: Current chat state containing image generation request
        
    Returns:
        Updated state with image generation result
    """
    try:
        user_message = state.get("user_message", "")
        if not user_message:
            return {
                "response": "❌ No image prompt provided.",
                "messages": [AIMessage(content="❌ No image prompt provided.")]
            }
        
        # Extract the actual prompt for image generation
        image_prompt = _extract_image_prompt(user_message)
        
        logger.info(f"Processing image generation request: {image_prompt}")
        
        # Call the image generation tool
        result = await execute_tool_async("generate_image", {"prompt": image_prompt})
        
        if result.get("success"):
            response_text = (
                f"🎨 I've generated an image for you based on the prompt: '{image_prompt}'\n\n"
                f"✅ Image saved to: {result['image_path']}\n"
                f"🔧 Parameters used:\n"
                f"   - Steps: {result['parameters']['num_inference_steps']}\n"
                f"   - Guidance: {result['parameters']['guidance_scale']}\n"
                f"   - Size: {result['parameters']['width']}x{result['parameters']['height']}\n"
                f"   - Device: {result['parameters']['device']}\n\n"
                f"The image has been saved to the tmps folder for you to view!"
            )
            
            # Include base64 image data for potential web display
            if "image_base64" in result:
                response_text += f"\n\n📷 Image data available for display."
                
        else:
            response_text = (
                f"❌ Sorry, I couldn't generate the image. "
                f"Error: {result.get('error', 'Unknown error')}\n\n"
                f"Make sure the Stable Diffusion model is properly installed in the demo/models folder."
            )
        
        return {
            "response": response_text,
            "messages": [AIMessage(content=response_text)],
            "tool_result": result  # Store tool result for potential use
        }
        
    except Exception as e:
        logger.error(f"Error in image generation node: {e}")
        error_response = f"❌ An error occurred while generating the image: {str(e)}"
        return {
            "response": error_response,
            "messages": [AIMessage(content=error_response)]
        }


def _extract_image_prompt(message: str) -> str:
    """
    Extract the image generation prompt from user message.
    
    Args:
        message: User message containing image request
        
    Returns:
        Cleaned prompt for image generation
    """
    # Remove common prefixes
    prefixes_to_remove = [
        "generate image of", "create image of", "draw", "generate picture of",
        "create picture of", "make image of", "make picture of", "visualize",
        "generate an image of", "create an image of", "show me", "paint"
    ]
    
    cleaned = message.lower()
    for prefix in prefixes_to_remove:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
            break
    
    # Remove common words at the beginning
    cleaned = re.sub(r'^(a|an|the)\s+', '', cleaned)
    
    return cleaned if cleaned else message
