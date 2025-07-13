"""
Demo Tools

Tools that the agent can use for various tasks.
"""

import os
import logging
import base64
import asyncio
import traceback
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
from langchain_experimental.tools import PythonREPLTool

logger = logging.getLogger(__name__)

# Tool registry
AVAILABLE_TOOLS = {}


def register_tool(name: str):
    """Decorator to register tools."""
    def decorator(func):
        AVAILABLE_TOOLS[name] = func
        return func
    return decorator


class ToolError(Exception):
    """Custom exception for tool errors."""
    pass


def ensure_output_directory():
    """Ensure the output directory exists."""
    output_dir = Path(__file__).parent.parent / "tmps"
    output_dir.mkdir(exist_ok=True)
    return output_dir


@register_tool("generate_image")
def generate_image_with_stable_diffusion(
    prompt: str,
    negative_prompt: Optional[str] = None,
    num_inference_steps: int = 20,
    guidance_scale: float = 7.5,
    width: int = 512,
    height: int = 512,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Generate an image using Stable Diffusion v1.4.
    
    Args:
        prompt: Text prompt for image generation
        negative_prompt: Optional negative prompt to avoid certain elements
        num_inference_steps: Number of denoising steps (default: 20)
        guidance_scale: Guidance scale for prompt adherence (default: 7.5)
        width: Image width (default: 512)
        height: Image height (default: 512)
        seed: Optional random seed for reproducibility
        
    Returns:
        Dictionary containing success status, image path, and metadata
    """
    try:
        logger.info(f"Starting image generation with prompt: {prompt[:50]}...")
        
        # Set up paths
        model_path = Path(__file__).parent.parent / "models" / "CompVis" / "stable-diffusion-v1-4"
        output_dir = ensure_output_directory()
        
        # Check if model exists
        if not model_path.exists():
            raise ToolError(
                f"Stable Diffusion model not found at {model_path}. "
                "Please download the model first."
            )
        
        # Load the pipeline
        logger.info("Loading Stable Diffusion pipeline...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        pipe = StableDiffusionPipeline.from_pretrained(
            str(model_path),
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None,  # Disable safety checker for demo
            requires_safety_checker=False
        )
        pipe = pipe.to(device)
        
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        # Generate image
        logger.info("Generating image...")
        with torch.no_grad():
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                generator=torch.Generator(device=device).manual_seed(seed) if seed else None
            )
        
        image = result.images[0]
        
        # Save image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_prompt = "".join(c for c in prompt[:30] if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_prompt = safe_prompt.replace(' ', '_')
        
        image_filename = f"generated_{timestamp}_{safe_prompt}.png"
        image_path = output_dir / image_filename
        
        image.save(image_path)
        
        # Convert to base64 for web display (optional)
        import io
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        logger.info(f"Image saved to: {image_path}")
        
        return {
            "success": True,
            "image_path": str(image_path),
            "image_base64": img_base64,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "parameters": {
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "width": width,
                "height": height,
                "seed": seed,
                "device": device
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating image: {e}")
        return {
            "success": False,
            "error": str(e),
            "prompt": prompt,
            "timestamp": datetime.now().isoformat()
        }


# Initialize PythonREPL tool
_python_repl = PythonREPLTool()


@register_tool("execute_python")
async def execute_python_code(code: str) -> Dict[str, Any]:
    """
    Execute Python code safely using PythonREPLTool in a separate thread.
    
    Args:
        code: Python code to execute
        
    Returns:
        Dictionary containing execution results, output, and error information
    """
    try:
        logger.info(f"Executing Python code: {code[:100]}...")
        
        # Run the blocking PythonREPL operation in a separate thread
        result = await asyncio.to_thread(_python_repl.run, code)
        
        # Parse the result and check for errors
        if isinstance(result, str):
            output = result.strip()
        else:
            output = str(result).strip()
        
        # Check if the output contains error indicators
        error_indicators = [
            'Traceback (most recent call last):',
            'Error:', 'Exception:', 'ValueError:', 'TypeError:', 'NameError:',
            'SyntaxError:', 'AttributeError:', 'KeyError:', 'IndexError:',
            'ZeroDivisionError:', 'ImportError:', 'ModuleNotFoundError:'
        ]
        
        has_error = any(indicator in output for indicator in error_indicators)
        
        logger.info(f"Code execution output: {output[:200]}...")
        logger.info(f"Error detected: {has_error}")
        
        if has_error:
            status = 'error'
            error = output  # The entire output is the error message
        else:
            status = 'success'
            error = None
            
        return {
            'success': status == 'success',
            'status': status,
            'output': output,
            'error': error,
            'traceback': output if has_error else None,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error executing Python code: {e}")
        return {
            'success': False,
            'status': 'error',
            'output': '',
            'error': str(e),
            'traceback': traceback.format_exc(),
            'timestamp': datetime.now().isoformat()
        }


def get_available_tools() -> Dict[str, Any]:
    """Get list of available tools with their descriptions."""
    return {
        "generate_image": {
            "name": "generate_image",
            "description": "Generate an image using Stable Diffusion based on a text prompt",
            "parameters": {
                "prompt": "Text description of the image to generate",
                "negative_prompt": "(Optional) Things to avoid in the image",
                "num_inference_steps": "(Optional) Number of denoising steps (default: 20)",
                "guidance_scale": "(Optional) How closely to follow the prompt (default: 7.5)",
                "width": "(Optional) Image width in pixels (default: 512)",
                "height": "(Optional) Image height in pixels (default: 512)",
                "seed": "(Optional) Random seed for reproducible results"
            },
            "example": "generate_image('a beautiful sunset over mountains')"
        },
        "execute_python": {
            "name": "execute_python",
            "description": "Execute Python code safely and return results",
            "parameters": {
                "code": "Python code to execute"
            },
            "example": "execute_python('print(5 + 7)')"
        }
    }


def execute_tool(tool_name: str, **kwargs) -> Dict[str, Any]:
    """
    Execute a tool by name with given parameters.
    
    Args:
        tool_name: Name of the tool to execute
        **kwargs: Tool parameters
        
    Returns:
        Tool execution result
    """
    if tool_name not in AVAILABLE_TOOLS:
        return {
            "success": False,
            "error": f"Tool '{tool_name}' not found. Available tools: {list(AVAILABLE_TOOLS.keys())}"
        }
    
    try:
        return AVAILABLE_TOOLS[tool_name](**kwargs)
    except Exception as e:
        return {
            "success": False,
            "error": f"Error executing tool '{tool_name}': {str(e)}"
        }


async def execute_tool_async(tool_name: str, **kwargs) -> Dict[str, Any]:
    """
    Execute an async tool by name with given parameters.
    
    Args:
        tool_name: Name of the tool to execute
        **kwargs: Tool parameters
        
    Returns:
        Tool execution result
    """
    if tool_name not in AVAILABLE_TOOLS:
        return {
            "success": False,
            "error": f"Tool '{tool_name}' not found. Available tools: {list(AVAILABLE_TOOLS.keys())}"
        }
    
    try:
        tool_func = AVAILABLE_TOOLS[tool_name]
        if asyncio.iscoroutinefunction(tool_func):
            return await tool_func(**kwargs)
        else:
            return tool_func(**kwargs)
    except Exception as e:
        return {
            "success": False,
            "error": f"Error executing tool '{tool_name}': {str(e)}"
        }
