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
    num_images_per_prompt: int = 1,
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
        num_images_per_prompt: Number of images to generate per prompt (default: 1)
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
                num_images_per_prompt=1,
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


@register_tool("execute_python_code")
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
        
        # Ensure output directory exists
        output_dir = ensure_output_directory()
        
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
        
        # Save the executed code and its output to the tmps directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        code_filename = f"executed_code_{timestamp}.py"
        output_filename = f"execution_output_{timestamp}.txt"
        
        code_path = output_dir / code_filename
        output_path = output_dir / output_filename
        
        # Save the code
        with open(code_path, 'w', encoding='utf-8') as f:
            f.write(f"# Executed on {datetime.now().isoformat()}\n")
            f.write(f"# Status: {status}\n\n")
            f.write(code)
        
        # Save the output
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"Execution timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Status: {status}\n")
            f.write(f"Code file: {code_filename}\n")
            f.write("=" * 50 + "\n")
            f.write("OUTPUT:\n")
            f.write("=" * 50 + "\n")
            f.write(output)
            if error:
                f.write("\n" + "=" * 50 + "\n")
                f.write("ERROR:\n")
                f.write("=" * 50 + "\n")
                f.write(error)
        
        logger.info(f"Code saved to: {code_path}")
        logger.info(f"Output saved to: {output_path}")
            
        return {
            'success': status == 'success',
            'status': status,
            'output': output,
            'error': error,
            'traceback': output if has_error else None,
            'code_path': str(code_path),
            'output_path': str(output_path),
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error executing Python code: {e}")
        
        # Try to save the code even if execution failed
        try:
            output_dir = ensure_output_directory()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            code_filename = f"failed_code_{timestamp}.py"
            error_filename = f"execution_error_{timestamp}.txt"
            
            code_path = output_dir / code_filename
            error_path = output_dir / error_filename
            
            # Save the code that failed
            with open(code_path, 'w', encoding='utf-8') as f:
                f.write(f"# Failed execution on {datetime.now().isoformat()}\n")
                f.write(f"# Error: {str(e)}\n\n")
                f.write(code)
            
            # Save the error details
            with open(error_path, 'w', encoding='utf-8') as f:
                f.write(f"Execution timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Status: error\n")
                f.write(f"Code file: {code_filename}\n")
                f.write("=" * 50 + "\n")
                f.write("ERROR:\n")
                f.write("=" * 50 + "\n")
                f.write(str(e))
                f.write("\n" + "=" * 50 + "\n")
                f.write("TRACEBACK:\n")
                f.write("=" * 50 + "\n")
                f.write(traceback.format_exc())
            
            logger.info(f"Failed code saved to: {code_path}")
            logger.info(f"Error details saved to: {error_path}")
            
            return {
                'success': False,
                'status': 'error',
                'output': '',
                'error': str(e),
                'traceback': traceback.format_exc(),
                'code_path': str(code_path),
                'output_path': str(error_path),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as save_error:
            logger.error(f"Failed to save code after execution error: {save_error}")
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
        "execute_python_code": {
            "name": "execute_python_code",
            "description": "Execute Python code safely and return results",
            "parameters": {
                "code": "Python code to execute"
            },
            "example": "execute_python_code('print(5 + 7)')"
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
