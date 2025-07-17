"""
Code Generation Nodes

AI Coding Agent implementation following LangGraph tutorial design.
This module implements a three-node workflow with PythonREPL support:
1. generate - Generates code solution and adapts based on previous errors
2. execute_and_check_code - Runs generated code and verifies correctness  
3. decide_to_finish - Determines whether to stop or continue refining
"""

import re
import logging
import traceback
import asyncio
from datetime import datetime
from typing import Dict, Any
from pathlib import Path

from langchain_experimental.tools import PythonREPLTool
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage

from state import ChatState
from configuration import DemoConfiguration

# Configuration instance for model settings
_config = DemoConfiguration()


def _get_model_name(state: ChatState) -> str:
    """Get the model name from state or configuration."""
    return state.get("model_name", _config.model_name)
from state import ChatState
from tools import execute_tool_async
import aiofiles  # For async file operations

logger = logging.getLogger(__name__)

# Remove the local PythonREPL tool since we'll use the one from tools.py


# ============================================================================
# Main Code Generation Node (Entry Point)
# ============================================================================

async def code_generation_node(state: ChatState, config=None) -> ChatState:
    """
    Main entry point for code generation workflow.
    This is called by the graph router and sets up the initial state.
    """
    logger.info("Starting code generation workflow")
    
    # Get the user message using the same pattern as chat_nodes.py
    user_message = state.get("user_message", "")
    if not user_message and state.get("messages"):
        # Extract from messages if available
        last_message = state["messages"][-1]
        if hasattr(last_message, 'content'):
            user_message = last_message.content
    
    if not user_message:
        logger.warning("No user message found in code generation node")
        return state
    
    logger.info(f"Processing code generation request: {user_message[:100]}...")
    
    # Extract requirements from user message
    requirements = _extract_code_requirements(user_message)
    logger.info(f"Extracted requirements: {requirements}")
    
    # Return state updates (don't modify state directly)
    return {
        "task_type": 'code_generation',
        "code_generation": {
            'status': 'initialized',
            'attempt': 0,
            'max_attempts': 3,
            'requirements': requirements
        }
    }


# ============================================================================
# Three-Node Workflow Implementation 
# ============================================================================

async def generate(state: ChatState, config=None) -> ChatState:
    """
    Node 1: Generate code solution and adapt based on previous errors.
    
    This node:
    - Analyzes the user request and any previous errors
    - Generates Python code solutions using LLM
    - Adapts approach based on execution feedback
    """
    logger.info("Starting code generation process")
    
    try:
        # Get current state
        code_gen = state.get("code_generation", {})
        attempt = code_gen.get("attempt", 0) + 1
        requirements = code_gen.get("requirements", "")
        
        # Extract previous errors for adaptive prompting
        previous_errors = _extract_previous_errors(state.get("messages", []))
        
        # Create adaptive prompt based on context
        prompt = _create_adaptive_code_prompt(requirements, previous_errors, attempt)
        logger.info(f"Generated prompt: {prompt[:200]}...")
        
        # Generate code using LLM
        model_name = _get_model_name(state)
        llm = ChatOllama(
            model=model_name,
            temperature=0.1,
            base_url="http://localhost:11434"
        )
        
        response = await llm.ainvoke(prompt)
        logger.info(f"LLM response: {response.content[:200]}...")
        
        # Parse and extract code from response
        parsed_result = _parse_code_response(response.content)
        logger.info(f"Parsed result: {parsed_result}")
        
        if not parsed_result.get('code'):
            error_msg = "Failed to generate valid Python code"
            logger.error(error_msg)
            return {
                "code_generation": {
                    **code_gen,
                    'status': 'error',
                    'error': error_msg,
                    'attempt': attempt
                }
            }
        
        # Store generated code in state
        logger.info(f"Successfully generated code (attempt {attempt})")
        return {
            "code_generation": {
                **code_gen,
                'status': 'generated',
                'code': parsed_result['code'],
                'explanation': parsed_result.get('explanation', ''),
                'attempt': attempt,
                'previous_errors': previous_errors
            }
        }
        
    except Exception as e:
        error_msg = f"Code generation error: {str(e)}"
        logger.error(error_msg)
        code_gen = state.get("code_generation", {})
        return {
            "code_generation": {
                **code_gen,
                'status': 'error',
                'error': error_msg,
                'attempt': code_gen.get("attempt", 0) + 1
            }
        }


async def execute_and_check_code(state: ChatState, config=None) -> ChatState:
    """
    Node 2: Execute the generated code and verify its correctness.
    
    This node:
    - Runs the generated Python code using PythonREPLTool
    - Captures output and any errors
    - Validates the execution results
    - Provides detailed feedback for refinement
    """
    logger.info("Starting code execution and verification")
    
    code_gen = state.get("code_generation", {})
    
    if not code_gen.get('code'):
        return {
            "code_generation": {
                **code_gen,
                'status': 'error',
                'error': 'No code available for execution',
                'execution_result': None
            }
        }
    
    code = code_gen['code']
    
    try:
        # Execute code using PythonREPLTool
        execution_result = await _execute_code_safely(code)
        
        # Validate execution results
        validation_result = _validate_execution_result(
            execution_result, 
            code_gen.get('requirements', '')
        )
        
        # Update state with execution results
        logger.info(f"Code execution completed: {execution_result['status']}")
        return {
            "code_generation": {
                **code_gen,
                'execution_result': execution_result,
                'validation_result': validation_result,
                'status': 'executed'
            }
        }
        
    except Exception as e:
        error_msg = f"Code execution error: {str(e)}"
        logger.error(error_msg)
        return {
            "code_generation": {
                **code_gen,
                'execution_result': {
                    'status': 'error',
                    'error': error_msg,
                    'output': '',
                    'traceback': traceback.format_exc()
                },
                'status': 'execution_failed'
            }
        }


async def decide_to_finish(state: ChatState, config=None) -> ChatState:
    """
    Node 3: Determine whether the agent should stop execution or continue refining.
    
    This node:
    - Analyzes execution results and code quality
    - Determines if the solution meets requirements
    - Decides whether to iterate or finalize
    - Saves successful code and provides final response
    """
    logger.info("Making decision to finish or continue")
    
    code_gen = state.get("code_generation", {})
    
    if not code_gen:
        return {
            "messages": [AIMessage(content="❌ No code generation context available.")]
        }
    
    max_attempts = code_gen.get('max_attempts', 3)
    
    try:
        # Analyze if we should continue or finish
        decision = _make_finish_decision(code_gen, max_attempts)
        
        if decision['action'] == 'finish_success':
            # Save successful code and provide final response
            file_path = await _save_code_to_file(
                code_gen['code'],
                code_gen.get('requirements', ''),
                code_gen.get('validation_result', {})
            )
            
            response = _format_success_response(code_gen, file_path)
            return {
                "messages": [AIMessage(content=response)],
                "response": response,
                "code_generation": {**code_gen, "status": "completed"}
            }
            
        elif decision['action'] == 'continue_refining':
            # Prepare for next iteration - update state for continuation
            error_context = _extract_error_context(code_gen)
            return {
                "code_generation": {
                    **code_gen, 
                    "status": "refining",
                    "error_context": error_context
                }
            }
            
        elif decision['action'] == 'finish_failure':
            # Maximum attempts reached, provide final failure response
            response = _format_failure_response(code_gen, max_attempts)
            return {
                "messages": [AIMessage(content=response)],
                "response": response,
                "code_generation": {**code_gen, "status": "failed"}
            }
        
        logger.info(f"Decision made: {decision['action']}")
        # Should not reach here, but return empty state if it does
        return {}
        
    except Exception as e:
        error_msg = f"Decision making error: {str(e)}"
        logger.error(error_msg)
        
        return {
            "messages": [AIMessage(content=f"❌ {error_msg}")]
        }


# ============================================================================
# Helper Functions for Code Generation Workflow
# ============================================================================

def _extract_code_requirements(message: str) -> str:
    """Extract code requirements from user message."""
    original_message = message
    
    # Clean the requirements by removing specific prefixes (more conservative)
    prefixes_to_remove = [
        "write code to", "generate code to", "create code to", 
        "write a script to", "generate a script to", "create a script to",
        "write python code to", "generate python code to", "create python code to"
    ]
    
    cleaned = message.lower()
    removed_prefix = False
    for prefix in prefixes_to_remove:
        if cleaned.startswith(prefix + " "):  # Ensure space after prefix
            cleaned = cleaned[len(prefix):].strip()
            removed_prefix = True
            break
    
    # If no specific prefix was removed, try general ones but be more careful
    if not removed_prefix:
        general_prefixes = ["write code for", "generate code for", "create code for"]
        for prefix in general_prefixes:
            if cleaned.startswith(prefix + " "):
                cleaned = cleaned[len(prefix):].strip()
                break
    
    # Don't remove python keywords if they're part of the actual task
    # Only remove if they appear redundantly
    
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    final_requirements = cleaned if cleaned else original_message
    
    logger.info(f"Requirements extraction: '{original_message}' -> '{final_requirements}'")
    return final_requirements


def _extract_previous_errors(messages: list) -> list:
    """Extract previous errors from conversation history."""
    errors = []
    for msg in messages:
        # Handle both LangChain message objects and dictionaries
        if hasattr(msg, 'content'):
            content = msg.content
        elif isinstance(msg, dict):
            content = msg.get("content", "")
        else:
            continue
            
        if "error" in content.lower():
            errors.append(content)
    return errors[-2:]  # Keep last 2 errors for context


def _create_adaptive_code_prompt(requirements: str, previous_errors: list, attempt: int) -> str:
    """Create adaptive prompt for code generation with error feedback."""
    base_prompt = f"""You are an expert Python programmer. Your task is to write Python code that exactly fulfills the user's request.

USER REQUEST: {requirements}

INSTRUCTIONS:
- Write ONLY the Python code needed to fulfill this specific request
- Keep the code simple and focused on the exact task
- Do NOT overthink or add unnecessary complexity
- Do NOT assume additional requirements that weren't specified
- If it's a simple calculation, write simple calculation code
- If it's a basic task, write basic code
- Include minimal error handling only if truly necessary
- Add brief comments only where needed for clarity

IMPORTANT: Focus ONLY on what the user actually asked for. Do not elaborate beyond their request.

"""
    
    if previous_errors and attempt > 1:
        base_prompt += f"""
PREVIOUS ERRORS TO FIX (Attempt #{attempt}):
{chr(10).join(previous_errors)}

Fix these specific issues in your new solution.

"""
    
    base_prompt += """
OUTPUT FORMAT:
```python
# Your focused Python code here
```

Brief explanation of what the code does.
"""
    
    return base_prompt


def _parse_code_response(response_text: str) -> Dict[str, Any]:
    """Parse LLM response to extract Python code and explanation."""
    try:
        # Extract code from markdown blocks
        code_pattern = r'```python\s*(.*?)```'
        code_match = re.search(code_pattern, response_text, re.DOTALL | re.IGNORECASE)
        
        if code_match:
            code = code_match.group(1).strip()
        else:
            # Fallback: try generic code blocks
            generic_pattern = r'```\s*(.*?)```'
            generic_match = re.search(generic_pattern, response_text, re.DOTALL)
            code = generic_match.group(1).strip() if generic_match else response_text.strip()
        
        # Extract explanation (text after code block)
        explanation_parts = response_text.split('```')
        explanation = explanation_parts[-1].strip() if len(explanation_parts) > 1 else "Generated Python code based on requirements."
        
        if not explanation or explanation == code:
            explanation = "Generated Python code based on requirements."
            
        return {
            'code': code,
            'explanation': explanation
        }
        
    except Exception as e:
        logger.error(f"Error parsing code response: {e}")
        return {
            'code': response_text.strip(),
            'explanation': "Generated Python code."
        }


async def _execute_code_safely(code: str) -> Dict[str, Any]:
    """Execute Python code safely using PythonREPLTool in a separate thread."""
    try:
        # Execute code using the centralized tool
        result = await execute_tool_async("execute_python_code", code=code)
        
        # The tool returns a structured response
        return {
            'status': result.get('status', 'success'),
            'output': result.get('output', ''),
            'error': result.get('error'),
            'execution_time': result.get('execution_time', 0)
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'output': '',
            'error': str(e),
            'execution_time': 0
        }


def _validate_execution_result(execution_result: Dict[str, Any], requirements: str) -> Dict[str, Any]:
    """Validate execution results against requirements."""
    status = execution_result.get('status', 'error')
    error = execution_result.get('error')
    output = execution_result.get('output', '')
    
    is_valid = (
        status == 'success' and 
        error is None and
        len(output) >= 0  # Allow empty output for some code
    )
    
    return {
        'valid': is_valid,
        'has_output': bool(output),
        'has_errors': bool(error),
        'feedback': error if error else 'Code executed successfully'
    }


def _make_finish_decision(code_gen: Dict[str, Any], max_attempts: int) -> Dict[str, str]:
    """Decide whether to finish or continue refining."""
    attempt = code_gen.get('attempt', 1)
    execution_result = code_gen.get('execution_result', {})
    validation_result = code_gen.get('validation_result', {})
    
    logger.info(f"Making finish decision - Attempt: {attempt}/{max_attempts}")
    logger.info(f"Execution result: {execution_result}")
    logger.info(f"Validation result: {validation_result}")
    
    # Check if maximum attempts reached
    if attempt >= max_attempts:
        logger.info("Max attempts reached - finishing with failure")
        return {'action': 'finish_failure'}
    
    # Check if execution was successful - be more strict
    execution_status = execution_result.get('status')
    execution_error = execution_result.get('error')
    validation_valid = validation_result.get('valid', False)
    
    if (execution_status == 'success' and 
        not execution_error and 
        validation_valid):
        logger.info("Code execution successful - finishing with success")
        return {'action': 'finish_success'}
    
    # Continue refining if we have attempts left
    logger.info("Code has issues - continuing to refine")
    return {'action': 'continue_refining'}


async def _save_code_to_file(code: str, requirements: str, validation_result: Dict[str, Any]) -> str:
    """Save successful code to file using async file operations."""
    try:
        # Get tmps directory - same pattern as image generation
        # Current file is at: backend/demo/srcs/nodes/code_nodes.py
        # Target is: backend/demo/tmps
        tmps_dir = Path(__file__).parent.parent / "tmps"
        tmps_dir.mkdir(exist_ok=True)
        
        logger.info(f"Saving code to tmps directory: {tmps_dir}")
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = re.sub(r'[^\w\s-]', '', requirements)[:30].replace(' ', '_')
        filename = f"generated_{timestamp}_{safe_name}.py"
        file_path = tmps_dir / filename
        
        # Add header and save using async file operations
        header = f"""# AI Generated Python Code
                # Requirements: {requirements}
                # Generated: {timestamp}
                # ===================================

                """
        
        # Use aiofiles for async file writing
        async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
            await f.write(header + code)
        
        logger.info(f"Code saved to: {file_path}")
        return str(file_path)
        
    except Exception as e:
        logger.error(f"Error saving code: {e}")
        return "error.py"


def _format_success_response(code_gen: Dict[str, Any], file_path: str) -> str:
    """Format successful completion response."""
    code = code_gen.get('code', '')
    explanation = code_gen.get('explanation', '')
    execution_result = code_gen.get('execution_result', {})
    
    response = f"""✅ **Python Code Generated Successfully!**

📋 **Requirements**: {code_gen.get('requirements', 'Code generation task')}
📁 **Saved to**: {Path(file_path).name}

📝 **Explanation**: {explanation}

📄 **Generated Code**:
```python
{code[:500] + '...' if len(code) > 500 else code}
```"""

    # Add execution output if available
    if execution_result.get('output'):
        response += f"""

🔍 **Execution Output**:
```
{execution_result['output'][:300] + '...' if len(execution_result['output']) > 300 else execution_result['output']}
```"""

    response += "\n\n🎯 **Your Python code has been saved to the tmps folder for you to use!**"
    return response


def _format_failure_response(code_gen: Dict[str, Any], max_attempts: int) -> str:
    """Format failure response after max attempts."""
    errors = []
    execution_result = code_gen.get('execution_result', {})
    
    if execution_result.get('error'):
        errors.append(execution_result['error'])
    
    return f"""❌ **Code Generation Failed**

After {max_attempts} attempts, I couldn't generate working Python code.

**Issues encountered:**
{chr(10).join(f'• {error}' for error in errors) if errors else '• Unknown errors occurred'}

Please try rephrasing your requirements or break down the task into smaller parts."""


def _extract_error_context(code_gen: Dict[str, Any]) -> str:
    """Extract error context for iteration feedback."""
    execution_result = code_gen.get('execution_result', {})
    error = execution_result.get('error', 'Unknown error')
    return error[:200] + '...' if len(error) > 200 else error