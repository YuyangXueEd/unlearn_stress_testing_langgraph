"""
Routing Nodes

Nodes responsible for determining the flow and routing messages to appropriate handlers.
"""

import logging
from typing import Dict
from state import ChatState
from configuration import DemoConfiguration

logger = logging.getLogger(__name__)

# Configuration instance for model settings
_config = DemoConfiguration()


def _get_model_name(state: ChatState) -> str:
    """Get the model name from state or configuration."""
    return state.get("model_name", _config.model_name)


def router_node(state: ChatState) -> ChatState:
    """
    Router node that determines the appropriate task type for the user message.
    
    This node analyzes the user message and routes to:
    - image_generation: For image creation requests
    - code_generation: For code writing and programming requests
    - database_search: For research paper queries and information retrieval
    - conversation: For general chat
    
    Args:
        state: Current chat state
        
    Returns:
        Updated state with routing decision
    """
    try:
        user_message = state.get("user_message", "")
        if not user_message and state.get("messages"):
            # Extract from messages if available
            last_message = state["messages"][-1]
            if hasattr(last_message, 'content'):
                user_message = last_message.content
        
        # Determine task type based on message content
        if _is_stress_testing_request(user_message):
            task_type = "stress_testing"
            # Extract stress testing context
            stress_context = _extract_stress_testing_components(user_message, state)
        elif _is_image_generation_request(user_message):
            task_type = "image_generation"
        elif _is_code_generation_request(user_message):
            task_type = "code_generation"
        elif _is_database_search_request(user_message):
            task_type = "database_search"
        else:
            task_type = "conversation"
        
        logger.info(f"Router decision: {task_type} for message: {user_message[:50]}...")
        
        # Update the existing state instead of returning a partial dictionary
        state["task_type"] = task_type
        state["user_message"] = user_message  # Ensure user_message is in state
        
        # Add stress testing context if it's a stress testing task
        if task_type == "stress_testing":
            state["stress_testing"] = stress_context
        
        return state
        
    except Exception as e:
        logger.error(f"Error in router node: {e}")
        # Default to conversation on error and preserve existing state
        state["task_type"] = "conversation"
        if user_message:
            state["user_message"] = user_message
        return state


def _is_database_search_request(message: str) -> bool:
    """
    Check if the user message is requesting database/paper search.
    
    Args:
        message: User message to analyze
        
    Returns:
        True if this appears to be a database search request
    """
    search_keywords = [
        # Direct search keywords
        "search", "find", "look for", "search for", "find information",
        "what does the paper say", "according to the paper", "in the research",
        "from the papers", "paper about", "research about", "study about",
        
        # Academic/research terms
        "literature", "publication", "research", "study", "analysis",
        "experiment", "methodology", "results", "findings", "conclusion",
        "abstract", "introduction", "related work", "evaluation",
        
        # Question patterns about research
        "what is", "how does", "explain", "describe", "tell me about",
        "what are the", "how to", "why does", "when was", "who proposed",
        
        # Technical terms that might be in papers
        "algorithm", "model", "method", "approach", "technique", "framework",
        "dataset", "benchmark", "evaluation", "performance", "accuracy",
        "neural network", "machine learning", "deep learning", "AI",
        "artificial intelligence", "computer vision", "NLP", "natural language"
    ]
    
    message_lower = message.lower()
    
    # First check for exact keyword matches
    if any(keyword in message_lower for keyword in search_keywords):
        return True
    
    # Check for question patterns that suggest academic inquiry
    import re
    
    academic_patterns = [
        r'\b(what|how|why|when|where|who)\s+(is|are|does|do|did|was|were)\s+.*(method|algorithm|model|approach|technique|research|study)',
        r'\b(explain|describe|tell me about|discuss)\s+.*(method|algorithm|model|approach|technique)',
        r'\b(paper|research|study|literature)\s+(says?|shows?|demonstrates?|proves?|suggests?)',
        r'\baccording to\s+(the\s+)?(paper|research|study|literature)',
        r'\bin\s+the\s+(paper|research|study|literature)',
        r'\bwhat\s+(is|are)\s+the\s+(results?|findings?|conclusions?)',
    ]
    
    for pattern in academic_patterns:
        if re.search(pattern, message_lower):
            return True
    
    return False


def _is_image_generation_request(message: str) -> bool:
    """
    Check if the user message is requesting image generation.
    
    Args:
        message: User message to analyze
        
    Returns:
        True if this appears to be an image generation request
    """
    image_keywords = [
        # Direct generation keywords
        "generate image", "create image", "generate picture", "create picture",
        "generate an image", "create an image", "generate a picture", "create a picture",
        "make image", "make picture", "make an image", "make a picture", "generate me"
        
        # Photo-related keywords
        "generate photo", "create photo", "generate a photo", "create a photo",
        "make photo", "make a photo", "photo of", "picture of",
        "generate me a photo", "generate me an image", "generate me a picture",
        "create me a photo", "create me an image", "create me a picture",
        
        # Art and drawing keywords
        "draw", "paint", "sketch", "illustrate", "visualize",
        "draw me", "paint me", "sketch me", "illustrate me",
        "draw a", "paint a", "sketch a", "illustrate a",
        
        # Show/display keywords
        "show me", "show me a", "display", "render"
    ]
    
    message_lower = message.lower()
    
    # First check for exact keyword matches
    if any(keyword in message_lower for keyword in image_keywords):
        return True
    
    # Additional pattern-based detection
    import re
    
    # Check for patterns like "generate [something] of [description]"
    generate_patterns = [
        r'\b(generate|create|make|draw|paint)\s+(me\s+)?(a\s+)?(photo|image|picture|drawing|painting|sketch)\s+(of|showing|depicting)',
        r'\b(photo|image|picture|drawing|painting|sketch)\s+of\b',
        r'\bgenerate\s+me\s+(a\s+)?(photo|image|picture)\b',
        r'\bcreate\s+me\s+(a\s+)?(photo|image|picture)\b'
    ]
    
    for pattern in generate_patterns:
        if re.search(pattern, message_lower):
            return True
    
    return False


def _is_code_generation_request(message: str) -> bool:
    """
    Check if the user message is requesting code generation.
    
    Args:
        message: User message to analyze
        
    Returns:
        True if this appears to be a code generation request
    """
    code_keywords = [
        # Direct code generation keywords
        "write code", "generate code", "create code", "make code", "code",
        "write a script", "generate a script", "create a script", "make a script",
        "write a program", "generate a program", "create a program", "make a program",
        "write a function", "create a function", "generate a function",
        
        # Code generation with "for" patterns - ENHANCED
        "generate code for", "write code for", "create code for", "make code for",
        "generate a script for", "write a script for", "create a script for",
        "generate a program for", "write a program for", "create a program for",
        "generate a function for", "write a function for", "create a function for",
        "code for", "script for", "program for", "function for",
        
        # Code modification keywords
        "write me code", "generate me code", "create me code",
        "write me a script", "generate me a script", "create me a script",
        "write me a program", "generate me a program", "create me a program",
        "write me a function", "create me a function", "generate me a function",
        
        # Programming language specific
        "write python", "write javascript", "write java", "write c++", "write go",
        "python code", "javascript code", "java code", "c++ code", "go code",
        "python script", "javascript script", "java script", "bash script",
        
        # Development tasks - ENHANCED
        "build a", "develop a", "implement", "code for", "script for",
        "programming", "algorithm", "function", "class", "module",
        "implement a", "implement an", "solve", "calculate", "compute",
        
        # Mathematical operations - NEW
        "add operation", "addition", "subtraction", "multiplication", "division",
        "arithmetic", "calculator", "compute", "calculation",
        
        # Common programming patterns - ENHANCED
        "write a web app", "create a web app", "build a web app",
        "write an api", "create an api", "build an api",
        "write a bot", "create a bot", "build a bot",
        "automate", "parse", "scrape", "process data", "sort", "filter",
        "data processing", "file handling", "database operations"
    ]
    
    message_lower = message.lower()
    
    # First check for exact keyword matches
    keyword_found = any(keyword in message_lower for keyword in code_keywords)
    logger.info(f"Code detection - Message: '{message}', Keyword found: {keyword_found}")
    
    if keyword_found:
        return True
    
    # Check for programming language mentions with action words
    programming_languages = [
        "python", "javascript", "java", "c++", "cpp", "c", "go", "rust", 
        "typescript", "php", "ruby", "swift", "kotlin", "scala", "r",
        "sql", "html", "css", "bash", "shell", "powershell", "matlab", "julia"
    ]
    
    action_words = [
        "write", "create", "generate", "make", "build", "develop", "code", "script"
    ]
    
    # Check for patterns like "write [language]" or "[language] code"
    import re
    
    for lang in programming_languages:
        for action in action_words:
            # Pattern: "write python" or "python code"
            if re.search(rf'\b({action}\s+{lang}|{lang}\s+(code|script))\b', message_lower):
                return True
    
    # Check for specific programming task patterns
    programming_patterns = [
        r'\b(write|create|generate|make|build)\s+(a\s+)?(function|class|module|library|package)\b',
        r'\b(implement|code)\s+(a\s+)?(algorithm|solution|program)\b',
        r'\b(automate|script)\s+.*(task|process|workflow)\b',
        r'\b(parse|scrape|extract)\s+.*(data|information|content)\b',
        r'\b(build|create|make)\s+.*(app|application|tool|utility)\b',
        r'\b(generate|write|create)\s+(code|script|program)\s+(for|to)\b',
        r'\b(code|script|program)\s+(for|to)\s+.*(add|calculate|compute|process|handle)\b'
    ]
    
    for pattern in programming_patterns:
        if re.search(pattern, message_lower):
            return True
    
    return False


def _is_stress_testing_request(message: str) -> bool:
    """
    Check if the user message is requesting stress testing.
    
    Args:
        message: User message to analyze
        
    Returns:
        True if this appears to be a stress testing request
    """
    stress_keywords = [
        # Direct stress testing keywords
        "stress test", "stress-test", "stress testing", "stress-testing",
        "run a stress test", "perform stress test", "execute stress test",
        "conduct stress test", "do stress test",
        
       
        # Model testing patterns
        "test the", "verify the", "check the", "evaluate the",
        "stress test the", "stress-test the",
        
        # Concept erasure specific
        "concept erasure", "concept removal", "concept unlearning",
        "erasure of", "removal of", "unlearning of"
    ]
    
    message_lower = message.lower()
    
    # First check for exact keyword matches
    if any(keyword in message_lower for keyword in stress_keywords):
        return True
    
    # Check for specific patterns like "stress-test the XXX erasure of YYY on ZZZ model"
    import re
    
    stress_patterns = [
        r'\b(stress.?test|verify|test|check|evaluate)\s+.*(erasure|removal|unlearning)\s+of\s+["\']?(\w+)["\']?\s+(on|from|in)\s+["\']?(\w+)["\']?\s+model',
        r'\brun\s+a\s+stress\s+test\s+to\s+(verify|test|check|evaluate)',
        r'\bstress.?test\s+the\s+\w+\s+erasure',
        r'\b(verify|test|check|evaluate)\s+.*concept.*erasure',
        r'\b(stress.?test|test)\s+.*unlearn',
    ]
    
    for pattern in stress_patterns:
        if re.search(pattern, message_lower):
            return True
    
    return False


def _extract_stress_testing_components(message: str, state: ChatState) -> Dict[str, str]:
    """
    Extract stress testing components from user message using LLM.
    
    Uses an LLM to intelligently parse the user's stress testing request
    and extract the target concept, erasure method, and model.
    
    Args:
        message: User message containing stress testing request
        state: Current chat state for accessing configuration
        
    Returns:
        Dictionary with concept, method, and model
    """
    import json
    from langchain_ollama import ChatOllama
    
    # Initialize default components
    components = {
        "concept": "",
        "method": "general",
        "model": "stable diffusion"
    }
    
    try:
        # Create extraction prompt
        extraction_prompt = f"""You are an expert in machine unlearning and concept erasure from diffusion models. 

Your task is to extract key components from a stress testing request.

USER MESSAGE: "{message}"

Extract the following components:
1. CONCEPT: The target concept/object to be erased (e.g., "pikachu", "van gogh style", "nudity", "violence")
2. METHOD: The erasure method being tested (e.g., "ESD", "FMN", "SLD", "DARE", "TOFU", "Exact Unlearning", or "general" if not specified)
3. MODEL: The target model (e.g., "stable diffusion", "SDXL", "midjourney", or infer from context)

IMPORTANT RULES:
- For CONCEPT: Extract the main concept being erased, remove quotes if present
- For METHOD: Use "general" if no specific method is mentioned
- For MODEL: Default to "stable diffusion" if not explicitly mentioned
- Return ONLY a valid JSON object, no additional text

OUTPUT FORMAT:
{{
    "concept": "extracted concept",
    "method": "extracted method or general",
    "model": "extracted model"
}}

Extract the components now:"""

        # Call LLM for extraction
        llm = ChatOllama(
            model=_get_model_name(state),
            temperature=0.0,
            base_url="http://localhost:11434"
        )
        
        response = llm.invoke(extraction_prompt)
        response_text = response.content.strip()
        
        # Parse JSON response
        try:
            # Find JSON in response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                extracted = json.loads(json_str)
                
                # Update components with extracted values
                if extracted.get("concept"):
                    components["concept"] = extracted["concept"].strip().strip('"\'')
                if extracted.get("method"):
                    components["method"] = extracted["method"].strip().lower()
                if extracted.get("model"):
                    components["model"] = extracted["model"].strip().lower()
                    
                logger.info(f"LLM extracted components: {components}")
                
            else:
                logger.warning("No JSON found in LLM response, using fallback extraction")
                components = _fallback_extraction(message)
                
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM JSON response: {e}, using fallback extraction")
            components = _fallback_extraction(message)
            
    except Exception as e:
        logger.error(f"Error in LLM extraction: {e}, using fallback extraction")
        components = _fallback_extraction(message)
    
    # Ensure we have at least a concept
    if not components["concept"]:
        logger.warning("No concept extracted, using fallback")
        components = _fallback_extraction(message)
    
    return components


def _fallback_extraction(message: str) -> Dict[str, str]:
    """
    Fallback extraction using simple regex patterns as backup.
    
    Args:
        message: User message
        
    Returns:
        Dictionary with extracted components
    """
    import re
    
    components = {
        "concept": "",
        "method": "general",
        "model": "stable diffusion"
    }
    
    message_lower = message.lower()
    
    # Simple concept extraction from common patterns
    concept_patterns = [
        r'erasure\s+of\s+["\']?([^"\']+?)["\']?',
        r'unlearning\s+of\s+["\']?([^"\']+?)["\']?',
        r'remove\s+["\']?([^"\']+?)["\']?',
        r'erase\s+["\']?([^"\']+?)["\']?',
        r'test.*["\']([^"\']+)["\']'
    ]
    
    for pattern in concept_patterns:
        match = re.search(pattern, message_lower)
        if match:
            components["concept"] = match.group(1).strip()
            break
    
    # Simple model extraction
    model_pattern = r'(?:on|from|in)\s+(?:the\s+)?["\']?([^"\']+?)["\']?\s+model'
    model_match = re.search(model_pattern, message_lower)
    if model_match:
        components["model"] = model_match.group(1).strip()
    
    # Method keywords
    method_keywords = ["esd", "fmn", "sld", "dare", "tofu", "exact"]
    for keyword in method_keywords:
        if keyword in message_lower:
            components["method"] = keyword
            break
    
    return components
