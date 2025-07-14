"""
Routing Nodes

Nodes responsible for determining the flow and routing messages to appropriate handlers.
"""

import logging
from typing import Dict
from state import ChatState

logger = logging.getLogger(__name__)


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
            stress_context = _extract_stress_testing_components(user_message)
        elif _is_image_generation_request(user_message):
            task_type = "image_generation"
        elif _is_code_generation_request(user_message):
            task_type = "code_generation"
        elif _is_database_search_request(user_message):
            task_type = "database_search"
        else:
            task_type = "conversation"
        
        logger.info(f"Router decision: {task_type} for message: {user_message[:50]}...")
        
        # Prepare state update
        state_update = {
            "task_type": task_type,
            "user_message": user_message  # Ensure user_message is in state
        }
        
        # Add stress testing context if it's a stress testing task
        if task_type == "stress_testing":
            state_update["stress_testing"] = stress_context
        
        return state_update
        
    except Exception as e:
        logger.error(f"Error in router node: {e}")
        # Default to conversation on error
        return {
            "task_type": "conversation",
            "user_message": user_message
        }


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
        "make image", "make picture", "make an image", "make a picture",
        
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
        
        # Verification and testing keywords
        "verify erasure", "test erasure", "validate erasure", "check erasure",
        "test unlearning", "verify unlearning", "validate unlearning",
        "check unlearning", "evaluate erasure", "evaluate unlearning",
        
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


def _extract_stress_testing_components(message: str) -> Dict[str, str]:
    """
    Extract stress testing components from user message.
    
    Patterns:
    - "Stress-test the YYY erasure of 'XXX' on the 'ZZZ' model"
    - "Run a stress test to verify the YYY erasure of 'XXX' from 'ZZZ' model"
    
    Args:
        message: User message containing stress testing request
        
    Returns:
        Dictionary with concept, method, and model
    """
    import re
    
    # Initialize components
    components = {
        "concept": "",
        "method": "",
        "model": ""
    }
    
    message_lower = message.lower()
    
    # Pattern 1: "stress-test the METHOD erasure of 'CONCEPT' on the 'MODEL' model"
    pattern1 = r'stress.?test\s+the\s+(\w+)\s+erasure\s+of\s+["\']?([^"\']+?)["\']?\s+on\s+(?:the\s+)?["\']?([^"\']+?)["\']?\s+model'
    match1 = re.search(pattern1, message_lower)
    
    if match1:
        components["method"] = match1.group(1).strip()
        components["concept"] = match1.group(2).strip()
        components["model"] = match1.group(3).strip()
        return components
    
    # Pattern 1.5: "stress-test the erasure of 'CONCEPT' on the 'MODEL' model" (no method specified)
    pattern1_5 = r'stress.?test\s+the\s+erasure\s+of\s+["\']([^"\']+)["\']?\s+on\s+(?:the\s+)?["\']?([^"\']+?)["\']?\s+model'
    match1_5 = re.search(pattern1_5, message_lower)
    
    if match1_5:
        components["concept"] = match1_5.group(1).strip()
        components["model"] = match1_5.group(2).strip()
        components["method"] = "general"  # Default method when not specified
        return components
    
    # Pattern 2: "verify the METHOD erasure of 'CONCEPT' from 'MODEL' model"
    pattern2 = r'(verify|test|check|evaluate)\s+the\s+(\w+)\s+erasure\s+of\s+["\']?([^"\']+?)["\']?\s+from\s+["\']?([^"\']+?)["\']?\s+model'
    match2 = re.search(pattern2, message_lower)
    
    if match2:
        components["method"] = match2.group(2).strip()
        components["concept"] = match2.group(3).strip()
        components["model"] = match2.group(4).strip()
        return components
    
    # Pattern 3: "stress test CONCEPT erasure on MODEL" (no quotes)
    pattern3 = r'stress.?test\s+(?:the\s+)?erasure\s+of\s+([a-zA-Z0-9_\-\s]+?)\s+on\s+([a-zA-Z0-9\-\s\.]+?)(?:\s*$|\s+model|\s*\.)'
    match3 = re.search(pattern3, message_lower)
    
    if match3:
        components["concept"] = match3.group(1).strip()
        components["model"] = match3.group(2).strip()
        return components
    
    # Pattern 4: "test the unlearning of CONCEPT concept on MODEL"
    pattern4 = r'(test|verify|check|evaluate)\s+(?:the\s+)?unlearning\s+of\s+([a-zA-Z0-9_\-\s]+?)\s+concept\s+on\s+([a-zA-Z0-9\-\s\.]+?)(?:\s*$|\s+model|\s*\.)'
    match4 = re.search(pattern4, message_lower)
    
    if match4:
        components["concept"] = match4.group(2).strip()
        components["model"] = match4.group(3).strip()
        components["method"] = "general"
        return components
    
    # Pattern 5: General concept and model extraction
    concept_pattern = r'erasure\s+of\s+["\']?([^"\']+?)["\']?'
    concept_match = re.search(concept_pattern, message_lower)
    if concept_match:
        components["concept"] = concept_match.group(1).strip()
    
    model_pattern = r'(?:on|from|in)\s+(?:the\s+)?["\']?([^"\']+?)["\']?\s+model'
    model_match = re.search(model_pattern, message_lower)
    if model_match:
        components["model"] = model_match.group(1).strip()
    
    # Method extraction (if not found above)
    if not components["method"]:
        method_keywords = ["esd", "fmn", "sld", "dare", "tofu", "exact", "gradient", "latent"]
        for keyword in method_keywords:
            if keyword in message_lower:
                components["method"] = keyword
                break
    
    return components
