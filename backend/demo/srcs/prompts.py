"""
Demo Prompts

Prompt templates for the simple chatbot demo including reflection and final answer prompts.
"""

from datetime import datetime


def get_current_date():
    """Get current date in a readable format."""
    return datetime.now().strftime("%B %d, %Y")


def get_chat_prompt(user_message: str, conversation_context: str = "") -> str:
    """
    Get the main chat prompt for responding to user messages.
    
    Args:
        user_message: The user's input message
        conversation_context: Previous conversation context
        
    Returns:
        Formatted prompt for the LLM
    """
    base_prompt = """You are a helpful AI assistant. You can help with various topics including:
- General questions and conversations
- Research and information
- Technical topics
- Creative tasks

"""
    
    if conversation_context:
        base_prompt += f"""Here is our conversation so far:
{conversation_context}

Please continue this conversation by responding to the latest message."""
    else:
        base_prompt += f"""Please provide a helpful and informative response to the following question:

{user_message}"""
    
    base_prompt += "\n\nKeep your response clear, concise, and helpful."
    
    return base_prompt


def get_error_prompt(error_message: str) -> str:
    """
    Get a prompt for handling errors gracefully.
    
    Args:
        error_message: The error that occurred
        
    Returns:
        User-friendly error message
    """
    return f"Sorry, I encountered an error while processing your request: {error_message}. Please try rephrasing your question or try again later."


def get_fallback_prompt() -> str:
    """
    Get a fallback prompt when no user message is received.
    
    Returns:
        Fallback response message
    """
    return "I didn't receive a message. Please try again with your question or request."


# RAG Database Search Reflection and Final Answer Prompts

REFLECTION_PROMPT = """You are an expert AI safety researcher analyzing search results about: "{research_topic}".

Your task is to determine if the search results provide sufficient information to answer the user's question, or if additional searches are needed.

Instructions:
- Analyze the provided search results to identify knowledge gaps or areas that need deeper exploration
- If the search results are sufficient to provide a comprehensive answer, mark as sufficient
- If there are knowledge gaps, generate specific follow-up queries that would help fill those gaps
- Focus on actionable information that can help provide a complete answer
- Consider if enough information is available to formulate a comprehensive response

Current search iteration: {search_iteration}/3 (max 3 iterations allowed)

Requirements:
- Be specific about what information is missing
- Generate follow-up queries that are distinct from previous searches
- Focus on getting information that would significantly improve the answer quality

Output Format:
- Format your response as a JSON object with these exact keys:
   - "is_sufficient": true or false
   - "knowledge_gap": Describe what information is missing (if any)
   - "follow_up_query": A single specific query to address the gap (if needed)

Example - Insufficient:
```json
{{
    "is_sufficient": false,
    "knowledge_gap": "The search results lack specific implementation details and code examples for the discussed methods",
    "follow_up_query": "implementation code examples for concept erasure in diffusion models"
}}
```

Example - Sufficient:
```json
{{
    "is_sufficient": true,
    "knowledge_gap": "",
    "follow_up_query": ""
}}
```

Previous Search Queries: {previous_queries}

Search Results to Analyze:
{search_results}
"""


FINAL_ANSWER_PROMPT = """You are an expert AI researcher providing a comprehensive answer based on research paper search results.

Your task is to synthesize the search results into a well-structured, informative response that directly addresses the user's question.

Instructions:
- The current date is {current_date}
- Analyze and synthesize ALL provided search results to create a comprehensive answer
- Structure your response logically with clear sections and headings
- Include specific details, methodologies, and findings from the papers
- Mention source papers when referencing specific information
- Focus on providing actionable insights and practical information
- Use markdown formatting for better readability

Your response should include:
1. **Summary**: Brief overview of the key findings
2. **Detailed Analysis**: In-depth analysis of the relevant information found
3. **Key Insights**: Important takeaways and implications
4. **Sources**: List of papers/sources referenced in your analysis

User Question: {research_topic}

Search Results to Synthesize:
{search_results}

Number of search iterations performed: {search_iteration}
"""
