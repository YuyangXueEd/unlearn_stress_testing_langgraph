import os
import requests
from bs4 import BeautifulSoup
import urllib.parse
from fastapi import HTTPException

from dotenv import load_dotenv
from duckduckgo_search import DDGS
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from agent.configuration import Configuration
from agent.prompts import (
    answer_instructions,
    get_current_date,
    query_writer_instructions,
    reflection_instructions,
)
from agent.state import (
    OverallState,
    QueryGenerationState,
    ReflectionState,
    WebSearchState,
)
from agent.tools_and_schemas import Reflection, SearchQueryList
from agent.utils import get_research_topic

load_dotenv()


DEFAULT_SEARCH_ENGINE_TIMEOUT = 100
REFERENCE_COUNT = 5
GOOGLE_SEARCH_ENDPOINT = "https://customsearch.googleapis.com/customsearch/v1"


# Nodes
def generate_query(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
    """LangGraph node that generates search queries based on the User's question.

    Uses the configured LLM to create optimized search queries for web research
    based on the user's question.

    Args:
        state: Current graph state containing the User's question
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated queries
    """
    configurable = Configuration.from_runnable_config(config, base_model=state.get("reasoning_model"))

    initial_search_query_count = state["initial_search_query_count"] or configurable.number_of_initial_queries
    
    # Initialize ChatOllama
    # Ensure Ollama server is running and the model specified in 
    # configurable.query_generator_model (e.g., "qwen:7b-chat") is available.
    chat = ChatOllama(
        model=configurable.query_generator_model, 
        temperature=1.0, # You can adjust temperature and other parameters
        base_url="localhost:11434", # If OLLAMA_BASE_URL is set in env or Configuration
    )
    
    structured_llm = chat.with_structured_output(SearchQueryList) # Simpler call, may need json_mode or parser

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        number_queries=initial_search_query_count,
    )
    # Generate the search queries
    result = structured_llm.invoke(formatted_prompt)
    return {"search_query": result.query}


def continue_to_web_research(state: QueryGenerationState):
    """LangGraph node that sends the search queries to the web research node.

    This is used to spawn n number of web research nodes, one for each search query.
    """
    return [
        Send("web_research", {"search_query": search_query, "id": int(idx)})
        for idx, search_query in enumerate(state["search_query"])
    ]

DEFAULT_SEARCH_ENGINE_TIMEOUT = 100
REFERENCE_COUNT = 5
GOOGLE_SEARCH_ENDPOINT = "https://customsearch.googleapis.com/customsearch/v1"

def search_with_google(query: str, subscription_key: str, cx: str):
    """
    Search with google and return the contexts.
    """
    params = {
        "key": subscription_key,
        "cx": cx,
        "q": query,
        "num": REFERENCE_COUNT,
    }
    response = requests.get(
        GOOGLE_SEARCH_ENDPOINT, params=params, timeout=DEFAULT_SEARCH_ENGINE_TIMEOUT
    )
    if not response.ok:
        print(f"{response.status_code} {response.text}")
        raise HTTPException(response.status_code, "Search engine error.")
    json_content = response.json()
    try:
        contexts = json_content["items"][:REFERENCE_COUNT]
    except KeyError:
        print(f"Error encountered: {json_content}")
        return []
    return contexts



def web_research(state: WebSearchState, config: RunnableConfig) -> OverallState:
    """LangGraph node that performs web research using DuckDuckGo.

    Executes a web search using the DuckDuckGo Search API and formats the
    results for later processing.

    Args:
        state: Current graph state containing the search query and research loop count
        config: Configuration for the runnable, including search API settings

    Returns:
        Dictionary with state update, including sources_gathered, research_loop_count, and web_research_results
    """
    query = state["search_query"]

    search_api_key = os.environ["GOOGLE_SEARCH_API_KEY"]
    # with DDGS() as ddgs:
    #     print(query, "xxx")
    #     results = list(ddgs.text(query, max_results=5))

    results = search_with_google(query, search_api_key, 
    os.environ["GOOGLE_SEARCH_CX"])

    formatted_lines = []
    sources_gathered = []
    for res in results:
        # print(res, "qqq")
        title = res.get("title", "")
        # href = res.get("href", "")
        # body = res.get("body", "")
        href = res.get("link", "")
        body = res.get("snippet", "")
        formatted_lines.append(f"{title}: {body} ({href})")
        sources_gathered.append({"label": title, "short_url": href, "value": href})

    modified_text = "\n".join(formatted_lines)

    return {
        "sources_gathered": sources_gathered,
        "search_query": [query],
        "web_research_result": [modified_text],
    }


def reflection(state: OverallState, config: RunnableConfig) -> ReflectionState:
    """LangGraph node that identifies knowledge gaps and generates potential follow-up queries.

    Analyzes the current summary to identify areas for further research and generates
    potential follow-up queries. Uses structured output to extract
    the follow-up query in JSON format.

    Args:
        state: Current graph state containing the running summary and research topic
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated follow-up query
    """
    configurable = Configuration.from_runnable_config(config, base_model=state.get("reasoning_model"))

    research_loop_count = state.get("research_loop_count", 0) + 1

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = reflection_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n\n---\n\n".join(state["web_research_result"]),
    )

    # Initialize ChatOllama for reflection
    chat_reflection = ChatOllama(
        model=configurable.reflection_model,
        temperature=1.0,
        base_url="localhost:11434", # If OLLAMA_BASE_URL is set in env or Configuration
    )
    
    # Assuming for now that with_structured_output with Ollama might work directly
    structured_llm_reflection = chat_reflection.with_structured_output(Reflection)
    result = structured_llm_reflection.invoke(formatted_prompt)


    return {
        "is_sufficient": result.is_sufficient,
        "knowledge_gap": result.knowledge_gap,
        "follow_up_queries": result.follow_up_queries,
        "research_loop_count": research_loop_count,
        "number_of_ran_queries": len(state["search_query"]),
    }


def evaluate_research(
    state: ReflectionState,
    config: RunnableConfig,
) -> OverallState:
    """LangGraph routing function that determines the next step in the research flow.

    Controls the research loop by deciding whether to continue gathering information
    or to finalize the summary based on the configured maximum number of research loops.

    Args:
        state: Current graph state containing the research loop count
        config: Configuration for the runnable, including max_research_loops setting

    Returns:
        String literal indicating the next node to visit ("web_research" or "finalize_summary")
    """
    configurable = Configuration.from_runnable_config(config, base_model=state.get("reasoning_model"))

    max_research_loops = (
        state.get("max_research_loops")
        if state.get("max_research_loops") is not None
        else configurable.max_research_loops
    )
    if state["is_sufficient"] or state["research_loop_count"] >= max_research_loops:
        return "finalize_answer"
    else:
        return [
            Send(
                "web_research",
                {
                    "search_query": follow_up_query,
                    "id": state["number_of_ran_queries"] + int(idx),
                },
            )
            for idx, follow_up_query in enumerate(state["follow_up_queries"])
        ]


def finalize_answer(state: OverallState, config: RunnableConfig):
    """LangGraph node that finalizes the research summary.

    Prepares the final output by deduplicating and formatting sources, then
    combining them with the running summary to create a well-structured
    research report with proper citations.

    Args:
        state: Current graph state containing the running summary and sources gathered

    Returns:
        Dictionary with state update, including running_summary key containing the formatted final summary with sources
    """
    configurable = Configuration.from_runnable_config(config, base_model=state.get("reasoning_model"))

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = answer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n---\n\n".join(state["web_research_result"]),
    )

    # Initialize ChatOllama for final answer
    chat_finalize = ChatOllama(
        model=configurable.answer_model,
        temperature=1.0, # Adjust as needed
        base_url="localhost:11434", # If OLLAMA_BASE_URL is set in env or Configuration
    )
    result = chat_finalize.invoke(formatted_prompt)
    # Replace the short urls with the original urls and add all used urls to the sources_gathered

    unique_sources = []
    for source in state["sources_gathered"]:
        if source["short_url"] in result.content:
            result.content = result.content.replace(
                source["short_url"], source["value"]
            )
            unique_sources.append(source)

    return {
        "messages": [AIMessage(content=result.content)],
        "sources_gathered": unique_sources,
    }


# Create our Agent Graph
builder = StateGraph(OverallState, config_schema=Configuration)

# Define the nodes we will cycle between
builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("reflection", reflection)
builder.add_node("finalize_answer", finalize_answer)

# Set the entrypoint as `generate_query`
# This means that this node is the first one called
builder.add_edge(START, "generate_query")
# Add conditional edge to continue with search queries in a parallel branch
builder.add_conditional_edges(
    "generate_query", continue_to_web_research, ["web_research"]
)
# Reflect on the web research
builder.add_edge("web_research", "reflection")
# Evaluate the research
builder.add_conditional_edges(
    "reflection", evaluate_research, ["web_research", "finalize_answer"]
)
# Finalize the answer
builder.add_edge("finalize_answer", END)

graph = builder.compile(name="pro-search-agent")
