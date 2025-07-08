import os
import requests
import logging
import time
import re
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
    rag_query_instructions,
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
from agent.rag_utils import RAGDatabase
from agent.rag_manager import get_rag_database

load_dotenv()

# Set up logger
logger = logging.getLogger(__name__)

# RAG database access is now managed by rag_manager module


DEFAULT_SEARCH_ENGINE_TIMEOUT = 100
REFERENCE_COUNT = 5
GOOGLE_SEARCH_ENDPOINT = "https://customsearch.googleapis.com/customsearch/v1"


# Nodes
def rag_query(state: OverallState, config: RunnableConfig) -> OverallState:
    """LangGraph node that generates search queries optimized for RAG database search.

    Uses the configured LLM to create search queries specifically optimized for 
    searching both academic papers and code implementations in the RAG database.
    The queries target research documents, Python files, Jupyter notebooks, and
    other implementation content related to machine unlearning and diffusion models.

    Args:
        state: Current graph state containing the User's question
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including rag_search_query key containing the generated queries
    """
    configurable = Configuration.from_runnable_config(config, base_model=state.get("reasoning_model"))

    initial_search_query_count = configurable.number_of_initial_queries or state["initial_search_query_count"]
    
    # Detect if this is a new conversation by checking if we have fresh messages
    messages = state.get("messages", [])
    is_new_conversation = True
    if messages:
        # Check if this looks like a new user query (simple heuristic)
        last_message = messages[-1]
        # If there's only one message or the last message is from user, it's likely a new conversation
        if len(messages) == 1 or (hasattr(last_message, 'type') and last_message.type == 'human'):
            is_new_conversation = True
        else:
            is_new_conversation = False
    
    if is_new_conversation:
        print("🔄 New conversation detected - resetting state flags")
    
    # Initialize ChatOllama for RAG query generation
    chat = ChatOllama(
        model=configurable.query_generator_model or state.get("reasoning_model"), 
        temperature=1.0,
        base_url="localhost:11434",
    )
    
    structured_llm = chat.with_structured_output(SearchQueryList)

    # Format the prompt specifically for RAG/academic search
    current_date = get_current_date()
    formatted_prompt = rag_query_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        number_queries=initial_search_query_count,
    )
    
    # Generate the RAG search queries
    result = structured_llm.invoke(formatted_prompt)
    
    print(f"🔍 Generated {len(result.query)} RAG search queries: {result.query}")
    
    # Detect paper titles in the research topic and queries
    detected_titles = []
    research_topic = get_research_topic(state["messages"])
    
    # Look for quoted paper titles or specific patterns that indicate academic papers
    paper_patterns = [
        r'"([^"]+)"',  # Quoted titles
        r'paper titled "([^"]+)"',  # "paper titled" pattern
        r'the paper "([^"]+)"',  # "the paper" pattern
        r'study "([^"]+)"',  # "study" pattern
        r'work "([^"]+)"',  # "work" pattern
        r'\b(\w+(?:\s+\w+)*)\s+et\s+al\.?',  # "Author et al." pattern (capture author)
        r'arXiv:(\d{4}\.\d{4,5})',  # arXiv ID pattern
    ]
    
    # Search in research topic
    for pattern in paper_patterns:
        matches = re.findall(pattern, research_topic, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                match = match[0] if match[0] else match[1]
            if len(match.strip()) > 5:  # Only consider substantial titles
                detected_titles.append(match.strip())
    
    # Also check if any query looks like a paper title (longer queries with capital words)
    for query in result.query:
        query_text = query if isinstance(query, str) else str(query)
        # Check if query has characteristics of a paper title
        if (len(query_text.split()) >= 3 and 
            any(word[0].isupper() for word in query_text.split() if len(word) > 2) and
            not query_text.lower().startswith(('how', 'what', 'why', 'when', 'where', 'which'))):
            detected_titles.append(query_text.strip())
    
    if detected_titles:
        print(f"📄 Detected potential paper titles: {detected_titles}")
    
    # Return state update with reset flags for new conversations
    return {
        "rag_search_query": result.query,
        "title": detected_titles,  # Store detected titles for paper search
        "rag_found": False,  # Always reset for new query
        "is_sufficient": False,  # Always reset for new query  
        "research_loop_count": 0,  # Always reset loop count
        "paper_found": False,  # Initialize paper search flags
        "papers_indexed": 0,
        "paper_search_attempted": False,  # Track if paper search has been tried
        "is_paper_search": False,  # Initialize paper search detection
        "paper_search_indicators": [],  # Initialize indicators list
        # Only clear results if it's truly a new conversation
        "rag_search_result": [] if is_new_conversation else state.get("rag_search_result", []),
        "web_research_result": [] if is_new_conversation else state.get("web_research_result", []),
        "sources_gathered": [] if is_new_conversation else state.get("sources_gathered", []),
        "search_query": [] if is_new_conversation else state.get("search_query", []),
    }


def generate_query(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
    """LangGraph node that generates search queries and determines if they are for academic papers.

    Uses the configured LLM to create optimized search queries for web research
    based on the user's question. The LLM also analyzes the research topic to 
    determine if this is an academic paper search or general web search.

    Args:
        state: Current graph state containing the User's question
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key and paper detection flags
    """
    configurable = Configuration.from_runnable_config(config, base_model=state.get("reasoning_model"))

    initial_search_query_count = configurable.number_of_initial_queries or state["initial_search_query_count"]
    
    # Initialize ChatOllama
    # Ensure Ollama server is running and the model specified in 
    # configurable.query_generator_model (e.g., "qwen:7b-chat") is available.
    chat = ChatOllama(
        model=configurable.query_generator_model or state.get("reasoning_model"), 
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
    # Generate the search queries and get paper search detection from LLM
    result = structured_llm.invoke(formatted_prompt)
    
    # The LLM now directly determines if this is a paper search
    is_paper_search = result.is_paper_search
    
    # Create indicators list based on LLM's analysis
    paper_indicators = [f"LLM analysis: {result.rationale}"]
    
    if is_paper_search:
        print(f"📄 LLM detected paper search: {result.rationale}")
    else:
        print(f"🌐 LLM detected general web search: {result.rationale}")
    
    return {
        "search_query": result.query,
        "is_paper_search": is_paper_search,
        "paper_search_indicators": paper_indicators
    }


def rag_search(state: OverallState, config: RunnableConfig) -> OverallState:
    """LangGraph node that searches the RAG database for relevant information.
    
    This node searches the local RAG database using the RAG-specific queries
    generated by the rag_query node. The database contains both academic papers
    and code implementations (Python files, Jupyter notebooks) related to 
    machine unlearning and diffusion models. If relevant information is found, 
    it formats it similarly to web search results.
    
    Args:
        state: Current graph state containing RAG search queries
        config: Configuration for the runnable
        
    Returns:
        Dictionary with state update including rag_results and rag_found flags
    """
    print("🔍 Starting RAG search...")
    
    # Check if we just indexed new papers from paper search
    papers_indexed = state.get("papers_indexed", 0)
    if papers_indexed > 0:
        print(f"📚 RAG search running after indexing {papers_indexed} new papers from Google search")
        logger.info(f"RAG search running after indexing {papers_indexed} new papers from Google search")
    else:
        logger.info("RAG search node started")
    
    
    try:
        print("📚 Getting RAG database instance...")
        rag_db = get_rag_database()
        
        if rag_db is None:
            print("⚠️  RAG database not initialized, will proceed to web search")
            logger.info("RAG database not initialized, proceeding to web search")
            
            # Record that database wasn't available
            no_db_turn = {
                "turn_timestamp": get_current_date(),
                "query_count": 0,
                "queries_used": [],
                "results": [],
                "sources": [],
                "note": "RAG database not initialized"
            }
            
            return {
                "rag_search_result": [no_db_turn],
                "sources_gathered": [],
                "rag_found": False,
                "research_loop_count": 0
            }
            
        print("✅ RAG database instance retrieved")
        logger.info("RAG database instance retrieved successfully")
        
        # Check if database is populated
        print("🔢 Checking database population...")
        is_populated = rag_db.is_database_populated()
        print(f"📊 Database populated: {is_populated}")
        
        if not is_populated:
            print("⚠️  RAG database is empty, will proceed to web search")
            logger.info("RAG database is empty, proceeding to web search")
            
            # Record that database was empty
            empty_db_turn = {
                "turn_timestamp": get_current_date(),
                "query_count": 0,
                "queries_used": [],
                "results": [],
                "sources": [],
                "note": "RAG database is empty"
            }
            
            return {
                "rag_search_result": [empty_db_turn],
                "sources_gathered": [],
                "rag_found": False,
                "research_loop_count": 0
            }
        
        # Get RAG search queries from state
        rag_queries = state.get("rag_search_query", [])
        if not rag_queries:
            print("⚠️  No RAG search queries found, will proceed to web search")
            logger.info("No RAG search queries found, proceeding to web search")
            
            # Record that no queries were provided
            no_queries_turn = {
                "turn_timestamp": get_current_date(),
                "query_count": 0,
                "queries_used": [],
                "results": [],
                "sources": [],
                "note": "No RAG search queries provided"
            }
            
            return {
                "rag_search_result": [no_queries_turn],
                "sources_gathered": [],
                "rag_found": False,
                "research_loop_count": 0
            }
        
        # Show document count
        try:
            collection = rag_db.client.get_collection(rag_db.collection_name)
            count = collection.count()
            print(f"📖 Database contains {count} document chunks")
            logger.info(f"Database contains {count} document chunks")
        except Exception as e:
            print(f"⚠️  Could not get document count: {e}")
        
        print(f"🔎 Processing {len(rag_queries)} RAG search queries...")
        logger.info(f"Processing {len(rag_queries)} RAG search queries")
        
        all_rag_results = []
        sources_gathered = []
        found_relevant = False
        
        # Search for each query in the database
        for i, query_obj in enumerate(rag_queries):
            query_text = query_obj if isinstance(query_obj, str) else query_obj.get("query", str(query_obj))
            print(f"🔍 RAG Query {i+1}/{len(rag_queries)}: '{query_text}'")
            logger.info(f"Searching RAG database for query: '{query_text}'")
            
            # Search the RAG database with a more permissive threshold
            print(f"   Searching with threshold 0.2...")
            rag_results = rag_db.search_database(query_text, k=3, score_threshold=0.2)
            print(f"   Found {len(rag_results)} results with threshold 0.2")
            
            if rag_results:
                found_relevant = True
                print(f"   ✅ Processing {len(rag_results)} relevant results")
                # Format results similar to web search
                formatted_lines = []
                for j, result in enumerate(rag_results):
                    content_preview = result["content"][:300] + "..." if len(result["content"]) > 300 else result["content"]
                    source = result["metadata"].get("filename", "Local Database")
                    score = result["score"]
                    print(f"      Result {j+1}: {source} (Score: {score:.3f})")
                    formatted_lines.append(f"Paper: {source} (Score: {score:.3f})\n{content_preview}")
                    
                    # Add source information
                    sources_gathered.append({
                        "label": f"RAG: {source}",
                        "short_url": f"local://{source}",
                        "value": f"Local Paper: {source}"
                    })
                
                combined_result = "\n\n---\n\n".join(formatted_lines)
                all_rag_results.append(combined_result)
                logger.info(f"Found {len(rag_results)} relevant documents in RAG database for query: {query_text}")
            else:
                # Try with an even lower threshold if nothing found
                print(f"   No results with threshold 0.2, trying 0.1...")
                rag_results_low = rag_db.search_database(query_text, k=2, score_threshold=0.1)
                print(f"   Found {len(rag_results_low)} results with threshold 0.1")
                
                if rag_results_low:
                    found_relevant = True
                    print(f"   ✅ Processing {len(rag_results_low)} low-threshold results")
                    formatted_lines = []
                    for j, result in enumerate(rag_results_low):
                        content_preview = result["content"][:300] + "..." if len(result["content"]) > 300 else result["content"]
                        source = result["metadata"].get("filename", "Local Database")
                        score = result["score"]
                        print(f"      Result {j+1}: {source} (Score: {score:.3f})")
                        formatted_lines.append(f"Paper: {source} (Score: {score:.3f})\n{content_preview}")
                        
                        sources_gathered.append({
                            "label": f"RAG: {source}",
                            "short_url": f"local://{source}",
                            "value": f"Local Paper: {source}"
                        })
                    
                    combined_result = "\n\n---\n\n".join(formatted_lines)
                    all_rag_results.append(combined_result)
                    logger.info(f"Found {len(rag_results_low)} documents with lower threshold for query: {query_text}")
                else:
                    print(f"   ❌ No relevant documents found for '{query_text}'")
                    logger.info(f"No relevant documents found in RAG database for query: {query_text}")
        
        if found_relevant:
            print(f"🎉 RAG search successful! Found content from {len(sources_gathered)} sources")
            logger.info(f"RAG search successful: Found relevant content from {len(sources_gathered)} sources")
            
            # Structure results as a single search turn - append as one grouped result
            search_turn_result = {
                "turn_timestamp": get_current_date(),
                "query_count": len(rag_queries),
                "queries_used": [q if isinstance(q, str) else str(q) for q in rag_queries],
                "results": all_rag_results,
                "sources": sources_gathered
            }
            
            result = {
                "rag_search_result": [search_turn_result],  # Wrap in list - will be appended to existing turns
                "sources_gathered": sources_gathered,        # Still accumulate sources for final answer
                "rag_found": True,
                "research_loop_count": 0  # Initialize research loop count
            }
            print("✅ Returning RAG results for next step")
            return result
        else:
            print("❌ No relevant content found in RAG database, will proceed to web search")
            logger.info("No relevant content found in RAG database")
            
            # Record this as an empty search turn for historical tracking
            empty_search_turn = {
                "turn_timestamp": get_current_date(),
                "query_count": len(rag_queries),
                "queries_used": [q if isinstance(q, str) else str(q) for q in rag_queries],
                "results": [],
                "sources": []
            }
            
            return {
                "rag_search_result": [empty_search_turn],  # Record the empty search attempt
                "sources_gathered": [],   # No sources found
                "rag_found": False,
                "research_loop_count": 0  # Initialize research loop count
            }
        
    except Exception as e:
        print(f"💥 RAG search failed with error: {e}")
        logger.error(f"RAG search failed: {e}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()
        
        # Record the failed search attempt
        failed_search_turn = {
            "turn_timestamp": get_current_date(),
            "query_count": len(state.get("rag_search_query", [])),
            "queries_used": state.get("rag_search_query", []),
            "results": [],
            "sources": [],
            "error": str(e)
        }
        
        return {
            "rag_search_result": [failed_search_turn],  # Record the failed attempt
            "sources_gathered": [],   # No sources due to error
            "rag_found": False,
            "research_loop_count": 0  # Initialize research loop count
        }


def paper_search(state: OverallState, config: RunnableConfig) -> OverallState:
    """LangGraph node that searches Google for academic papers and indexes them into RAG database.
    
    This node analyzes detected paper titles and uses Google search to find PDFs
    and other academic content that can be indexed into the RAG database.
    
    Args:
        state: Current graph state containing detected paper titles
        config: Configuration for the runnable
        
    Returns:
        Dictionary with state update including paper_found flag and indexed papers count
    """
    print("📚 Starting Google paper search for detected papers...")
    logger.info("Paper search node started")
    
    try:
        # Get detected paper titles and search queries
        detected_titles = state.get("title", [])
        search_queries = state.get("search_query", [])
        
        # Combine both sources for comprehensive paper search
        titles_to_search = detected_titles.copy()
        
        # Also add search queries that look like paper titles
        for query in search_queries:
            query_text = query if isinstance(query, str) else str(query)
            if query_text not in titles_to_search:
                titles_to_search.append(query_text)
        
        if not titles_to_search:
            print("⚠️  No paper titles or queries detected, skipping paper search")
            return {
                "paper_found": False,
                "papers_indexed": 0,
                "paper_search_attempted": True
            }
        
        print(f"🔎 Searching Google for {len(titles_to_search)} detected titles: {titles_to_search}")
        
        # Get RAG database instance
        rag_db = get_rag_database()
        if rag_db is None:
            print("⚠️  RAG database not available, cannot index new papers")
            return {
                "paper_found": False,
                "papers_indexed": 0,
                "paper_search_attempted": True
            }
        
        papers_indexed = 0
        paper_found = False
        
        # Check if Google Search API is available
        try:
            search_api_key = os.environ.get("GOOGLE_SEARCH_API_KEY")
            search_cx = os.environ.get("GOOGLE_SEARCH_CX")
            use_google_api = search_api_key and search_cx
        except:
            use_google_api = False
        
        for title in titles_to_search:
            try:
                print(f"📖 Searching for: '{title}'")
                
                # Format search queries to find academic papers
                search_queries = [
                    f'"{title}" filetype:pdf',  # Look for PDF files
                    f'"{title}" site:arxiv.org',  # arXiv papers
                    f'"{title}" site:scholar.google.com',  # Google Scholar
                    f'"{title}" paper academic',  # General academic search
                ]
                
                paper_urls = []
                
                # Try each search query
                for query in search_queries:
                    try:
                        print(f"   Trying query: {query}")
                        
                        if use_google_api:
                            # Use existing Google Search API
                            results = search_with_google(query, search_api_key, search_cx)
                            for result in results:
                                url = result.get('link', '')
                                title_result = result.get('title', '')
                                snippet = result.get('snippet', '')
                                
                                # Check if this looks like an academic paper
                                if any(indicator in url.lower() for indicator in ['.pdf', 'arxiv.org', 'scholar.google', 'researchgate', 'ieee.org', 'acm.org', 'springer.com']):
                                    paper_urls.append({
                                        'url': url,
                                        'title': title_result,
                                        'snippet': snippet
                                    })
                                    print(f"   ✅ Found potential paper: {title_result[:50]}...")
                        else:
                            # Fallback to DuckDuckGo if Google API not available
                            print("   Using DuckDuckGo as fallback...")
                            with DDGS() as ddgs:
                                results = list(ddgs.text(query, max_results=3))
                                
                            for result in results:
                                url = result.get('href', '')
                                title_result = result.get('title', '')
                                
                                # Check if this looks like an academic paper
                                if any(indicator in url.lower() for indicator in ['.pdf', 'arxiv.org', 'scholar.google', 'researchgate', 'ieee.org', 'acm.org', 'springer.com']):
                                    paper_urls.append({
                                        'url': url,
                                        'title': title_result,
                                        'snippet': result.get('body', '')
                                    })
                                    print(f"   ✅ Found potential paper: {title_result[:50]}...")
                        
                        if paper_urls:
                            break  # Found papers, no need to try more queries
                            
                    except Exception as e:
                        print(f"   ⚠️  Search query failed: {e}")
                        continue
                
                if not paper_urls:
                    print(f"❌ No papers found for: '{title}'")
                    continue
                
                paper_found = True
                print(f"✅ Found {len(paper_urls)} potential papers for '{title}'")
                
                # For each found paper URL, try to download and index
                for paper_info in paper_urls:
                    try:
                        url = paper_info['url']
                        paper_title = paper_info['title']
                        
                        print(f"📄 Attempting to download and index: {paper_title[:50]}...")
                        
                        # Check if this paper is already indexed
                        existing_results = rag_db.search_database(paper_title, k=1, score_threshold=0.8)
                        
                        already_exists = False
                        for result in existing_results:
                            if (url in result["metadata"].get("source_url", "") or 
                                paper_title.lower() in result["content"].lower()):
                                already_exists = True
                                break
                        
                        if already_exists:
                            print(f"⏭️  Paper already indexed: {paper_title[:40]}...")
                            continue
                        
                        # Check if this is a PDF that can be downloaded
                        if url.endswith('.pdf') or 'arxiv.org' in url.lower():
                            try:
                                # Create a safe filename from the paper title
                                safe_title = "".join(c for c in paper_title if c.isalnum() or c in (' ', '-', '_')).rstrip()
                                safe_title = safe_title[:100]  # Limit filename length
                                
                                # Handle arXiv URLs - convert to PDF download URL
                                if 'arxiv.org' in url.lower() and not url.endswith('.pdf'):
                                    # Extract arXiv ID and convert to PDF URL
                                    import re
                                    arxiv_match = re.search(r'arxiv\.org/abs/(\d{4}\.\d{4,5})', url)
                                    if arxiv_match:
                                        arxiv_id = arxiv_match.group(1)
                                        url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                                        print(f"   🔄 Converted arXiv URL to PDF: {url}")
                                
                                # Create the paper download directory
                                # Get the backend directory path correctly
                                backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                                paper_dir = os.path.join(backend_dir, "paper")
                                os.makedirs(paper_dir, exist_ok=True)
                                
                                # Create filename with hash to avoid conflicts
                                file_hash = hash(url) % 100000
                                filename = f"{safe_title}_{file_hash}.pdf"
                                save_path = os.path.join(paper_dir, filename)
                                
                                print(f"   📥 Downloading PDF to: {filename}")
                                
                                # Download and index the PDF using RAG utility
                                additional_metadata = {
                                    "title": paper_title,
                                    "source_url": url,
                                    "source": "google_search",
                                    "search_title": title,  # Original search term
                                    "snippet": paper_info.get('snippet', '')
                                }
                                
                                success = rag_db.download_and_index_pdf(url, save_path, additional_metadata)
                                
                                if success:
                                    papers_indexed += 1
                                    print(f"✅ Successfully downloaded and indexed: {paper_title[:40]}...")
                                else:
                                    print(f"❌ Failed to download PDF: {paper_title[:40]}...")
                                    # Fallback to web content extraction
                                    print(f"   🔄 Falling back to web content extraction...")
                                    content = _extract_web_content(url, paper_info, title)
                                    if content:
                                        success = _index_content_chunks(rag_db, content, paper_title, url, title, paper_info)
                                        if success:
                                            papers_indexed += 1
                                            
                            except Exception as e:
                                print(f"   ⚠️  PDF download failed: {e}")
                                # Fallback to web content extraction
                                print(f"   🔄 Falling back to web content extraction...")
                                content = _extract_web_content(url, paper_info, title)
                                if content:
                                    success = _index_content_chunks(rag_db, content, paper_title, url, title, paper_info)
                                    if success:
                                        papers_indexed += 1
                        else:
                            # For non-PDF URLs, extract web content
                            print(f"   🌐 Extracting web content from: {url}")
                            content = _extract_web_content(url, paper_info, title)
                            if content:
                                success = _index_content_chunks(rag_db, content, paper_title, url, title, paper_info)
                                if success:
                                    papers_indexed += 1
                        
                        # Limit to avoid overwhelming the system
                        if papers_indexed >= 2:
                            print("🛑 Reached indexing limit (2 papers per search)")
                            break
                        
                    except Exception as e:
                        print(f"❌ Error processing paper {paper_info['title'][:30]}...: {e}")
                        logger.error(f"Error processing paper: {e}")
                        continue
                    
            except Exception as e:
                print(f"❌ Error searching for '{title}': {e}")
                logger.error(f"Error searching for '{title}': {e}")
                continue
        
        if papers_indexed > 0:
            print(f"🎉 Successfully indexed {papers_indexed} new papers from Google search!")
            logger.info(f"Indexed {papers_indexed} new papers from Google search")
        else:
            print("📚 No new papers were indexed from Google search")
            
        return {
            "paper_found": paper_found,
            "papers_indexed": papers_indexed,
            "paper_search_attempted": True
        }
        
    except Exception as e:
        print(f"💥 Paper search failed with error: {e}")
        logger.error(f"Paper search failed: {e}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()
        
        return {
            "paper_found": False,
            "papers_indexed": 0,
            "paper_search_attempted": True
        }


def continue_to_web_research(state: QueryGenerationState):
    """LangGraph node that sends the search queries to the web research node.

    This is used to spawn n number of web research nodes, one for each search query.
    """
    return [
        Send("web_research", {"search_query": search_query, "id": int(idx)})
        for idx, search_query in enumerate(state["search_query"])
    ]


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
    """Stubbed LangGraph node that skips real web research and returns mock data."""
    query = state["search_query"]
    ## The following code is commented out to avoid real API calls:
    configurable = Configuration.from_runnable_config(config, base_model=state.get("reasoning_model"))
    search_api_key = os.environ["GOOGLE_SEARCH_API_KEY"]
    results = search_with_google(query, search_api_key, os.environ["GOOGLE_SEARCH_CX"])
    formatted_lines = []
    sources_gathered = []
    for res in results:
        title = res.get("title", "")
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
    ## Return mock/fake research results to avoid API calls
    # mock_result = f"[MOCKED] No real web search performed for query: {query}"
    # return {
    #     "sources_gathered": [{"label": "Mock Source", "short_url": "https://example.com", "value": "https://example.com"}],
    #     "search_query": [query],
    #     "web_research_result": [mock_result],
    # }


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
    
    # Combine RAG and web research results for reflection
    all_summaries = []
    
    # Add RAG results if available
    rag_results = state.get("rag_search_result", [])
    if rag_results:
        print(f"📚 Including {len(rag_results)} RAG search turns in reflection")
        # Extract actual content from the structured search turns
        for turn in rag_results:
            if turn.get("results"):  # Only include turns that have actual results
                all_summaries.extend(turn["results"])
        print(f"📚 Total RAG content pieces: {len([r for turn in rag_results for r in turn.get('results', [])])}")
    
    # Add web research results if available
    web_results = state.get("web_research_result", [])
    if web_results:
        print(f"🌐 Including {len(web_results)} web research results in reflection")
        all_summaries.extend(web_results)
    
    if not all_summaries:
        print("⚠️  No research results found for reflection")
        all_summaries = ["No research results available for analysis."]
    
    formatted_prompt = reflection_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n\n---\n\n".join(all_summaries),
    )

    # Initialize ChatOllama for reflection
    chat_reflection = ChatOllama(
        model=configurable.reflection_model or state.get("reasoning_model"),
        temperature=1.0,
        base_url="localhost:11434",
    )
    
    # Assuming for now that with_structured_output with Ollama might work directly
    structured_llm_reflection = chat_reflection.with_structured_output(Reflection)
    result = structured_llm_reflection.invoke(formatted_prompt)

    print(state)

    return {
        "is_sufficient": result.is_sufficient,
        "knowledge_gap": result.knowledge_gap,
        "follow_up_queries": result.follow_up_queries,
        "research_loop_count": research_loop_count,  # Use the calculated value
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
    
    # Combine RAG and web research results for final answer
    all_summaries = []
    
    # Add RAG results if available
    rag_results = state.get("rag_search_result", [])
    if rag_results:
        print(f"📚 Including {len(rag_results)} RAG search turns in final answer")
        # Extract actual content from the structured search turns
        for turn in rag_results:
            if turn.get("results"):  # Only include turns that have actual results
                all_summaries.extend(turn["results"])
        print(f"📚 Total RAG content pieces: {len([r for turn in rag_results for r in turn.get('results', [])])}")
    
    # Add web research results if available
    web_results = state.get("web_research_result", [])
    if web_results:
        print(f"🌐 Including {len(web_results)} web research results in final answer")
        all_summaries.extend(web_results)
    
    if not all_summaries:
        print("⚠️  No research results found for final answer")
        all_summaries = ["No research results available for synthesis."]
    
    formatted_prompt = answer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n---\n\n".join(all_summaries),
    )

    # Initialize ChatOllama for final answer
    chat_finalize = ChatOllama(
        model=configurable.answer_model or state.get("reasoning_model"),
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


def decide_after_rag_search(state: OverallState):
    """LangGraph routing function that decides next step after RAG search.
    
    Based on the updated flowchart:
    - If RAG found results: go directly to reflection (skip paper search)
    - If RAG found no results: go to generate_query for web search query generation
    
    Args:
        state: Current graph state containing RAG search results
        
    Returns:
        String indicating next node ("reflection" or "generate_query")
    """
    rag_found = state.get("rag_found", False)
    
    if rag_found:
        print("✅ RAG found results! Skipping web search and going directly to reflection...")
        logger.info("RAG found results, skipping web search and proceeding to reflection")
        return "reflection"
    else:
        print("❌ RAG found no results, generating web search queries...")
        logger.info("RAG found no results, proceeding to web query generation")
        return "generate_query"


def decide_search_strategy(state: OverallState):
    """LangGraph routing function that decides between paper search and web search.
    
    Based on the analysis from generate_query:
    - If is_paper_search: go to paper_search
    - Otherwise: spawn parallel web_research tasks
    
    Args:
        state: Current graph state containing search type detection
        
    Returns:
        String or list of Send objects
    """
    is_paper_search = state.get("is_paper_search", False)
    search_indicators = state.get("paper_search_indicators", [])
    
    if is_paper_search:
        print(f"📄 Routing to paper search based on indicators: {', '.join(search_indicators)}")
        logger.info(f"Routing to paper search. Indicators: {search_indicators}")
        return "paper_search"
    else:
        print("🌐 Routing to general web search...")
        logger.info("Routing to general web search")
        
        # Spawn parallel web research tasks for each query
        search_queries = state.get("search_query", [])
        print(f"🌐 Creating {len(search_queries)} web search tasks")
        
        return [
            Send("web_research", {"search_query": search_query, "id": int(idx)})
            for idx, search_query in enumerate(search_queries)
        ]


def _extract_web_content(url: str, paper_info: dict, original_title: str) -> str:
    """
    Helper function to extract content from web URLs.
    
    Args:
        url: URL to extract content from
        paper_info: Paper information dictionary
        original_title: Original search title
        
    Returns:
        Extracted content string
    """
    try:
        print(f"   🌐 Fetching web content from: {url}")
        response = requests.get(url, timeout=10, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            # Extract text content
            text = soup.get_text()
            # Clean up the text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            content = ' '.join(chunk for chunk in chunks if chunk)
            
            # Limit content length
            if len(content) > 10000:
                content = content[:10000] + "..."
            return content
        else:
            print(f"   ❌ Failed to fetch {url}: {response.status_code}")
            return f"Paper: {paper_info['title']}\nURL: {url}\nSnippet: {paper_info['snippet']}"
    except Exception as e:
        print(f"   ⚠️  Error fetching content: {e}")
        return f"Paper: {paper_info['title']}\nURL: {url}\nSnippet: {paper_info['snippet']}"


def _index_content_chunks(rag_db, content: str, paper_title: str, url: str, original_title: str, paper_info: dict) -> bool:
    """
    Helper function to split content into chunks and index them.
    
    Args:
        rag_db: RAG database instance
        content: Content to index
        paper_title: Title of the paper
        url: Source URL
        original_title: Original search title
        paper_info: Paper information dictionary
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if not content.strip():
            print(f"   ❌ No content to index for {paper_title[:30]}...")
            return False
        
        # Split content into chunks for indexing
        if len(content) > 2000:
            # Simple chunking - split by paragraphs and combine to ~1500 chars
            paragraphs = content.split('\n\n')
            chunks = []
            current_chunk = ""
            
            for para in paragraphs:
                if len(current_chunk) + len(para) > 1500 and current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = para
                else:
                    current_chunk += f"\n\n{para}" if current_chunk else para
            
            if current_chunk:
                chunks.append(current_chunk.strip())
        else:
            chunks = [content]
        
        # Create a unique source identifier
        source_id = f"google_paper_{hash(url) % 100000}"
        
        # Index each chunk
        for i, chunk in enumerate(chunks):
            chunk_metadata = {
                "filename": f"{source_id}.pdf",
                "title": paper_title,
                "source_url": url,
                "chunk_id": i,
                "source": "google_search",
                "search_title": original_title,  # Original search term
                "snippet": paper_info.get('snippet', '')
            }
            
            # Add to RAG database
            rag_db.add_document(chunk, chunk_metadata)
        
        print(f"✅ Successfully indexed: {paper_title[:40]}... ({len(chunks)} chunks)")
        return True
        
    except Exception as e:
        print(f"❌ Error indexing content chunks: {e}")
        logger.error(f"Error indexing content chunks: {e}")
        return False


# Create our Agent Graph
builder = StateGraph(OverallState, config_schema=Configuration)

# Define the nodes we will cycle between
builder.add_node("rag_query", rag_query)
builder.add_node("paper_search", paper_search)
builder.add_node("generate_query", generate_query)
builder.add_node("rag_search", rag_search)
builder.add_node("web_research", web_research)
builder.add_node("reflection", reflection)
builder.add_node("finalize_answer", finalize_answer)

# Set the entrypoint as `rag_query` to follow RAG-first strategy
# This means that this node is the first one called
builder.add_edge(START, "rag_query")

# After generating RAG queries, search RAG database first
builder.add_edge("rag_query", "rag_search")

# After RAG search, decide next step based on results
builder.add_conditional_edges(
    "rag_search", 
    decide_after_rag_search, 
    ["reflection", "generate_query"]
)

# After query generation, decide between paper search and web search
builder.add_conditional_edges(
    "generate_query",
    decide_search_strategy,
    ["paper_search", "web_research"]
)

# After paper search, go back to web research as fallback
builder.add_edge("paper_search", "web_research")

# Reflect on the research (from RAG and potentially web results)
builder.add_edge("web_research", "reflection")

# From reflection, we can also get there directly from RAG if it found results
# Evaluate the research - this handles both RAG-only and RAG+web scenarios
builder.add_conditional_edges(
    "reflection", evaluate_research, ["web_research", "finalize_answer"]
)

# Finalize the answer
builder.add_edge("finalize_answer", END)

graph = builder.compile(name="pro-search-agent")