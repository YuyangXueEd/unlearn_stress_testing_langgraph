"""
Database Nodes

Nodes responsible for handling database operations and RAG (Retrieval-Augmented Generation).
Uses ChromaDB with nomic-embed-text for document indexing and retrieval.
"""

import logging
import os
import requests
import json
from pathlib import Path
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from langchain_core.messages import AIMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from state import ChatState
from prompts import REFLECTION_PROMPT, FINAL_ANSWER_PROMPT, get_current_date
from configuration import DemoConfiguration

logger = logging.getLogger(__name__)


class OllamaEmbeddingFunction:
    """Custom embedding function that uses local Ollama nomic-embed-text model."""
    
    def __init__(self, model_name: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api/embeddings"
        self.name = f"ollama_{model_name}"  # Required by ChromaDB
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using Ollama API.
        
        Args:
            input: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        for text in input:
            try:
                response = requests.post(
                    self.api_url,
                    json={
                        "model": self.model_name,
                        "prompt": text
                    },
                    timeout=30
                )
                response.raise_for_status()
                
                result = response.json()
                embedding = result.get("embedding", [])
                
                if not embedding:
                    logger.error(f"No embedding returned for text: {text[:50]}...")
                    # Return a zero vector as fallback - use actual dimension from first successful call
                    embedding = [0.0] * 384  # nomic-embed-text actually has 384 dimensions
                
                embeddings.append(embedding)
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Ollama API request failed: {e}")
                # Return a zero vector as fallback
                embeddings.append([0.0] * 384)  # nomic-embed-text has 384 dimensions
            except Exception as e:
                logger.error(f"Error generating embedding: {e}")
                # Return a zero vector as fallback
                embeddings.append([0.0] * 384)  # nomic-embed-text has 384 dimensions
        
        return embeddings


# Global variables for database connection
_chroma_client = None
_collection = None
_embedding_function = None


def initialize_database():
    """
    Initialize ChromaDB connection and load documents from papers folder.
    
    Returns:
        bool: True if successful, False otherwise
    """
    global _chroma_client, _collection, _embedding_function
    
    try:
        logger.info("Initializing ChromaDB connection...")
        
        # Initialize ChromaDB client
        db_path = Path(__file__).parent.parent.parent / "chroma_db"
        db_path.mkdir(exist_ok=True)
        
        _chroma_client = chromadb.PersistentClient(
            path=str(db_path),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize Ollama nomic-embed-text embedding function
        logger.info("Initializing Ollama nomic-embed-text embedding function...")
        try:
            _embedding_function = OllamaEmbeddingFunction(model_name="nomic-embed-text")
            
            # Test the embedding function with a simple text
            test_embedding = _embedding_function(["test"])
            if test_embedding and len(test_embedding[0]) > 0:
                logger.info(f"Successfully connected to Ollama nomic-embed-text (dimension: {len(test_embedding[0])})")
            else:
                raise Exception("Empty embedding returned from Ollama")
                
        except Exception as e:
            logger.error(f"Failed to initialize Ollama embedding function: {e}")
            logger.info("Falling back to default embedding function")
            _embedding_function = embedding_functions.DefaultEmbeddingFunction()
        
        # Get or create collection
        collection_name = "papers_collection_ollama"  # Use different name to avoid dimension conflicts
        
        # First, check if collection exists
        try:
            existing_collections = _chroma_client.list_collections()
            collection_exists = any(col.name == collection_name for col in existing_collections)
            
            if collection_exists:
                logger.info(f"Collection {collection_name} exists, attempting to retrieve it...")
                try:
                    # Try to get the existing collection
                    _collection = _chroma_client.get_collection(name=collection_name)
                    logger.info(f"Successfully retrieved existing collection: {collection_name}")
                    
                    # Test if the embedding dimensions match by trying a small query
                    try:
                        test_result = _collection.query(
                            query_embeddings=[_embedding_function(["test"])[0]],
                            n_results=1
                        )
                        logger.info("Embedding dimensions match existing collection")
                    except Exception as dim_error:
                        logger.warning(f"Dimension mismatch detected: {dim_error}")
                        logger.info("Deleting and recreating collection with correct dimensions...")
                        
                        # Delete the existing collection
                        _chroma_client.delete_collection(name=collection_name)
                        logger.info(f"Deleted existing collection: {collection_name}")
                        
                        # Create new collection
                        _collection = _chroma_client.create_collection(
                            name=collection_name,
                            embedding_function=_embedding_function,
                            metadata={"description": "Research papers collection with Ollama embeddings", "dimension": "384"}
                        )
                        logger.info(f"Created new collection with correct dimensions: {collection_name}")
                        _load_papers_into_database()
                        return True
                    
                    # Check if collection is empty and needs to be populated
                    doc_count = _collection.count()
                    if doc_count == 0:
                        logger.info("Collection exists but is empty, loading papers...")
                        _load_papers_into_database()
                    else:
                        logger.info(f"Collection has {doc_count} documents")
                        
                except Exception as get_error:
                    logger.warning(f"Failed to get existing collection: {get_error}")
                    logger.info("Deleting and recreating collection...")
                    
                    # Delete the existing collection
                    _chroma_client.delete_collection(name=collection_name)
                    logger.info(f"Deleted existing collection: {collection_name}")
                    
                    # Create new collection
                    _collection = _chroma_client.create_collection(
                        name=collection_name,
                        embedding_function=_embedding_function,
                        metadata={"description": "Research papers collection with Ollama embeddings", "dimension": "384"}
                    )
                    logger.info(f"Created new collection: {collection_name}")
                    _load_papers_into_database()
            else:
                logger.info(f"Creating new collection: {collection_name}")
                _collection = _chroma_client.create_collection(
                    name=collection_name,
                    embedding_function=_embedding_function,
                    metadata={"description": "Research papers collection with Ollama embeddings", "dimension": "384"}
                )
                logger.info(f"Created new collection: {collection_name}")
                _load_papers_into_database()
                
        except Exception as e:
            logger.error(f"Error in collection management: {e}")
            # Try a simple creation as fallback
            try:
                _collection = _chroma_client.get_or_create_collection(
                    name=collection_name,
                    embedding_function=_embedding_function,
                    metadata={"description": "Research papers collection with Ollama embeddings", "dimension": "384"}
                )
                logger.info(f"Used get_or_create for collection: {collection_name}")
                
                # Check if we need to load documents
                doc_count = _collection.count()
                if doc_count == 0:
                    logger.info("Collection is empty, loading papers...")
                    _load_papers_into_database()
                else:
                    logger.info(f"Collection has {doc_count} documents")
                    
            except Exception as fallback_error:
                logger.error(f"Fallback collection creation failed: {fallback_error}")
                raise fallback_error
        
        logger.info("Database initialization completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        return False


def _load_papers_into_database():
    """Load PDF documents from papers folder into ChromaDB."""
    global _collection
    
    if not _collection:
        logger.error("Collection not initialized")
        return
    
    try:
        # Get papers directory - it's in backend/demo/paper
        # Current file is in backend/demo/srcs/nodes/, so go up 2 levels to get to demo/
        papers_dir = Path(__file__).parent.parent.parent / "paper"
        
        if not papers_dir.exists():
            logger.warning(f"Papers directory not found: {papers_dir}")
            return
        
        # Find all PDF files
        pdf_files = list(papers_dir.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning("No PDF files found in papers directory")
            return
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        # Text splitter for chunking documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        all_documents = []
        all_metadatas = []
        all_ids = []
        
        for pdf_file in pdf_files:
            try:
                logger.info(f"Processing: {pdf_file.name}")
                
                # Load PDF
                loader = PyPDFLoader(str(pdf_file))
                documents = loader.load()
                
                # Split into chunks
                chunks = text_splitter.split_documents(documents)
                
                for i, chunk in enumerate(chunks):
                    doc_id = f"{pdf_file.stem}_chunk_{i}"
                    
                    all_documents.append(chunk.page_content)
                    all_metadatas.append({
                        "source": pdf_file.name,
                        "chunk_id": i,
                        "page": chunk.metadata.get("page", 0),
                        "file_path": str(pdf_file)
                    })
                    all_ids.append(doc_id)
                
                logger.info(f"Processed {pdf_file.name}: {len(chunks)} chunks")
                
            except Exception as e:
                logger.error(f"Error processing {pdf_file.name}: {e}")
                continue
        
        if all_documents:
            # Add all documents to collection in batch
            _collection.add(
                documents=all_documents,
                metadatas=all_metadatas,
                ids=all_ids
            )
            
            logger.info(f"Successfully added {len(all_documents)} document chunks to database")
        else:
            logger.warning("No documents were successfully processed")
            
    except Exception as e:
        logger.error(f"Error loading papers into database: {e}")


def database_search_node(state: ChatState) -> ChatState:
    """
    Database search node for retrieving relevant information from indexed papers.
    
    This node:
    1. Extracts the search query from user message
    2. Performs similarity search in ChromaDB
    3. Returns relevant documents with sources
    4. Sets up state for reflection analysis
    
    Args:
        state: Current chat state containing search query
        
    Returns:
        Updated state with search results and routing to reflection
    """
    try:
        global _collection
        
        if not _collection:
            return {
                "response": "❌ Database not initialized. Please check the setup.",
                "messages": [AIMessage(content="❌ Database not initialized. Please check the setup.")],
                "task_type": "final_answer"  # Skip reflection if DB not available
            }
        
        user_message = state.get("user_message", "")
        if not user_message:
            return {
                "response": "❌ No search query provided.",
                "messages": [AIMessage(content="❌ No search query provided.")],
                "task_type": "final_answer"  # Skip reflection if no query
            }
        
        # Initialize search tracking if not present
        search_iteration = state.get("search_iteration", 0) + 1
        previous_queries = state.get("previous_queries", [])
        
        # Add current query to previous queries if it's new
        if user_message not in previous_queries:
            previous_queries.append(user_message)
        
        # Extract search query
        search_query = _extract_search_query(user_message)
        
        logger.info(f"Performing database search for: {search_query} (iteration {search_iteration})")
        
        # Perform similarity search
        results = _collection.query(
            query_texts=[search_query],
            n_results=5,  # Get top 5 most relevant chunks
            include=["documents", "metadatas", "distances"]
        )
        
        if not results["documents"] or not results["documents"][0]:
            response_text = (
                f"🔍 I searched the research papers for '{search_query}' but couldn't find relevant information.\n\n"
                f"The database contains papers from the 'paper' folder. "
                f"Try rephrasing your query or asking about topics covered in the indexed papers."
            )
            task_type = "final_answer"  # Skip reflection if no results
        else:
            # Format the results
            response_text = f"📚 **Search Results for: '{search_query}' (Iteration {search_iteration})**\n\n"
            
            for i, (doc, metadata, distance) in enumerate(zip(
                results["documents"][0], 
                results["metadatas"][0], 
                results["distances"][0]
            )):
                source = metadata.get("source", "Unknown")
                page = metadata.get("page", "Unknown")
                
                response_text += f"**Result {i+1}** (Relevance: {1-distance:.3f})\n"
                response_text += f"📄 **Source**: {source} (Page {page})\n"
                response_text += f"**Content**: {doc[:300]}{'...' if len(doc) > 300 else ''}\n\n"
            
            response_text += "� **Analyzing results for completeness...**"
            task_type = "reflection"  # Route to reflection for analysis
        
        return {
            "response": response_text,
            "messages": [AIMessage(content=response_text)],
            "search_results": results,
            "search_iteration": search_iteration,
            "previous_queries": previous_queries,
            "task_type": task_type
        }
        
    except Exception as e:
        logger.error(f"Error in database search node: {e}")
        error_response = f"❌ An error occurred while searching the database: {str(e)}"
        return {
            "response": error_response,
            "messages": [AIMessage(content=error_response)],
            "task_type": "final_answer"  # Skip reflection on error
        }


def _extract_search_query(message: str) -> str:
    """
    Extract the search query from user message.
    
    Args:
        message: User message containing search request
        
    Returns:
        Cleaned search query
    """
    # Remove common prefixes
    prefixes_to_remove = [
        "search for", "find", "look for", "search", "find information about",
        "what does the paper say about", "tell me about", "explain", "describe"
    ]
    
    cleaned = message.lower()
    for prefix in prefixes_to_remove:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
            break
    
    # Remove common question words
    import re
    cleaned = re.sub(r'^(what|how|why|when|where|who)\s+(is|are|does|do|did|can|could|would|should)\s+', '', cleaned)
    
    return cleaned if cleaned else message


def get_database_status() -> Dict[str, Any]:
    """
    Get the current status of the database connection.
    
    Returns:
        Dictionary containing database status information
    """
    global _chroma_client, _collection
    
    status = {
        "initialized": _chroma_client is not None and _collection is not None,
        "client_available": _chroma_client is not None,
        "collection_available": _collection is not None,
        "document_count": 0
    }
    
    if _collection:
        try:
            status["document_count"] = _collection.count()
        except Exception as e:
            logger.error(f"Error getting document count: {e}")
    
    return status


def _call_ollama_llm(prompt: str, model_name: str = "gemma3") -> str:
    """
    Call Ollama LLM API for text generation.
    
    Args:
        prompt: The input prompt for the LLM
        model_name: The Ollama model to use (default: gemma3)
        
    Returns:
        Generated text response
    """
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )
        response.raise_for_status()
        
        result = response.json()
        return result.get("response", "")
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Ollama API request failed: {e}")
        return f"Error calling Ollama API: {str(e)}"
    except Exception as e:
        logger.error(f"Error calling Ollama LLM: {e}")
        return f"Error processing LLM request: {str(e)}"


def reflection_node(state: ChatState) -> ChatState:
    """
    Reflection node to analyze search results and determine if additional searches are needed.
    
    This node:
    1. Analyzes the current search results
    2. Determines if the results are sufficient to answer the user's question
    3. If insufficient, generates a new search query and routes back to database search
    4. If sufficient, routes to the final answer node
    
    Args:
        state: Current chat state containing search results
        
    Returns:
        Updated state with reflection analysis
    """
    try:
        # Get configuration
        config = DemoConfiguration()
        max_iterations = config.max_database_search_iterations
        
        # Get current state information
        search_results = state.get("search_results", {})
        search_iteration = state.get("search_iteration", 1)
        previous_queries = state.get("previous_queries", [])
        research_topic = state.get("user_message", "")
        
        # Format search results for analysis
        if search_results and "documents" in search_results and search_results["documents"]:
            results_text = ""
            for i, (doc, metadata) in enumerate(zip(
                search_results["documents"][0][:3],  # Limit to top 3 for analysis
                search_results["metadatas"][0][:3]
            )):
                source = metadata.get("source", "Unknown")
                results_text += f"Result {i+1} from {source}:\n{doc[:500]}...\n\n"
        else:
            results_text = "No search results found."
        
        # Prepare reflection prompt
        reflection_prompt = REFLECTION_PROMPT.format(
            research_topic=research_topic,
            search_iteration=search_iteration,
            previous_queries=previous_queries,
            search_results=results_text
        )
        
        logger.info(f"Performing reflection analysis (iteration {search_iteration}/{max_iterations})")
        
        # Call Ollama for reflection analysis
        reflection_response = _call_ollama_llm(reflection_prompt)
        
        # Parse reflection result
        try:
            # Extract JSON from the response
            start_idx = reflection_response.find('{')
            end_idx = reflection_response.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = reflection_response[start_idx:end_idx]
                reflection_result = json.loads(json_str)
            else:
                # Fallback if JSON parsing fails
                reflection_result = {
                    "is_sufficient": search_iteration >= max_iterations,  # Force sufficient if max iterations reached
                    "knowledge_gap": "Unable to parse reflection analysis",
                    "follow_up_query": ""
                }
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse reflection JSON: {e}")
            reflection_result = {
                "is_sufficient": search_iteration >= max_iterations,  # Force sufficient if max iterations reached
                "knowledge_gap": "Unable to parse reflection analysis",
                "follow_up_query": ""
            }
        
        # Determine next action based on reflection
        is_sufficient = reflection_result.get("is_sufficient", False)
        
        # Force sufficient if we've reached max iterations
        if search_iteration >= max_iterations:
            is_sufficient = True
            reflection_result["is_sufficient"] = True
        
        # Update task type for routing
        if is_sufficient:
            task_type = "final_answer"
        else:
            task_type = "database_search"
            # Update user_message with the new query for the next search
            new_query = reflection_result.get("follow_up_query", "")
            if new_query:
                state["user_message"] = new_query
                # Add to previous queries
                previous_queries.append(new_query)
        
        response_text = f"🔍 **Reflection Analysis (Iteration {search_iteration}/{max_iterations})**\n\n"
        
        if is_sufficient:
            response_text += "✅ **Analysis Complete**: The search results provide sufficient information to answer your question.\n"
            response_text += "📋 Proceeding to generate a comprehensive final answer..."
        else:
            gap = reflection_result.get("knowledge_gap", "Additional information needed")
            new_query = reflection_result.get("follow_up_query", "")
            response_text += f"⚠️ **Knowledge Gap Identified**: {gap}\n\n"
            response_text += f"🔄 **Next Search**: {new_query}\n"
            response_text += "🔍 Performing additional search to gather more comprehensive information..."
        
        return {
            "response": response_text,
            "messages": [AIMessage(content=response_text)],
            "reflection_result": reflection_result,
            "task_type": task_type,
            "search_iteration": search_iteration,
            "previous_queries": previous_queries
        }
        
    except Exception as e:
        logger.error(f"Error in reflection node: {e}")
        error_response = f"❌ An error occurred during reflection analysis: {str(e)}"
        return {
            "response": error_response,
            "messages": [AIMessage(content=error_response)],
            "task_type": "final_answer"  # Force to final answer to avoid loops
        }


def final_answer_node(state: ChatState) -> ChatState:
    """
    Final answer node to generate a comprehensive response based on all search results.
    
    This node:
    1. Synthesizes all accumulated search results
    2. Generates a comprehensive, well-formatted final answer
    3. Provides detailed analysis and insights
    
    Args:
        state: Current chat state containing all search results
        
    Returns:
        Updated state with final comprehensive answer
    """
    try:
        # Get all accumulated search results
        search_results = state.get("search_results", {})
        search_iteration = state.get("search_iteration", 1)
        research_topic = state.get("user_message", "")
        
        # Format all search results for synthesis
        if search_results and "documents" in search_results and search_results["documents"]:
            results_text = ""
            for i, (doc, metadata, distance) in enumerate(zip(
                search_results["documents"][0],
                search_results["metadatas"][0],
                search_results["distances"][0]
            )):
                source = metadata.get("source", "Unknown")
                page = metadata.get("page", "Unknown")
                relevance = 1 - distance
                
                results_text += f"**Source {i+1}**: {source} (Page {page}, Relevance: {relevance:.3f})\n"
                results_text += f"Content: {doc}\n\n"
        else:
            results_text = "No search results available for synthesis."
        
        # Prepare final answer prompt
        final_prompt = FINAL_ANSWER_PROMPT.format(
            current_date=get_current_date(),
            research_topic=research_topic,
            search_results=results_text,
            search_iteration=search_iteration
        )
        
        logger.info(f"Generating final comprehensive answer after {search_iteration} search iteration(s)")
        
        # Call Ollama for final answer generation
        final_response = _call_ollama_llm(final_prompt)
        
        # Format the final response
        response_text = f"📚 **Comprehensive Analysis**\n\n"
        response_text += final_response
        response_text += f"\n\n---\n*Analysis completed after {search_iteration} search iteration(s)*"
        
        return {
            "response": response_text,
            "messages": [AIMessage(content=response_text)]
        }
        
    except Exception as e:
        logger.error(f"Error in final answer node: {e}")
        error_response = f"❌ An error occurred while generating the final answer: {str(e)}"
        return {
            "response": error_response,
            "messages": [AIMessage(content=error_response)]
        }
