"""
RAG Database Manager

Centralized management of RAG database initialization and access.
Used by both FastAPI app and CLI to ensure consistent RAG database handling.
"""
import os
import logging
import asyncio
from agent.rag_utils import RAGDatabase

# Set up logger
logger = logging.getLogger(__name__)

# Global RAG database instance
_rag_database = None


async def initialize_rag_database_async(persist_directory="./chroma_db", papers_directory="../paper"):
    """Initialize the RAG database asynchronously with specified directories.
    
    Args:
        persist_directory: Directory to store the ChromaDB database
        papers_directory: Directory containing papers to index
        
    Returns:
        RAGDatabase instance or None if initialization fails
    """
    global _rag_database
    
    print("🚀 Initializing RAG database asynchronously...")
    logger.info("Starting async RAG database initialization")
    
    try:
        # Run the blocking RAG initialization in a thread pool
        _rag_database = await asyncio.to_thread(_initialize_rag_sync, persist_directory, papers_directory)
        
        if _rag_database:
            print("🎉 RAG database initialization completed successfully!")
            logger.info("RAG database initialization completed")
        else:
            print("⚠️  RAG database initialization returned None")
            logger.warning("RAG database initialization returned None")
            
        return _rag_database
        
    except Exception as e:
        print(f"💥 Async RAG database initialization failed: {e}")
        logger.error(f"Async RAG database initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def _initialize_rag_sync(persist_directory, papers_directory):
    """Synchronous RAG initialization helper (runs in thread)."""
    try:
        # Initialize with default settings
        rag_db = RAGDatabase(
            persist_directory=persist_directory,
            collection_name="research_papers",
            embedding_model="nomic-embed-text",
            ollama_base_url="http://localhost:11434"
        )
        print("✅ RAG database instance created")
        logger.info("RAG database instance created successfully")
        
        # Auto-index papers from the paper directory if database is empty
        if not rag_db.is_database_populated():
            print(f"📁 Checking for papers in: {papers_directory}")
            
            if os.path.exists(papers_directory):
                print(f"📚 Indexing papers from {papers_directory}...")
                logger.info(f"Auto-indexing papers from {papers_directory}")
                success = rag_db.index_papers_from_directory(papers_directory)
                if success:
                    print("✅ Papers indexed successfully")
                    logger.info("Papers indexed successfully")
                else:
                    print("⚠️  Paper indexing failed")
                    logger.warning("Paper indexing failed")
            else:
                print(f"📁 Paper directory not found: {papers_directory}")
                logger.info(f"Paper directory not found: {papers_directory}")
        else:
            print("📚 RAG database already contains indexed papers")
            logger.info("RAG database already contains indexed papers")
            
        return rag_db
        
    except Exception as e:
        print(f"💥 Sync RAG database initialization failed: {e}")
        logger.error(f"Sync RAG database initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def initialize_rag_database(persist_directory="./chroma_db", papers_directory="../paper"):
    """Initialize the RAG database with specified directories (synchronous version for CLI).
    
    Args:
        persist_directory: Directory to store the ChromaDB database
        papers_directory: Directory containing papers to index
        
    Returns:
        RAGDatabase instance or None if initialization fails
    """
    global _rag_database
    
    print("🚀 Initializing RAG database...")
    logger.info("Starting RAG database initialization")
    
    _rag_database = _initialize_rag_sync(persist_directory, papers_directory)
    
    if _rag_database:
        print("🎉 RAG database initialization completed successfully!")
        logger.info("RAG database initialization completed")
    
    return _rag_database


def get_rag_database():
    """Get the globally initialized RAG database instance.
    
    Returns:
        RAGDatabase instance or None if not initialized
    """
    global _rag_database
    if _rag_database is None:
        print("⚠️  RAG database not initialized! Call initialize_rag_database() first.")
        logger.warning("RAG database accessed before initialization")
    return _rag_database


def is_rag_initialized():
    """Check if RAG database has been initialized.
    
    Returns:
        bool: True if RAG database is initialized, False otherwise
    """
    return _rag_database is not None
