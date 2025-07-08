import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

import chromadb
from chromadb.config import Settings
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Set up logging
logger = logging.getLogger(__name__)

class RAGDatabase:
    """
    RAG (Retrieval-Augmented Generation) database utility using ChromaDB
    for storing and retrieving document embeddings.
    """
    
    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "research_papers",
        embedding_model: str = "nomic-embed-text",
        ollama_base_url: str = "http://localhost:11434",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize the RAG database.
        
        Args:
            persist_directory: Directory to persist the ChromaDB
            collection_name: Name of the collection in ChromaDB
            embedding_model: Ollama embedding model to use (default: nomic-embed-text)
            ollama_base_url: Base URL for Ollama server
            chunk_size: Size of text chunks for splitting documents
            chunk_overlap: Overlap between chunks
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.ollama_base_url = ollama_base_url
        
        # Initialize embeddings with Ollama
        self.embeddings = OllamaEmbeddings(
            model=embedding_model,
            base_url=ollama_base_url
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Initialize vector store
        self.vector_store = None
        self._initialize_vector_store()
        
        logger.info(f"RAG Database initialized with collection: {collection_name}")
    
    def _initialize_vector_store(self):
        """Initialize the Chroma vector store."""
        try:
            self.vector_store = Chroma(
                client=self.client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
            )
            logger.info("Vector store initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise
    
    def search_database(self, query: str, k: int = 5, score_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Search the RAG database for relevant documents.
        
        Args:
            query: Search query string
            k: Number of top results to return
            score_threshold: Minimum similarity score threshold (0.0 to 1.0, higher is more similar)
            
        Returns:
            List of dictionaries containing document content and metadata
        """
        try:
            if self.vector_store is None:
                logger.warning("Vector store not initialized")
                return []
            
            # Perform similarity search with scores
            results = self.vector_store.similarity_search_with_score(query, k=k)
            logger.info(f"Raw search returned {len(results)} results for query: '{query[:50]}...'")
            
            # Filter by score threshold and format results
            relevant_results = []
            for doc, distance in results:
                # ChromaDB uses distance (lower is better), convert to similarity score
                similarity_score = 1.0 - distance
                
                logger.debug(f"Document distance: {distance:.4f}, similarity: {similarity_score:.4f}")
                
                # Apply threshold (now correctly: similarity_score >= threshold)
                if similarity_score >= score_threshold:
                    relevant_results.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": similarity_score,
                        "source": doc.metadata.get("source", "unknown")
                    })
                    logger.debug(f"Document passed threshold: {similarity_score:.4f} >= {score_threshold}")
                else:
                    logger.debug(f"Document failed threshold: {similarity_score:.4f} < {score_threshold}")
            
            logger.info(f"Found {len(relevant_results)} relevant documents above threshold {score_threshold}")
            return relevant_results
            
        except Exception as e:
            logger.error(f"Database search failed: {e}")
            return []
    
    def is_database_populated(self) -> bool:
        """
        Check if the database contains any documents.
        
        Returns:
            True if database has documents, False otherwise
        """
        try:
            if self.vector_store is None:
                return False
            
            # Try to get a few documents to check if collection exists and has data
            collection = self.client.get_collection(self.collection_name)
            count = collection.count()
            return count > 0
            
        except Exception as e:
            logger.debug(f"Database population check failed: {e}")
            return False
    
    def load_pdf_documents(self, pdf_paths: List[str]) -> List[Document]:
        """
        Load PDF documents and split them into chunks.
        
        Args:
            pdf_paths: List of paths to PDF files
            
        Returns:
            List of Document objects with content and metadata
        """
        all_documents = []
        
        for pdf_path in pdf_paths:
            try:
                logger.info(f"Loading PDF: {pdf_path}")
                
                # Load PDF using PyPDFLoader
                loader = PyPDFLoader(pdf_path)
                pages = loader.load()
                
                # Split documents into chunks
                chunks = self.text_splitter.split_documents(pages)
                
                # Add source metadata
                for chunk in chunks:
                    chunk.metadata.update({
                        "source": pdf_path,
                        "filename": os.path.basename(pdf_path),
                        "document_type": "research_paper"
                    })
                
                all_documents.extend(chunks)
                logger.info(f"Loaded {len(chunks)} chunks from {pdf_path}")
                
            except Exception as e:
                logger.error(f"Failed to load PDF {pdf_path}: {e}")
                continue
        
        logger.info(f"Total documents loaded: {len(all_documents)}")
        return all_documents
    
    def add_documents_to_database(self, documents: List[Document]) -> bool:
        """
        Add documents to the vector database.
        
        Args:
            documents: List of Document objects to add
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not documents:
                logger.warning("No documents to add")
                return False
            
            if self.vector_store is None:
                logger.error("Vector store not initialized")
                return False
            
            # Add documents to the vector store
            self.vector_store.add_documents(documents)
            logger.info(f"Successfully added {len(documents)} documents to database")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents to database: {e}")
            return False
    
    def index_papers_from_directory(self, directory_path: str) -> bool:
        """
        Index all PDF papers from a directory into the database.
        
        Args:
            directory_path: Path to directory containing PDF files
            
        Returns:
            True if successful, False otherwise
        """
        try:
            directory = Path(directory_path)
            if not directory.exists():
                logger.error(f"Directory does not exist: {directory_path}")
                return False
            
            # Find all PDF files in the directory
            pdf_files = list(directory.glob("*.pdf"))
            
            if not pdf_files:
                logger.warning(f"No PDF files found in {directory_path}")
                return False
            
            pdf_paths = [str(pdf_file) for pdf_file in pdf_files]
            logger.info(f"Found {len(pdf_paths)} PDF files to index")
            
            # Load and process PDF documents
            documents = self.load_pdf_documents(pdf_paths)
            
            if not documents:
                logger.error("No documents were successfully loaded")
                return False
            
            # Add documents to the database
            success = self.add_documents_to_database(documents)
            
            if success:
                logger.info(f"Successfully indexed {len(pdf_files)} papers from {directory_path}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to index papers from directory {directory_path}: {e}")
            return False

    def add_document(self, content: str, metadata: Dict[str, Any]) -> bool:
        """
        Add a single document with content and metadata to the database.
        
        Args:
            content: Text content of the document
            metadata: Metadata dictionary for the document
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not content.strip():
                logger.warning("Empty content provided")
                return False
            
            if self.vector_store is None:
                logger.error("Vector store not initialized")
                return False
            
            # Create a Document object
            document = Document(
                page_content=content,
                metadata=metadata
            )
            
            # Add document to the vector store
            self.vector_store.add_documents([document])
            logger.debug(f"Successfully added document: {metadata.get('title', 'Unknown')[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add document to database: {e}")
            return False
    
    def download_and_index_pdf(self, url: str, save_path: str, metadata: Dict[str, Any] = None) -> bool:
        """
        Download a PDF from URL, save it locally, and index it into the database.
        
        Args:
            url: URL of the PDF to download
            save_path: Local path where to save the PDF
            metadata: Additional metadata to include
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import requests
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Download the PDF
            logger.info(f"Downloading PDF from: {url}")
            response = requests.get(url, timeout=30, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            
            if response.status_code == 200:
                # Save the PDF file
                with open(save_path, 'wb') as f:
                    f.write(response.content)
                logger.info(f"PDF saved to: {save_path}")
                
                # Load and index the PDF
                documents = self.load_pdf_documents([save_path])
                
                if documents:
                    # Add additional metadata if provided
                    if metadata:
                        for doc in documents:
                            doc.metadata.update(metadata)
                    
                    success = self.add_documents_to_database(documents)
                    if success:
                        logger.info(f"Successfully indexed PDF: {os.path.basename(save_path)}")
                        return True
                else:
                    logger.error(f"Failed to load PDF documents from: {save_path}")
                    return False
            else:
                logger.error(f"Failed to download PDF: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to download and index PDF: {e}")
            return False