import os
import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from typing import List, Dict, Any
from dotenv import load_dotenv
from light_embed import TextEmbedding
from langchain_community.embeddings import HuggingFaceEmbeddings


# Load environment variables
load_dotenv()


class VectorStore:
    """
    Manages the ChromaDB vector store for document embeddings.
    Handles storing and retrieving document chunks based on semantic similarity.
    Uses Hugging Face model for embeddings.
    """
    
    def __init__(self):
        """Initialize the vector store """

        self.chroma_db_path = os.getenv("CHROMA_DB_PATH", "./chroma_db")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
     
        # Initialize ChromaDB client with persistent storage
        self.client = chromadb.PersistentClient(
            path=self.chroma_db_path,
            settings=Settings(
                anonymized_telemetry=False,  # Disable usage tracking
                allow_reset=True  # Allow database reset for development
            )
        )
        
        # Initialize the vector store (will connect to existing or create new)
        self.vector_store = None
        self._initialize_vector_store()
    
    def _initialize_vector_store(self):
        """Initialize or connect to existing Chroma vector store"""
        try:
            # Try to connect to existing collection
            self.vector_store = Chroma(
                client=self.client,
                collection_name="knowledge_base",
                embedding_function=self.embeddings
            )
            print(f"✅ Connected to existing vector store at {self.chroma_db_path}")
        except Exception as e:
            print(f"⚠️  Creating new vector store: {e}")
            # Create new collection if it doesn't exist
            self.vector_store = Chroma(
                client=self.client,
                collection_name="knowledge_base",
                embedding_function=self.embeddings
            )
    
    def add_documents(self, texts: List[str], metadatas: List[Dict[str, Any]]) -> int:
        """
        Add document chunks to the vector store
        
        Args:
            texts: List of text chunks to add
            metadatas: List of metadata dicts (e.g., {"source": "doc.pdf", "chunk_id": 1})
        
        Returns:
            Number of documents added
        """
        try:
            # Add texts with their metadata to the vector store
            # LangChain automatically generates embeddings and stores them
            self.vector_store.add_texts(
                texts=texts,
                metadatas=metadatas
            )
            print(f"✅ Added {len(texts)} chunks to vector store")
            return len(texts)
        except Exception as e:
            print(f"❌ Error adding documents: {e}")
            raise
    
    def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for documents similar to the query
        
        Args:
            query: The search query
            k: Number of results to return (top_k)
        
        Returns:
            List of dicts with 'content' and 'metadata' keys
        """
        try:
            # Perform similarity search
            # This converts query to embedding and finds closest document embeddings
            results = self.vector_store.similarity_search(
                query=query,
                k=k
            )
            
            # Format results for easier consumption
            formatted_results = []
            for doc in results:
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })
            
            print(f"✅ Found {len(formatted_results)} relevant chunks")
            return formatted_results
            
        except Exception as e:
            print(f"❌ Error during search: {e}")
            return []
    
    def similarity_search_with_score(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search with similarity scores (distance metrics)
        
        Args:
            query: The search query
            k: Number of results to return
        
        Returns:
            List of dicts with 'content', 'metadata', and 'score' keys
        """
        try:
            # Get results with similarity scores
            # Score = how close the embedding is (lower = more similar in L2 distance)
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=k
            )
            
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score)  # Distance score (lower is better)
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"❌ Error during search with score: {e}")
            return []
    
    def get_collection_count(self) -> int:
        """Get the total number of documents in the vector store"""
        try:
            collection = self.client.get_collection("knowledge_base")
            return collection.count()
        except Exception as e:
            print(f"❌ Error getting collection count: {e}")
            return 0
    
    def delete_collection(self):
        """Delete the entire collection (useful for testing/reset)"""
        try:
            self.client.delete_collection("knowledge_base")
            print("✅ Collection deleted successfully")
            # Reinitialize
            self._initialize_vector_store()
        except Exception as e:
            print(f"❌ Error deleting collection: {e}")
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """
        Retrieve all documents from the vector store
        Useful for debugging or showing what's in the knowledge base
        """
        try:
            collection = self.client.get_collection("knowledge_base")
            results = collection.get()
            
            documents = []
            if results['documents']:
                for i, doc in enumerate(results['documents']):
                    documents.append({
                        "content": doc,
                        "metadata": results['metadatas'][i] if results['metadatas'] else {}
                    })
            
            return documents
        except Exception as e:
            print(f"❌ Error retrieving all documents: {e}")
            return []


# Singleton instance (one vector store for the entire app)
_vector_store_instance = None

def get_vector_store() -> VectorStore:
    """
    Get or create the vector store instance (Singleton pattern)
    This ensures we only have one connection to ChromaDB
    """
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = VectorStore()
    return _vector_store_instance