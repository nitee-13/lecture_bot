"""
Vector store module for managing embeddings and searching for similar documents.
"""
from typing import List, Dict, Any, Optional, Tuple
import json
import os
import numpy as np
from pathlib import Path
import faiss

from src.config import VECTOR_DB_PATH, EMBEDDING_DIMENSIONS


class VectorStore:
    """Vector store for managing embeddings."""
    
    def __init__(self, store_path: str = VECTOR_DB_PATH):
        """Initialize the vector store.
        
        Args:
            store_path: Path to the vector store
        """
        self.store_path = Path(store_path)
        self.metadata_path = self.store_path / "metadata.json"
        self.index_path = self.store_path / "index.faiss"
        
        # Ensure the store directory exists
        self.store_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize the index and metadata
        self.index = None
        self.metadata = {}
        self.chunks = []
        
        # Load existing data if available
        self._load()
    
    def _load(self):
        """Load the index and metadata from disk."""
        # Load the metadata
        if self.metadata_path.exists():
            with open(self.metadata_path, "r") as file:
                self.metadata = json.load(file)
            
            # Load the chunks
            chunks_path = self.store_path / self.metadata.get("chunks_file", "chunks.json")
            if chunks_path.exists():
                with open(chunks_path, "r") as file:
                    self.chunks = json.load(file)
        
        # Load the index
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
    
    def _save(self):
        """Save the index and metadata to disk."""
        # Save the metadata
        with open(self.metadata_path, "w") as file:
            json.dump(self.metadata, file, indent=2)
        
        # Save the chunks
        chunks_path = self.store_path / self.metadata.get("chunks_file", "chunks.json")
        with open(chunks_path, "w") as file:
            json.dump(self.chunks, file, indent=2)
        
        # Save the index
        if self.index is not None:
            faiss.write_index(self.index, str(self.index_path))
    
    def add_document(self, document_id: str, chunks_with_embeddings: List[Dict[str, Any]]):
        """Add a document to the vector store.
        
        Args:
            document_id: ID of the document
            chunks_with_embeddings: Chunks with embeddings to add
        """
        # Update the metadata
        if "documents" not in self.metadata:
            self.metadata["documents"] = {}
        
        # Get the current index size
        current_index_size = 0
        if self.index is not None:
            current_index_size = self.index.ntotal
        
        # Record the document's chunk indices
        chunk_indices = list(range(current_index_size, current_index_size + len(chunks_with_embeddings)))
        self.metadata["documents"][document_id] = {
            "chunk_indices": chunk_indices,
            "num_chunks": len(chunks_with_embeddings)
        }
        
        # Update the default chunks file if not set
        if "chunks_file" not in self.metadata:
            self.metadata["chunks_file"] = "chunks.json"
        
        # Add the chunks to the list
        for chunk in chunks_with_embeddings:
            # Remove the embedding from the chunk data to save space
            chunk_data = {k: v for k, v in chunk.items() if k != "embedding"}
            self.chunks.append(chunk_data)
        
        # Convert embeddings to numpy array
        embeddings = np.array([chunk["embedding"] for chunk in chunks_with_embeddings], dtype=np.float32)
        
        # Create or update the index
        if self.index is None:
            # Create a new index
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
            self.index.add(embeddings)
        else:
            # Add to existing index
            self.index.add(embeddings)
        
        # Save to disk
        self._save()
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar chunks.
        
        Args:
            query_embedding: Embedding of the query
            k: Number of results to return
            
        Returns:
            List of similar chunks with metadata
        """
        if self.index is None or len(self.chunks) == 0:
            return []
        
        # Ensure the query embedding is 2D
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Search the index
        distances, indices = self.index.search(query_embedding, k)
        
        # Get the results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks) and idx >= 0:
                result = self.chunks[idx].copy()
                result["distance"] = float(distances[0][i])
                results.append(result)
        
        return results
    
    def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a document.
        
        Args:
            document_id: ID of the document
            
        Returns:
            List of chunks for the document
        """
        if "documents" not in self.metadata or document_id not in self.metadata["documents"]:
            return []
        
        # Get the chunk indices
        chunk_indices = self.metadata["documents"][document_id]["chunk_indices"]
        
        # Get the chunks
        chunks = []
        for idx in chunk_indices:
            if idx < len(self.chunks):
                chunks.append(self.chunks[idx])
        
        return chunks
    
    def remove_document(self, document_id: str):
        """Remove a document from the vector store.
        
        Args:
            document_id: ID of the document
        """
        if "documents" not in self.metadata or document_id not in self.metadata["documents"]:
            return
        
        # Get the chunk indices for this document
        chunk_indices = self.metadata["documents"][document_id]["chunk_indices"]
        
        # Remove the chunks from the chunks list
        # We need to remove them in reverse order to maintain correct indices
        for idx in sorted(chunk_indices, reverse=True):
            if idx < len(self.chunks):
                self.chunks.pop(idx)
        
        # Remove the document from metadata
        del self.metadata["documents"][document_id]
        
        # Rebuild the index with remaining chunks
        if self.chunks:
            # Generate new embeddings for remaining chunks
            from src.embedding.embedder import Embedder
            embedder = Embedder()
            
            # Generate embeddings for all chunks at once
            texts = [chunk["text"] for chunk in self.chunks]
            embeddings = []
            for text in texts:
                embedding = embedder.generate_embedding(text)
                embeddings.append(embedding)
            
            # Convert to numpy array and ensure correct shape
            embeddings = np.array(embeddings, dtype=np.float32)
            
            # Create a new index with the correct dimensions
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
            
            # Add all embeddings at once
            self.index.add(embeddings)
        else:
            self.index = None
        
        # Save to disk
        self._save()
    
    def list_documents(self) -> List[str]:
        """List all documents in the vector store.
        
        Returns:
            List of document IDs
        """
        if "documents" not in self.metadata:
            return []
        
        # Return only non-removed documents
        documents = []
        for doc_id, doc_info in self.metadata["documents"].items():
            if not doc_info.get("removed", False):
                documents.append(doc_id)
        
        return documents
    
    def clear_all(self):
        """Clear all documents from the vector store."""
        # Reset metadata
        self.metadata = {
            "documents": {},
            "chunks_file": "chunks.json"
        }
        
        # Reset chunks
        self.chunks = []
        
        # Reset index
        self.index = None
        
        # Save to disk
        self._save()
