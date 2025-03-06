"""
Retriever module for retrieving relevant chunks from the vector store.
"""
from typing import List, Dict, Any
import numpy as np

from src.vector_store.store import VectorStore
from src.embedding.embedder import Embedder
from src.config import TOP_K_RESULTS


class Retriever:
    """Retrieve relevant chunks from the vector store."""
    
    def __init__(self, vector_store: VectorStore, embedder: Embedder):
        """Initialize the retriever.
        
        Args:
            vector_store: Vector store to retrieve from
            embedder: Embedder to use for query embedding
        """
        self.vector_store = vector_store
        self.embedder = embedder
    
    def retrieve(self, query: str, k: int = TOP_K_RESULTS) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks for a query.
        
        Args:
            query: Query to retrieve for
            k: Number of results to retrieve
            
        Returns:
            List of relevant chunks
        """
        # Generate an embedding for the query
        query_embedding = self.embedder.generate_embedding(query)
        
        # Search the vector store
        results = self.vector_store.search(query_embedding, k=k)
        
        # Sort by distance (ascending)
        results.sort(key=lambda x: x.get("distance", float("inf")))
        
        return results
    
    def retrieve_from_document(self, query: str, document_id: str, k: int = TOP_K_RESULTS) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks from a specific document.
        
        Args:
            query: Query to retrieve for
            document_id: ID of the document to retrieve from
            k: Number of results to retrieve
            
        Returns:
            List of relevant chunks
        """
        # Get all chunks for the document
        document_chunks = self.vector_store.get_document_chunks(document_id)
        
        if not document_chunks:
            return []
        
        # Generate an embedding for the query
        query_embedding = self.embedder.generate_embedding(query)
        
        # Calculate distances for each chunk
        results = []
        for chunk in document_chunks:
            # We need to retrieve the embedding from the vector store
            # This is a simplistic approach - in a real system, we'd want a more efficient way to do this
            chunk_index = chunk.get("id", "").split("_chunk_")[1] if "_chunk_" in chunk.get("id", "") else None
            if chunk_index and chunk_index.isdigit():
                idx = int(chunk_index)
                # This assumes the chunks are stored in order in the vector store
                # In a real system, we'd want a more robust way to handle this
                if self.vector_store.index is not None and idx < self.vector_store.index.ntotal:
                    # Reshape query embedding for distance calculation
                    q_emb = query_embedding.reshape(1, -1)
                    
                    # Get the distances
                    distances, _ = self.vector_store.index.search(q_emb, 1)
                    
                    # Add distance to the chunk
                    chunk_with_distance = chunk.copy()
                    chunk_with_distance["distance"] = float(distances[0][0])
                    results.append(chunk_with_distance)
        
        # Sort by distance (ascending)
        results.sort(key=lambda x: x.get("distance", float("inf")))
        
        # Return top k results
        return results[:k]
    
    def retrieve_contexts(self, query: str, k: int = TOP_K_RESULTS) -> List[str]:
        """Retrieve relevant contexts for a query.
        
        Args:
            query: Query to retrieve for
            k: Number of results to retrieve
            
        Returns:
            List of relevant context strings
        """
        # Retrieve relevant chunks
        results = self.retrieve(query, k=k)
        
        # Extract the text from each chunk
        contexts = []
        for result in results:
            context = result.get("text", "")
            
            # Add slide information
            slide_number = result.get("slide_number", "")
            slide_title = result.get("slide_title", "")
            if slide_number and slide_title:
                context = f"[Slide {slide_number}: {slide_title}]\n{context}"
            
            # Add equations if available
            equations = result.get("equations", [])
            if equations:
                equation_text = "\nEquations:\n" + "\n".join(equations)
                context += equation_text
            
            contexts.append(context)
        
        return contexts
