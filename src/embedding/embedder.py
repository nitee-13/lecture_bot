"""
Embedding module for generating embeddings from text chunks.
"""
from typing import List, Dict, Any, Optional, Union
import json
import os
import numpy as np
from pathlib import Path
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss

from src.config import GOOGLE_API_KEY, EMBEDDING_DIMENSIONS, VECTOR_DB_PATH

# Configure the Gemini API
genai.configure(api_key=GOOGLE_API_KEY)

class Embedder:
    """Generate embeddings for text chunks."""
    
    def __init__(self, use_gemini: bool = True, embedding_dimensions: int = EMBEDDING_DIMENSIONS):
        """Initialize the embedder.
        
        Args:
            use_gemini: Whether to use Gemini for embeddings
            embedding_dimensions: Dimensions of the embeddings
        """
        self.use_gemini = use_gemini
        self.embedding_dimensions = embedding_dimensions
        
        if use_gemini:
            self.embedding_model = genai.GenerativeModel(model_name="models/embedding-001")
        else:
            # Fallback to TF-IDF vectorizer
            self.vectorizer = TfidfVectorizer(max_features=embedding_dimensions)
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate an embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding as a numpy array
        """
        if self.use_gemini:
            # Use Gemini to generate the embedding
            try:
                result = self.embedding_model.embed_content(text)
                embedding = np.array(result.embedding.values)
                return embedding
            except Exception as e:
                print(f"Error generating embedding with Gemini: {e}")
                # Fall back to TF-IDF
                return self._generate_tfidf_embedding(text)
        else:
            # Use TF-IDF to generate the embedding
            return self._generate_tfidf_embedding(text)
    
    def _generate_tfidf_embedding(self, text: str) -> np.ndarray:
        """Generate a TF-IDF embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding as a numpy array
        """
        # Fit the vectorizer if it hasn't been fit yet
        if not hasattr(self.vectorizer, 'vocabulary_'):
            self.vectorizer.fit([text])
        
        # Transform the text
        embedding = self.vectorizer.transform([text]).toarray()[0]
        
        # Pad or truncate to the required dimensions
        if len(embedding) < self.embedding_dimensions:
            padding = np.zeros(self.embedding_dimensions - len(embedding))
            embedding = np.concatenate([embedding, padding])
        elif len(embedding) > self.embedding_dimensions:
            embedding = embedding[:self.embedding_dimensions]
        
        return embedding
    
    def generate_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate embeddings for a list of text chunks.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            List of chunks with embeddings added
        """
        for chunk in chunks:
            # Generate the embedding for the chunk text
            embedding = self.generate_embedding(chunk["text"])
            
            # Add the embedding to the chunk
            chunk["embedding"] = embedding.tolist()
        
        return chunks
    
    def save_embeddings(self, chunks_with_embeddings: List[Dict[str, Any]], output_path: str) -> str:
        """Save chunks with embeddings to a file.
        
        Args:
            chunks_with_embeddings: List of chunks with embeddings
            output_path: Path to save the chunks
            
        Returns:
            Path to the saved file
        """
        # Ensure the directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the chunks
        with open(output_path, "w") as file:
            json.dump(chunks_with_embeddings, file, indent=2)
        
        return str(output_path)
    
    def create_faiss_index(self, chunks_with_embeddings: List[Dict[str, Any]], index_path: str) -> Any:
        """Create a FAISS index from the embeddings.
        
        Args:
            chunks_with_embeddings: List of chunks with embeddings
            index_path: Path to save the index
            
        Returns:
            FAISS index
        """
        # Convert embeddings to numpy array
        embeddings = np.array([chunk["embedding"] for chunk in chunks_with_embeddings], dtype=np.float32)
        
        # Create the index
        index = faiss.IndexFlatL2(self.embedding_dimensions)
        index.add(embeddings)
        
        # Save the index
        index_path = Path(index_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(index_path))
        
        return index
