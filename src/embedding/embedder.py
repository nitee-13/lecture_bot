"""
Generate embeddings for text chunks using sentence-transformers with GPU support.
"""
from typing import List, Dict, Any, Optional, Union
import json
import os
import numpy as np
from pathlib import Path
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
from sentence_transformers import SentenceTransformer
import torch
import logging

from src.config import GOOGLE_API_KEY, EMBEDDING_DIMENSIONS, VECTOR_DB_PATH, BATCH_SIZE

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure the Gemini API
genai.configure(api_key=GOOGLE_API_KEY)

class Embedder:
    """Generate embeddings for text chunks using sentence-transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", embedding_dimensions: int = 384):
        """Initialize the embedder.
        
        Args:
            model_name: Name of the sentence-transformers model to use
            embedding_dimensions: Expected embedding dimensions
        """
        self.embedding_dimensions = embedding_dimensions
        
        # Safely determine device
        try:
            if torch.cuda.is_available():
                self.device = "cuda"
                logger.info("Using CUDA for embeddings")
            else:
                self.device = "cpu"
                logger.info("Using CPU for embeddings")
        except Exception as e:
            logger.warning(f"Error checking CUDA availability: {e}")
            self.device = "cpu"
        
        try:
            # Initialize the model
            self.model = SentenceTransformer(model_name)
            self.model.to(self.device)
            logger.info(f"Successfully initialized {model_name} on {self.device}")
        except Exception as e:
            logger.error(f"Error initializing sentence transformer: {e}")
            raise
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate an embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding as a numpy array
        """
        try:
            # Generate embedding using sentence-transformers
            embedding = self.model.encode(text, convert_to_numpy=True)
            
            # Ensure correct dimensions
            if len(embedding) != self.embedding_dimensions:
                if len(embedding) < self.embedding_dimensions:
                    padding = np.zeros(self.embedding_dimensions - len(embedding))
                    embedding = np.concatenate([embedding, padding])
                else:
                    embedding = embedding[:self.embedding_dimensions]
            
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            # Fallback to TF-IDF if sentence transformer fails
            return self._generate_tfidf_embedding(text)
    
    def _generate_tfidf_embedding(self, text: str) -> np.ndarray:
        """Generate a TF-IDF embedding as fallback.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding as a numpy array
        """
        try:
            # Initialize TF-IDF vectorizer if not already done
            if not hasattr(self, 'vectorizer'):
                self.vectorizer = TfidfVectorizer(max_features=self.embedding_dimensions)
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
        except Exception as e:
            logger.error(f"Error generating TF-IDF embedding: {e}")
            # Return zero vector as last resort
            return np.zeros(self.embedding_dimensions)
    
    def generate_embeddings(self, chunks: List[Dict[str, Any]], batch_size: int = BATCH_SIZE) -> List[Dict[str, Any]]:
        """Generate embeddings for a list of text chunks in batches.
        
        Args:
            chunks: List of text chunks
            batch_size: Number of chunks to process at once
            
        Returns:
            List of chunks with embeddings added
        """
        try:
            # Extract texts for batch processing
            texts = [chunk["text"] for chunk in chunks]
            
            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                try:
                    batch_embeddings = self.model.encode(
                        batch_texts,
                        convert_to_numpy=True,
                        show_progress_bar=False
                    )
                    
                    # Add embeddings to chunks
                    for j, embedding in enumerate(batch_embeddings):
                        chunk_idx = i + j
                        if chunk_idx < len(chunks):
                            chunks[chunk_idx]["embedding"] = embedding.tolist()
                except Exception as e:
                    logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
                    # Process remaining chunks individually with fallback
                    for j in range(len(batch_texts)):
                        chunk_idx = i + j
                        if chunk_idx < len(chunks):
                            chunks[chunk_idx]["embedding"] = self.generate_embedding(batch_texts[j]).tolist()
            
            return chunks
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            # Process all chunks individually as fallback
            for chunk in chunks:
                chunk["embedding"] = self.generate_embedding(chunk["text"]).tolist()
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
