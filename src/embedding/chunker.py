"""
Text chunking module for splitting documents into chunks for embedding.
"""
from typing import List, Dict, Any, Tuple
import re
from pathlib import Path
import json

from src.config import CHUNK_SIZE, CHUNK_OVERLAP


class TextChunker:
    """Split text into chunks for embedding."""
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        """Initialize the text chunker.
        
        Args:
            chunk_size: Maximum chunk size in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        # Clean the text
        text = self._clean_text(text)
        
        # If the text is shorter than the chunk size, return it as is
        if len(text) <= self.chunk_size:
            return [text]
        
        # Split the text into chunks
        chunks = []
        start = 0
        
        while start < len(text):
            # Calculate the end position
            end = start + self.chunk_size
            
            # Adjust the end position to avoid cutting words
            if end < len(text):
                # Try to find a sentence boundary
                sentence_end = self._find_sentence_boundary(text, end)
                if sentence_end > start:
                    end = sentence_end
                else:
                    # Try to find a word boundary
                    word_end = self._find_word_boundary(text, end)
                    if word_end > start:
                        end = word_end
            else:
                end = len(text)
            
            # Add the chunk
            chunks.append(text[start:end])
            
            # Move the start position for the next chunk
            start = end - self.chunk_overlap
            
            # Ensure we make progress
            if start >= end:
                start = end
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean the text.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        
        # Replace multiple newlines with a single newline
        text = re.sub(r'\n+', '\n', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _find_sentence_boundary(self, text: str, position: int) -> int:
        """Find a sentence boundary near the specified position.
        
        Args:
            text: Text to search
            position: Position to search around
            
        Returns:
            Position of the sentence boundary, or the original position if none found
        """
        # Look for a sentence boundary within 100 characters of the position
        search_range = 100
        start = max(0, position - search_range)
        end = min(len(text), position + search_range)
        
        # Search for sentence-ending punctuation followed by a space or newline
        for i in range(position, start, -1):
            if i > 0 and i < len(text) and text[i-1] in '.!?' and (i == len(text) or text[i] in ' \n'):
                return i
        
        # If no sentence boundary found before the position, look after it
        for i in range(position, end):
            if i > 0 and i < len(text) and text[i-1] in '.!?' and (i == len(text) or text[i] in ' \n'):
                return i
        
        return position
    
    def _find_word_boundary(self, text: str, position: int) -> int:
        """Find a word boundary near the specified position.
        
        Args:
            text: Text to search
            position: Position to search around
            
        Returns:
            Position of the word boundary, or the original position if none found
        """
        # Look for a space within 20 characters of the position
        search_range = 20
        start = max(0, position - search_range)
        end = min(len(text), position + search_range)
        
        # Search for a space before the position
        for i in range(position, start, -1):
            if text[i] in ' \n\t':
                return i + 1
        
        # If no space found before the position, look after it
        for i in range(position, end):
            if text[i] in ' \n\t':
                return i
        
        return position
    
    def chunk_lecture_content(self, content: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split lecture content into chunks.
        
        Args:
            content: Lecture content dictionary
            
        Returns:
            List of chunked content items
        """
        chunks = []
        
        # Check if the content has the expected structure
        if "slide_content" not in content:
            return chunks
        
        slides = content["slide_content"]
        
        for slide in slides:
            slide_number = slide.get("slide_number", 0)
            slide_title = slide.get("title", "")
            slide_content = slide.get("content", "")
            equations = slide.get("equations", [])
            concepts = slide.get("concepts", [])
            
            # Combine the slide content with equations and concepts
            full_text = slide_content
            
            # Add equations if any
            if equations:
                equations_text = "\nEquations:\n" + "\n".join(equations)
                full_text += equations_text
            
            # Add concepts if any
            if concepts:
                concepts_text = "\nConcepts:\n" + ", ".join(concepts)
                full_text += concepts_text
            
            # Chunk the full text
            text_chunks = self.chunk_text(full_text)
            
            # Create a chunk item for each text chunk
            for i, chunk_text in enumerate(text_chunks):
                chunk_item = {
                    "id": f"slide_{slide_number}_chunk_{i}",
                    "slide_number": slide_number,
                    "slide_title": slide_title,
                    "chunk_index": i,
                    "text": chunk_text,
                    "equations": equations,
                    "concepts": concepts,
                    "source": f"Slide {slide_number}"
                }
                chunks.append(chunk_item)
        
        return chunks
    
    def save_chunks(self, chunks: List[Dict[str, Any]], output_path: str) -> str:
        """Save chunks to a file.
        
        Args:
            chunks: List of chunks
            output_path: Path to save the chunks
            
        Returns:
            Path to the saved file
        """
        # Ensure the directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the chunks
        with open(output_path, "w") as file:
            json.dump(chunks, file, indent=2)
        
        return str(output_path)
