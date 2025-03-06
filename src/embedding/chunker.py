"""
Text chunking module for splitting documents into chunks for embedding.
"""
from typing import List, Dict, Any, Tuple
import re
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

from src.config import CHUNK_SIZE, CHUNK_OVERLAP


class TextChunker:
    """Split text into chunks for embedding."""
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP, max_workers: int = 4):
        """Initialize the text chunker.
        
        Args:
            chunk_size: Maximum chunk size in characters
            chunk_overlap: Overlap between chunks in characters
            max_workers: Maximum number of worker threads for parallel processing
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_workers = max_workers
    
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
    
    def chunk_lecture_content(self, content: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split lecture content into chunks.
        
        Args:
            content: Lecture content dictionary
            
        Returns:
            List of chunked content items
        """
        chunks = []
        
        # Handle slide_content structure
        if "slide_content" in content:
            slides = content["slide_content"]
            
            # Process slides in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Create a partial function with the slide processing logic
                process_slide = partial(self._process_single_slide)
                
                # Submit all slides for processing
                future_to_slide = {
                    executor.submit(process_slide, slide): slide 
                    for slide in slides
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_slide):
                    slide_chunks = future.result()
                    chunks.extend(slide_chunks)
        
        # Handle concepts/equations/diagrams/relationships structure
        elif all(key in content for key in ["concepts", "equations", "diagrams", "relationships"]):
            # Create a single chunk for each concept
            for concept in content["concepts"]:
                chunk_text = f"Concept: {concept['name']}\n"
                chunk_text += f"Definition: {concept['definition']}\n"
                
                # Add prerequisites if any
                if concept.get("prerequisites"):
                    chunk_text += f"Prerequisites: {', '.join(concept['prerequisites'])}\n"
                
                # Add related concepts if any
                if concept.get("related_concepts"):
                    chunk_text += f"Related Concepts: {', '.join(concept['related_concepts'])}\n"
                
                # Add related equations
                related_equations = [
                    eq for eq in content["equations"]
                    if concept["name"] in eq.get("related_concepts", [])
                ]
                if related_equations:
                    chunk_text += "\nRelated Equations:\n"
                    for eq in related_equations:
                        chunk_text += f"- {eq['latex']}\n"
                
                # Add related diagrams
                related_diagrams = [
                    d for d in content["diagrams"]
                    if concept["name"] in d.get("related_concepts", [])
                ]
                if related_diagrams:
                    chunk_text += "\nRelated Diagrams:\n"
                    for d in related_diagrams:
                        chunk_text += f"- {d['description']}\n"
                
                # Add relationships
                related_relationships = [
                    r for r in content["relationships"]
                    if r["from"] == concept["name"] or r["to"] == concept["name"]
                ]
                if related_relationships:
                    chunk_text += "\nRelationships:\n"
                    for r in related_relationships:
                        chunk_text += f"- {r['from']} {r['type']} {r['to']}\n"
                
                chunk_item = {
                    "id": f"concept_{concept['name'].lower().replace(' ', '_')}",
                    "slide_number": 0,
                    "slide_title": f"Concept: {concept['name']}",
                    "chunk_index": 0,
                    "text": chunk_text,
                    "equations": [eq["latex"] for eq in related_equations],
                    "concepts": [concept["name"]],
                    "source": "Concept Extraction"
                }
                chunks.append(chunk_item)
        
        return chunks
    
    def _process_single_slide(self, slide: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process a single slide into chunks.
        
        Args:
            slide: Slide dictionary
            
        Returns:
            List of chunks for the slide
        """
        slide_chunks = []
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
            slide_chunks.append(chunk_item)
        
        return slide_chunks
    
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
