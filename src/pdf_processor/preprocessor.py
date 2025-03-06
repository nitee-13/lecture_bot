"""
Text preprocessing module for cleaning and preparing text from PDFs.
"""
import re
from typing import List, Dict, Any, Tuple


class TextPreprocessor:
    """Preprocess and clean text extracted from PDFs."""
    
    def __init__(self):
        """Initialize the text preprocessor."""
        pass
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess and clean text.
        
        Args:
            text: Text to preprocess
            
        Returns:
            Preprocessed text
        """
        # Replace multiple newlines with a single newline
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Replace multiple spaces with a single space
        text = re.sub(r' {2,}', ' ', text)
        
        # Clean up whitespace around punctuation
        text = re.sub(r' +([.,;:!?])', r'\1', text)
        
        # Remove references like [1], [2], etc.
        text = re.sub(r'\[\d+\]', '', text)
        
        # Clean up bullets and numbering
        text = re.sub(r'^\s*[•●○◆▪]\s*', '• ', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*(\d+)[\.\)]\s*', r'\1. ', text, flags=re.MULTILINE)
        
        return text.strip()
    
    def extract_slides(self, text: str) -> List[Dict[str, str]]:
        """Split text into slides.
        
        Args:
            text: Text to split
            
        Returns:
            List of slide dictionaries with title and content
        """
        # Split the text into slides
        slides = []
        
        # Simple heuristic: Pages usually start with a number and have a title
        slide_pattern = r'(?:Page|Slide)?\s*(\d+)[\.\):]?\s*(.*?)(?=(?:Page|Slide)?\s*\d+[\.\):]|\Z)'
        slide_matches = re.finditer(slide_pattern, text, re.DOTALL)
        
        for match in slide_matches:
            slide_number = match.group(1)
            slide_content = match.group(2).strip()
            
            # Extract title from the first line
            slide_lines = slide_content.split('\n')
            title = slide_lines[0] if slide_lines else f"Slide {slide_number}"
            content = '\n'.join(slide_lines[1:]) if len(slide_lines) > 1 else ""
            
            slides.append({
                "slide_number": int(slide_number),
                "title": title,
                "content": content
            })
        
        # If no slides were extracted, treat the whole text as one slide
        if not slides:
            slides.append({
                "slide_number": 1,
                "title": "Extracted Content",
                "content": text
            })
        
        return slides
    
    def identify_section_headers(self, text: str) -> List[Tuple[str, int]]:
        """Identify section headers in text.
        
        Args:
            text: Text to process
            
        Returns:
            List of (header, position) tuples
        """
        headers = []
        
        # Match common section header patterns
        header_patterns = [
            r'^#+\s+(.+)$',  # Markdown-style headers
            r'^(\d+\.\d*\s+[A-Z][^\n]+)$',  # Numbered sections
            r'^([A-Z][a-zA-Z\s]{2,30})$'  # Capitalized short phrases
        ]
        
        lines = text.split('\n')
        line_position = 0
        
        for line in lines:
            line = line.strip()
            
            for pattern in header_patterns:
                match = re.match(pattern, line, re.MULTILINE)
                if match:
                    header_text = match.group(1)
                    headers.append((header_text, line_position))
                    break
            
            line_position += len(line) + 1  # +1 for the newline
        
        return headers
    
    def cleanup_equations(self, text: str) -> str:
        """Clean up equation formatting.
        
        Args:
            text: Text with equations
            
        Returns:
            Text with cleaned equations
        """
        # Ensure consistent delimiter spacing
        text = re.sub(r'(?<!\$)\$(?!\$)', ' $ ', text)
        text = re.sub(r'(?<!\$)\$\$(?!\$)', ' $$ ', text)
        
        # Fix equation formatting
        text = re.sub(r'\s*\$\s*([^$]+?)\s*\$\s*', r'$\1$', text)
        text = re.sub(r'\s*\$\$\s*([^$]+?)\s*\$\$\s*', r'$$\1$$', text)
        
        return text
