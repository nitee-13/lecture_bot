"""
PDF extraction module for reading and processing PDF files.
"""
import os
import re
import fitz  # PyMuPDF
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import logging

from src.config import GOOGLE_API_KEY
from src.llm.gemini_client import GeminiClient
from src.pdf_processor.preprocessor import TextPreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFExtractor:
    """Extract text content from PDF files."""
    
    def __init__(self, use_gemini: bool = True):
        """Initialize the PDF extractor.
        
        Args:
            use_gemini: Whether to use Gemini for extraction
        """
        self.use_gemini = use_gemini
        self.preprocessor = TextPreprocessor()
        
        if use_gemini:
            try:
                self.gemini_client = GeminiClient()
                logger.info("Initialized Gemini client for PDF extraction")
            except Exception as e:
                logger.warning(f"Could not initialize Gemini client: {e}")
                self.use_gemini = False
                logger.info("Falling back to PyMuPDF for extraction")
    
    def extract_text(self, pdf_path: str) -> str:
        """Extract text from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text content
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if self.use_gemini:
            try:
                logger.info(f"Extracting text from PDF using Gemini: {pdf_path}")
                return self.gemini_client.process_pdf(pdf_path)
            except Exception as e:
                logger.warning(f"Error using Gemini for extraction: {e}")
                logger.info("Falling back to PyMuPDF for extraction")
        
        # Fallback to PyMuPDF
        return self._extract_with_pymupdf(pdf_path)
    
    def _extract_with_pymupdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file using PyMuPDF.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text content
        """
        try:
            logger.info(f"Extracting text from PDF using PyMuPDF: {pdf_path}")
            doc = fitz.open(pdf_path)
            text = ""
            
            for i, page in enumerate(doc):
                page_text = page.get_text("text")
                text += f"Page {i+1}:\n\n{page_text}\n\n"
            
            return text
        except Exception as e:
            logger.error(f"Error extracting text with PyMuPDF: {e}")
            raise
    
    def process_pdf(self, pdf_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Process a PDF file and extract content.
        
        Args:
            pdf_path: Path to the PDF file
            output_path: Optional path to save the extracted content
            
        Returns:
            Dictionary with extracted content
        """
        try:
            # Extract text from PDF
            raw_text = self.extract_text(pdf_path)
            
            # Preprocess the extracted text
            cleaned_text = self.preprocessor.preprocess_text(raw_text)
            
            # Extract slides from the text
            slides = self.preprocessor.extract_slides(cleaned_text)
            
            # Clean up equations
            for slide in slides:
                slide["content"] = self.preprocessor.cleanup_equations(slide["content"])
            
            # Create result dictionary
            result = {
                "filename": os.path.basename(pdf_path),
                "path": pdf_path,
                "raw_text": raw_text,
                "cleaned_text": cleaned_text,
                "slides": slides,
                "num_slides": len(slides)
            }
            
            # Save to file if output path is provided
            if output_path:
                self._save_to_file(result, output_path)
            
            return result
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            raise
    
    def _save_to_file(self, data: Dict[str, Any], output_path: str) -> None:
        """Save extracted content to a file.
        
        Args:
            data: Dictionary with extracted content
            output_path: Path to save the extracted content
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Write to file
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"# {data['filename']}\n\n")
                
                for slide in data["slides"]:
                    f.write(f"## Slide {slide['slide_number']}: {slide['title']}\n\n")
                    f.write(f"{slide['content']}\n\n")
            
            logger.info(f"Saved extracted content to: {output_path}")
        except Exception as e:
            logger.error(f"Error saving to file: {e}")
            raise
