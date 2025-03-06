"""
PDF extraction module for reading and processing PDF files.
"""
import os
import re
import fitz  # PyMuPDF
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import logging
import json
import traceback

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
                error_msg = str(e)
                if "429" in error_msg or "quota" in error_msg.lower():
                    logger.warning("Gemini API quota exceeded. Falling back to PyMuPDF.")
                    self.use_gemini = False  # Disable Gemini for future extractions
                else:
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
    
    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Process a PDF file and extract its content.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing extracted content
        """
        try:
            # Validate PDF file
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            # Try using Gemini first if enabled
            if self.use_gemini:
                try:
                    logger.info(f"Extracting text from PDF using Gemini: {pdf_path}")
                    extracted_data = self.gemini_client.process_pdf(pdf_path)
                    
                    # If we got a dictionary response, validate and return it
                    if isinstance(extracted_data, dict):
                        # Ensure all required keys are present
                        required_keys = ["concepts", "equations", "diagrams", "relationships"]
                        for key in required_keys:
                            if key not in extracted_data:
                                extracted_data[key] = []
                        
                        # Validate the structure of each section
                        for key in required_keys:
                            if not isinstance(extracted_data[key], list):
                                extracted_data[key] = []
                        
                        return extracted_data
                    
                    # If we got a string response, try to parse it as JSON
                    if isinstance(extracted_data, str):
                        try:
                            # Try to parse the string as JSON
                            parsed_data = json.loads(extracted_data)
                            if isinstance(parsed_data, dict):
                                return parsed_data
                        except json.JSONDecodeError:
                            # If parsing fails, convert it to the expected format
                            return {
                                "concepts": [],
                                "equations": [],
                                "diagrams": [],
                                "relationships": [],
                                "raw_text": extracted_data
                            }
                    
                except Exception as e:
                    logger.warning(f"Error using Gemini for extraction: {e}")
                    logger.info("Falling back to PyMuPDF for extraction")
            
            # Fallback to PyMuPDF
            try:
                logger.info(f"Extracting text from PDF using PyMuPDF: {pdf_path}")
                raw_text = self._extract_with_pymupdf(pdf_path)
                
                if not raw_text or len(raw_text.strip()) == 0:
                    raise ValueError("No text extracted from PDF")
                
                # Extract basic structure from raw text
                content = {
                    "concepts": [],
                    "equations": [],
                    "diagrams": [],
                    "relationships": [],
                    "raw_text": raw_text
                }
                
                # Try to extract equations
                equations = re.findall(r'\$.*?\$|\\\(.*?\\\)|\\\[.*?\\\]', raw_text)
                for eq in equations:
                    content["equations"].append({
                        "latex": eq,
                        "context": "Extracted from text",
                        "variables": [],
                        "related_concepts": []
                    })
                
                # Try to extract concepts (basic approach)
                sentences = raw_text.split('.')
                for sentence in sentences:
                    if len(sentence.strip()) > 10:  # Avoid very short sentences
                        content["concepts"].append({
                            "name": sentence.strip(),
                            "definition": sentence.strip(),
                            "prerequisites": [],
                            "related_concepts": []
                        })
                
                return content
                
            except Exception as e:
                logger.error(f"Error in PyMuPDF extraction: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                raise
            
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
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
                
                for slide in data["slide_content"]:
                    f.write(f"## Slide {slide['slide_number']}: {slide['title']}\n\n")
                    f.write(f"{slide['content']}\n\n")
            
            logger.info(f"Saved extracted content to: {output_path}")
        except Exception as e:
            logger.error(f"Error saving to file: {e}")
            raise
