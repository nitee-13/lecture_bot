"""
Google Gemini API client for generating text responses.
"""
import os
import json
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from pathlib import Path
import logging
import fitz  # PyMuPDF
import time
import gc  # Garbage collection
import io
from PIL import Image
import numpy as np

from src.config import GOOGLE_API_KEY, GEMINI_MODEL, SYSTEM_PROMPT

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure the Gemini API
genai.configure(api_key=GOOGLE_API_KEY)

class GeminiClient:
    """Client for interacting with Google Gemini API."""
    
    def __init__(self, model: str = GEMINI_MODEL):
        """Initialize the Gemini client.
        
        Args:
            model: The model to use for generation
        """
        # Configure the Gemini API
        try:
            if not GOOGLE_API_KEY:
                raise ValueError("Google API Key not found. Please add it to your .env file.")
            
            genai.configure(api_key=GOOGLE_API_KEY)
            self.model = genai.GenerativeModel(model)
            self.file_model = genai.GenerativeModel(model)  # Use the same model for file processing
            
            # Define generation config
            self.generation_config = {
                "temperature": 0.2,
                "top_p": 0.95,
                "top_k": 0,
                "max_output_tokens": 8192,
            }
            
            logger.info(f"Initialized Gemini client with model: {model}")
        except Exception as e:
            logger.error(f"Error initializing Gemini client: {e}")
            raise
    
    def generate_response(self, prompt: str, temperature: float = 0.2) -> str:
        """Generate a response based on a prompt.
        
        Args:
            prompt: The prompt to generate from
            temperature: The temperature for generation
            
        Returns:
            The generated text
        """
        try:
            response = self.model.generate_content(prompt, generation_config={
                "temperature": temperature,
                "top_p": 0.95,
                "top_k": 0,
                "max_output_tokens": 2048,
            })
            
            return response.text
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"
    
    def generate_embeddings(self, text: str) -> List[float]:
        """Generate embeddings for text.
        
        Args:
            text: The text to embed
            
        Returns:
            The embedding vector
        """
        try:
            embedding_model = genai.GenerativeModel("embedding-001")
            result = embedding_model.embed_content(text)
            
            if hasattr(result, "embedding"):
                return result.embedding
            else:
                logger.error("No embedding returned")
                return []
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return []
    
    def _process_page(self, page: fitz.Page, page_num: int, temperature: float) -> str:
        """Process a single page with memory optimization.
        
        Args:
            page: PyMuPDF page object
            page_num: Page number
            temperature: Temperature for generation
            
        Returns:
            Extracted text from the page
        """
        try:
            # Get page dimensions
            width, height = page.rect.width, page.rect.height
            
            # Calculate scale to reduce memory usage while maintaining quality
            scale = min(2.0, 2000 / max(width, height))
            
            # Create pixmap with reduced resolution
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
            
            # Convert to PIL Image for memory-efficient processing
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Convert to JPEG with compression to reduce memory
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG', quality=85, optimize=True)
            img_bytes = img_byte_arr.getvalue()
            
            # Clear memory
            del pix
            del img
            del img_byte_arr
            gc.collect()
            
            # Create prompt for this page
            prompt = f"""
            Extract all text content from this page (Page {page_num + 1}). 
            Preserve the structure including titles and bullet points.
            Correctly format any mathematical equations using LaTeX notation.
            Include all text, even if it appears in diagrams or charts.
            Format the output as a structured text with clear section breaks.
            """
            
            # Generate response for this page
            response = self.file_model.generate_content(
                [prompt, img_bytes],
                generation_config={
                    "temperature": temperature,
                    "top_p": 0.95,
                    "top_k": 0,
                    "max_output_tokens": 4096,
                }
            )
            
            # Clear memory
            del img_bytes
            gc.collect()
            
            return f"Page {page_num + 1}:\n{response.text}"
            
        except Exception as e:
            logger.warning(f"Error processing page {page_num + 1}: {e}")
            # Fallback to PyMuPDF text extraction
            return f"Page {page_num + 1}:\n{page.get_text('text')}"
    
    def _fallback_extract(self, pdf_path: str) -> Dict[str, Any]:
        """Fallback method to extract content using PyMuPDF.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing extracted content
        """
        try:
            # Open the PDF
            doc = fitz.open(pdf_path)
            
            # Extract content from each page
            content = {
                "concepts": [],
                "equations": [],
                "diagrams": [],
                "relationships": []
            }
            
            # Process each page
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                
                # Extract equations (basic pattern matching)
                import re
                equations = re.findall(r'\$.*?\$|\\\(.*?\\\)|\\\[.*?\\\]', text)
                for eq in equations:
                    content["equations"].append({
                        "latex": eq,
                        "context": "Extracted from text",
                        "variables": [],
                        "related_concepts": []
                    })
                
                # Extract concepts (basic approach)
                sentences = text.split('.')
                for sentence in sentences:
                    if len(sentence.strip()) > 10:  # Avoid very short sentences
                        content["concepts"].append({
                            "name": sentence.strip(),
                            "definition": sentence.strip(),
                            "prerequisites": [],
                            "related_concepts": []
                        })
            
            doc.close()
            return content
            
        except Exception as e:
            logger.error(f"Error in fallback extraction: {e}")
            return {
                "concepts": [],
                "equations": [],
                "diagrams": [],
                "relationships": []
            }
    
    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Process a PDF file using Gemini Vision.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        try:
            # Validate PDF file
            pdf_path = Path(pdf_path)
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            # Upload the PDF file to Gemini
            file_data = genai.upload_file(str(pdf_path))
            
            # Create a prompt that handles varying slide formats
            prompt = """Extract structured information from this lecture slide. 
            Note that equations and concepts may be placed differently across slides.
            
            For each slide:
            1. Identify key concepts and their definitions
            2. Extract mathematical equations in LaTeX format, regardless of their placement
            3. Identify diagrams and their relationships to concepts
            4. Note any prerequisites or related concepts mentioned
            
            Format the output as JSON with the following structure:
            {
                "concepts": [
                    {
                        "name": "concept name",
                        "definition": "clear definition",
                        "prerequisites": ["prereq1", "prereq2"],
                        "related_concepts": ["related1", "related2"]
                    }
                ],
                "equations": [
                    {
                        "latex": "equation in LaTeX",
                        "context": "where/how it's used",
                        "variables": ["var1", "var2"],
                        "related_concepts": ["concept1", "concept2"]
                    }
                ],
                "diagrams": [
                    {
                        "description": "what the diagram shows",
                        "type": "flowchart/circuit/etc",
                        "relationships": ["rel1", "rel2"],
                        "related_concepts": ["concept1", "concept2"]
                    }
                ],
                "relationships": [
                    {
                        "from": "source concept",
                        "to": "target concept",
                        "type": "prerequisite/part_of/etc",
                        "description": "how they're related"
                    }
                ]
            }
            
            Important:
            - Look for equations anywhere on the slide, not just in specific sections
            - Consider both explicit and implicit relationships
            - Include context for equations to understand their usage
            - Note any visual elements that help explain concepts
            """
            
            # Process the PDF with Gemini
            response = self.model.generate_content(
                [prompt, file_data],
                generation_config=self.generation_config
            )
            
            # Parse the response as JSON
            try:
                # Extract the JSON part from the response
                json_text = response.text
                # Check if the response is wrapped in code blocks
                if "```json" in json_text:
                    json_text = json_text.split("```json")[1].split("```")[0]
                elif "```" in json_text:
                    json_text = json_text.split("```")[1].split("```")[0]
                
                # Parse the JSON
                extracted_data = json.loads(json_text.strip())
                
                # Log the extracted JSON for debugging
                print("\nExtracted JSON from PDF:")
                print(json.dumps(extracted_data, indent=2))
                
                return extracted_data
                
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON response: {e}")
                print("Raw response:", response.text)
                raise
                
        except Exception as e:
            print(f"Error in process_pdf: {e}")
            import traceback
            print(traceback.format_exc())
            # Fallback to PyMuPDF
            return self._fallback_extract(pdf_path)
    
    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
        """Generate a response to a conversation.
        
        Args:
            messages: List of message dictionaries with roles and content
            temperature: The temperature for generation
            
        Returns:
            The generated response
        """
        try:
            # Convert messages to the format expected by genai
            chat = self.model.start_chat(history=[])
            
            # Add messages to the chat
            for message in messages:
                if message["role"] == "user":
                    chat.send_message(message["content"])
                elif message["role"] == "assistant":
                    # For assistant messages, we just need to add them to history
                    # We'll handle this differently if the API changes
                    pass
            
            # Generate response to the last message
            response = chat.send_message(
                messages[-1]["content"] if messages[-1]["role"] == "user" else "Please respond.",
                generation_config={
                    "temperature": temperature,
                    "top_p": 0.95,
                    "top_k": 0,
                    "max_output_tokens": 2048,
                }
            )
            
            return response.text
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return f"Error generating response: {str(e)}"
    
    def generate_response(self, 
                          prompt: str, 
                          context: Optional[List[str]] = None,
                          system_prompt: str = SYSTEM_PROMPT,
                          temperature: float = 0.2) -> str:
        """Generate a response using the Gemini model.
        
        Args:
            prompt: The user prompt
            context: Optional context from retrieved documents
            system_prompt: The system prompt to use
            temperature: The temperature for generation
            
        Returns:
            The generated response as a string
        """
        # Construct the full prompt with context if provided
        full_prompt = prompt
        if context:
            context_text = "\n\n".join(context)
            full_prompt = f"Context information is below.\n\n{context_text}\n\nGiven the context information and not prior knowledge, answer the query: {prompt}"
            
        # Create the generation config
        generation_config = {
            "temperature": temperature,
            "top_p": 0.95,
            "top_k": 0,
            "max_output_tokens": 4096,
        }
        
        # Create safety settings
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        
        # Generate the response
        response = self.model.generate_content(
            [system_prompt, full_prompt],
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        return response.text
    
    def extract_pdf_content(self, pdf_path: str) -> Dict[str, Any]:
        """Extract content from a PDF file using Gemini's vision capabilities.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted content as a dictionary
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Load the PDF file
        file_data = genai.upload_file(str(pdf_path))
        
        # Create a prompt for content extraction
        system_prompt = """Extract all content from this lecture slide PDF, including:
1. All text content
2. All mathematical equations (preserve LaTeX format)
3. Key concepts and their definitions
4. Figures and diagrams descriptions
5. Important relationships between concepts

Format the output as a JSON with the following structure:
{
  "slide_content": [
    {
      "slide_number": 1,
      "title": "Slide title",
      "content": "Text content",
      "equations": ["Equation 1", "Equation 2"],
      "concepts": ["Concept 1", "Concept 2"],
      "relationships": [{"from": "Concept A", "to": "Concept B", "type": "is related to"}]
    }
  ]
}"""
        
        # Generate the response
        response = self.file_model.generate_content(
            [
                system_prompt,
                "Extract all content from this lecture slide PDF, maintaining the mathematical equations in LaTeX format.",
                file_data
            ],
            generation_config={"temperature": 0.2, "max_output_tokens": 8192}
        )
        
        # Parse the JSON response
        try:
            # Extract the JSON part from the response
            json_text = response.text
            # Check if the response is wrapped in code blocks
            if "```json" in json_text:
                json_text = json_text.split("```json")[1].split("```")[0]
            elif "```" in json_text:
                json_text = json_text.split("```")[1].split("```")[0]
                
            # Parse the JSON
            result = json.loads(json_text.strip())
            return result
        except json.JSONDecodeError:
            # If JSON parsing fails, return the raw text
            return {"raw_text": response.text}
        
    def build_knowledge_graph(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Build a knowledge graph from extracted content.
        
        Args:
            content: Extracted content from PDF
            
        Returns:
            Knowledge graph data as a dictionary
        """
        # Extract slide content
        if "slide_content" not in content:
            return {"nodes": [], "edges": []}
        
        slides = content["slide_content"]
        
        # Prepare a prompt for knowledge graph generation
        slide_texts = [
            f"Slide {slide.get('slide_number', i+1)}: {slide.get('title', 'Untitled')}\n"
            f"Content: {slide.get('content', '')}\n"
            f"Equations: {', '.join(slide.get('equations', []))}\n"
            f"Concepts: {', '.join(slide.get('concepts', []))}\n"
            for i, slide in enumerate(slides)
        ]
        
        combined_text = "\n\n".join(slide_texts)
        
        system_prompt = """Analyze the lecture content and create a knowledge graph. 
Identify key concepts, entities, and their relationships.
Output the graph as a JSON with the following structure:
{
  "nodes": [
    {"id": "concept1", "label": "Concept 1", "type": "concept"},
    {"id": "entity1", "label": "Entity 1", "type": "entity"}
  ],
  "edges": [
    {"from": "concept1", "to": "entity1", "label": "relates to"}
  ]
}
Node types can be: concept, equation, definition, example, figure
Edge labels should describe the relationship between nodes."""
        
        # Generate the response
        response = self.model.generate_content(
            [
                system_prompt,
                f"Create a knowledge graph from the following lecture content:\n\n{combined_text}"
            ],
            generation_config={"temperature": 0.2, "max_output_tokens": 8192}
        )
        
        # Parse the JSON response
        try:
            # Extract the JSON part from the response
            json_text = response.text
            # Check if the response is wrapped in code blocks
            if "```json" in json_text:
                json_text = json_text.split("```json")[1].split("```")[0]
            elif "```" in json_text:
                json_text = json_text.split("```")[1].split("```")[0]
                
            # Parse the JSON
            result = json.loads(json_text.strip())
            return result
        except json.JSONDecodeError:
            # If JSON parsing fails, return an empty graph
            return {"nodes": [], "edges": []}
            
    def generate_quiz(self, context: List[str], num_questions: int = 5) -> Dict[str, Any]:
        """Generate a quiz based on the lecture content.
        
        Args:
            context: Context information from retrieved documents
            num_questions: Number of questions to generate
            
        Returns:
            Quiz data as a dictionary
        """
        context_text = "\n\n".join(context)
        
        system_prompt = f"""Generate a quiz with {num_questions} questions based on the lecture content.
Include different types of questions: multiple choice, true/false, and open-ended.
For mathematical questions, include proper LaTeX formatting.
Output the quiz as a JSON with the following structure:
{{
  "quiz": [
    {{
      "question": "Question text",
      "type": "multiple_choice", 
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "correct_answer": "Option A",
      "explanation": "Explanation of the correct answer"
    }},
    {{
      "question": "Question text",
      "type": "true_false",
      "correct_answer": true,
      "explanation": "Explanation of the correct answer"
    }},
    {{
      "question": "Question text",
      "type": "open_ended",
      "sample_answer": "Sample correct answer",
      "explanation": "Explanation of the sample answer"
    }}
  ]
}}"""
        
        # Generate the response
        response = self.model.generate_content(
            [
                system_prompt,
                f"Generate a quiz based on the following lecture content:\n\n{context_text}"
            ],
            generation_config={"temperature": 0.3, "max_output_tokens": 4096}
        )
        
        # Parse the JSON response
        try:
            # Extract the JSON part from the response
            json_text = response.text
            # Check if the response is wrapped in code blocks
            if "```json" in json_text:
                json_text = json_text.split("```json")[1].split("```")[0]
            elif "```" in json_text:
                json_text = json_text.split("```")[1].split("```")[0]
                
            # Parse the JSON
            result = json.loads(json_text.strip())
            return result
        except json.JSONDecodeError:
            # If JSON parsing fails, return an empty quiz
            return {"quiz": []}
