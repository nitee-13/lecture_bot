"""
Upload page for Streamlit UI to upload and process PDF files.
"""
import streamlit as st
import os
from pathlib import Path
import uuid
import time
import logging
import traceback

from src.pdf_processor.extractor import PDFExtractor
from src.pdf_processor.preprocessor import TextPreprocessor
from src.embedding.chunker import TextChunker
from src.embedding.embedder import Embedder
from src.knowledge_graph.builder import KnowledgeGraphBuilder
from src.config import PDF_STORAGE_PATH
from src.vector_store.store import VectorStore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UploadPage:
    """Upload page for Streamlit UI."""
    
    def __init__(self, vector_store: VectorStore):
        """Initialize the upload page.
        
        Args:
            vector_store: VectorStore instance for storing embeddings
        """
        self.vector_store = vector_store
        self.pdf_extractor = PDFExtractor(use_gemini=True)
        self.preprocessor = TextPreprocessor()
        self.chunker = TextChunker()
        self.embedder = Embedder()
        self.kg_builder = KnowledgeGraphBuilder()
    
    def render(self):
        """Render the upload page."""
        st.title("Upload Lecture Files")
        
        with st.container():
            st.write("""
            Upload your lecture PDFs here. The system will process them, extract content, 
            and build a knowledge base for you to chat with.
            """)
            
            uploaded_file = st.file_uploader(
                "Choose a PDF file", 
                type=["pdf"],
                help="Upload lecture slides in PDF format"
            )
            
            if uploaded_file:
                try:
                    # Create a temporary file for the uploaded content
                    with st.status("Processing your file...", expanded=True) as status:
                        # Save the uploaded file
                        file_path = self._save_uploaded_file(uploaded_file)
                        status.update(label="✅ File uploaded", state="running")
                        
                        # Process the file
                        document_id = self._process_file(file_path, status)
                        
                        if document_id:
                            status.update(label="✅ Processing complete!", state="complete")
                            st.success(f"Successfully processed {uploaded_file.name}")
                            st.session_state.active_document = document_id
                        else:
                            status.update(label="❌ Processing failed", state="error")
                            st.error("Failed to process the file. Please try again.")
                except Exception as e:
                    logger.error(f"Error processing file: {e}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    st.error(f"Error processing file: {str(e)}")
                
                # Show details about the processed file outside the status container
                if document_id:
                    self._show_processing_details(document_id)
    
    def _save_uploaded_file(self, uploaded_file) -> str:
        """Save the uploaded file to disk.
        
        Args:
            uploaded_file: Uploaded file from Streamlit
            
        Returns:
            Path to the saved file
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(PDF_STORAGE_PATH, exist_ok=True)
            
            # Generate a unique filename
            filename = f"{int(time.time())}_{uploaded_file.name}"
            file_path = os.path.join(PDF_STORAGE_PATH, filename)
            
            # Save the file
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            logger.info(f"Saved uploaded file to: {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Error saving uploaded file: {e}")
            raise
    
    def _process_file(self, file_path: str, status) -> str:
        """Process the uploaded file.
        
        Args:
            file_path: Path to the file to process
            status: Streamlit status object for updates
            
        Returns:
            Document ID if successful, None otherwise
        """
        try:
            # Extract document ID from file path
            document_id = Path(file_path).stem
            filename = Path(file_path).name
            
            # 1. Process PDF page by page
            status.update(label="🔍 Processing PDF page by page...", state="running")
            page_contents = self.pdf_extractor.process_pdf_by_pages(file_path)
            
            if not page_contents:
                logger.error("No valid content extracted from PDF")
                raise ValueError("Failed to extract valid content from PDF")
            
            # 2. Process each page and build chunks and embeddings
            all_chunks = []
            
            for page_idx, page_content in enumerate(page_contents):
                page_num = page_idx + 1
                status.update(label=f"✂️ Processing page {page_num}/{len(page_contents)}...", state="running")
                
                # Create chunks from the page content
                page_chunks = self.chunker.chunk_lecture_content(page_content)
                
                if page_chunks:
                    # Add page number to each chunk
                    for chunk in page_chunks:
                        chunk["page"] = page_num
                    
                    # Add to all chunks
                    all_chunks.extend(page_chunks)
            
            if not all_chunks:
                logger.error("No chunks created from extracted content")
                raise ValueError("Failed to create text chunks")
            
            # 3. Generate embeddings for all chunks
            status.update(label="🧠 Generating embeddings...", state="running")
            chunks_with_embeddings = self.embedder.generate_embeddings(all_chunks)
            
            if not chunks_with_embeddings or not all("embedding" in chunk for chunk in chunks_with_embeddings):
                logger.error("Failed to generate embeddings for chunks")
                raise ValueError("Failed to generate embeddings")
            
            # 4. Build knowledge graph incrementally
            status.update(label="🔄 Building knowledge graph incrementally...", state="running")
            graph_data = self.kg_builder.process_content_incrementally(
                page_contents, 
                filename
            )
            
            # 5. Add to vector store
            status.update(label="💾 Adding to vector store...", state="running")
            self.vector_store.add_document(
                document_id=document_id,
                chunks_with_embeddings=chunks_with_embeddings
            )
            
            return document_id
        except Exception as e:
            logger.error(f"Error processing file: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def _show_processing_details(self, document_id: str):
        """Show details about the processed file.
        
        Args:
            document_id: ID of the processed document
        """
        try:
            # Get document chunks
            chunks = self.vector_store.get_document_chunks(document_id)
            
            if not chunks:
                st.warning("No chunks found for this document.")
                return
            
            # Display summary statistics
            st.write(f"**Document ID:** {document_id}")
            st.write(f"**Number of chunks:** {len(chunks)}")
            
            # Display sample chunks in a single expander
            with st.expander("View Sample Chunks"):
                for i, chunk in enumerate(chunks[:3]):
                    st.markdown(f"**Chunk {i+1}**")
                    st.text(chunk["text"][:200] + "...")
                    st.markdown("---")
            
            # Show knowledge graph info
            graph_path = os.path.join("data/knowledge_graph", f"{document_id}_graph.json")
            if os.path.exists(graph_path):
                st.success("✅ Knowledge Graph created successfully")
                st.write("You can view it in the Knowledge Graph tab.")
            else:
                st.warning("Knowledge Graph could not be created for this document.")
        except Exception as e:
            logger.error(f"Error showing processing details: {e}")
            st.error("Error displaying document details")
