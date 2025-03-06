"""
Upload page for Streamlit UI to upload and process PDF files.
"""
import streamlit as st
import os
from pathlib import Path
import uuid
import time
import logging

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
                            
                            # Show details about the processed file
                            self._show_processing_details(document_id)
                        else:
                            status.update(label="❌ Processing failed", state="error")
                            st.error("Failed to process the file. Please try again.")
                except Exception as e:
                    logger.error(f"Error processing file: {e}")
                    st.error(f"Error processing file: {str(e)}")
    
    def _save_uploaded_file(self, uploaded_file) -> str:
        """Save the uploaded file to disk.
        
        Args:
            uploaded_file: Uploaded file from Streamlit
            
        Returns:
            Path to the saved file
        """
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
    
    def _process_file(self, file_path: str, status) -> str:
        """Process the uploaded file.
        
        Args:
            file_path: Path to the file to process
            status: Streamlit status object for updates
            
        Returns:
            Document ID if successful, None otherwise
        """
        try:
            # 1. Extract text from PDF
            status.update(label="🔍 Extracting text from PDF...", state="running")
            extracted_data = self.pdf_extractor.process_pdf(file_path)
            
            # 2. Create chunks from the extracted text
            status.update(label="✂️ Chunking text...", state="running")
            chunks = self.chunker.chunk_lecture_content(extracted_data)
            
            # 3. Generate embeddings for chunks
            status.update(label="🧠 Generating embeddings...", state="running")
            chunks_with_embeddings = self.embedder.generate_embeddings(chunks)
            
            # 4. Build knowledge graph
            status.update(label="🔄 Building knowledge graph...", state="running")
            document_id = Path(file_path).stem
            filename = Path(file_path).name
            graph_data = self.kg_builder.process_content(
                extracted_data, 
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
            raise
    
    def _show_processing_details(self, document_id: str):
        """Show details about the processed file.
        
        Args:
            document_id: ID of the processed document
        """
        # Get document chunks
        chunks = self.vector_store.get_document_chunks(document_id)
        
        if not chunks:
            st.warning("No chunks found for this document.")
            return
        
        # Display summary statistics
        st.write(f"**Document ID:** {document_id}")
        st.write(f"**Number of chunks:** {len(chunks)}")
        
        # Display sample chunks
        with st.expander("View Sample Chunks", expanded=False):
            for i, chunk in enumerate(chunks[:3]):
                st.markdown(f"**Chunk {i+1}**")
                st.text(chunk["text"][:200] + "...")
                st.markdown("---")
        
        # Show knowledge graph info
        graph_path = os.path.join("data/knowledge_graph", f"{document_id}_graph.json")
        if os.path.exists(graph_path):
            st.write("**Knowledge Graph:** Created successfully")
            st.write("You can view it in the Knowledge Graph tab.")
        else:
            st.warning("Knowledge Graph could not be created for this document.")
