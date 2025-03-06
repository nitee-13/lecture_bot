"""
Sidebar component for Streamlit UI.
"""
import streamlit as st
from pathlib import Path
import os

from src.config import APP_TITLE, APP_EMOJI, APP_DESCRIPTION
from src.vector_store.store import VectorStore


class Sidebar:
    """Sidebar component for Streamlit UI."""
    
    def __init__(self, vector_store: VectorStore):
        """Initialize the sidebar.
        
        Args:
            vector_store: Vector store instance
        """
        self.vector_store = vector_store
    
    def render(self):
        """Render the sidebar."""
        with st.sidebar:
            # App title and description
            st.title(f"{APP_EMOJI} {APP_TITLE}")
            st.markdown(APP_DESCRIPTION)
            
            st.divider()
            
            # Navigation
            st.subheader("Navigation")
            page = st.radio(
                "Go to",
                ["Chat", "Upload", "Knowledge Graph", "Quiz"],
                index=0,
                label_visibility="collapsed"
            )
            
            st.divider()
            
            # Documents section
            self._render_documents_section()
            
            st.divider()
            
            # Settings section
            self._render_settings_section()
            
        return page
    
    def _render_documents_section(self):
        """Render the documents section in the sidebar."""
        st.subheader("Documents")
        
        # Get list of documents
        documents = self.vector_store.list_documents()
        
        if not documents:
            st.info("No documents uploaded yet. Go to the Upload page to add some!")
            return
        
        # Show documents with delete buttons
        st.write(f"You have {len(documents)} documents:")
        for doc in documents:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"- {Path(doc).stem}")
            with col2:
                if st.button("🗑️", key=f"delete_{doc}"):
                    # Delete the document from vector store
                    self.vector_store.remove_document(doc)
                    
                    # Delete the knowledge graph file if it exists
                    graph_path = Path("data/knowledge_graph") / f"{Path(doc).stem}_graph.json"
                    if graph_path.exists():
                        graph_path.unlink()
                    
                    # Delete the PDF file if it exists
                    pdf_path = Path("data/pdfs") / doc
                    if pdf_path.exists():
                        pdf_path.unlink()
                    
                    st.rerun()
        
        # Add clear all button
        if st.button("🗑️ Clear All Documents", type="primary"):
            if st.warning("This will delete all documents and their associated data. Are you sure?"):
                # Clear all documents from vector store
                self.vector_store.clear_all()
                
                # Delete all knowledge graph files
                graph_dir = Path("data/knowledge_graph")
                if graph_dir.exists():
                    for graph_file in graph_dir.glob("*_graph.json"):
                        graph_file.unlink()
                
                # Delete all PDF files
                pdf_dir = Path("data/pdfs")
                if pdf_dir.exists():
                    for pdf_file in pdf_dir.glob("*.pdf"):
                        pdf_file.unlink()
                
                st.rerun()
    
    def _render_settings_section(self):
        """Render the settings section in the sidebar."""
        st.subheader("Settings")
        
        # Model selection
        model = st.selectbox(
            "Model",
            ["gemini-1.5-flash", "gemini-1.5-pro"],
            index=0
        )
        
        # Temperature
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.1
        )
        
        # Retrieval settings
        retrieval_k = st.slider(
            "Number of chunks to retrieve",
            min_value=1,
            max_value=10,
            value=5,
            step=1
        )
        
        # Store settings in session state
        if "settings" not in st.session_state:
            st.session_state.settings = {}
        
        st.session_state.settings["model"] = model
        st.session_state.settings["temperature"] = temperature
        st.session_state.settings["retrieval_k"] = retrieval_k
