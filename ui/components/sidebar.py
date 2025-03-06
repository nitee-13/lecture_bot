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
        
        # Show documents
        st.write(f"You have {len(documents)} documents:")
        for doc in documents:
            st.markdown(f"- {Path(doc).stem}")
    
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
