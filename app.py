"""
Lecture Bot - A personalized chatbot for lecture slides with RAG capabilities.
"""
import streamlit as st
from pathlib import Path
import os
import sys

# Fix for semaphore leak issue on macOS
if sys.platform == 'darwin':
    os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
    # Limit the number of threads used by numpy and other libraries
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'

# Import components
from ui.components.sidebar import Sidebar
from ui.pages.chat import ChatPage
from ui.pages.upload import UploadPage
from ui.pages.knowledge_graph import KnowledgeGraphPage
from ui.pages.quiz import QuizPage

# Import core modules
from src.vector_store.store import VectorStore
from src.vector_store.retriever import Retriever
from src.embedding.embedder import Embedder
from src.config import APP_TITLE

# Set page configuration
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main application function."""
    # Initialize vector store
    vector_store = VectorStore()
    
    # Initialize embedder
    embedder = Embedder()
    
    # Initialize retriever
    retriever = Retriever(vector_store, embedder)
    
    # Initialize UI components
    sidebar = Sidebar(vector_store)
    chat_page = ChatPage(retriever)
    upload_page = UploadPage(vector_store)
    knowledge_graph_page = KnowledgeGraphPage(vector_store)
    quiz_page = QuizPage(retriever, vector_store)
    
    # Render sidebar and get selected page
    selected_page = sidebar.render()
    
    # Render the selected page
    if selected_page == "Chat":
        chat_page.render()
    elif selected_page == "Upload":
        upload_page.render()
    elif selected_page == "Knowledge Graph":
        knowledge_graph_page.render()
    elif selected_page == "Quiz":
        quiz_page.render()

# Add custom CSS for better styling
def add_custom_css():
    """Add custom CSS to the app."""
    custom_css = """
    <style>
    .block-container {
        padding-top: 2rem;
    }
    .stApp a {
        color: #4287f5;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 5px;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

if __name__ == "__main__":
    add_custom_css()
    main()
