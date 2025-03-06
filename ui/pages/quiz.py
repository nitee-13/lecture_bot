"""
Quiz page for Streamlit UI to generate quizzes based on lecture content.
"""
import streamlit as st
import random
from typing import List, Dict, Any

from src.llm.gemini_client import GeminiClient
from src.llm.response_parser import ResponseParser
from src.vector_store.retriever import Retriever
from src.vector_store.store import VectorStore
from ui.components.math_renderer import MathRenderer


class QuizPage:
    """Quiz page for Streamlit UI."""
    
    def __init__(self, retriever: Retriever, vector_store: VectorStore):
        """Initialize the quiz page.
        
        Args:
            retriever: Retriever instance
            vector_store: Vector store instance
        """
        self.retriever = retriever
        self.vector_store = vector_store
        self.response_parser = ResponseParser()
        self.math_renderer = MathRenderer()
        
        # Initialize quiz in session state if not exists
        if "quiz" not in st.session_state:
            st.session_state.quiz = []
        if "document_for_quiz" not in st.session_state:
            st.session_state.document_for_quiz = None
    
    def render(self):
        """Render the quiz page."""
        st.title("Quiz Yourself")
        
        st.markdown("""
        Generate a quiz based on your lecture materials to test your knowledge.
        Choose the document to generate questions from and set your preferences.
        """)
        
        # Get documents from vector store
        documents = self.vector_store.list_documents()
        
        if not documents:
            st.info("No documents available. Upload lecture materials to generate quizzes.")
            return
        
        # Document selection
        col1, col2 = st.columns([3, 1])
        
        with col1:
            selected_document = st.selectbox(
                "Select a document",
                options=documents,
                format_func=lambda x: x
            )
        
        with col2:
            num_questions = st.number_input(
                "Number of questions",
                min_value=1,
                max_value=10,
                value=5,
                step=1
            )
        
        # Quiz generation button
        if st.button("Generate Quiz"):
            with st.spinner("Generating quiz questions..."):
                self._generate_quiz(selected_document, num_questions)
        
        # Display the quiz
        if st.session_state.quiz:
            self._display_quiz()
    
    def _generate_quiz(self, document_id: str, num_questions: int):
        """Generate a quiz for a document.
        
        Args:
            document_id: Document ID
            num_questions: Number of questions to generate
        """
        # Update the document in session state
        st.session_state.document_for_quiz = document_id
        
        # Get all chunks for the document
        chunks = self.vector_store.get_document_chunks(document_id)
        
        if not chunks:
            st.warning("Could not find content for this document.")
            return
        
        # Extract text from chunks to use as context
        contexts = [chunk.get("text", "") for chunk in chunks]
        
        # If there are many chunks, select a random subset for the quiz
        if len(contexts) > 10:
            contexts = random.sample(contexts, 10)
        
        # Get settings from session state
        settings = st.session_state.get("settings", {})
        model_name = settings.get("model", "gemini-1.5-flash")
        temperature = settings.get("temperature", 0.3)
        
        # Create Gemini client with the selected model
        gemini_client = GeminiClient(model_name=model_name)
        
        # Generate the quiz
        quiz_data = gemini_client.generate_quiz(contexts, num_questions)
        
        # Parse the quiz
        if "quiz" in quiz_data:
            st.session_state.quiz = quiz_data["quiz"]
        else:
            st.error("Failed to generate quiz. Please try again.")
    
    def _display_quiz(self):
        """Display the quiz questions."""
        st.subheader(f"Quiz on {st.session_state.document_for_quiz}")
        
        # Display each question in the quiz
        for i, question in enumerate(st.session_state.quiz):
            # Create a container for the question
            question_container = st.container()
            
            with question_container:
                st.markdown(f"#### Question {i+1}")
                self.math_renderer.render_quiz_question(question)
                
                # Add a small divider between questions
                st.markdown("---")
        
        # Reset button
        if st.button("Generate New Quiz"):
            st.session_state.quiz = []
            st.session_state.document_for_quiz = None
            st.rerun() 