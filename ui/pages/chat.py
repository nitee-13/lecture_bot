"""
Chat interface page for Streamlit UI.
"""
import streamlit as st
from typing import List, Dict, Any

from src.llm.gemini_client import GeminiClient
from src.llm.prompt_builder import PromptBuilder
from src.llm.response_parser import ResponseParser
from src.vector_store.retriever import Retriever
from ui.components.math_renderer import MathRenderer


class ChatPage:
    """Chat interface page for Streamlit UI."""
    
    def __init__(self, retriever: Retriever):
        """Initialize the chat page.
        
        Args:
            retriever: Retriever instance
        """
        self.retriever = retriever
        self.prompt_builder = PromptBuilder()
        self.response_parser = ResponseParser()
        self.math_renderer = MathRenderer()
        
        # Initialize message history in session state if not exists
        if "messages" not in st.session_state:
            st.session_state.messages = []
    
    def render(self):
        """Render the chat page."""
        st.title("Chat with Your Lecture Bot")
        
        # Display the chat messages
        self._display_chat_messages()
        
        # Chat input
        prompt = st.chat_input("Ask a question about your lectures...")
        
        if prompt:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Generate response
            with st.spinner("Thinking..."):
                response = self._generate_response(prompt)
            
            # Add assistant message to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Force a rerun to display the new messages
            st.rerun()
    
    def _display_chat_messages(self):
        """Display the chat messages."""
        # Display chat messages from history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message["role"] == "assistant":
                    # Render assistant messages with math support
                    self.math_renderer.render_math(message["content"])
                else:
                    # Render user messages normally
                    st.write(message["content"])
    
    def _generate_response(self, prompt: str) -> str:
        """Generate a response for a prompt.
        
        Args:
            prompt: User prompt
            
        Returns:
            Generated response
        """
        # Get settings from session state
        settings = st.session_state.get("settings", {})
        model_name = settings.get("model", "gemini-1.5-flash")
        temperature = settings.get("temperature", 0.2)
        retrieval_k = settings.get("retrieval_k", 5)
        
        # Create Gemini client with the selected model
        gemini_client = GeminiClient(model_name=model_name)
        
        # Retrieve relevant contexts
        contexts = self.retriever.retrieve_contexts(prompt, k=retrieval_k)
        
        if not contexts:
            # No relevant contexts found, generate response without context
            response = gemini_client.generate_response(
                prompt=prompt,
                temperature=temperature
            )
        else:
            # Build prompt with contexts
            contextualized_prompt = self.prompt_builder.build_qa_prompt(prompt, contexts)
            
            # Generate response
            response = gemini_client.generate_response(
                prompt=contextualized_prompt,
                temperature=temperature
            )
        
        # Parse and format the response
        formatted_response = self.response_parser.format_response(response)
        
        return formatted_response
