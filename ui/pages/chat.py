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
        
        # Define available response types
        self.response_types = [
            "Question Answer",
            "Summary",
            "Explanation",
            "Comparison",
            "Problem Solving"
        ]
        
        # Initialize message history in session state if not exists
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Initialize response type in session state if not exists
        if "response_type" not in st.session_state:
            st.session_state.response_type = "Question Answer"
            
        # Initialize debug mode in session state if not exists
        if "debug_mode" not in st.session_state:
            st.session_state.debug_mode = False
    
    def render(self):
        """Render the chat page."""
        st.title("Chat with Your Lecture Bot")
        
        # Response type selector
        response_type = st.selectbox(
            "Response Type",
            self.response_types,
            index=self.response_types.index(st.session_state.response_type)
        )
        
        # Update response type in session state
        st.session_state.response_type = response_type
        
        # Debug mode toggle
        debug_mode = st.sidebar.checkbox("Debug Mode", value=st.session_state.debug_mode)
        st.session_state.debug_mode = debug_mode
        
        # Display the chat messages
        self._display_chat_messages()
        
        # Chat input with context based on response type
        prompt_placeholder = {
            "Question Answer": "Ask a question about your lectures...",
            "Summary": "Enter a topic to summarize...",
            "Explanation": "Enter a concept to explain...",
            "Comparison": "Enter two concepts to compare (separated by 'vs')...",
            "Problem Solving": "Enter a problem to solve..."
        }
        
        prompt = st.chat_input(prompt_placeholder[response_type])
        
        if prompt:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Generate response
            with st.spinner("Thinking..."):
                response = self._generate_response(prompt, response_type)
            
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
    
    def _generate_response(self, prompt: str, response_type: str) -> str:
        """Generate a response for a prompt.
        
        Args:
            prompt: User prompt
            response_type: Type of response to generate
            
        Returns:
            Generated response
        """
        # Get settings from session state
        settings = st.session_state.get("settings", {})
        model_name = settings.get("model", "gemini-1.5-flash")
        temperature = settings.get("temperature", 0.7)
        retrieval_k = settings.get("retrieval_k", 10)
        debug_mode = st.session_state.get("debug_mode", False)
        
        # Create Gemini client with the selected model
        gemini_client = GeminiClient(model=model_name)
        
        # Retrieve relevant contexts
        contexts = self.retriever.retrieve_contexts(prompt, k=retrieval_k)
        
        # In debug mode, show the retrieved contexts
        if debug_mode and st.sidebar.checkbox("Show Retrieved Contexts", value=True):
            with st.sidebar.expander("Retrieved Contexts", expanded=True):
                if contexts:
                    for i, context in enumerate(contexts):
                        st.markdown(f"**Context {i+1}:**")
                        st.text(context[:500] + "..." if len(context) > 500 else context)
                        st.markdown("---")
                else:
                    st.warning("No contexts retrieved")
        
        if not contexts:
            # No relevant contexts found, inform the user and try to generate a general response
            print("No relevant contexts found for query:", prompt)
            
            # In debug mode, show a warning
            if debug_mode:
                st.sidebar.warning("No relevant contexts found for this query")
            
            # Try to generate a response with a special prompt for no context
            no_context_prompt = f"""You are a helpful teaching assistant. The user has asked about: "{prompt}"
            
Unfortunately, I couldn't find specific information about this in the lecture materials. 
Please provide a general response based on your knowledge, but make it clear that this is general information 
and not specifically from the lecture materials. If this is a very specific concept that should be in the 
lecture materials, suggest that the user upload the relevant lecture slides if they haven't already done so.

Response:"""
            
            response = gemini_client.generate_response(
                prompt=no_context_prompt,
                temperature=temperature
            )
        else:
            # Build prompt with contexts based on response type
            if response_type == "Question Answer":
                contextualized_prompt = self.prompt_builder.build_qa_prompt(prompt, contexts)
            elif response_type == "Summary":
                contextualized_prompt = self.prompt_builder.build_summary_prompt(contexts)
            elif response_type == "Explanation":
                contextualized_prompt = self.prompt_builder.build_explain_prompt(prompt, contexts)
            elif response_type == "Comparison":
                # Split prompt into two concepts
                concepts = [c.strip() for c in prompt.split("vs")]
                if len(concepts) != 2:
                    return "Please provide two concepts to compare, separated by 'vs'"
                contextualized_prompt = self.prompt_builder.build_compare_prompt(concepts[0], concepts[1], contexts)
            else:  # Problem Solving
                contextualized_prompt = self.prompt_builder.build_solve_prompt(prompt, contexts)
            
            # In debug mode, show the contextualized prompt
            if debug_mode and st.sidebar.checkbox("Show Prompt", value=False):
                with st.sidebar.expander("Contextualized Prompt", expanded=True):
                    st.text(contextualized_prompt)
            
            # Generate response with higher temperature for more creative responses
            response = gemini_client.generate_response(
                prompt=contextualized_prompt,
                temperature=temperature
            )
        
        # Parse and format the response
        formatted_response = self.response_parser.format_response(response)
        
        return formatted_response
