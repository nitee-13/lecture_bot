"""
Math renderer component for rendering mathematical equations in Streamlit.
"""
import re
import streamlit as st


class MathRenderer:
    """Render mathematical equations in Streamlit."""
    
    def __init__(self):
        """Initialize the math renderer."""
        pass
    
    def render_math(self, text: str):
        """Render text with math equations.
        
        Args:
            text: Text with math equations
        """
        # Split the text into parts with and without equations
        parts = self._split_by_equations(text)
        
        # Render each part
        for part in parts:
            if part.startswith("$$") and part.endswith("$$"):
                # Display math
                equation = part[2:-2].strip()
                st.latex(equation)
            elif "$" in part:
                # Inline math mixed with text
                self._render_mixed_text(part)
            else:
                # Regular text
                st.write(part)
    
    def _split_by_equations(self, text: str) -> list:
        """Split text by display equations.
        
        Args:
            text: Text to split
            
        Returns:
            List of text parts
        """
        # Split by display equations ($$...$$)
        pattern = r'(\$\$[^$]+\$\$)'
        parts = re.split(pattern, text)
        
        # Filter out empty parts
        return [part for part in parts if part.strip()]
    
    def _render_mixed_text(self, text: str):
        """Render text with inline math equations.
        
        Args:
            text: Text with inline math equations
        """
        # Use st.markdown with raw LaTeX
        # Streamlit supports LaTeX in markdown with $...$ syntax
        st.markdown(text)
    
    def render_quiz_question(self, question: dict):
        """Render a quiz question with math equations.
        
        Args:
            question: Quiz question dictionary
        """
        # Render the question text
        st.markdown("### " + question["question"])
        
        # Render options for multiple choice questions
        if question["type"] == "multiple_choice":
            for option in question.get("options", []):
                st.markdown(option)
        
        # Create an expander for the answer
        with st.expander("Show Answer"):
            if question["type"] == "multiple_choice":
                st.markdown(f"**Correct Answer:** {question.get('correct_answer', '')}")
            elif question["type"] == "true_false":
                st.markdown(f"**Correct Answer:** {'True' if question.get('correct_answer') else 'False'}")
            elif question["type"] == "open_ended":
                st.markdown("**Sample Answer:**")
                self.render_math(question.get("sample_answer", ""))
            
            # Render explanation
            if "explanation" in question and question["explanation"]:
                st.markdown("**Explanation:**")
                self.render_math(question["explanation"])
    
    def render_equation_block(self, equation: str):
        """Render a block of LaTeX equation.
        
        Args:
            equation: LaTeX equation string
        """
        # Remove any surrounding $$ if present
        if equation.startswith("$$") and equation.endswith("$$"):
            equation = equation[2:-2]
        elif equation.startswith("$") and equation.endswith("$"):
            equation = equation[1:-1]
            
        # Render the equation
        st.latex(equation) 