"""
Response parser module for formatting responses from the Gemini model.
"""
from typing import Dict, Any, List, Optional, Union
import re
import json


class ResponseParser:
    """Parse and format responses from the Gemini model."""
    
    def __init__(self):
        """Initialize the response parser."""
        pass
    
    def format_response(self, response: str) -> str:
        """Format a response from the Gemini model.
        
        Args:
            response: Response from the Gemini model
            
        Returns:
            Formatted response
        """
        # Clean the response
        response = self._clean_response(response)
        
        # Format any LaTeX equations
        response = self._format_latex(response)
        
        return response
    
    def _clean_response(self, response: str) -> str:
        """Clean a response from the Gemini model.
        
        Args:
            response: Response from the Gemini model
            
        Returns:
            Cleaned response
        """
        # Remove redundant prefixes that the model might generate
        prefixes_to_remove = [
            "Answer:",
            "Solution:",
            "Response:",
            "Here's the answer:",
            "Here's the solution:",
            "Here's the explanation:"
        ]
        
        for prefix in prefixes_to_remove:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()
        
        return response
    
    def _format_latex(self, text: str) -> str:
        """Format LaTeX equations in a text.
        
        Args:
            text: Text with LaTeX equations
            
        Returns:
            Text with formatted LaTeX equations
        """
        # In Streamlit, we will render LaTeX equations with st.latex() or st.markdown()
        # For now, we'll just ensure that the equations are properly formatted for later rendering
        
        # Ensure inline math is properly formatted
        text = re.sub(r'(?<!\$)\$(?!\$)([^$]+?)(?<!\$)\$(?!\$)', r'$\1$', text)
        
        # Ensure display math is properly formatted
        text = re.sub(r'(?<!\$)\$\$([^$]+?)\$\$', r'$$\1$$', text)
        
        return text
    
    def parse_quiz(self, response: str) -> List[Dict[str, Any]]:
        """Parse a quiz response.
        
        Args:
            response: Quiz response from the Gemini model
            
        Returns:
            List of questions
        """
        # Try to extract JSON if the response contains it
        json_match = re.search(r'```json(.*?)```', response, re.DOTALL)
        if json_match:
            try:
                json_str = json_match.group(1).strip()
                quiz_data = json.loads(json_str)
                if isinstance(quiz_data, dict) and "quiz" in quiz_data:
                    return quiz_data["quiz"]
            except json.JSONDecodeError:
                pass
        
        # If JSON extraction fails, parse the text manually
        questions = []
        question_blocks = re.split(r'\n\s*\d+[\.\)]\s+', response)
        
        # Skip the first block if it's empty or contains text before the first question
        if not question_blocks[0].strip() or "Question" not in question_blocks[0]:
            question_blocks = question_blocks[1:]
        
        for i, block in enumerate(question_blocks):
            if not block.strip():
                continue
                
            # Try to determine question type
            is_multiple_choice = "A)" in block or "A." in block or "Option A" in block
            is_true_false = "True or False" in block.lower() or "true/false" in block.lower()
            
            # Extract question text
            question_text = block.split("\n")[0].strip()
            
            # For multiple choice, extract options and correct answer
            options = []
            correct_answer = ""
            explanation = ""
            
            if is_multiple_choice:
                # Extract options
                option_matches = re.finditer(r'([A-D])[\.|\)]([^A-D\.\)]*)(?=\s*[A-D][\.\)]|$)', block, re.DOTALL)
                for match in option_matches:
                    option_letter = match.group(1)
                    option_text = match.group(2).strip()
                    options.append(f"{option_letter}) {option_text}")
                
                # Try to find the correct answer and explanation
                answer_match = re.search(r'(?:Correct Answer|Answer):\s*([A-D])', block, re.IGNORECASE)
                if answer_match:
                    correct_answer = answer_match.group(1)
                
                explanation_match = re.search(r'(?:Explanation|Reason):\s*(.*?)(?=$|\n\s*(?:Question|$))', block, re.DOTALL | re.IGNORECASE)
                if explanation_match:
                    explanation = explanation_match.group(1).strip()
                
                questions.append({
                    "question": question_text,
                    "type": "multiple_choice",
                    "options": options,
                    "correct_answer": correct_answer,
                    "explanation": explanation
                })
            elif is_true_false:
                # Try to find the correct answer and explanation
                answer_match = re.search(r'(?:Correct Answer|Answer):\s*(True|False)', block, re.IGNORECASE)
                if answer_match:
                    correct_answer = answer_match.group(1)
                
                explanation_match = re.search(r'(?:Explanation|Reason):\s*(.*?)(?=$|\n\s*(?:Question|$))', block, re.DOTALL | re.IGNORECASE)
                if explanation_match:
                    explanation = explanation_match.group(1).strip()
                
                questions.append({
                    "question": question_text,
                    "type": "true_false",
                    "correct_answer": correct_answer.lower() == "true",
                    "explanation": explanation
                })
            else:
                # Open-ended question
                explanation_match = re.search(r'(?:Sample Answer|Answer):\s*(.*?)(?=$|\n\s*(?:Explanation|Reason))', block, re.DOTALL | re.IGNORECASE)
                sample_answer = ""
                if explanation_match:
                    sample_answer = explanation_match.group(1).strip()
                
                explanation_match = re.search(r'(?:Explanation|Reason):\s*(.*?)(?=$|\n\s*(?:Question|$))', block, re.DOTALL | re.IGNORECASE)
                if explanation_match:
                    explanation = explanation_match.group(1).strip()
                
                questions.append({
                    "question": question_text,
                    "type": "open_ended",
                    "sample_answer": sample_answer,
                    "explanation": explanation
                })
        
        return questions
