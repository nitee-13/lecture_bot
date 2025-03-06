"""
Prompt builder module for constructing prompts for the Gemini model.
"""
from typing import List, Dict, Any, Optional


class PromptBuilder:
    """Build prompts for the Gemini model."""
    
    def __init__(self):
        """Initialize the prompt builder."""
        pass
    
    def build_qa_prompt(self, query: str, contexts: List[str]) -> str:
        """Build a prompt for question answering.
        
        Args:
            query: User query
            contexts: List of context strings from retrieval
            
        Returns:
            Formatted prompt string
        """
        # Format contexts
        formatted_contexts = "\n\n".join([f"Context {i+1}:\n{ctx}" for i, ctx in enumerate(contexts)])
        
        # Build the prompt
        prompt = f"""You are a knowledgeable teaching assistant helping a student understand their lecture material. 
Please provide a detailed and comprehensive answer to the following question based on the provided contexts from lecture slides.

Your response should:
1. Directly answer the question using information from the provided contexts
2. Include relevant examples and analogies to help explain the concepts
3. Connect related concepts and show relationships between ideas
4. If equations are involved, explain them step by step
5. If the answer involves multiple steps or concepts, break it down clearly
6. If you need to make assumptions or use general knowledge, clearly state that
7. If you cannot find the answer in the contexts, acknowledge that and provide a general response based on your knowledge

Contexts:
{formatted_contexts}

Question: {query}

Answer:"""
        
        return prompt
    
    def build_quiz_prompt(self, contexts: List[str], num_questions: int = 5) -> str:
        """Build a prompt for quiz generation.
        
        Args:
            contexts: List of context strings from retrieval
            num_questions: Number of questions to generate
            
        Returns:
            Formatted prompt string
        """
        # Format contexts
        formatted_contexts = "\n\n".join([f"Context {i+1}:\n{ctx}" for i, ctx in enumerate(contexts)])
        
        # Build the prompt
        prompt = f"""Please generate a quiz with {num_questions} questions based on the provided contexts from lecture slides.
Include a mix of question types: multiple choice, true/false, and open-ended questions.
For mathematical or technical topics, include proper equation formatting.
For each question, provide the correct answer and an explanation.

Contexts:
{formatted_contexts}

Quiz:"""
        
        return prompt
    
    def build_summary_prompt(self, contexts: List[str]) -> str:
        """Build a prompt for generating a summary.
        
        Args:
            contexts: List of context strings from retrieval
            
        Returns:
            Formatted prompt string
        """
        # Format contexts
        formatted_contexts = "\n\n".join([f"Context {i+1}:\n{ctx}" for i, ctx in enumerate(contexts)])
        
        # Build the prompt
        prompt = f"""Please provide a comprehensive summary of the following lecture material.
Highlight key concepts, equations, and relationships between ideas.
Format any mathematical equations properly.

Contexts:
{formatted_contexts}

Summary:"""
        
        return prompt
    
    def build_explain_prompt(self, concept: str, contexts: List[str]) -> str:
        """Build a prompt for explaining a concept.
        
        Args:
            concept: Concept to explain
            contexts: List of context strings from retrieval
            
        Returns:
            Formatted prompt string
        """
        # Format contexts
        formatted_contexts = "\n\n".join([f"Context {i+1}:\n{ctx}" for i, ctx in enumerate(contexts)])
        
        # Build the prompt
        prompt = f"""Please explain the concept of "{concept}" based on the provided lecture materials.
Provide a clear explanation with examples if available.
Format any mathematical equations properly.

Contexts:
{formatted_contexts}

Explanation of {concept}:"""
        
        return prompt
    
    def build_compare_prompt(self, concept1: str, concept2: str, contexts: List[str]) -> str:
        """Build a prompt for comparing two concepts.
        
        Args:
            concept1: First concept to compare
            concept2: Second concept to compare
            contexts: List of context strings from retrieval
            
        Returns:
            Formatted prompt string
        """
        # Format contexts
        formatted_contexts = "\n\n".join([f"Context {i+1}:\n{ctx}" for i, ctx in enumerate(contexts)])
        
        # Build the prompt
        prompt = f"""Please compare and contrast the concepts of "{concept1}" and "{concept2}" based on the provided lecture materials.
Explain their similarities, differences, and relationships.
Format any mathematical equations properly.

Contexts:
{formatted_contexts}

Comparison of {concept1} and {concept2}:"""
        
        return prompt
    
    def build_solve_prompt(self, problem: str, contexts: List[str]) -> str:
        """Build a prompt for solving a problem.
        
        Args:
            problem: Problem to solve
            contexts: List of context strings from retrieval
            
        Returns:
            Formatted prompt string
        """
        # Format contexts
        formatted_contexts = "\n\n".join([f"Context {i+1}:\n{ctx}" for i, ctx in enumerate(contexts)])
        
        # Build the prompt
        prompt = f"""Please solve the following problem using the concepts and equations from the provided lecture materials.
Show your work step by step, explaining the reasoning at each step.
Format mathematical equations properly.

Problem: {problem}

Contexts:
{formatted_contexts}

Solution:"""
        
        return prompt
