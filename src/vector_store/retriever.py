"""
Retriever module for retrieving relevant chunks from the vector store.
"""
from typing import List, Dict, Any
import numpy as np
import networkx as nx
from pathlib import Path
import json
import re
from difflib import SequenceMatcher

from src.vector_store.store import VectorStore
from src.embedding.embedder import Embedder
from src.config import TOP_K_RESULTS, KNOWLEDGE_GRAPH_PATH


class Retriever:
    """Retrieve relevant chunks from the vector store."""
    
    def __init__(self, vector_store: VectorStore, embedder: Embedder):
        """Initialize the retriever.
        
        Args:
            vector_store: Vector store to retrieve from
            embedder: Embedder to use for query embedding
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.similarity_threshold = 0.8  # Threshold for fuzzy matching
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for better matching.
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text
        """
        # Convert to lowercase
        text = text.lower()
        # Replace special characters with spaces
        text = re.sub(r'[_-]', ' ', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def _similarity_ratio(self, s1: str, s2: str) -> float:
        """Calculate similarity ratio between two strings.
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            Similarity ratio between 0 and 1
        """
        return SequenceMatcher(None, s1, s2).ratio()
    
    def retrieve(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks using text-based similarity.
        
        Args:
            query: Query to retrieve for
            k: Number of results to retrieve
            
        Returns:
            List of relevant chunks
        """
        # Generate embedding for the query
        query_embedding = self.embedder.generate_embedding(query)
        
        # Search the vector store
        results = self.vector_store.search(query_embedding, k=k)
        
        # Sort by distance (ascending)
        results.sort(key=lambda x: x.get("distance", float("inf")))
        
        return results
    
    def retrieve_contexts(self, query: str, k: int = 10) -> List[str]:
        """Retrieve relevant contexts for a query using both text and graph-based retrieval.
        
        Args:
            query: Query to retrieve for
            k: Number of results to retrieve
            
        Returns:
            List of relevant context strings
        """
        # Get text-based retrieval results
        text_results = self.retrieve(query, k=k)
        
        # Extract concepts from the query
        query_concepts = self._extract_concepts_from_query(query)
        
        # Get graph-based results
        graph_results = self._retrieve_from_graph(query_concepts, k=k)
        
        # Combine and deduplicate results
        all_results = text_results + graph_results
        unique_results = self._deduplicate_results(all_results)
        
        # Extract and format contexts
        contexts = []
        for result in unique_results[:k]:
            context = result.get("text", "")
            
            # Add slide information
            slide_number = result.get("slide_number", "")
            slide_title = result.get("slide_title", "")
            if slide_number and slide_title:
                context = f"[Slide {slide_number}: {slide_title}]\n{context}"
            
            # Add equations if available
            equations = result.get("equations", [])
            if equations:
                equation_text = "\nEquations:\n" + "\n".join(equations)
                context += equation_text
            
            # Add related concepts if available
            related_concepts = result.get("related_concepts", [])
            if related_concepts:
                concepts_text = "\nRelated Concepts:\n" + ", ".join(related_concepts)
                context += concepts_text
            
            contexts.append(context)
        
        return contexts
    
    def _extract_concepts_from_query(self, query: str) -> List[str]:
        """Extract potential concepts from the query.
        
        Args:
            query: Query to extract concepts from
            
        Returns:
            List of extracted concepts
        """
        concepts = []
        
        # Split query into words and handle special characters
        words = re.findall(r'[\w-]+', query)
        
        # Look for technical terms (words with underscores, hyphens, or camelCase)
        for i, word in enumerate(words):
            # Handle underscore-separated terms
            if '_' in word:
                concepts.append(word)
                continue
                
            # Handle hyphen-separated terms
            if '-' in word:
                concepts.append(word)
                continue
                
            # Handle camelCase terms
            if re.search(r'[a-z][A-Z]', word):
                concepts.append(word)
                continue
            
            # Handle capitalized words
            if word[0].isupper() if word else False:
                concept = word
                # Try to extend to phrases
                j = i + 1
                while j < len(words) and words[j][0].isupper():
                    concept += " " + words[j]
                    j += 1
                concepts.append(concept)
                continue
            
            # Handle technical terms that might be in lowercase
            if len(word) > 3 and word.islower():
                concepts.append(word)
        
        # Normalize concepts
        normalized_concepts = []
        for concept in concepts:
            # Normalize the concept
            normalized = self._normalize_text(concept)
            normalized_concepts.append(normalized)
        
        return normalized_concepts
    
    def _retrieve_from_graph(self, concepts: List[str], k: int) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks using the knowledge graph.
        
        Args:
            concepts: List of concepts to search for
            k: Number of results to retrieve
            
        Returns:
            List of relevant chunks
        """
        results = []
        
        # Get all documents
        documents = self.vector_store.list_documents()
        if not documents:
            return results
        
        # Create a NetworkX graph for all documents
        G = nx.Graph()
        
        # Load and combine graphs from each document
        for doc_id in documents:
            graph_path = Path(KNOWLEDGE_GRAPH_PATH) / f"{doc_id}_graph.json"
            if graph_path.exists():
                try:
                    with open(graph_path, "r") as file:
                        graph_data = json.load(file)
                        if "nodes" in graph_data:
                            # Add nodes
                            for node in graph_data["nodes"]:
                                G.add_node(node["id"], **node)
                        if "edges" in graph_data:
                            # Add edges
                            for edge in graph_data["edges"]:
                                G.add_edge(edge["from"], edge["to"], **edge)
                except Exception as e:
                    print(f"Error loading graph for {doc_id}: {e}")
        
        # Find relevant nodes and their neighbors
        relevant_nodes = set()
        for concept in concepts:
            # Find nodes containing the concept
            for node in G.nodes(data=True):
                node_label = node[1].get("label", "").lower()
                node_id = node[0].lower()
                
                # Normalize node label and ID for comparison
                normalized_label = self._normalize_text(node_label)
                normalized_id = self._normalize_text(node_id)
                
                # Check for exact matches
                if (concept in normalized_label or 
                    concept in normalized_id or 
                    concept.replace(' ', '-') in node_label or
                    concept.replace(' ', '_') in node_id):
                    relevant_nodes.add(node[0])
                    relevant_nodes.update(G.neighbors(node[0]))
                    continue
                
                # Check for fuzzy matches
                if (self._similarity_ratio(concept, normalized_label) >= self.similarity_threshold or
                    self._similarity_ratio(concept, normalized_id) >= self.similarity_threshold):
                    relevant_nodes.add(node[0])
                    relevant_nodes.update(G.neighbors(node[0]))
        
        # Get chunks associated with relevant nodes
        for node_id in relevant_nodes:
            node_data = G.nodes[node_id]
            if "chunk_id" in node_data:
                chunk = self.vector_store.get_chunk(node_data["chunk_id"])
                if chunk:
                    results.append(chunk)
        
        return results
    
    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate results based on chunk IDs.
        
        Args:
            results: List of results to deduplicate
            
        Returns:
            Deduplicated list of results
        """
        seen_ids = set()
        unique_results = []
        
        for result in results:
            chunk_id = result.get("id")
            if chunk_id and chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                unique_results.append(result)
        
        return unique_results