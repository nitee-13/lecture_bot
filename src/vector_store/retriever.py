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
        self.similarity_threshold = 0.6  # Lower threshold for fuzzy matching (was 0.8)
    
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
        # Special case for listing all concepts
        if "list" in query.lower() and "concept" in query.lower():
            print("Special case: listing all concepts")
            return self._list_all_concepts()
            
        # Get text-based retrieval results
        text_results = self.retrieve(query, k=k)
        
        # Extract concepts from the query
        query_concepts = self._extract_concepts_from_query(query)
        
        # Log for debugging
        print(f"Extracted concepts from query: {query_concepts}")
        
        # Get graph-based results
        graph_results = self._retrieve_from_graph(query_concepts, k=k)
        
        # Combine and deduplicate results
        all_results = text_results + graph_results
        unique_results = self._deduplicate_results(all_results)
        
        # If no results found, try a more aggressive approach with the full query
        if not unique_results:
            print("No results found with standard approach, trying with full query")
            # Try using the full query as a concept
            full_query_concepts = [self._normalize_text(query)]
            graph_results = self._retrieve_from_graph(full_query_concepts, k=k)
            all_results = text_results + graph_results
            unique_results = self._deduplicate_results(all_results)
            
            # If still no results, try to get all documents
            if not unique_results:
                print("Still no results, retrieving all documents")
                documents = self.vector_store.list_documents()
                for doc_id in documents:
                    doc_chunks = self.vector_store.get_document_chunks(doc_id)
                    all_results.extend(doc_chunks[:5])  # Get first 5 chunks from each document
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
        
        # Log for debugging
        print(f"Retrieved {len(contexts)} contexts")
        
        return contexts
    
    def _extract_concepts_from_query(self, query: str) -> List[str]:
        """Extract potential concepts from the query.
        
        Args:
            query: Query to extract concepts from
            
        Returns:
            List of extracted concepts
        """
        concepts = []
        
        # Add the full query as a concept (normalized)
        full_query = self._normalize_text(query)
        concepts.append(full_query)
        
        # Split query into words and handle special characters
        words = re.findall(r'[\w-]+', query)
        
        # Extract noun phrases (simple approach)
        for i in range(len(words)):
            # Single words (might be concepts)
            if len(words[i]) > 2:  # Consider words with at least 3 characters
                concepts.append(words[i])
            
            # Two-word phrases
            if i < len(words) - 1 and len(words[i]) > 2 and len(words[i+1]) > 2:
                concepts.append(f"{words[i]} {words[i+1]}")
            
            # Three-word phrases
            if i < len(words) - 2 and len(words[i]) > 2 and len(words[i+1]) > 2 and len(words[i+2]) > 2:
                concepts.append(f"{words[i]} {words[i+1]} {words[i+2]}")
        
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
            if normalized and normalized not in normalized_concepts:
                normalized_concepts.append(normalized)
        
        return normalized_concepts
    
    def _list_all_concepts(self) -> List[str]:
        """List all concepts from the knowledge graph.
        
        Returns:
            List of contexts containing all concepts
        """
        # Get all documents
        documents = self.vector_store.list_documents()
        if not documents:
            return ["No documents found in the vector store."]
            
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
        
        if not G.nodes:
            return ["No concepts found in the knowledge graph."]
            
        # Extract all concept nodes
        concept_nodes = []
        for node_id, node_data in G.nodes(data=True):
            if node_data.get("type") == "concept" or "concept" in node_id.lower():
                concept_nodes.append(node_data)
                
        # If no concept nodes found, try to get all nodes
        if not concept_nodes:
            concept_nodes = [data for _, data in G.nodes(data=True)]
            
        # Format the concepts
        concepts_text = "Here are all the concepts from the lecture slides:\n\n"
        
        # Group concepts by type if available
        concept_groups = {}
        
        for node in concept_nodes:
            node_type = node.get("type", "concept")
            if node_type not in concept_groups:
                concept_groups[node_type] = []
            
            label = node.get("label", node.get("id", "Unknown"))
            description = node.get("description", "")
            
            if description:
                concept_entry = f"- {label}: {description}"
            else:
                concept_entry = f"- {label}"
                
            concept_groups[node_type].append(concept_entry)
            
        # Format the grouped concepts
        for group_type, entries in concept_groups.items():
            concepts_text += f"\n{group_type.capitalize()}s:\n"
            concepts_text += "\n".join(entries)
            concepts_text += "\n"
            
        return [concepts_text]
    
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
            print("No documents found in vector store")
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
        
        if not G.nodes:
            print("No nodes found in knowledge graph")
            return results
            
        print(f"Loaded graph with {len(G.nodes)} nodes and {len(G.edges)} edges")
        
        # Print some sample nodes for debugging
        print("Sample nodes:")
        node_count = 0
        for node_id, node_data in list(G.nodes(data=True))[:5]:
            print(f"  Node ID: {node_id}")
            print(f"  Node data: {node_data}")
            node_count += 1
            if node_count >= 5:
                break
        
        # Find relevant nodes and their neighbors
        relevant_nodes = set()
        for concept in concepts:
            print(f"Looking for concept: '{concept}'")
            # Find nodes containing the concept
            for node in G.nodes(data=True):
                node_id = node[0]
                node_data = node[1]
                
                # Get node label and other text fields that might contain the concept
                node_label = node_data.get("label", "").lower()
                node_description = node_data.get("description", "").lower()
                node_text = node_data.get("text", "").lower()
                
                # Normalize for comparison
                normalized_id = self._normalize_text(node_id.lower())
                normalized_label = self._normalize_text(node_label)
                normalized_description = self._normalize_text(node_description)
                normalized_text = self._normalize_text(node_text)
                
                # Check for exact substring matches
                if (concept in normalized_id or 
                    concept in normalized_label or 
                    concept in normalized_description or
                    concept in normalized_text or
                    concept.replace(' ', '-') in node_id.lower() or
                    concept.replace(' ', '_') in node_id.lower()):
                    print(f"  Found exact match in node: {node_id}")
                    relevant_nodes.add(node_id)
                    # Add neighbors
                    for neighbor in G.neighbors(node_id):
                        relevant_nodes.add(neighbor)
                    continue
                
                # Check for fuzzy matches with node label
                if normalized_label and self._similarity_ratio(concept, normalized_label) >= self.similarity_threshold:
                    print(f"  Found fuzzy match in node label: {node_id} - {normalized_label}")
                    relevant_nodes.add(node_id)
                    # Add neighbors
                    for neighbor in G.neighbors(node_id):
                        relevant_nodes.add(neighbor)
                    continue
                    
                # Check for fuzzy matches with node description
                if normalized_description and self._similarity_ratio(concept, normalized_description) >= self.similarity_threshold:
                    print(f"  Found fuzzy match in node description: {node_id}")
                    relevant_nodes.add(node_id)
                    # Add neighbors
                    for neighbor in G.neighbors(node_id):
                        relevant_nodes.add(neighbor)
                    continue
        
        print(f"Found {len(relevant_nodes)} relevant nodes in the graph")
        
        # If no relevant nodes found but we have a graph, try to get some general nodes
        if not relevant_nodes and G.nodes:
            print("No specific nodes found, getting general nodes")
            # Get nodes with highest degree (most connected)
            node_degrees = dict(G.degree())
            sorted_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)
            for node_id, degree in sorted_nodes[:10]:  # Get top 10 most connected nodes
                relevant_nodes.add(node_id)
        
        # Get chunks associated with relevant nodes
        for node_id in relevant_nodes:
            node_data = G.nodes[node_id]
            print(f"Processing node: {node_id}")
            
            # Try to get chunk by chunk_id
            if "chunk_id" in node_data:
                chunk_id = node_data["chunk_id"]
                print(f"  Looking for chunk with ID: {chunk_id}")
                chunk = self.vector_store.get_chunk(chunk_id)
                if chunk:
                    print(f"  Found chunk with ID: {chunk_id}")
                    results.append(chunk)
                else:
                    print(f"  No chunk found with ID: {chunk_id}")
            
            # If no chunk_id or chunk not found, try to find chunks by slide number
            elif "slide_number" in node_data:
                slide_number = node_data["slide_number"]
                print(f"  Looking for chunks with slide number: {slide_number}")
                for doc_id in documents:
                    doc_chunks = self.vector_store.get_document_chunks(doc_id)
                    for chunk in doc_chunks:
                        if chunk.get("slide_number") == slide_number:
                            print(f"  Found chunk for slide: {slide_number}")
                            results.append(chunk)
            
            # If no slide_number, try to find chunks by concept name/label
            elif "label" in node_data:
                concept_name = node_data["label"]
                print(f"  Looking for chunks containing concept: {concept_name}")
                for doc_id in documents:
                    doc_chunks = self.vector_store.get_document_chunks(doc_id)
                    for chunk in doc_chunks:
                        chunk_text = chunk.get("text", "").lower()
                        chunk_concepts = [c.lower() for c in chunk.get("concepts", [])]
                        
                        if concept_name.lower() in chunk_text or concept_name.lower() in chunk_concepts:
                            print(f"  Found chunk containing concept: {concept_name}")
                            results.append(chunk)
            
            # If nothing else, create a synthetic chunk from the node data
            else:
                print(f"  Creating synthetic chunk from node data")
                # Create a synthetic chunk from the node data
                synthetic_chunk = {
                    "id": f"synthetic_{node_id}",
                    "text": f"Concept: {node_data.get('label', node_id)}\n"
                }
                
                if "description" in node_data:
                    synthetic_chunk["text"] += f"Description: {node_data['description']}\n"
                    
                if "type" in node_data:
                    synthetic_chunk["text"] += f"Type: {node_data['type']}\n"
                    
                # Add related nodes
                related_nodes = []
                for neighbor in G.neighbors(node_id):
                    neighbor_data = G.nodes[neighbor]
                    neighbor_label = neighbor_data.get("label", neighbor)
                    related_nodes.append(neighbor_label)
                    
                if related_nodes:
                    synthetic_chunk["text"] += f"Related concepts: {', '.join(related_nodes)}\n"
                    synthetic_chunk["related_concepts"] = related_nodes
                    
                results.append(synthetic_chunk)
        
        print(f"Retrieved {len(results)} chunks from graph-based retrieval")
        
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