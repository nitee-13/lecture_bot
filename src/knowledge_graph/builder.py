"""
Knowledge graph builder module for constructing knowledge graphs from lecture content.
"""
from typing import Dict, Any, List, Optional
import json
import os
from pathlib import Path
import networkx as nx

from src.llm.gemini_client import GeminiClient
from src.config import KNOWLEDGE_GRAPH_PATH


class KnowledgeGraphBuilder:
    """Build knowledge graphs from lecture content."""
    
    def __init__(self, use_gemini: bool = True):
        """Initialize the knowledge graph builder.
        
        Args:
            use_gemini: Whether to use Gemini for graph construction
        """
        self.use_gemini = use_gemini
        if use_gemini:
            self.gemini_client = GeminiClient()
    
    def build_graph(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Build a knowledge graph from lecture content.
        
        Args:
            content: Lecture content dictionary
            
        Returns:
            Knowledge graph data as a dictionary
        """
        if self.use_gemini:
            # Use Gemini to build the graph
            return self.gemini_client.build_knowledge_graph(content)
        
        # Fallback to basic graph construction
        return self._build_basic_graph(content)
    
    def _build_basic_graph(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Build a basic knowledge graph from lecture content.
        
        Args:
            content: Lecture content dictionary
            
        Returns:
            Knowledge graph data as a dictionary
        """
        graph_data = {"nodes": [], "edges": []}
        
        # Check if the content has the expected structure
        if "slide_content" not in content:
            return graph_data
        
        # Extract concepts and relationships from slides
        concepts = set()
        relationships = []
        
        for slide in content["slide_content"]:
            slide_concepts = slide.get("concepts", [])
            slide_relationships = slide.get("relationships", [])
            
            # Add concepts to the set
            for concept in slide_concepts:
                concepts.add(concept)
            
            # Add relationships to the list
            relationships.extend(slide_relationships)
            
            # Extract equations as concepts
            for equation in slide.get("equations", []):
                concepts.add(f"Equation: {equation}")
                
                # Create relationships between equations and concepts
                for concept in slide_concepts:
                    relationships.append({
                        "from": concept,
                        "to": f"Equation: {equation}",
                        "type": "related to"
                    })
        
        # Create nodes for all concepts
        for i, concept in enumerate(concepts):
            node = {
                "id": f"concept_{i}",
                "label": concept,
                "type": "concept" if not concept.startswith("Equation:") else "equation"
            }
            graph_data["nodes"].append(node)
        
        # Create edges for all relationships
        concept_to_id = {node["label"]: node["id"] for node in graph_data["nodes"]}
        
        for i, rel in enumerate(relationships):
            source = rel.get("from", "")
            target = rel.get("to", "")
            rel_type = rel.get("type", "related to")
            
            # Only add the edge if both nodes exist
            if source in concept_to_id and target in concept_to_id:
                edge = {
                    "from": concept_to_id[source],
                    "to": concept_to_id[target],
                    "label": rel_type
                }
                graph_data["edges"].append(edge)
        
        return graph_data
    
    def save_graph(self, graph_data: Dict[str, Any], filename: str) -> str:
        """Save the knowledge graph to a file.
        
        Args:
            graph_data: Knowledge graph data
            filename: Name to use for the saved file
            
        Returns:
            Path to the saved file
        """
        # Create the output filename
        base_name = Path(filename).stem
        output_path = Path(KNOWLEDGE_GRAPH_PATH) / f"{base_name}_graph.json"
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the graph
        with open(output_path, "w") as file:
            json.dump(graph_data, file, indent=2)
        
        return str(output_path)
    
    def build_networkx_graph(self, graph_data: Dict[str, Any]) -> nx.Graph:
        """Build a NetworkX graph from graph data.
        
        Args:
            graph_data: Knowledge graph data
            
        Returns:
            NetworkX graph
        """
        # Create a new graph
        G = nx.Graph()
        
        # Add nodes
        for node in graph_data.get("nodes", []):
            G.add_node(
                node["id"],
                label=node.get("label", ""),
                type=node.get("type", "concept")
            )
        
        # Add edges
        for edge in graph_data.get("edges", []):
            G.add_edge(
                edge["from"],
                edge["to"],
                label=edge.get("label", "")
            )
        
        return G
    
    def merge_graphs(self, graphs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple knowledge graphs.
        
        Args:
            graphs: List of knowledge graph data dictionaries
            
        Returns:
            Merged knowledge graph data
        """
        if not graphs:
            return {"nodes": [], "edges": []}
        
        merged_nodes = {}
        merged_edges = {}
        
        # Merge nodes
        for graph in graphs:
            for node in graph.get("nodes", []):
                node_id = node["id"]
                if node_id not in merged_nodes:
                    merged_nodes[node_id] = node
        
        # Merge edges
        for graph in graphs:
            for edge in graph.get("edges", []):
                edge_key = f"{edge['from']}__{edge['to']}__{edge.get('label', '')}"
                if edge_key not in merged_edges:
                    merged_edges[edge_key] = edge
        
        # Create merged graph
        merged_graph = {
            "nodes": list(merged_nodes.values()),
            "edges": list(merged_edges.values())
        }
        
        return merged_graph
    
    def process_content(self, content: Dict[str, Any], filename: str) -> Dict[str, Any]:
        """Process lecture content and build a knowledge graph.
        
        Args:
            content: Lecture content dictionary
            filename: Name to use for the saved file
            
        Returns:
            Knowledge graph data
        """
        # Build the graph
        graph_data = self.build_graph(content)
        
        # Save the graph
        self.save_graph(graph_data, filename)
        
        return graph_data
