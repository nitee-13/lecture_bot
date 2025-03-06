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
            content: Lecture content dictionary with concepts, equations, and relationships
            
        Returns:
            Knowledge graph data as a dictionary
        """
        graph_data = {"nodes": [], "edges": []}
        
        # Handle slide_content structure
        if "slide_content" in content:
            slides = content["slide_content"]
            
            # Add nodes for each slide
            for slide in slides:
                slide_id = f"slide_{slide['slide_number']}"
                graph_data["nodes"].append({
                    "id": slide_id,
                    "label": slide["title"],
                    "type": "slide"
                })
                
                # Add edges between consecutive slides
                if slide["slide_number"] > 1:
                    prev_slide_id = f"slide_{slide['slide_number'] - 1}"
                    graph_data["edges"].append({
                        "from": prev_slide_id,
                        "to": slide_id,
                        "label": "next"
                    })
                
                # Add nodes and edges for concepts in the slide
                for concept in slide.get("concepts", []):
                    concept_id = f"concept_{concept.lower().replace(' ', '_')}"
                    graph_data["nodes"].append({
                        "id": concept_id,
                        "label": concept,
                        "type": "concept"
                    })
                    graph_data["edges"].append({
                        "from": slide_id,
                        "to": concept_id,
                        "label": "contains"
                    })
                
                # Add nodes and edges for equations
                for equation in slide.get("equations", []):
                    eq_id = f"equation_{len(graph_data['nodes'])}"
                    graph_data["nodes"].append({
                        "id": eq_id,
                        "label": equation,
                        "type": "equation"
                    })
                    graph_data["edges"].append({
                        "from": slide_id,
                        "to": eq_id,
                        "label": "contains"
                    })
        
        # Handle concepts/equations/diagrams/relationships structure
        elif all(key in content for key in ["concepts", "equations", "diagrams", "relationships"]):
            # Add nodes for concepts
            for concept in content["concepts"]:
                concept_id = f"concept_{concept['name'].lower().replace(' ', '_')}"
                graph_data["nodes"].append({
                    "id": concept_id,
                    "label": concept["name"],
                    "type": "concept",
                    "definition": concept["definition"]
                })
                
                # Add edges for prerequisites
                for prereq in concept.get("prerequisites", []):
                    prereq_id = f"concept_{prereq.lower().replace(' ', '_')}"
                    graph_data["edges"].append({
                        "from": prereq_id,
                        "to": concept_id,
                        "label": "prerequisite"
                    })
                
                # Add edges for related concepts
                for related in concept.get("related_concepts", []):
                    related_id = f"concept_{related.lower().replace(' ', '_')}"
                    graph_data["edges"].append({
                        "from": concept_id,
                        "to": related_id,
                        "label": "related_to"
                    })
            
            # Add nodes for equations
            for equation in content["equations"]:
                eq_id = f"equation_{len(graph_data['nodes'])}"
                graph_data["nodes"].append({
                    "id": eq_id,
                    "label": equation["latex"],
                    "type": "equation",
                    "context": equation["context"]
                })
                
                # Add edges to related concepts
                for concept in equation.get("related_concepts", []):
                    concept_id = f"concept_{concept.lower().replace(' ', '_')}"
                    graph_data["edges"].append({
                        "from": concept_id,
                        "to": eq_id,
                        "label": "expressed_by"
                    })
            
            # Add nodes for diagrams
            for diagram in content["diagrams"]:
                diagram_id = f"diagram_{len(graph_data['nodes'])}"
                graph_data["nodes"].append({
                    "id": diagram_id,
                    "label": diagram["description"],
                    "type": "diagram"
                })
                
                # Add edges to related concepts
                for concept in diagram.get("related_concepts", []):
                    concept_id = f"concept_{concept.lower().replace(' ', '_')}"
                    graph_data["edges"].append({
                        "from": concept_id,
                        "to": diagram_id,
                        "label": "illustrated_by"
                    })
            
            # Add edges from relationships
            for rel in content["relationships"]:
                from_id = f"concept_{rel['from'].lower().replace(' ', '_')}"
                to_id = f"concept_{rel['to'].lower().replace(' ', '_')}"
                graph_data["edges"].append({
                    "from": from_id,
                    "to": to_id,
                    "label": rel["type"]
                })
        
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
