"""
Knowledge graph visualizer module for visualizing knowledge graphs.
"""
from typing import Dict, Any, Optional, Union
import json
import os
from pathlib import Path
import networkx as nx
from pyvis.network import Network
import matplotlib.pyplot as plt
import streamlit as st

from src.config import KG_NODE_SIZE, KG_EDGE_LENGTH, KG_WIDTH, KG_HEIGHT


class KnowledgeGraphVisualizer:
    """Visualize knowledge graphs."""
    
    def __init__(self, node_size: int = KG_NODE_SIZE, edge_length: int = KG_EDGE_LENGTH):
        """Initialize the knowledge graph visualizer.
        
        Args:
            node_size: Size of the nodes in the visualization
            edge_length: Length of the edges in the visualization
        """
        self.node_size = node_size
        self.edge_length = edge_length
    
    def visualize_pyvis(self, graph_data: Dict[str, Any], output_path: Optional[str] = None, 
                         height: str = KG_HEIGHT, width: str = KG_WIDTH) -> Optional[str]:
        """Visualize a knowledge graph using pyvis.
        
        Args:
            graph_data: Knowledge graph data
            output_path: Path to save the visualization HTML
            height: Height of the visualization
            width: Width of the visualization
            
        Returns:
            Path to the saved visualization HTML, or None if visualization failed
        """
        # Create a NetworkX graph
        G = nx.Graph()
        
        # Add nodes
        node_types = {}
        for node in graph_data.get("nodes", []):
            node_id = node["id"]
            label = node.get("label", "")
            node_type = node.get("type", "concept")
            
            G.add_node(node_id, label=label, title=label)
            node_types[node_id] = node_type
        
        # Add edges
        for edge in graph_data.get("edges", []):
            source = edge["from"]
            target = edge["to"]
            label = edge.get("label", "")
            
            if source in G.nodes and target in G.nodes:
                G.add_edge(source, target, title=label)
        
        # Create a pyvis network
        net = Network(height=height, width=width, notebook=False, directed=False)
        
        # Set physics options for better layout
        physics_options = {
            "physics": {
                "barnesHut": {
                    "gravitationalConstant": -2000,
                    "centralGravity": 0.1,
                    "springLength": self.edge_length,
                    "springConstant": 0.01,
                    "damping": 0.09,
                    "avoidOverlap": 0.2
                },
                "maxVelocity": 50,
                "minVelocity": 0.1,
                "solver": "barnesHut",
                "stabilization": {
                    "enabled": True,
                    "iterations": 1000,
                    "updateInterval": 100,
                    "fit": True
                },
                "timestep": 0.5,
                "adaptiveTimestep": True
            }
        }
        net.set_options(json.dumps(physics_options))
        
        # Define node colors based on type
        color_map = {
            "concept": "#4CAF50",  # Green
            "equation": "#2196F3",  # Blue
            "definition": "#FFC107",  # Amber
            "example": "#9C27B0",  # Purple
            "figure": "#F44336"   # Red
        }
        
        # Add the nodes and edges to the network
        for node_id in G.nodes:
            node_label = G.nodes[node_id].get("label", "")
            node_type = node_types.get(node_id, "concept")
            color = color_map.get(node_type, "#4CAF50")
            
            net.add_node(
                node_id, 
                label=node_label, 
                title=node_label,
                color=color,
                size=self.node_size
            )
        
        for edge in G.edges:
            source, target = edge
            label = G.edges[edge].get("title", "")
            
            net.add_edge(source, target, title=label)
        
        # Save the visualization if an output path is provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            net.save_graph(str(output_path))
            return str(output_path)
        
        # Return the network HTML string if no output path is provided
        return net.generate_html()
    
    def visualize_matplotlib(self, graph_data: Dict[str, Any], output_path: Optional[str] = None, 
                             figsize: tuple = (12, 8)) -> Optional[str]:
        """Visualize a knowledge graph using matplotlib.
        
        Args:
            graph_data: Knowledge graph data
            output_path: Path to save the visualization image
            figsize: Figure size
            
        Returns:
            Path to the saved visualization image, or None if visualization failed
        """
        # Create a NetworkX graph
        G = nx.Graph()
        
        # Add nodes
        node_types = {}
        node_labels = {}
        for node in graph_data.get("nodes", []):
            node_id = node["id"]
            label = node.get("label", "")
            node_type = node.get("type", "concept")
            
            G.add_node(node_id)
            node_types[node_id] = node_type
            node_labels[node_id] = label
        
        # Add edges
        for edge in graph_data.get("edges", []):
            source = edge["from"]
            target = edge["to"]
            
            if source in G.nodes and target in G.nodes:
                G.add_edge(source, target)
        
        # Define node colors based on type
        color_map = {
            "concept": "green",
            "equation": "blue",
            "definition": "orange",
            "example": "purple",
            "figure": "red"
        }
        
        # Map node types to colors
        node_colors = [color_map.get(node_types.get(node, "concept"), "green") for node in G.nodes]
        
        # Create the figure
        plt.figure(figsize=figsize)
        
        # Draw the graph
        pos = nx.spring_layout(G, k=0.3, iterations=50)
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=self.node_size * 10, alpha=0.8)
        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, font_color="black")
        
        plt.title("Knowledge Graph")
        plt.axis("off")
        
        # Save the visualization if an output path is provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(output_path), dpi=300, bbox_inches="tight")
            plt.close()
            return str(output_path)
        
        # Return the figure if no output path is provided
        return plt
    
    def display_in_streamlit(self, graph_data: Dict[str, Any], height: str = KG_HEIGHT, width: str = KG_WIDTH):
        """Display a knowledge graph in Streamlit.
        
        Args:
            graph_data: Knowledge graph data
            height: Height of the visualization (e.g., "400px" or "100%")
            width: Width of the visualization (e.g., "100%" or "800px")
        """
        # Convert height to integer (remove 'px' suffix)
        height_int = int(height.replace('px', '')) if 'px' in height else 400
        
        # Convert width to integer (remove 'px' suffix)
        # For percentage values, use a default width
        width_int = int(width.replace('px', '')) if 'px' in width else 800
        
        # Generate the HTML
        html = self.visualize_pyvis(graph_data, height=height, width=width)
        
        # Display the HTML in Streamlit
        st.components.v1.html(html, height=height_int, width=width_int)
