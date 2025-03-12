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
                         height: str = KG_HEIGHT, width: str = KG_WIDTH, memory_efficient: bool = True) -> Optional[str]:
        """Visualize a knowledge graph using pyvis.
        
        Args:
            graph_data: Knowledge graph data
            output_path: Path to save the visualization HTML
            height: Height of the visualization
            width: Width of the visualization
            memory_efficient: Whether to use memory-efficient settings
            
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
            
            G.add_node(node_id, label=label)
            node_types[node_id] = node_type
        
        # Add edges
        for edge in graph_data.get("edges", []):
            source = edge.get("from", "")
            target = edge.get("to", "")
            label = edge.get("label", "")
            
            if source in G.nodes and target in G.nodes:
                G.add_edge(source, target, title=label)
        
        # Create a pyvis network
        net = Network(height=height, width=width, notebook=False, directed=False)
        
        # Set physics options for better layout with minimal resource usage
        if memory_efficient:
            physics_options = {
                "physics": {
                    "enabled": True,
                    "solver": "repulsion",  # Even simpler solver
                    "repulsion": {
                        "nodeDistance": 100,
                        "centralGravity": 0.2,
                        "springLength": 100,
                        "springConstant": 0.05,
                        "damping": 0.09
                    },
                    "stabilization": {
                        "enabled": True,
                        "iterations": 50,  # Very few iterations
                        "updateInterval": 100,
                        "fit": True
                    },
                    "maxVelocity": 5,
                    "minVelocity": 2,
                    "timestep": 0.2
                }
            }
        else:
            physics_options = {
                "physics": {
                    "enabled": True,
                    "solver": "forceAtlas2Based",
                    "forceAtlas2Based": {
                        "gravitationalConstant": -50,
                        "centralGravity": 0.01,
                        "springLength": self.edge_length,
                        "springConstant": 0.08,
                        "damping": 0.4,
                        "avoidOverlap": 0
                    },
                    "stabilization": {
                        "enabled": True,
                        "iterations": 100,
                        "updateInterval": 50,
                        "fit": True
                    },
                    "maxVelocity": 10,
                    "minVelocity": 1,
                    "timestep": 0.3,
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
                             figsize: tuple = (12, 8)) -> Optional[Union[str, plt.Figure]]:
        """Visualize a knowledge graph using matplotlib.
        
        Args:
            graph_data: Knowledge graph data
            output_path: Path to save the visualization image
            figsize: Figure size
            
        Returns:
            Path to the saved visualization image, matplotlib figure, or None if visualization failed
        """
        try:
            # Create a NetworkX graph
            G = nx.Graph()
            
            # Add nodes
            node_types = {}
            node_labels = {}
            for node in graph_data.get("nodes", []):
                node_id = node.get("id", "")
                if not node_id:
                    continue
                    
                label = node.get("label", "")
                node_type = node.get("type", "concept")
                
                G.add_node(node_id)
                node_types[node_id] = node_type
                node_labels[node_id] = label
            
            # Add edges
            for edge in graph_data.get("edges", []):
                source = edge.get("from", "")
                target = edge.get("to", "")
                
                if not source or not target:
                    continue
                    
                if source in G.nodes and target in G.nodes:
                    G.add_edge(source, target)
            
            # If the graph is empty, raise an exception
            if len(G.nodes) == 0:
                raise ValueError("Graph has no nodes")
            
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
            return plt.gcf()
        except Exception as e:
            import logging
            logging.error(f"Error visualizing graph with matplotlib: {str(e)}")
            return None
    
    def display_in_streamlit(self, graph_data: Dict[str, Any], height: str = KG_HEIGHT, width: str = KG_WIDTH):
        """Display a knowledge graph in Streamlit.
        
        Args:
            graph_data: Knowledge graph data
            height: Height of the visualization (e.g., "400px" or "100%")
            width: Width of the visualization (e.g., "100%" or "800px")
        """
        try:
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
            
            # Generate the HTML
            html = net.generate_html()
            
            # Add custom CSS to ensure the graph is interactive
            custom_css = """
            <style>
                .vis-network {
                    touch-action: none;
                    user-select: none;
                }
                .vis-network:focus {
                    outline: none;
                }
            </style>
            """
            html = custom_css + html
            
            # Display the HTML in Streamlit
            st.components.v1.html(html, height=int(height.replace('px', '')) if 'px' in height else 400)
            
        except Exception as e:
            st.error(f"Error displaying knowledge graph: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
    
    def _save_and_provide_download_link(self, html_content: str, filename: str, graph_data: Dict[str, Any] = None):
        """Save HTML content to a file and provide a download link.
        
        Args:
            html_content: HTML content to save
            filename: Name of the file to save
            graph_data: Knowledge graph data for simplified visualization
        """
        import tempfile
        import base64
        from pathlib import Path
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp:
            tmp.write(html_content.encode('utf-8'))
            tmp_path = tmp.name
        
        # Read the file and create a download link
        with open(tmp_path, 'rb') as f:
            data = f.read()
            b64 = base64.b64encode(data).decode('utf-8')
            href = f'<a href="data:text/html;base64,{b64}" download="{filename}">Download Knowledge Graph HTML</a>'
            st.markdown(href, unsafe_allow_html=True)
            
        # Display instructions
        st.info("The knowledge graph visualization couldn't be displayed directly. Please download the HTML file using the link above and open it in your browser.")
        
        # Try to display a simplified version using matplotlib if graph_data is provided
        if graph_data:
            try:
                st.subheader("Simplified Graph Preview")
                fig = self.visualize_matplotlib(graph_data)
                st.pyplot(fig)
            except Exception:
                pass
    
    def _clean_resources(self):
        """Clean up resources to prevent memory leaks."""
        import gc
        import sys
        
        # Force garbage collection
        gc.collect()
        
        # Clear matplotlib figures if any
        try:
            import matplotlib.pyplot as plt
            plt.close('all')
        except Exception:
            pass
            
        # Clear any networkx graphs
        try:
            import networkx as nx
            # Clear any global references to graphs
            for obj in gc.get_objects():
                if isinstance(obj, nx.Graph):
                    del obj
        except Exception:
            pass
            
        # Clear pyvis networks
        try:
            from pyvis.network import Network
            # Clear any global references to networks
            for obj in gc.get_objects():
                if isinstance(obj, Network):
                    del obj
        except Exception:
            pass
            
        # Run garbage collection again
        gc.collect()
        
        # Try to release memory back to the system
        if hasattr(gc, 'mem_free'):
            gc.mem_free()
