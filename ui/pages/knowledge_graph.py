"""
Knowledge graph visualization page for Streamlit UI.
"""
import streamlit as st
import os
import json
from pathlib import Path

from src.knowledge_graph.visualizer import KnowledgeGraphVisualizer
from src.knowledge_graph.builder import KnowledgeGraphBuilder
from src.vector_store.store import VectorStore
from src.config import KNOWLEDGE_GRAPH_PATH, KG_HEIGHT, KG_WIDTH


class KnowledgeGraphPage:
    """Knowledge graph visualization page for Streamlit UI."""
    
    def __init__(self, vector_store: VectorStore):
        """Initialize the knowledge graph page.
        
        Args:
            vector_store: Vector store instance
        """
        self.vector_store = vector_store
        self.kg_visualizer = KnowledgeGraphVisualizer()
        self.kg_builder = KnowledgeGraphBuilder()
    
    def render(self):
        """Render the knowledge graph page."""
        st.title("Knowledge Graph")
        
        st.markdown("""
        The knowledge graph visualizes concepts and their relationships from your lecture materials.
        Use it to understand how different topics are connected.
        """)
        
        # Get available knowledge graphs
        graph_files = self._get_available_graphs()
        
        if not graph_files:
            st.info("No knowledge graphs available. Upload lecture materials to generate graphs.")
            return
        
        # Document selection
        selected_document = st.selectbox(
            "Select a document",
            options=list(graph_files.keys()),
            format_func=lambda x: x
        )
        
        # Visualization options
        col1, col2 = st.columns(2)
        with col1:
            height = st.selectbox(
                "Height",
                options=["400px", "500px", "600px", "700px", "800px"],
                index=2
            )
        with col2:
            width = st.selectbox(
                "Width",
                options=["100%", "80%", "1200px", "1000px", "800px"],
                index=0
            )
        
        # Load and visualize the selected graph
        if selected_document:
            self._visualize_graph(graph_files[selected_document], height, width)
    
    def _get_available_graphs(self) -> dict:
        """Get available knowledge graph files.
        
        Returns:
            Dictionary of document names to graph file paths
        """
        graph_files = {}
        
        # Get documents from vector store
        documents = self.vector_store.list_documents()
        
        # Check if each document has a corresponding graph file
        for doc_id in documents:
            doc_name = Path(doc_id).stem
            graph_path = Path(KNOWLEDGE_GRAPH_PATH) / f"{doc_name}_graph.json"
            
            if graph_path.exists():
                graph_files[doc_name] = str(graph_path)
        
        return graph_files
    
    def _visualize_graph(self, graph_path: str, height: str = KG_HEIGHT, width: str = KG_WIDTH):
        """Visualize a knowledge graph.
        
        Args:
            graph_path: Path to the graph file
            height: Height of the visualization
            width: Width of the visualization
        """
        try:
            # Load the graph data
            with open(graph_path, "r") as file:
                graph_data = json.load(file)
            
            # Check if the graph has nodes
            if not graph_data.get("nodes"):
                st.warning("This knowledge graph is empty. Try uploading more detailed lecture materials.")
                return
            
            # Display graph statistics
            st.write(f"**Concepts:** {len(graph_data.get('nodes', []))}")
            st.write(f"**Relationships:** {len(graph_data.get('edges', []))}")
            
            # Visualize the graph
            st.subheader("Graph Visualization")
            self.kg_visualizer.display_in_streamlit(graph_data, height=height, width=width)
            
            # Show concept list
            with st.expander("View Concept List"):
                # Extract and sort concept labels
                concepts = [node.get("label", "") for node in graph_data.get("nodes", [])]
                concepts.sort()
                
                # Group concepts by first letter
                concept_groups = {}
                for concept in concepts:
                    first_letter = concept[0].upper() if concept else ""
                    if first_letter not in concept_groups:
                        concept_groups[first_letter] = []
                    concept_groups[first_letter].append(concept)
                
                # Display concept groups
                for letter, letter_concepts in sorted(concept_groups.items()):
                    st.write(f"**{letter}**")
                    st.write(", ".join(letter_concepts))
            
        except Exception as e:
            st.error(f"Error visualizing knowledge graph: {str(e)}")
    
    def _merge_all_graphs(self) -> dict:
        """Merge all available knowledge graphs.
        
        Returns:
            Merged graph data
        """
        # Get available graph files
        graph_files = self._get_available_graphs()
        
        # Load all graphs
        graphs = []
        for graph_path in graph_files.values():
            try:
                with open(graph_path, "r") as file:
                    graph_data = json.load(file)
                graphs.append(graph_data)
            except Exception:
                pass
        
        # Merge graphs
        merged_graph = self.kg_builder.merge_graphs(graphs)
        
        return merged_graph
