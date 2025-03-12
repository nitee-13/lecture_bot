"""
Test script for knowledge graph visualization.
"""
import streamlit as st
import json
import os
from pathlib import Path

from src.knowledge_graph.visualizer import KnowledgeGraphVisualizer
from src.config import KNOWLEDGE_GRAPH_PATH

def main():
    """Main function for testing knowledge graph visualization."""
    st.title("Knowledge Graph Visualization Test")
    
    # Find all knowledge graph files
    kg_path = Path(KNOWLEDGE_GRAPH_PATH)
    graph_files = list(kg_path.glob("*.json"))
    
    if not graph_files:
        st.error("No knowledge graph files found in the knowledge graph directory.")
        return
    
    # Let the user select a graph file
    selected_file = st.selectbox(
        "Select a knowledge graph file",
        options=graph_files,
        format_func=lambda x: x.name
    )
    
    if selected_file:
        st.write(f"Selected file: {selected_file.name}")
        
        try:
            # Load the graph data
            with open(selected_file, "r") as file:
                graph_data = json.load(file)
            
            # Display graph statistics
            st.write(f"**Nodes:** {len(graph_data.get('nodes', []))}")
            st.write(f"**Edges:** {len(graph_data.get('edges', []))}")
            
            # Create a visualizer
            visualizer = KnowledgeGraphVisualizer()
            
            # Display the graph
            visualizer.display_in_streamlit(graph_data)
            
        except Exception as e:
            st.error(f"Error visualizing knowledge graph: {str(e)}")
            st.exception(e)

if __name__ == "__main__":
    main() 