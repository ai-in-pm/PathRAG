import os
import networkx as nx
import argparse
from typing import List, Dict, Any

from agents.graph_construction_expert import GraphConstructionExpert
from utils.data_loader import load_sample_dataset, load_pathrag_paper
from visualization.graph_visualizer import GraphVisualizer

def generate_visualizations(use_paper_data=True, output_dir="visualization_examples"):
    """
    Generate a series of visualizations to showcase the PathRAG visualization capabilities.
    
    Args:
        use_paper_data: Whether to use the PathRAG paper as data source
        output_dir: Directory to save the visualizations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating visualizations in {output_dir}...")
    
    # Load data
    if use_paper_data:
        print("Loading data from PathRAG paper...")
        documents = load_pathrag_paper()
    else:
        print("Loading sample dataset...")
        documents = load_sample_dataset()
    
    # Initialize graph expert
    graph_expert = GraphConstructionExpert()
    
    # Build graph
    graph_expert.load_documents(documents)
    graph = graph_expert.get_graph()
    
    print(f"Created graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    
    # Example 1: Basic graph visualization
    print("Generating basic graph visualization...")
    GraphVisualizer.visualize_graph(graph, output_path=os.path.join(output_dir, "basic_graph.png"))
    
    # Example 2: Subgraph visualization
    print("Generating subgraph visualization...")
    # Take a smaller subgraph for better visualization
    subgraph_nodes = list(graph.nodes())[:50]  # First 50 nodes
    subgraph = graph.subgraph(subgraph_nodes)
    GraphVisualizer.visualize_graph(subgraph, output_path=os.path.join(output_dir, "subgraph.png"))
    
    # Example 3: Highlighted nodes
    print("Generating visualization with highlighted nodes...")
    highlight_nodes = set(list(graph.nodes())[:5])  # Highlight first 5 nodes
    GraphVisualizer.visualize_graph(
        subgraph, 
        output_path=os.path.join(output_dir, "highlighted_graph.png"),
        highlight_nodes=highlight_nodes
    )
    
    # Example 4: Path visualization with correct format (adds score as third element)
    print("Generating path visualization...")
    # Create a simple path for demonstration with score
    if len(subgraph_nodes) >= 10:
        example_path = ([subgraph_nodes[0], subgraph_nodes[2], subgraph_nodes[5]], 
                        ["related_to", "contains"],
                        0.85)  # Added score as third element
        GraphVisualizer.visualize_paths(
            subgraph, 
            [example_path], 
            output_path=os.path.join(output_dir, "path_visualization.png")
        )
    
    # Example 5: Interactive 3D visualization
    print("Generating interactive 3D visualization...")
    GraphVisualizer.create_interactive_visualization(
        subgraph,
        output_path=os.path.join(output_dir, "interactive_graph.html")
    )
    
    # Example 6: Performance metrics visualization
    print("Generating performance metrics visualization...")
    metrics = {
        "PathRAG": {
            "comprehensiveness": 4.5,
            "diversity": 4.3,
            "logicality": 4.7,
            "relevance": 4.9,
            "coherence": 4.6,
            "tokens": 850
        },
        "Traditional RAG": {
            "comprehensiveness": 3.8,
            "diversity": 3.5,
            "logicality": 3.9,
            "relevance": 4.1,
            "coherence": 3.7,
            "tokens": 1500
        },
        "GraphRAG": {
            "comprehensiveness": 4.2,
            "diversity": 4.0,
            "logicality": 4.3,
            "relevance": 4.5,
            "coherence": 4.1,
            "tokens": 1200
        }
    }
    GraphVisualizer.visualize_performance_comparison(
        metrics,
        output_path=os.path.join(output_dir, "performance_comparison.png")
    )
    
    print(f"All visualizations generated in {output_dir}")
    
    return output_dir

def main():
    parser = argparse.ArgumentParser(description='Generate PathRAG visualizations')
    parser.add_argument('--use-paper-data', action='store_true', help='Use PathRAG paper as data source')
    parser.add_argument('--output-dir', default='visualization_examples', help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    output_dir = generate_visualizations(args.use_paper_data, args.output_dir)
    print(f"Visualization examples saved to {output_dir}")
    print(f"To view the interactive visualization, open {os.path.join(output_dir, 'interactive_graph.html')} in a web browser")

if __name__ == "__main__":
    main()
