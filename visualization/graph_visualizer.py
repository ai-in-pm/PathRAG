import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple, Set
import plotly.graph_objects as go

class GraphVisualizer:
    """
    The GraphVisualizer provides visualization tools for graphs, paths, and performance metrics.
    """
    
    @staticmethod
    def visualize_graph(graph: nx.DiGraph, output_path: str = None, 
                       highlight_nodes: Set[str] = None) -> None:
        """
        Visualize the graph using matplotlib.
        
        Args:
            graph: The knowledge graph
            output_path: Path to save the visualization
            highlight_nodes: Set of node IDs to highlight
        """
        plt.figure(figsize=(12, 10))
        
        # Use spring layout for node positioning
        pos = nx.spring_layout(graph, seed=42)
        
        # Default node color
        node_color = ['lightblue'] * len(graph.nodes())
        
        # Highlight specific nodes if provided
        if highlight_nodes:
            for i, node in enumerate(graph.nodes()):
                if node in highlight_nodes:
                    node_color[i] = 'orange'
        
        # Draw the graph
        nx.draw_networkx_nodes(graph, pos, node_size=500, node_color=node_color, alpha=0.8)
        nx.draw_networkx_edges(graph, pos, width=1, alpha=0.5, arrowsize=15)
        
        # Add labels
        labels = {}
        for node in graph.nodes():
            node_text = graph.nodes[node].get("text", "")
            labels[node] = node[:10] + ("..." if len(node) > 10 else "")
        
        nx.draw_networkx_labels(graph, pos, labels=labels, font_size=10)
        
        plt.title("Knowledge Graph Visualization")
        plt.axis('off')
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight')
            print(f"Graph visualization saved to {output_path}")
        
        plt.close()
    
    @staticmethod
    def visualize_paths(graph: nx.DiGraph, paths: List[Tuple[List[str], List[str], float]], 
                      output_path: str = None) -> None:
        """
        Visualize the extracted paths using matplotlib.
        
        Args:
            graph: The knowledge graph
            paths: List of paths with reliability scores
            output_path: Path to save the visualization
        """
        plt.figure(figsize=(15, 10))
        
        # Use spring layout for node positioning
        pos = nx.spring_layout(graph, seed=42)
        
        # First, draw the full graph in light gray
        nx.draw_networkx_nodes(graph, pos, node_size=300, node_color='lightgray', alpha=0.3)
        nx.draw_networkx_edges(graph, pos, width=0.5, alpha=0.2, arrowsize=10)
        
        # Then highlight the paths
        path_colors = plt.cm.rainbow(np.linspace(0, 1, min(len(paths), 10)))
        
        for i, (node_ids, edge_types, score) in enumerate(paths[:10]):  # Visualize up to 10 paths
            path_edges = [(node_ids[j], node_ids[j+1]) for j in range(len(node_ids)-1)]
            
            # Draw path nodes and edges
            nx.draw_networkx_nodes(graph, pos, nodelist=node_ids, node_size=500, 
                                  node_color=[path_colors[i]] * len(node_ids), alpha=0.8)
            
            nx.draw_networkx_edges(graph, pos, edgelist=path_edges, width=2, 
                                  edge_color=path_colors[i], alpha=0.8, arrowsize=15)
            
            # Add label for the first node in path for identification
            path_label = {}
            path_label[node_ids[0]] = f"Path {i+1}"
            nx.draw_networkx_labels(graph, pos, labels=path_label, 
                                  font_size=12, font_weight='bold')
        
        plt.title("Extracted Paths Visualization")
        plt.axis('off')
        
        # Add legend for paths
        legend_elements = [plt.Line2D([0], [0], color=path_colors[i], lw=2, 
                                     label=f"Path {i+1} (Score: {paths[i][2]:.2f})")
                          for i in range(min(len(paths), 10))]
        
        plt.legend(handles=legend_elements, loc='upper right')
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight')
            print(f"Path visualization saved to {output_path}")
        
        plt.close()
    
    @staticmethod
    def visualize_comparison(methods: Dict[str, Dict[str, float]], 
                           metrics: List[str], output_path: str = None) -> None:
        """
        Visualize the performance comparison between different methods.
        
        Args:
            methods: Dictionary mapping method names to dictionaries of metric scores
            metrics: List of metric names to compare
            output_path: Path to save the visualization
        """
        method_names = list(methods.keys())
        metric_values = {metric: [methods[method].get(metric, 0) for method in method_names] 
                        for metric in metrics}
        
        # Create a figure with a subplot for each metric
        fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 3*len(metrics)))
        
        if len(metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            axes[i].bar(method_names, metric_values[metric], color=plt.cm.viridis(np.linspace(0, 1, len(method_names))))
            axes[i].set_title(f"{metric.capitalize()} Comparison")
            axes[i].set_ylim(0, 5.5 if metric != "token_reduction" else 1.1)
            
            # Add values on top of bars
            for j, v in enumerate(metric_values[metric]):
                axes[i].text(j, v + 0.1, f"{v:.2f}", ha='center')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight')
            print(f"Comparison visualization saved to {output_path}")
        
        plt.close()
    
    @staticmethod
    def create_interactive_visualization(graph: nx.DiGraph, paths: List[Tuple[List[str], List[str], float]], 
                                       output_path: str = None) -> None:
        """
        Create an interactive visualization of the graph and paths using Plotly.
        
        Args:
            graph: The knowledge graph
            paths: List of paths with reliability scores
            output_path: Path to save the HTML visualization
        """
        # Use spring layout for node positioning
        pos = nx.spring_layout(graph, seed=42, dim=3)
        
        # Create node traces
        node_x = []
        node_y = []
        node_z = []
        node_text = []
        node_size = []
        node_color = []
        
        for node in graph.nodes():
            x, y, z = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)
            
            # Node text
            node_data = graph.nodes[node]
            node_type = node_data.get("type", "unknown")
            node_content = node_data.get("text", "")[:50] + "..." if len(node_data.get("text", "")) > 50 else node_data.get("text", "")
            node_text.append(f"ID: {node}<br>Type: {node_type}<br>Content: {node_content}")
            
            # Node size and color
            is_in_path = any(node in node_ids for node_ids, _, _ in paths)
            node_size.append(15 if is_in_path else 8)
            node_color.append('#FF9500' if is_in_path else '#97C2FC')
        
        node_trace = go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                size=node_size,
                color=node_color,
                opacity=0.8
            )
        )
        
        # Create edge traces for the main graph
        edge_x = []
        edge_y = []
        edge_z = []
        edge_text = []
        
        for edge in graph.edges(data=True):
            x0, y0, z0 = pos[edge[0]]
            x1, y1, z1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_z.extend([z0, z1, None])
            
            relation = edge[2].get("relation", "related_to")
            edge_text.append(f"Source: {edge[0]}<br>Target: {edge[1]}<br>Relation: {relation}")
        
        edge_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(width=1, color='#888'),
            hoverinfo='text',
            text=edge_text,
            opacity=0.3
        )
        
        # Create edge traces for the paths
        path_traces = []
        path_colors = ['#FF0000', '#00FF00', '#0000FF', '#FF00FF', '#FFFF00', 
                      '#00FFFF', '#FF8000', '#8000FF', '#0080FF', '#FF0080']
        
        for i, (node_ids, edge_types, score) in enumerate(paths[:10]):  # Visualize up to 10 paths
            path_edge_x = []
            path_edge_y = []
            path_edge_z = []
            path_edge_text = []
            
            for j in range(len(node_ids) - 1):
                x0, y0, z0 = pos[node_ids[j]]
                x1, y1, z1 = pos[node_ids[j+1]]
                path_edge_x.extend([x0, x1, None])
                path_edge_y.extend([y0, y1, None])
                path_edge_z.extend([z0, z1, None])
                
                relation = edge_types[j]
                path_edge_text.append(f"Path {i+1}<br>Source: {node_ids[j]}<br>Target: {node_ids[j+1]}<br>Relation: {relation}")
            
            path_trace = go.Scatter3d(
                x=path_edge_x, y=path_edge_y, z=path_edge_z,
                mode='lines',
                line=dict(width=4, color=path_colors[i % len(path_colors)]),
                hoverinfo='text',
                text=path_edge_text,
                name=f"Path {i+1} (Score: {score:.2f})"
            )
            
            path_traces.append(path_trace)
        
        # Create the figure
        fig = go.Figure(data=[edge_trace, node_trace] + path_traces)
        
        # Update layout
        fig.update_layout(
            title="Interactive PathRAG Visualization",
            showlegend=True,
            hovermode='closest',
            scene=dict(
                xaxis=dict(showbackground=False, showticklabels=False, title=''),
                yaxis=dict(showbackground=False, showticklabels=False, title=''),
                zaxis=dict(showbackground=False, showticklabels=False, title='')
            ),
            margin=dict(b=0, l=0, r=0, t=40)
        )
        
        if output_path:
            fig.write_html(output_path)
            print(f"Interactive visualization saved to {output_path}")
        
        return fig
