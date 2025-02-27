import networkx as nx
import numpy as np
from typing import List, Dict, Set, Tuple
from collections import defaultdict, deque

class PathAnalysisEngineer:
    """
    The Path Analysis Engineer implements the flow-based pruning algorithm 
    with distance awareness to extract key relational paths between retrieved nodes.
    """
    
    def __init__(self, decay_rate: float = 0.8, pruning_threshold: float = 0.01, max_path_length: int = 4):
        """
        Initialize the Path Analysis Engineer.
        
        Args:
            decay_rate: Decay rate for information propagation along edges (α in the paper)
            pruning_threshold: Threshold for early stopping (θ in the paper)
            max_path_length: Maximum path length to consider
        """
        self.decay_rate = decay_rate
        self.pruning_threshold = pruning_threshold
        self.max_path_length = max_path_length
    
    def extract_paths(self, graph: nx.DiGraph, retrieved_nodes: Set[str]) -> List[Tuple[List[str], List[str]]]:
        """
        Extract key relational paths between each pair of retrieved nodes.
        
        Args:
            graph: The knowledge graph
            retrieved_nodes: Set of retrieved node IDs
            
        Returns:
            List of paths, where each path is represented as ([node_ids], [edge_types])
        """
        paths = []
        
        # Generate all possible node pairs from retrieved nodes
        retrieved_nodes_list = list(retrieved_nodes)
        
        for i in range(len(retrieved_nodes_list)):
            for j in range(i+1, len(retrieved_nodes_list)):
                start_node = retrieved_nodes_list[i]
                end_node = retrieved_nodes_list[j]
                
                # For each pair, extract paths in both directions
                paths_i_to_j = self._flow_based_pruning(graph, start_node, end_node)
                paths_j_to_i = self._flow_based_pruning(graph, end_node, start_node)
                
                paths.extend(paths_i_to_j)
                paths.extend(paths_j_to_i)
        
        return paths
    
    def _flow_based_pruning(self, graph: nx.DiGraph, start_node: str, end_node: str) -> List[Tuple[List[str], List[str]]]:
        """
        Apply flow-based pruning algorithm to extract key paths from start_node to end_node.
        
        Args:
            graph: The knowledge graph
            start_node: Starting node ID
            end_node: Target node ID
            
        Returns:
            List of paths from start_node to end_node
        """
        # Initialize resource distribution
        resources = defaultdict(float)
        resources[start_node] = 1.0
        visited = set([start_node])
        
        # Queue for BFS traversal: (node_id, path_so_far, edges_so_far)
        queue = deque([(start_node, [start_node], [])])
        valid_paths = []
        
        while queue:
            current_node, current_path, current_edges = queue.popleft()
            
            # Skip if path exceeds max length
            if len(current_path) > self.max_path_length + 1:
                continue
            
            # If reached end node, add path to valid paths
            if current_node == end_node and len(current_path) > 2:  # Path must have at least one intermediate node
                valid_paths.append((current_path, current_edges))
                continue
            
            # Get outgoing edges
            out_edges = list(graph.out_edges(current_node, data=True))
            
            # Early stopping if resource is too small
            if resources[current_node] / max(1, len(out_edges)) < self.pruning_threshold:
                continue
            
            # Distribute resources to neighbors
            for _, neighbor, edge_data in out_edges:
                # Skip if node already in path (avoid cycles)
                if neighbor in current_path:
                    continue
                
                # Calculate resource flow to neighbor
                neighbor_resource = self.decay_rate * resources[current_node] / max(1, len(out_edges))
                
                # If neighbor not visited or new resource is higher, update
                if neighbor not in visited or neighbor_resource > resources[neighbor]:
                    resources[neighbor] = neighbor_resource
                    visited.add(neighbor)
                    
                    # Add new path to queue
                    new_path = current_path + [neighbor]
                    new_edges = current_edges + [edge_data.get('relation', 'related_to')]
                    queue.append((neighbor, new_path, new_edges))
        
        return valid_paths
    
    def explain_path_analysis(self, graph: nx.DiGraph, paths: List[Tuple[List[str], List[str]]]) -> str:
        """
        Provide an explanation of the path analysis process.
        
        Args:
            graph: The knowledge graph
            paths: List of extracted paths
            
        Returns:
            A detailed explanation of the path analysis process
        """
        explanation = [
            "# Path Analysis Explanation",
            "",
            "## Flow-based Pruning Algorithm",
            "",
            "The path analysis process uses a flow-based pruning algorithm with distance awareness:",
            "",
            f"1. **Resource Initialization:**",
            f"   - The starting node is assigned a resource value of 1.0",
            f"   - All other nodes start with 0 resources",
            "",
            f"2. **Resource Propagation:**",
            f"   - Resources flow from nodes to their neighbors",
            f"   - A decay rate of {self.decay_rate} reduces resource values with distance",
            f"   - Resources are evenly distributed among outgoing edges",
            "",
            f"3. **Early Stopping:**",
            f"   - Paths are pruned when resource values fall below {self.pruning_threshold}",
            f"   - This prevents exploration of paths with negligible contributions",
            "",
            f"4. **Path Length Limitation:**",
            f"   - Maximum path length is set to {self.max_path_length} to focus on concise connections",
            "",
            "## Extracted Paths",
            ""
        ]
        
        for i, (node_ids, edge_types) in enumerate(paths[:5]):  # Show first 5 paths
            explanation.append(f"### Path {i+1}:")
            path_str = ""
            for j in range(len(node_ids)):
                node_id = node_ids[j]
                node_text = graph.nodes[node_id].get("text", node_id)
                path_str += f"{node_text}"
                
                if j < len(node_ids) - 1:
                    edge_type = edge_types[j]
                    path_str += f" --[{edge_type}]--> "
            
            explanation.append(f"{path_str}")
            explanation.append("")
        
        if len(paths) > 5:
            explanation.append(f"... and {len(paths)-5} more paths")
            
        explanation.append("")
        explanation.append("This pruning approach effectively reduces redundant information while")
        explanation.append("preserving the most relevant connections between retrieved nodes.")
        
        return "\n".join(explanation)
