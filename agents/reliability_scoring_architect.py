import networkx as nx
import numpy as np
from typing import List, Dict, Tuple, Set
from collections import defaultdict

class ReliabilityScoringArchitect:
    """
    The Reliability Scoring Architect calculates and explains path reliability scores
    to determine which paths are most valuable for answering the query.
    """
    
    def __init__(self, max_paths: int = 10):
        """
        Initialize the Reliability Scoring Architect.
        
        Args:
            max_paths: Maximum number of paths to return after scoring
        """
        self.max_paths = max_paths
    
    def calculate_reliability_scores(self, graph: nx.DiGraph, 
                                    paths: List[Tuple[List[str], List[str]]],
                                    resources: Dict[str, float]) -> List[Tuple[List[str], List[str], float]]:
        """
        Calculate reliability scores for paths based on resource values.
        
        Args:
            graph: The knowledge graph
            paths: List of paths, where each path is represented as ([node_ids], [edge_types])
            resources: Dictionary mapping node IDs to resource values
            
        Returns:
            List of paths with reliability scores: [(node_ids, edge_types, score)]
        """
        scored_paths = []
        
        for node_ids, edge_types in paths:
            # Calculate average resource value for the path
            path_score = self._compute_path_score(node_ids, resources)
            
            # Add path with score to list
            scored_paths.append((node_ids, edge_types, path_score))
        
        # Sort paths by score in descending order
        scored_paths.sort(key=lambda x: x[2], reverse=True)
        
        # Take top paths based on max_paths
        top_paths = scored_paths[:self.max_paths]
        
        print(f"Selected {len(top_paths)} paths with highest reliability scores")
        for i, (node_ids, _, score) in enumerate(top_paths[:3]):  # Print top 3 for brevity
            start_text = graph.nodes[node_ids[0]].get("text", "")[:30]
            end_text = graph.nodes[node_ids[-1]].get("text", "")[:30]
            print(f"  Path {i+1}: {start_text}... -> ... {end_text}... (Score: {score:.4f})")
        
        return top_paths
    
    def _compute_path_score(self, node_ids: List[str], resources: Dict[str, float]) -> float:
        """
        Compute reliability score for a path based on resource values.
        
        Args:
            node_ids: List of node IDs in the path
            resources: Dictionary mapping node IDs to resource values
            
        Returns:
            Reliability score for the path
        """
        # If resources not provided, use path length as inverse score
        if not resources:
            return 1.0 / (len(node_ids) - 1)  # Shorter paths get higher scores
        
        # Otherwise, calculate average resource value
        path_resources = [resources.get(node_id, 0.0) for node_id in node_ids]
        
        # Normalize by path length (average resource per node)
        # We use a geometric mean to penalize paths with any very low-resource nodes
        non_zero_resources = [r for r in path_resources if r > 0]
        if not non_zero_resources:
            return 0.0
        
        # Use geometric mean for resources
        log_resources = [np.log(r) for r in non_zero_resources if r > 0]
        if not log_resources:
            return 0.0
            
        geom_mean = np.exp(sum(log_resources) / len(log_resources))
        
        # Factor in path length (shorter paths get higher scores, all else equal)
        length_penalty = 1.0 / (1 + np.log(len(node_ids)))
        
        return geom_mean * length_penalty
    
    def generate_resource_values(self, graph: nx.DiGraph, start_nodes: Set[str], 
                               decay_rate: float = 0.8) -> Dict[str, float]:
        """
        Generate resource values for all nodes starting from the retrieved nodes.
        
        Args:
            graph: The knowledge graph
            start_nodes: Set of starting node IDs (usually the retrieved nodes)
            decay_rate: Decay rate for resource propagation
            
        Returns:
            Dictionary mapping node IDs to resource values
        """
        resources = defaultdict(float)
        
        # Initialize resources for start nodes
        for node_id in start_nodes:
            resources[node_id] = 1.0
        
        # Propagate resources through the graph
        visited = set(start_nodes)
        frontier = list(start_nodes)
        
        while frontier:
            next_frontier = []
            
            for node_id in frontier:
                # Get outgoing edges
                out_edges = list(graph.out_edges(node_id))
                
                # Skip if no outgoing edges
                if not out_edges:
                    continue
                
                # Calculate resource flow to each neighbor
                flow_per_edge = resources[node_id] * decay_rate / len(out_edges)
                
                for _, neighbor in out_edges:
                    # Update resource value for neighbor
                    resources[neighbor] += flow_per_edge
                    
                    # Add to next frontier if not visited
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.append(neighbor)
            
            frontier = next_frontier
        
        return resources
    
    def explain_reliability_scoring(self, graph: nx.DiGraph, 
                                   scored_paths: List[Tuple[List[str], List[str], float]]) -> str:
        """
        Provide an explanation of the reliability scoring process.
        
        Args:
            graph: The knowledge graph
            scored_paths: List of paths with scores
            
        Returns:
            A detailed explanation of the reliability scoring process
        """
        explanation = [
            "# Reliability Scoring Explanation",
            "",
            "## Scoring Methodology",
            "",
            "Reliability scores quantify how much we can trust each path for answering the query:",
            "",
            "1. **Resource Propagation:**",
            "   - Initial resources start at 1.0 for query-relevant nodes",
            "   - Resources flow through the graph with decay over distance",
            "",
            "2. **Path Score Calculation:**",
            "   - Based on average resource values along the path",
            "   - Geometric mean is used to penalize paths with any low-resource nodes",
            "   - Length penalty factor prefers shorter, more direct paths",
            "",
            "3. **Ranking Mechanism:**",
            "   - Paths are ranked by their reliability scores",
            f"   - Top {self.max_paths} paths are selected for the final retrieval set",
            "",
            "## Reliability Scores for Selected Paths",
            ""
        ]
        
        for i, (node_ids, edge_types, score) in enumerate(scored_paths):
            explanation.append(f"### Path {i+1}: (Score: {score:.4f})")
            
            path_str = ""
            for j in range(len(node_ids)):
                node_id = node_ids[j]
                node_text = graph.nodes[node_id].get("text", node_id)
                node_text_short = node_text[:50] + "..." if len(node_text) > 50 else node_text
                path_str += f"{node_text_short}"
                
                if j < len(node_ids) - 1:
                    edge_type = edge_types[j]
                    path_str += f" --[{edge_type}]--> "
            
            explanation.append(f"{path_str}")
            explanation.append("")
        
        explanation.append("Higher reliability scores indicate paths that are more likely to")
        explanation.append("provide accurate and relevant information for answering the query.")
        
        return "\n".join(explanation)
