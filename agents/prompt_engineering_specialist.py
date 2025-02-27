import networkx as nx
from typing import List, Dict, Tuple, Any
import re

class PromptEngineeringSpecialist:
    """
    The Prompt Engineering Specialist creates path-based prompts with ascending reliability order
    to optimize LLM response generation.
    """
    
    def __init__(self, template_type: str = "ascending"):
        """
        Initialize the Prompt Engineering Specialist.
        
        Args:
            template_type: Type of prompt template to use ("ascending", "descending", or "random")
        """
        self.template_type = template_type
        self.templates = {
            "standard": self._standard_template,
            "ascending": self._ascending_template,
            "descending": self._descending_template,
            "random": self._random_template
        }
    
    def generate_prompt(self, query: str, graph: nx.DiGraph, 
                        scored_paths: List[Tuple[List[str], List[str], float]]) -> str:
        """
        Generate a prompt for the LLM based on the query and retrieved paths.
        
        Args:
            query: The user query
            graph: The knowledge graph
            scored_paths: List of paths with reliability scores
            
        Returns:
            A prompt for the LLM
        """
        # Sort paths based on template type
        if self.template_type == "ascending":
            # Sort by ascending reliability score (least reliable first)
            sorted_paths = sorted(scored_paths, key=lambda x: x[2])
        elif self.template_type == "descending":
            # Sort by descending reliability score (most reliable first)
            sorted_paths = sorted(scored_paths, key=lambda x: x[2], reverse=True)
        else:  # "random" or any other
            # Keep original order
            sorted_paths = scored_paths
        
        # Convert paths to textual form
        textual_paths = self._convert_paths_to_text(graph, sorted_paths)
        
        # Generate prompt using selected template
        template_func = self.templates.get(self.template_type, self.templates["standard"])
        prompt = template_func(query, textual_paths)
        
        print(f"Generated {self.template_type} prompt with {len(textual_paths)} paths")
        
        return prompt
    
    def _convert_paths_to_text(self, graph: nx.DiGraph, 
                              scored_paths: List[Tuple[List[str], List[str], float]]) -> List[str]:
        """
        Convert paths to textual form for inclusion in the prompt.
        
        Args:
            graph: The knowledge graph
            scored_paths: List of paths with reliability scores
            
        Returns:
            List of textual representations of paths
        """
        textual_paths = []
        
        for node_ids, edge_types, score in scored_paths:
            path_text = []
            
            for i in range(len(node_ids)):
                node_id = node_ids[i]
                node_data = graph.nodes[node_id]
                node_text = node_data.get("text", "")
                node_type = node_data.get("type", "")
                
                # Clean the node text
                node_text = re.sub(r'\s+', ' ', node_text).strip()
                
                # Add node information
                path_text.append(f"[{node_type.upper()}] {node_text}")
                
                # Add edge information if not the last node
                if i < len(node_ids) - 1:
                    edge_type = edge_types[i]
                    path_text.append(f"--[{edge_type}]-->")
            
            # Join all elements with spaces
            textual_paths.append(" ".join(path_text))
        
        return textual_paths
    
    def _standard_template(self, query: str, textual_paths: List[str]) -> str:
        """
        Standard prompt template that includes all paths without specific ordering.
        
        Args:
            query: The user query
            textual_paths: List of textual representations of paths
            
        Returns:
            Formatted prompt
        """
        prompt = [
            "Please answer the following query based on the information provided in the relational paths below.",
            "",
            f"Query: {query}",
            "",
            "Relevant Information:"
        ]
        
        for i, path in enumerate(textual_paths):
            prompt.append(f"Path {i+1}: {path}")
        
        prompt.append("")
        prompt.append("Answer:")
        
        return "\n".join(prompt)
    
    def _ascending_template(self, query: str, textual_paths: List[str]) -> str:
        """
        Ascending reliability prompt template (from least to most reliable).
        
        Args:
            query: The user query
            textual_paths: List of textual representations of paths in ascending order of reliability
            
        Returns:
            Formatted prompt
        """
        prompt = [
            "Please answer the following query based on the information provided in the relational paths below.",
            "The paths are arranged in ascending order of reliability, with the most reliable information at the end.",
            "",
            f"Query: {query}",
            "",
            "Relevant Information (ascending reliability):"
        ]
        
        for i, path in enumerate(textual_paths):
            reliability_level = "Low" if i < len(textual_paths) // 3 else "Medium" if i < 2 * len(textual_paths) // 3 else "High"
            prompt.append(f"Path {i+1} [{reliability_level} Reliability]: {path}")
        
        prompt.append("")
        prompt.append("Answer the query comprehensively, prioritizing information from the high reliability paths.")
        prompt.append("Ensure your answer is logical, coherent, and directly addresses the query.")
        prompt.append("")
        prompt.append("Answer:")
        
        return "\n".join(prompt)
    
    def _descending_template(self, query: str, textual_paths: List[str]) -> str:
        """
        Descending reliability prompt template (from most to least reliable).
        
        Args:
            query: The user query
            textual_paths: List of textual representations of paths in descending order of reliability
            
        Returns:
            Formatted prompt
        """
        prompt = [
            "Please answer the following query based on the information provided in the relational paths below.",
            "The paths are arranged in descending order of reliability, with the most reliable information at the beginning.",
            "",
            f"Query: {query}",
            "",
            "Relevant Information (descending reliability):"
        ]
        
        for i, path in enumerate(textual_paths):
            reliability_level = "High" if i < len(textual_paths) // 3 else "Medium" if i < 2 * len(textual_paths) // 3 else "Low"
            prompt.append(f"Path {i+1} [{reliability_level} Reliability]: {path}")
        
        prompt.append("")
        prompt.append("Answer the query comprehensively, prioritizing information from the high reliability paths.")
        prompt.append("Ensure your answer is logical, coherent, and directly addresses the query.")
        prompt.append("")
        prompt.append("Answer:")
        
        return "\n".join(prompt)
    
    def _random_template(self, query: str, textual_paths: List[str]) -> str:
        """
        Random order prompt template (no specific ordering of paths).
        
        Args:
            query: The user query
            textual_paths: List of textual representations of paths
            
        Returns:
            Formatted prompt
        """
        prompt = [
            "Please answer the following query based on the information provided in the relational paths below.",
            "",
            f"Query: {query}",
            "",
            "Relevant Information (in no particular order):"
        ]
        
        for i, path in enumerate(textual_paths):
            prompt.append(f"Path {i+1}: {path}")
        
        prompt.append("")
        prompt.append("Answer the query comprehensively based on the provided information.")
        prompt.append("")
        prompt.append("Answer:")
        
        return "\n".join(prompt)
    
    def explain_prompt_engineering(self) -> str:
        """
        Provide an explanation of the prompt engineering strategies.
        
        Returns:
            A detailed explanation of the prompt engineering approach
        """
        explanation = [
            "# Prompt Engineering Explanation",
            "",
            "## PathRAG Prompting Strategy",
            "",
            "Our prompt engineering approach leverages key insights about large language models (LLMs):",
            "",
            "1. **Path-Based Organization:**",
            "   - Information is structured as relational paths rather than flat chunks",
            "   - Each path shows explicit connections between entities",
            "   - This preserves the graph structure in a textual format LLMs can understand",
            "",
            "2. **Reliability-Based Ordering:**",
            "   - Paths are arranged by reliability score to optimize LLM attention",
            "   - We use **ascending reliability order** (least to most reliable)",
            "   - This leverages the 'recency bias' of LLMs, which tend to focus more on",
            "     information at the beginning and end of prompts ('lost in the middle' effect)",
            "",
            "3. **Template Components:**",
            "   - Clear query statement at the beginning",
            "   - Path information with explicit reliability indicators",
            "   - Specific instruction to prioritize high-reliability paths",
            "   - Guidance for comprehensive, logical, and coherent answers",
            "",
            "4. **Alternative Templates:**",
            "   - Descending: Most reliable paths first (helps with very long contexts)",
            "   - Random: No specific ordering (baseline for comparison)",
            "   - Standard: Simplified version without reliability indicators",
            "",
            "This prompting approach guides the LLM to generate more accurate and coherent responses",
            "while reducing the impact of redundant or less reliable information."
        ]
        
        return "\n".join(explanation)
