import networkx as nx
import numpy as np
from typing import List, Dict, Set, Tuple
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class NodeRetrievalSpecialist:
    """
    The Node Retrieval Specialist is responsible for extracting keywords from queries
    and identifying relevant nodes in the graph.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the Node Retrieval Specialist.
        
        Args:
            model_name: The name of the sentence transformer model to use for embeddings
        """
        self.nlp = spacy.load("en_core_web_sm")
        self.sentence_model = SentenceTransformer(model_name)
    
    def extract_keywords(self, query: str) -> Tuple[List[str], List[str]]:
        """
        Extract both global and local keywords from the query.
        
        Args:
            query: The user query
            
        Returns:
            Tuple of (global_keywords, local_keywords)
        """
        doc = self.nlp(query)
        
        # Extract global keywords (nouns, named entities)
        global_keywords = []
        for ent in doc.ents:
            global_keywords.append(ent.text)
        
        # Extract local keywords (important words not captured in global keywords)
        local_keywords = []
        for token in doc:
            if (token.pos_ in ["NOUN", "VERB", "ADJ"] and 
                token.is_alpha and 
                not token.is_stop and 
                token.text.lower() not in [k.lower() for k in global_keywords]):
                local_keywords.append(token.text)
        
        print(f"Extracted global keywords: {global_keywords}")
        print(f"Extracted local keywords: {local_keywords}")
        
        return global_keywords, local_keywords
    
    def identify_relevant_nodes(self, graph: nx.DiGraph, query: str, 
                               top_k: int = 5, similarity_threshold: float = 0.4) -> Set[str]:
        """
        Identify the most relevant nodes in the graph for the given query.
        
        Args:
            graph: The knowledge graph
            query: The user query
            top_k: Number of top nodes to retrieve
            similarity_threshold: Minimum similarity score to consider a node relevant
            
        Returns:
            Set of node IDs that are relevant to the query
        """
        # Extract keywords from query
        global_keywords, local_keywords = self.extract_keywords(query)
        all_keywords = global_keywords + local_keywords
        
        # Encode query
        query_embedding = self.sentence_model.encode(query)
        
        # Find matching nodes based on keywords and semantic similarity
        relevant_nodes = set()
        node_scores = {}
        
        for node_id, node_data in graph.nodes(data=True):
            # Skip nodes without text
            if "text" not in node_data:
                continue
                
            node_text = node_data["text"]
            
            # Keyword matching score
            keyword_score = 0
            for keyword in all_keywords:
                if keyword.lower() in node_text.lower():
                    keyword_score += 1
            
            # Semantic similarity score
            if "embedding" in node_data:
                node_embedding = node_data["embedding"]
                similarity_score = cosine_similarity([query_embedding], [node_embedding])[0][0]
            else:
                node_embedding = self.sentence_model.encode(node_text)
                similarity_score = cosine_similarity([query_embedding], [node_embedding])[0][0]
            
            # Combine scores (normalize keyword score)
            normalized_keyword_score = keyword_score / max(1, len(all_keywords))
            combined_score = 0.4 * normalized_keyword_score + 0.6 * similarity_score
            
            # Store node score
            node_scores[node_id] = combined_score
        
        # Get top-k nodes with scores above threshold
        sorted_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)
        for node_id, score in sorted_nodes:
            if score >= similarity_threshold and len(relevant_nodes) < top_k:
                relevant_nodes.add(node_id)
        
        print(f"Identified {len(relevant_nodes)} relevant nodes:")
        for node_id in relevant_nodes:
            node_text = graph.nodes[node_id].get("text", "")
            print(f"  - {node_id}: {node_text[:50]}...")
        
        return relevant_nodes
    
    def explain_node_retrieval(self, graph: nx.DiGraph, query: str, 
                              retrieved_nodes: Set[str]) -> str:
        """
        Provide an explanation of the node retrieval process.
        
        Args:
            graph: The knowledge graph
            query: The user query
            retrieved_nodes: Set of retrieved node IDs
            
        Returns:
            A detailed explanation of the node retrieval process
        """
        global_keywords, local_keywords = self.extract_keywords(query)
        
        explanation = [
            "# Node Retrieval Explanation",
            "",
            f"For the query: '{query}'",
            "",
            "1. **Keyword Extraction:**",
            f"   - Global Keywords: {', '.join(global_keywords) if global_keywords else 'None'}",
            f"   - Local Keywords: {', '.join(local_keywords) if local_keywords else 'None'}",
            "",
            "2. **Node Identification Process:**",
            "   - Keyword Matching: Nodes containing query keywords are scored higher",
            "   - Semantic Similarity: Cosine similarity between query and node embeddings",
            "   - Combined Score: Weighted combination of keyword and semantic scores",
            "",
            "3. **Retrieved Nodes:**"
        ]
        
        for node_id in retrieved_nodes:
            node_data = graph.nodes[node_id]
            node_text = node_data.get("text", "")
            node_type = node_data.get("type", "unknown")
            
            explanation.append(f"   - Node ID: {node_id}")
            explanation.append(f"     Type: {node_type}")
            explanation.append(f"     Text: {node_text[:100]}..." if len(node_text) > 100 else f"     Text: {node_text}")
            explanation.append("")
        
        explanation.append("This retrieval approach combines symbolic (keyword-based) and semantic (embedding-based)")
        explanation.append("methods to identify the most relevant nodes for the given query.")
        
        return "\n".join(explanation)
