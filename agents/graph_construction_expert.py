import networkx as nx
import spacy
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Tuple, Set

class GraphConstructionExpert:
    """
    The Graph Construction Expert builds the indexing graph structure from text data.
    It extracts entities and relationships to form nodes and edges in the graph.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the Graph Construction Expert.
        
        Args:
            model_name: The name of the sentence transformer model to use for embeddings
        """
        self.nlp = spacy.load("en_core_web_sm")
        self.sentence_model = SentenceTransformer(model_name)
        self.graph = nx.DiGraph()
        
    def load_documents(self, documents: List[str]) -> None:
        """
        Process documents and extract information to build the graph.
        
        Args:
            documents: List of text documents to process
        """
        print("Building graph from documents...")
        doc_nodes = {}
        
        # Create nodes for documents
        for i, doc in enumerate(documents):
            doc_id = f"doc_{i}"
            doc_nodes[doc_id] = {
                "text": doc,
                "type": "document",
                "embedding": self.sentence_model.encode(doc)
            }
            self.graph.add_node(doc_id, **doc_nodes[doc_id])
        
        # Extract entities and create nodes
        entity_nodes = {}
        for doc_id, doc_info in doc_nodes.items():
            entities = self._extract_entities(doc_info["text"])
            
            # Add entities as nodes
            for entity, entity_type in entities:
                entity_id = f"entity_{entity.lower().replace(' ', '_')}"
                
                if entity_id not in entity_nodes:
                    entity_nodes[entity_id] = {
                        "text": entity,
                        "type": "entity",
                        "entity_type": entity_type,
                        "embedding": self.sentence_model.encode(entity)
                    }
                    self.graph.add_node(entity_id, **entity_nodes[entity_id])
                
                # Connect document to entity
                self.graph.add_edge(doc_id, entity_id, relation="contains")
                
            # Connect entities that co-occur in the same document
            entity_ids = [f"entity_{e[0].lower().replace(' ', '_')}" for e in entities]
            for i, e1 in enumerate(entity_ids):
                for e2 in entity_ids[i+1:]:
                    if e1 != e2:
                        # Bidirectional edges for co-occurrence
                        self.graph.add_edge(e1, e2, relation="co_occurs_with")
                        self.graph.add_edge(e2, e1, relation="co_occurs_with")
        
        print(f"Graph construction complete: {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
    
    def _extract_entities(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract entities from text using SpaCy.
        
        Args:
            text: Text to extract entities from
            
        Returns:
            List of (entity_text, entity_type) tuples
        """
        doc = self.nlp(text)
        entities = []
        
        # Extract named entities
        for ent in doc.ents:
            entities.append((ent.text, ent.label_))
        
        # Add noun chunks as additional entities if they're not already included
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text
            if not any(chunk_text == e[0] for e in entities):
                entities.append((chunk_text, "NOUN_CHUNK"))
        
        return entities
    
    def get_graph(self) -> nx.DiGraph:
        """
        Get the constructed graph.
        
        Returns:
            The directed graph with document and entity nodes
        """
        return self.graph
    
    def explain_graph_structure(self) -> str:
        """
        Provide an explanation of the graph structure.
        
        Returns:
            A detailed explanation of the graph structure
        """
        explanation = [
            "# Graph Structure Explanation",
            "",
            "The indexing graph is constructed as follows:",
            "",
            "1. **Node Types:**",
            "   - Document nodes: Represent full text documents",
            "   - Entity nodes: Represent entities extracted from documents (people, places, organizations, etc.)",
            "",
            "2. **Edge Types:**",
            "   - 'contains' edges: Connect documents to entities they contain",
            "   - 'co_occurs_with' edges: Connect entities that appear in the same document",
            "",
            f"3. **Graph Statistics:**",
            f"   - Total nodes: {self.graph.number_of_nodes()}",
            f"   - Total edges: {self.graph.number_of_edges()}",
            f"   - Document nodes: {sum(1 for _, data in self.graph.nodes(data=True) if data.get('type') == 'document')}",
            f"   - Entity nodes: {sum(1 for _, data in self.graph.nodes(data=True) if data.get('type') == 'entity')}",
            "",
            "This graph structure captures both the content of documents and the relationships between entities,",
            "enabling efficient retrieval and path-based analysis for answering complex queries."
        ]
        
        return "\n".join(explanation)
