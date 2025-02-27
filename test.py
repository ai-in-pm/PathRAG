import os
import networkx as nx
from agents.graph_construction_expert import GraphConstructionExpert
from agents.node_retrieval_specialist import NodeRetrievalSpecialist
from agents.path_analysis_engineer import PathAnalysisEngineer
from agents.reliability_scoring_architect import ReliabilityScoringArchitect
from agents.prompt_engineering_specialist import PromptEngineeringSpecialist
from utils.data_loader import load_sample_dataset

def test_basic_functionality():
    """Test the basic functionality of the PathRAG components."""
    print("Testing basic PathRAG functionality...")
    
    # Test data loading
    documents = load_sample_dataset()
    print(f"✓ Loaded {len(documents)} sample documents")
    
    # Test graph construction
    graph_expert = GraphConstructionExpert()
    graph_expert.load_documents(documents)
    graph = graph_expert.get_graph()
    print(f"✓ Built graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    
    # Test node retrieval
    query = "How does PathRAG reduce redundancy in graph-based retrieval?"
    node_specialist = NodeRetrievalSpecialist()
    retrieved_nodes = node_specialist.identify_relevant_nodes(graph, query)
    print(f"✓ Retrieved {len(retrieved_nodes)} nodes relevant to the query")
    
    # Test path analysis
    path_engineer = PathAnalysisEngineer()
    paths = path_engineer.extract_paths(graph, retrieved_nodes)
    print(f"✓ Extracted {len(paths)} paths between retrieved nodes")
    
    # Test reliability scoring
    reliability_architect = ReliabilityScoringArchitect()
    resources = reliability_architect.generate_resource_values(graph, retrieved_nodes)
    scored_paths = reliability_architect.calculate_reliability_scores(graph, paths, resources)
    print(f"✓ Calculated reliability scores for {len(scored_paths)} paths")
    
    # Test prompt engineering
    prompt_specialist = PromptEngineeringSpecialist()
    prompt = prompt_specialist.generate_prompt(query, graph, scored_paths)
    print(f"✓ Generated prompt with {len(prompt.split())} words")
    
    print("All basic functionality tests passed!")

if __name__ == "__main__":
    test_basic_functionality()
