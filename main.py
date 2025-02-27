import os
import time
import argparse
from typing import List, Dict, Any

from agents.graph_construction_expert import GraphConstructionExpert
from agents.node_retrieval_specialist import NodeRetrievalSpecialist
from agents.path_analysis_engineer import PathAnalysisEngineer
from agents.reliability_scoring_architect import ReliabilityScoringArchitect
from agents.prompt_engineering_specialist import PromptEngineeringSpecialist
from agents.evaluation_researcher import EvaluationResearcher

from utils.data_loader import load_sample_dataset, load_pathrag_paper
from utils.config import get_config, setup_config
from visualization.graph_visualizer import GraphVisualizer

class PathRAGDemo:
    """
    The main PathRAG demonstration class that coordinates the team of agents.
    """
    
    def __init__(self, use_paper_data: bool = True, visualization_dir: str = None, env_file: str = None):
        """
        Initialize the PathRAG demonstration.
        
        Args:
            use_paper_data: Whether to use the PathRAG paper as data source
            visualization_dir: Directory to save visualizations (overrides config)
            env_file: Path to custom environment file
        """
        # Load configuration
        self.config = setup_config(env_file) if env_file else get_config()
        
        # Override visualization directory if provided
        self.visualization_dir = visualization_dir or self.config.visualization_directory
        os.makedirs(self.visualization_dir, exist_ok=True)
        
        # Display configuration
        self.config.display_config()
        
        # Initialize the team of agents
        print("\n=== 👩‍🔬 INITIALIZING AGENT TEAM ===")
        
        if self.config.enable_graph_construction:
            self.graph_expert = GraphConstructionExpert()
            print("✅ Graph Construction Expert initialized")
        else:
            print("❌ Graph Construction Expert disabled")
        
        if self.config.enable_node_retrieval:
            self.node_specialist = NodeRetrievalSpecialist()
            print("✅ Node Retrieval Specialist initialized")
        else:
            print("❌ Node Retrieval Specialist disabled")
        
        if self.config.enable_path_analysis:
            self.path_engineer = PathAnalysisEngineer(
                decay_rate=self.config.decay_rate, 
                pruning_threshold=self.config.pruning_threshold,
                max_path_length=self.config.max_path_length
            )
            print("✅ Path Analysis Engineer initialized")
        else:
            print("❌ Path Analysis Engineer disabled")
        
        if self.config.enable_reliability_scoring:
            self.reliability_architect = ReliabilityScoringArchitect(max_paths=self.config.max_paths)
            print("✅ Reliability Scoring Architect initialized")
        else:
            print("❌ Reliability Scoring Architect disabled")
        
        if self.config.enable_prompt_engineering:
            self.prompt_specialist = PromptEngineeringSpecialist(template_type="ascending")
            print("✅ Prompt Engineering Specialist initialized")
        else:
            print("❌ Prompt Engineering Specialist disabled")
        
        if self.config.enable_evaluation:
            self.evaluation_researcher = EvaluationResearcher()
            print("✅ Evaluation Researcher initialized")
        else:
            print("❌ Evaluation Researcher disabled")
        
        # Run the demo with paper or sample data as specified
        self.use_paper_data = use_paper_data

    def run_demonstration(self, query: str) -> Dict[str, Any]:
        """
        Run the full PathRAG demonstration with all agents.
        
        Args:
            query: The query to process
            
        Returns:
            Dictionary with demonstration results
        """
        results = {}
        
        # Step 1: Graph Construction
        print("\n=== 🔬 GRAPH CONSTRUCTION EXPERT ===")
        print("Building knowledge graph from documents...")
        start_time = time.time()
        if hasattr(self, 'graph_expert'):
            self.graph_expert.load_documents(self.documents)
            graph = self.graph_expert.get_graph()
        else:
            graph = None
        graph_time = time.time() - start_time
        
        # Save graph explanation
        if hasattr(self, 'graph_expert'):
            graph_explanation = self.graph_expert.explain_graph_structure()
            print(graph_explanation)
            results["graph_explanation"] = graph_explanation
        
        # Visualize the graph
        print("\nGenerating graph visualization...")
        if graph:
            graph_viz_path = os.path.join(self.visualization_dir, "knowledge_graph.png")
            GraphVisualizer.visualize_graph(graph, output_path=graph_viz_path)
        
        # Step 2: Node Retrieval
        print("\n=== 🔍 NODE RETRIEVAL SPECIALIST ===")
        print(f"Processing query: '{query}'")
        start_time = time.time()
        if hasattr(self, 'node_specialist'):
            retrieved_nodes = self.node_specialist.identify_relevant_nodes(graph, query)
        else:
            retrieved_nodes = None
        node_time = time.time() - start_time
        
        # Save node retrieval explanation
        if hasattr(self, 'node_specialist'):
            node_explanation = self.node_specialist.explain_node_retrieval(graph, query, retrieved_nodes)
            print(node_explanation)
            results["node_explanation"] = node_explanation
        
        # Step 3: Path Analysis
        print("\n=== 🛣️ PATH ANALYSIS ENGINEER ===")
        print("Extracting paths between retrieved nodes...")
        start_time = time.time()
        if hasattr(self, 'path_engineer'):
            paths = self.path_engineer.extract_paths(graph, retrieved_nodes)
        else:
            paths = None
        path_time = time.time() - start_time
        
        # Save path analysis explanation
        if hasattr(self, 'path_engineer'):
            path_explanation = self.path_engineer.explain_path_analysis(graph, paths)
            print(path_explanation)
            results["path_explanation"] = path_explanation
        
        # Visualize the paths
        print("\nGenerating path visualization...")
        if paths:
            path_viz_path = os.path.join(self.visualization_dir, "extracted_paths.png")
            GraphVisualizer.visualize_paths(graph, [(p[0], p[1], 1.0) for p in paths], output_path=path_viz_path)
        
        # Step 4: Reliability Scoring
        print("\n=== ⭐ RELIABILITY SCORING ARCHITECT ===")
        print("Calculating reliability scores for paths...")
        start_time = time.time()
        if hasattr(self, 'reliability_architect'):
            resources = self.reliability_architect.generate_resource_values(graph, retrieved_nodes)
            scored_paths = self.reliability_architect.calculate_reliability_scores(
                graph, paths, resources
            )
        else:
            scored_paths = None
        reliability_time = time.time() - start_time
        
        # Save reliability scoring explanation
        if hasattr(self, 'reliability_architect'):
            reliability_explanation = self.reliability_architect.explain_reliability_scoring(graph, scored_paths)
            print(reliability_explanation)
            results["reliability_explanation"] = reliability_explanation
        
        # Step 5: Prompt Engineering
        print("\n=== 💬 PROMPT ENGINEERING SPECIALIST ===")
        print("Generating LLM prompt with path-based structure...")
        start_time = time.time()
        if hasattr(self, 'prompt_specialist'):
            prompt = self.prompt_specialist.generate_prompt(query, graph, scored_paths)
        else:
            prompt = None
        
        # Basic simulated answer
        answer = f"This is a simulated answer to the query: '{query}'. In a real implementation, this would be generated by an LLM using the prompt."
        prompt_time = time.time() - start_time
        
        # Save prompt engineering explanation
        if hasattr(self, 'prompt_specialist'):
            prompt_explanation = self.prompt_specialist.explain_prompt_engineering()
            print(prompt_explanation)
            results["prompt_explanation"] = prompt_explanation
        results["prompt"] = prompt
        results["answer"] = answer
        
        # Step 6: Evaluation
        print("\n=== 📊 EVALUATION RESEARCHER ===")
        print("Evaluating the quality and efficiency of the answer...")
        start_time = time.time()
        
        # Track token efficiency for PathRAG
        if hasattr(self, 'evaluation_researcher'):
            pathrag_efficiency = self.evaluation_researcher.measure_token_efficiency(
                "PathRAG", prompt, answer, start_time, time.time()
            )
        
        # Simulate efficiency for traditional RAG
        traditional_prompt = "Traditional RAG prompt would be much longer due to redundant information..."
        traditional_answer = "Traditional RAG answer for comparison."
        traditional_efficiency = self.evaluation_researcher.measure_token_efficiency(
            "Traditional RAG", traditional_prompt, traditional_answer, start_time, time.time()
        )
        
        # Compare methods
        efficiency_comparison = self.evaluation_researcher.compare_methods(
            ["Traditional RAG", "PathRAG"]
        )
        
        # Evaluate answer quality
        evaluation_scores = self.evaluation_researcher.evaluate_answer(query, answer)
        eval_time = time.time() - start_time
        
        # Save evaluation explanation
        evaluation_explanation = self.evaluation_researcher.explain_evaluation(
            evaluation_scores, efficiency_comparison
        )
        print(evaluation_explanation)
        results["evaluation_explanation"] = evaluation_explanation
        
        # Create interactive visualization
        print("\n=== 🌐 CREATING INTERACTIVE VISUALIZATION ===")
        if graph and scored_paths:
            interactive_viz_path = os.path.join(self.visualization_dir, "interactive_visualization.html")
            GraphVisualizer.create_interactive_visualization(
                graph, scored_paths, output_path=interactive_viz_path
            )
        
        # Compile timing information
        results["timing"] = {
            "graph_construction": graph_time,
            "node_retrieval": node_time,
            "path_analysis": path_time,
            "reliability_scoring": reliability_time,
            "prompt_engineering": prompt_time,
            "evaluation": eval_time,
            "total": graph_time + node_time + path_time + reliability_time + prompt_time + eval_time
        }
        
        print("\n=== 🎉 DEMONSTRATION COMPLETE ===")
        print(f"Results saved to {self.visualization_dir}")
        print(f"Total processing time: {results['timing']['total']:.2f} seconds")
        
        return results

def main():
    """Main function to run the PathRAG demonstration."""
    parser = argparse.ArgumentParser(description='PathRAG Demonstration')
    parser.add_argument('--query', type=str, default="How does PathRAG reduce redundancy in graph-based retrieval?",
                      help='Query to process with PathRAG')
    parser.add_argument('--sample-data', action='store_true',
                      help='Use sample data instead of PathRAG paper')
    parser.add_argument('--viz-dir', type=str, default="visualization_output",
                      help='Directory to save visualizations')
    parser.add_argument('--env-file', type=str, default=None,
                      help='Path to custom environment file')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🚀 PathRAG Demonstration: Pruning Graph-based RAG with Relational Paths")
    print("=" * 80)
    
    # Initialize and run the demonstration
    demo = PathRAGDemo(use_paper_data=not args.sample_data, visualization_dir=args.viz_dir, env_file=args.env_file)
    if demo.use_paper_data:
        print("Loading data from PathRAG paper...")
        demo.documents = load_pathrag_paper()
    else:
        print("Loading sample dataset...")
        demo.documents = load_sample_dataset()
    
    print(f"✅ Loaded {len(demo.documents)} documents")
    
    results = demo.run_demonstration(args.query)
    
    # Display conclusion
    print("\n" + "=" * 80)
    print("📑 CONCLUSION")
    print("=" * 80)
    print("The PathRAG demonstration has shown how graph-based RAG can be improved by:")
    print("1. Building an indexing graph that captures entity relationships")
    print("2. Using efficient node retrieval based on query keywords")
    print("3. Applying flow-based pruning to identify key relational paths")
    print("4. Scoring paths by reliability to prioritize the most relevant information")
    print("5. Structuring prompts with paths in ascending reliability order")
    print("")
    print("These improvements lead to:")
    print("- Reduced redundancy in retrieved information")
    print("- Lower token consumption")
    print("- More logical and coherent answers")
    print("- Better handling of complex queries")
    print("")
    print("To learn more, see the visualizations and explanations in the output directory.")
    print("=" * 80)

if __name__ == "__main__":
    main()
