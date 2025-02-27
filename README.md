# PathRAG Demo: Pruning Graph-based Retrieval Augmented Generation

This project demonstrates the PathRAG implementation using a team of 6 AI agents at PhD level to show how PathRAG prunes graph-based RAG using relational paths.

The development of this repository was inspired by the "PathRAG: Pruning Graph-based Retrieval Augmented Generation with Relational Paths" paper. To read the full paper, visit https://arxiv.org/pdf/2502.14902

## Team Structure

1. **Graph Construction Expert**: Builds and explains the indexing graph structure from text data
2. **Node Retrieval Specialist**: Demonstrates keyword extraction and relevant node identification
3. **Path Analysis Engineer**: Implements the flow-based pruning algorithm with distance awareness
4. **Reliability Scoring Architect**: Calculates and explains path reliability scores
5. **Prompt Engineering Specialist**: Shows path-based prompting techniques with ascending reliability order
6. **Evaluation Researcher**: Measures performance across comprehensiveness, diversity, logicality, relevance, and coherence

## Installation

```bash
# Create and activate a virtual environment
python -m venv .venv
\.venv\Scripts\activate  # Windows

# Ensure you have the latest pip, setuptools, and wheel
pip install --upgrade pip setuptools wheel

# Install dependencies
pip install -r requirements.txt

# Download SpaCy model
python -m spacy download en_core_web_sm

# Configure environment variables
cp .env.example .env
# Edit .env with your API keys
```

> **Note:** If you encounter any installation issues, particularly with numpy or other packages requiring compilation, try installing the problematic packages individually first before running the full requirements installation.

## Environment Configuration

PathRAG uses environment variables for API keys and configuration settings. The framework supports multiple AI providers for different components:

1. Copy the example configuration file:
   ```bash
   cp .env.example .env
   ```

2. Edit the `.env` file with your API keys:
   - `OPENAI_API_KEY`: Required for LLM components
   - `GOOGLE_API_KEY`: Used for semantic search and embeddings
   - Other optional keys for alternate providers

3. Configure agent settings in the `.env` file:
   - Enable/disable specific agents
   - Adjust graph parameters
   - Customize visualization settings

## Running the Demo

```bash
python main.py
```

## Project Structure

- `main.py`: Main script to run the PathRAG demonstration
- `test.py`: Test script to verify basic functionality
- `agents/`: Contains the implementation of each AI agent
  - `graph_construction_expert.py`: Builds indexing graph from text documents
  - `node_retrieval_specialist.py`: Extracts keywords and identifies relevant nodes
  - `path_analysis_engineer.py`: Implements flow-based pruning algorithm
  - `reliability_scoring_architect.py`: Calculates reliability scores for paths
  - `prompt_engineering_specialist.py`: Creates path-based prompts for LLMs
  - `evaluation_researcher.py`: Measures performance across multiple dimensions
- `utils/`: Utility functions for data processing and manipulation
  - `data_loader.py`: Functions for loading and processing text data
- `visualization/`: Code for visualizing the graph and pruned paths
  - `graph_visualizer.py`: Static and interactive graph visualization tools

## Key Features

### 1. Graph-Based Knowledge Representation
- Constructs knowledge graphs from document collections
- Represents entities and their relationships as nodes and edges
- Preserves semantic connections between information pieces

### 2. Flow-Based Path Pruning
- Implements resource propagation algorithm to identify important paths
- Uses distance awareness to prioritize direct connections
- Reduces redundancy while maintaining information quality

### 3. Reliability Scoring
- Assigns reliability scores to extracted paths
- Uses a combination of resource values and path characteristics
- Provides explainable scoring metrics for transparency

### 4. Path-Based Prompt Engineering
- Generates prompts with paths ordered by reliability
- Addresses the "lost in the middle" problem in LLM attention
- Offers multiple template strategies for different scenarios

### 5. Comprehensive Evaluation
- Evaluates across five dimensions: comprehensiveness, diversity, logicality, relevance, and coherence
- Provides token efficiency comparisons with traditional RAG
- Includes visualization of performance metrics

### 6. Rich Visualization
- Static graph visualization with node highlighting
- Path extraction visualization to show pruning results
- Interactive 3D visualization for in-depth exploration

## Research Implementation Details

The PathRAG implementation follows the approach described in the paper with these key components:

1. **Resource Propagation**: Resources flow from starting nodes through the graph with decay over distance
2. **Path Extraction**: Paths are extracted between relevant nodes based on resource values
3. **Reliability Scoring**: Paths are scored using a combination of resource values and path characteristics
4. **Prompt Construction**: Paths are ordered by reliability in the final LLM prompt

For more technical details, see the code documentation and the original paper.

## Example Usage

```python
from pathrag_demo import PathRAGDemo

# Initialize the PathRAG demonstration
demo = PathRAGDemo()

# Run the complete demonstration with a query
demo.run_demo("How does PathRAG reduce redundancy in graph-based retrieval?")

# Access individual components
graph = demo.graph_expert.get_graph()
nodes = demo.node_specialist.retrieve_nodes(graph, "my query")
paths = demo.path_engineer.extract_paths(graph, nodes)
```

## License

This project is available under the MIT License.
