import os
import dotenv
from typing import Dict, Any, Optional

class PathRAGConfig:
    """
    Configuration manager for the PathRAG framework.
    Loads settings from .env file and provides access to configuration values.
    """
    
    def __init__(self, env_file: str = ".env"):
        """
        Initialize the configuration manager.
        
        Args:
            env_file: Path to the environment file
        """
        # Load environment variables
        self.env_file = env_file
        dotenv.load_dotenv(env_file)
        
        # API keys
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.cohere_api_key = os.getenv("COHERE_API_KEY")
        self.emergenceai_api_key = os.getenv("EMERGENCEAI_API_KEY")
        
        # Model configuration
        self.default_llm_model = os.getenv("DEFAULT_LLM_MODEL", "gpt-4-turbo")
        self.default_embedding_model = os.getenv("DEFAULT_EMBEDDING_MODEL", "text-embedding-ada-002")
        self.max_tokens = int(os.getenv("MAX_TOKENS", "4096"))
        self.temperature = float(os.getenv("TEMPERATURE", "0.1"))
        
        # Path settings
        self.output_directory = os.getenv("OUTPUT_DIRECTORY", "pathrag_output")
        self.visualization_directory = os.getenv("VISUALIZATION_DIRECTORY", "visualization_output")
        self.data_directory = os.getenv("DATA_DIRECTORY", "data")
        
        # Agent configuration
        self.enable_graph_construction = self._parse_bool(os.getenv("ENABLE_GRAPH_CONSTRUCTION", "true"))
        self.enable_node_retrieval = self._parse_bool(os.getenv("ENABLE_NODE_RETRIEVAL", "true"))
        self.enable_path_analysis = self._parse_bool(os.getenv("ENABLE_PATH_ANALYSIS", "true"))
        self.enable_reliability_scoring = self._parse_bool(os.getenv("ENABLE_RELIABILITY_SCORING", "true"))
        self.enable_prompt_engineering = self._parse_bool(os.getenv("ENABLE_PROMPT_ENGINEERING", "true"))
        self.enable_evaluation = self._parse_bool(os.getenv("ENABLE_EVALUATION", "true"))
        
        # Graph settings
        self.max_paths = int(os.getenv("MAX_PATHS", "10"))
        self.decay_rate = float(os.getenv("DECAY_RATE", "0.8"))
        self.pruning_threshold = float(os.getenv("PRUNING_THRESHOLD", "0.01"))
        self.max_path_length = int(os.getenv("MAX_PATH_LENGTH", "4"))
        
        # Visualization settings
        self.enable_interactive_visualization = self._parse_bool(os.getenv("ENABLE_INTERACTIVE_VISUALIZATION", "true"))
        self.enable_path_visualization = self._parse_bool(os.getenv("ENABLE_PATH_VISUALIZATION", "true"))
        self.enable_performance_visualization = self._parse_bool(os.getenv("ENABLE_PERFORMANCE_VISUALIZATION", "true"))
        
        # Create required directories
        os.makedirs(self.output_directory, exist_ok=True)
        os.makedirs(self.visualization_directory, exist_ok=True)
        os.makedirs(self.data_directory, exist_ok=True)
    
    def _parse_bool(self, value: str) -> bool:
        """
        Parse a string as a boolean value.
        
        Args:
            value: String to parse
            
        Returns:
            Boolean value
        """
        return value.lower() in ('true', 't', 'yes', 'y', '1')
    
    def validate(self) -> Dict[str, bool]:
        """
        Validate that all required configuration is present.
        
        Returns:
            Dictionary with validation results
        """
        results = {
            "openai_api_key": bool(self.openai_api_key),
            "google_api_key": bool(self.google_api_key)
        }
        
        # Other validations can be added here
        
        return results
    
    def get_llm_provider(self) -> str:
        """
        Determine which LLM provider to use based on available API keys.
        
        Returns:
            String identifying the LLM provider
        """
        if self.openai_api_key:
            return "openai"
        elif self.anthropic_api_key:
            return "anthropic"
        elif self.groq_api_key:
            return "groq"
        else:
            return "mock"  # Use mock responses when no API keys are available
    
    def get_embedding_provider(self) -> str:
        """
        Determine which embedding provider to use based on available API keys.
        
        Returns:
            String identifying the embedding provider
        """
        if self.openai_api_key:
            return "openai"
        elif self.cohere_api_key:
            return "cohere"
        elif self.google_api_key:
            return "google"
        else:
            return "sentence-transformers"  # Fallback to local models
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get all configuration as a dictionary.
        
        Returns:
            Dictionary of configuration values
        """
        config = {}
        
        # Add all attributes except API keys
        for key, value in self.__dict__.items():
            if not key.endswith("_api_key") and not key == "env_file":
                config[key] = value
        
        # Add provider information
        config["llm_provider"] = self.get_llm_provider()
        config["embedding_provider"] = self.get_embedding_provider()
        
        return config
    
    def display_config(self) -> None:
        """
        Display the current configuration.
        """
        config = self.get_config()
        
        print("\n=== 🔧 PathRAG Configuration ===")
        
        print("\nProviders:")
        print(f"  LLM Provider: {config['llm_provider']}")
        print(f"  Embedding Provider: {config['embedding_provider']}")
        
        print("\nModel Configuration:")
        print(f"  Default LLM Model: {config['default_llm_model']}")
        print(f"  Default Embedding Model: {config['default_embedding_model']}")
        print(f"  Max Tokens: {config['max_tokens']}")
        print(f"  Temperature: {config['temperature']}")
        
        print("\nEnabled Agents:")
        print(f"  Graph Construction: {config['enable_graph_construction']}")
        print(f"  Node Retrieval: {config['enable_node_retrieval']}")
        print(f"  Path Analysis: {config['enable_path_analysis']}")
        print(f"  Reliability Scoring: {config['enable_reliability_scoring']}")
        print(f"  Prompt Engineering: {config['enable_prompt_engineering']}")
        print(f"  Evaluation: {config['enable_evaluation']}")
        
        print("\nGraph Settings:")
        print(f"  Max Paths: {config['max_paths']}")
        print(f"  Decay Rate: {config['decay_rate']}")
        print(f"  Pruning Threshold: {config['pruning_threshold']}")
        print(f"  Max Path Length: {config['max_path_length']}")
        
        print("\nDirectories:")
        print(f"  Output Directory: {config['output_directory']}")
        print(f"  Visualization Directory: {config['visualization_directory']}")
        print(f"  Data Directory: {config['data_directory']}")


# Global instance
config = PathRAGConfig()


def get_config() -> PathRAGConfig:
    """
    Get the global configuration instance.
    
    Returns:
        PathRAGConfig instance
    """
    return config


def setup_config(env_file: Optional[str] = None) -> PathRAGConfig:
    """
    Set up a new configuration instance.
    
    Args:
        env_file: Path to environment file
        
    Returns:
        PathRAGConfig instance
    """
    global config
    config = PathRAGConfig(env_file) if env_file else PathRAGConfig()
    return config


if __name__ == "__main__":
    # Display configuration when run as script
    config = get_config()
    config.display_config()
