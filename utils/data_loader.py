import os
import json
import pandas as pd
import pickle
from typing import List, Dict, Any
from pypdf import PdfReader

def load_sample_dataset() -> List[str]:
    """
    Load a sample dataset for demonstration purposes.
    
    Returns:
        List of text documents
    """
    # Sample data - in a real application, this would load from files
    sample_documents = [
        "PathRAG is a method for pruning graph-based retrieval augmented generation using relational paths. "
        "It improves upon traditional RAG by organizing textual information into an indexing graph.",
        
        "Traditional RAG approaches split text into chunks and organize them in a flat structure. "
        "This fails to capture inherent dependencies and structured relationships across texts.",
        
        "Graph-based RAG methods organize textual information into an indexing graph. "
        "Nodes represent entities and edges denote relationships between entities.",
        
        "GraphRAG applies community detection on the graph and summarizes information in each community. "
        "The final answer is generated based on the most query-relevant communities.",
        
        "LightRAG extracts both local and global keywords from input queries and retrieves relevant nodes. "
        "The ego-network information of retrieved nodes is used as retrieval results.",
        
        "The limitation of current graph-based RAG methods lies in the redundancy of retrieved information, "
        "which introduces noise, degrades model performance, and increases token consumption.",
        
        "PathRAG performs key path retrieval among retrieved nodes and converts these paths into textual form. "
        "This reduces redundant information with flow-based pruning.",
        
        "Flow-based pruning with distance awareness identifies key relational paths between retrieved nodes. "
        "The algorithm enjoys low time complexity and assigns reliability scores to paths.",
        
        "PathRAG places textual paths into the prompt in ascending order of reliability scores. "
        "This addresses the 'lost in the middle' issue of LLMs for better answer generation.",
        
        "Experimental results show that PathRAG generates better answers across all evaluation dimensions "
        "compared to state-of-the-art baselines, with significant advantages for larger datasets."
    ]
    
    return sample_documents

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text as a string
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    reader = PdfReader(pdf_path)
    text = ""
    
    for page in reader.pages:
        text += page.extract_text() + "\n"
    
    return text

def split_text_into_documents(text: str, max_length: int = 500) -> List[str]:
    """
    Split a long text into smaller documents.
    
    Args:
        text: The input text
        max_length: Maximum length of each document (in characters)
        
    Returns:
        List of text documents
    """
    # First split by paragraphs
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    
    documents = []
    current_doc = ""
    
    for para in paragraphs:
        # If adding this paragraph exceeds max_length, start a new document
        if len(current_doc) + len(para) > max_length and current_doc:
            documents.append(current_doc.strip())
            current_doc = para
        else:
            current_doc += " " + para if current_doc else para
    
    # Add the last document if not empty
    if current_doc:
        documents.append(current_doc.strip())
    
    return documents

def save_documents(documents: List[str], output_path: str) -> None:
    """
    Save the documents to a file.
    
    Args:
        documents: List of text documents
        output_path: Path to save the documents
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(documents, f, indent=2)
    
    print(f"Saved {len(documents)} documents to {output_path}")

def load_documents(input_path: str) -> List[str]:
    """
    Load documents from a file.
    
    Args:
        input_path: Path to the documents file
        
    Returns:
        List of text documents
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    print(f"Loaded {len(documents)} documents from {input_path}")
    
    return documents

def load_pathrag_paper() -> List[str]:
    """
    Load the PathRAG paper text and split it into documents.
    
    Returns:
        List of text documents from the PathRAG paper
    """
    paper_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                              "PathRAG Paper.pdf")
    
    # Check if the extracted text already exists
    extracted_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                 "data", "pathrag_paper_documents.json")
    
    if os.path.exists(extracted_path):
        return load_documents(extracted_path)
    
    if not os.path.exists(paper_path):
        print(f"PathRAG paper not found at {paper_path}")
        return load_sample_dataset()
    
    # Extract text from PDF
    paper_text = extract_text_from_pdf(paper_path)
    
    # Split into documents
    documents = split_text_into_documents(paper_text)
    
    # Save documents
    os.makedirs(os.path.dirname(extracted_path), exist_ok=True)
    save_documents(documents, extracted_path)
    
    return documents
