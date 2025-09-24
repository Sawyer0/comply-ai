"""
RAG (Retrieval-Augmented Generation) system for compliance AI.

This module provides dynamic access to regulatory knowledge, enabling models
to provide expert-level compliance guidance while maintaining current regulatory
information through retrieval rather than training.
"""

from .core.vector_store import VectorStore, Document, SearchResult
from .core.embeddings import EmbeddingModel
from .core.retriever import DocumentRetriever
from .integration.model_enhancement import RAGContextEnhancer, RAGEnhancedContext

__all__ = [
    "VectorStore",
    "Document", 
    "SearchResult",
    "EmbeddingModel",
    "DocumentRetriever",
    "RAGContextEnhancer",
    "RAGEnhancedContext"
]