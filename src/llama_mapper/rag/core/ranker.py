"""
Document ranking and re-ranking for RAG system.

Provides interfaces and implementations for ranking and re-ranking retrieved documents.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import logging
import numpy as np
from .vector_store import SearchResult, Document

logger = logging.getLogger(__name__)


class DocumentRanker(ABC):
    """Abstract interface for document ranking."""
    
    @abstractmethod
    async def rank(self, query: str, documents: List[SearchResult]) -> List[SearchResult]:
        """Rank documents based on query relevance.
        
        Args:
            query: Search query
            documents: List of documents to rank
            
        Returns:
            List of ranked documents
        """
        pass
    
    @abstractmethod
    async def rerank(self, query: str, documents: List[SearchResult], 
                    top_k: int = 10) -> List[SearchResult]:
        """Re-rank documents and return top-k results.
        
        Args:
            query: Search query
            documents: List of documents to re-rank
            top_k: Number of top results to return
            
        Returns:
            List of top-k ranked documents
        """
        pass


class CrossEncoderRanker(DocumentRanker):
    """Cross-encoder based document ranking."""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 device: str = "cpu", batch_size: int = 32):
        """Initialize cross-encoder ranker.
        
        Args:
            model_name: Name of the cross-encoder model
            device: Device to run model on
            batch_size: Batch size for processing
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self.logger = logging.getLogger(__name__)
    
    async def _get_model(self):
        """Get or load the cross-encoder model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                
                self._model = CrossEncoder(self.model_name, device=self.device)
                self.logger.info(f"Loaded CrossEncoder model: {self.model_name}")
            except ImportError:
                raise ImportError("sentence-transformers is required. Install with: pip install sentence-transformers")
            except Exception as e:
                self.logger.error(f"Failed to load CrossEncoder model: {e}")
                raise
        
        return self._model
    
    async def rank(self, query: str, documents: List[SearchResult]) -> List[SearchResult]:
        """Rank documents using cross-encoder."""
        try:
            model = await self._get_model()
            
            if not documents:
                return []
            
            # Prepare query-document pairs
            pairs = [(query, doc.document.content) for doc in documents]
            
            # Get relevance scores
            scores = model.predict(pairs)
            
            # Update document scores
            for i, doc in enumerate(documents):
                doc.score = float(scores[i])
            
            # Sort by score
            documents.sort(key=lambda x: x.score, reverse=True)
            
            return documents
            
        except Exception as e:
            self.logger.error(f"Failed to rank documents with cross-encoder: {e}")
            return documents
    
    async def rerank(self, query: str, documents: List[SearchResult], 
                    top_k: int = 10) -> List[SearchResult]:
        """Re-rank documents and return top-k results."""
        try:
            # Rank all documents
            ranked_docs = await self.rank(query, documents)
            
            # Return top-k
            return ranked_docs[:top_k]
            
        except Exception as e:
            self.logger.error(f"Failed to re-rank documents: {e}")
            return documents[:top_k]


class BM25Ranker(DocumentRanker):
    """BM25-based document ranking."""
    
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        """Initialize BM25 ranker.
        
        Args:
            k1: BM25 k1 parameter
            b: BM25 b parameter
        """
        self.k1 = k1
        self.b = b
        self.logger = logging.getLogger(__name__)
    
    async def rank(self, query: str, documents: List[SearchResult]) -> List[SearchResult]:
        """Rank documents using BM25."""
        try:
            if not documents:
                return []
            
            # Tokenize query
            query_terms = self._tokenize(query)
            if not query_terms:
                return documents
            
            # Calculate BM25 scores
            for doc in documents:
                doc.score = self._calculate_bm25_score(query_terms, doc.document.content)
            
            # Sort by score
            documents.sort(key=lambda x: x.score, reverse=True)
            
            return documents
            
        except Exception as e:
            self.logger.error(f"Failed to rank documents with BM25: {e}")
            return documents
    
    async def rerank(self, query: str, documents: List[SearchResult], 
                    top_k: int = 10) -> List[SearchResult]:
        """Re-rank documents and return top-k results."""
        try:
            # Rank all documents
            ranked_docs = await self.rank(query, documents)
            
            # Return top-k
            return ranked_docs[:top_k]
            
        except Exception as e:
            self.logger.error(f"Failed to re-rank documents with BM25: {e}")
            return documents[:top_k]
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        import re
        # Remove punctuation and convert to lowercase
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        # Split into words and filter out empty strings
        return [word for word in text.split() if word]
    
    def _calculate_bm25_score(self, query_terms: List[str], document_text: str) -> float:
        """Calculate BM25 score for a document."""
        # Simple BM25 implementation
        doc_terms = self._tokenize(document_text)
        if not doc_terms:
            return 0.0
        
        # Calculate term frequencies
        term_freq = {}
        for term in doc_terms:
            term_freq[term] = term_freq.get(term, 0) + 1
        
        # Calculate document length
        doc_length = len(doc_terms)
        
        # Calculate BM25 score
        score = 0.0
        for term in query_terms:
            if term in term_freq:
                tf = term_freq[term]
                # Simple BM25 formula (without IDF for simplicity)
                score += tf / (tf + self.k1 * (1 - self.b + self.b * (doc_length / 100)))
        
        return score


class HybridRanker(DocumentRanker):
    """Hybrid ranking combining multiple ranking methods."""
    
    def __init__(self, rankers: List[DocumentRanker], weights: List[float] = None):
        """Initialize hybrid ranker.
        
        Args:
            rankers: List of ranking methods
            weights: Weights for each ranking method
        """
        self.rankers = rankers
        self.weights = weights or [1.0 / len(rankers)] * len(rankers)
        
        if len(self.weights) != len(self.rankers):
            raise ValueError("Number of weights must match number of rankers")
        
        self.logger = logging.getLogger(__name__)
    
    async def rank(self, query: str, documents: List[SearchResult]) -> List[SearchResult]:
        """Rank documents using hybrid approach."""
        try:
            if not documents:
                return []
            
            # Get rankings from each ranker
            all_rankings = []
            for ranker in self.rankers:
                ranking = await ranker.rank(query, documents.copy())
                all_rankings.append(ranking)
            
            # Combine rankings
            combined_ranking = self._combine_rankings(all_rankings)
            
            return combined_ranking
            
        except Exception as e:
            self.logger.error(f"Failed to rank documents with hybrid approach: {e}")
            return documents
    
    async def rerank(self, query: str, documents: List[SearchResult], 
                    top_k: int = 10) -> List[SearchResult]:
        """Re-rank documents and return top-k results."""
        try:
            # Rank all documents
            ranked_docs = await self.rank(query, documents)
            
            # Return top-k
            return ranked_docs[:top_k]
            
        except Exception as e:
            self.logger.error(f"Failed to re-rank documents with hybrid approach: {e}")
            return documents[:top_k]
    
    def _combine_rankings(self, rankings: List[List[SearchResult]]) -> List[SearchResult]:
        """Combine multiple rankings into a single ranking."""
        if not rankings:
            return []
        
        # Create a dictionary to store combined scores
        doc_scores = {}
        
        for ranking, weight in zip(rankings, self.weights):
            for i, result in enumerate(ranking):
                doc_id = result.document.id
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {
                        'result': result,
                        'scores': [],
                        'ranks': []
                    }
                
                # Normalize score by rank (higher rank = higher score)
                normalized_score = 1.0 / (i + 1)
                doc_scores[doc_id]['scores'].append(normalized_score * weight)
                doc_scores[doc_id]['ranks'].append(i + 1)
        
        # Calculate combined scores
        combined_results = []
        for doc_id, data in doc_scores.items():
            # Use weighted average of scores
            combined_score = sum(data['scores']) / len(data['scores'])
            
            result = data['result']
            result.score = combined_score
            combined_results.append(result)
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x.score, reverse=True)
        
        return combined_results


class ContextualRanker(DocumentRanker):
    """Contextual ranking that considers user context and preferences."""
    
    def __init__(self, base_ranker: DocumentRanker):
        """Initialize contextual ranker.
        
        Args:
            base_ranker: Base ranking method
        """
        self.base_ranker = base_ranker
        self.logger = logging.getLogger(__name__)
    
    async def rank(self, query: str, documents: List[SearchResult]) -> List[SearchResult]:
        """Rank documents using base ranker."""
        try:
            return await self.base_ranker.rank(query, documents)
        except Exception as e:
            self.logger.error(f"Failed to rank documents: {e}")
            return documents
    
    async def rerank(self, query: str, documents: List[SearchResult], 
                    top_k: int = 10) -> List[SearchResult]:
        """Re-rank documents with contextual adjustments."""
        try:
            # Get base ranking
            ranked_docs = await self.base_ranker.rerank(query, documents, top_k)
            
            # Apply contextual adjustments
            ranked_docs = self._apply_contextual_adjustments(ranked_docs, query)
            
            return ranked_docs
            
        except Exception as e:
            self.logger.error(f"Failed to re-rank documents with contextual adjustments: {e}")
            return documents[:top_k]
    
    def _apply_contextual_adjustments(self, documents: List[SearchResult], 
                                    query: str) -> List[SearchResult]:
        """Apply contextual adjustments to document scores."""
        # Apply contextual adjustments based on query and document features
        adjusted_results = []
        
        for result in results:
            # Boost score for recent documents
            recency_boost = 1.0
            if hasattr(result, 'metadata') and result.metadata.get('pub_date'):
                try:
                    pub_date = datetime.fromisoformat(result.metadata['pub_date'])
                    days_old = (datetime.now() - pub_date).days
                    recency_boost = max(0.8, 1.0 - (days_old / 365) * 0.2)  # Decay over year
                except (ValueError, TypeError):
                    pass
            
            # Boost score for documents matching query domain
            domain_boost = 1.0
            if 'compliance' in query.lower() and 'compliance' in result.content.lower():
                domain_boost = 1.1
            elif 'privacy' in query.lower() and any(term in result.content.lower() for term in ['gdpr', 'privacy', 'pii']):
                domain_boost = 1.15
            
            # Apply adjustments
            adjusted_score = result.score * recency_boost * domain_boost
            adjusted_result = SearchResult(
                content=result.content,
                score=min(adjusted_score, 1.0),  # Cap at 1.0
                metadata=result.metadata
            )
            adjusted_results.append(adjusted_result)
        # - Historical interactions
        # - Domain expertise
        # - Regulatory context
        
        for doc in documents:
            # Example: Boost score for recent documents
            if doc.document.last_updated:
                # Simple recency boost
                doc.score *= 1.1
            
            # Example: Boost score for exact framework matches
            if 'framework' in query.lower() and doc.document.regulatory_framework:
                doc.score *= 1.2
        
        # Re-sort by adjusted scores
        documents.sort(key=lambda x: x.score, reverse=True)
        
        return documents


class DiversityRanker(DocumentRanker):
    """Diversity-aware ranking to ensure diverse results."""
    
    def __init__(self, base_ranker: DocumentRanker, diversity_threshold: float = 0.7):
        """Initialize diversity ranker.
        
        Args:
            base_ranker: Base ranking method
            diversity_threshold: Threshold for diversity filtering
        """
        self.base_ranker = base_ranker
        self.diversity_threshold = diversity_threshold
        self.logger = logging.getLogger(__name__)
    
    async def rank(self, query: str, documents: List[SearchResult]) -> List[SearchResult]:
        """Rank documents using base ranker."""
        try:
            return await self.base_ranker.rank(query, documents)
        except Exception as e:
            self.logger.error(f"Failed to rank documents: {e}")
            return documents
    
    async def rerank(self, query: str, documents: List[SearchResult], 
                    top_k: int = 10) -> List[SearchResult]:
        """Re-rank documents ensuring diversity."""
        try:
            # Get base ranking
            ranked_docs = await self.base_ranker.rerank(query, documents, top_k * 2)
            
            # Apply diversity filtering
            diverse_docs = self._apply_diversity_filtering(ranked_docs, top_k)
            
            return diverse_docs
            
        except Exception as e:
            self.logger.error(f"Failed to re-rank documents with diversity filtering: {e}")
            return documents[:top_k]
    
    def _apply_diversity_filtering(self, documents: List[SearchResult], 
                                  top_k: int) -> List[SearchResult]:
        """Apply diversity filtering to ensure diverse results."""
        if len(documents) <= top_k:
            return documents
        
        diverse_docs = []
        used_frameworks = set()
        used_industries = set()
        used_types = set()
        
        for doc in documents:
            if len(diverse_docs) >= top_k:
                break
            
            # Check diversity constraints
            is_diverse = True
            
            # Check framework diversity
            if doc.document.regulatory_framework:
                if doc.document.regulatory_framework in used_frameworks:
                    is_diverse = False
            
            # Check industry diversity
            if doc.document.industry:
                if doc.document.industry in used_industries:
                    is_diverse = False
            
            # Check document type diversity
            if doc.document.document_type:
                if doc.document.document_type in used_types:
                    is_diverse = False
            
            # Add document if diverse or if we need more results
            if is_diverse or len(diverse_docs) < top_k // 2:
                diverse_docs.append(doc)
                
                # Update used sets
                if doc.document.regulatory_framework:
                    used_frameworks.add(doc.document.regulatory_framework)
                if doc.document.industry:
                    used_industries.add(doc.document.industry)
                if doc.document.document_type:
                    used_types.add(doc.document.document_type)
        
        return diverse_docs
