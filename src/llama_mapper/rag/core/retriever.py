"""
Document retrieval logic for RAG system.

Provides interfaces and implementations for retrieving relevant documents.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import logging
import numpy as np
from .vector_store import VectorStore, SearchResult, Document
from .embeddings import EmbeddingModel

logger = logging.getLogger(__name__)


class DocumentRetriever(ABC):
    """Abstract interface for document retrieval."""
    
    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 10, 
                      filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve
            filters: Optional filters for metadata
            
        Returns:
            List of search results ordered by relevance
        """
        pass
    
    @abstractmethod
    async def retrieve_with_context(self, query: str, context: Dict[str, Any], 
                                   top_k: int = 10) -> List[SearchResult]:
        """Retrieve documents with additional context.
        
        Args:
            query: Search query
            context: Additional context for retrieval
            top_k: Number of documents to retrieve
            
        Returns:
            List of search results ordered by relevance
        """
        pass


class SemanticRetriever(DocumentRetriever):
    """Semantic retrieval using vector similarity search."""
    
    def __init__(self, vector_store: VectorStore, embedding_model: EmbeddingModel):
        """Initialize semantic retriever.
        
        Args:
            vector_store: Vector store for document storage
            embedding_model: Model for generating embeddings
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.logger = logging.getLogger(__name__)
    
    async def retrieve(self, query: str, top_k: int = 10, 
                      filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Retrieve relevant documents using semantic search."""
        try:
            # Generate query embedding
            query_embedding = await self.embedding_model.embed_text(query)
            
            # Search vector store
            results = await self.vector_store.search(
                query=query,
                filters=filters,
                top_k=top_k
            )
            
            self.logger.info(f"Retrieved {len(results)} documents for query: {query}")
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve documents: {e}")
            return []
    
    async def retrieve_with_context(self, query: str, context: Dict[str, Any], 
                                   top_k: int = 10) -> List[SearchResult]:
        """Retrieve documents with additional context."""
        try:
            # Build enhanced query with context
            enhanced_query = self._build_enhanced_query(query, context)
            
            # Apply context-based filters
            filters = self._build_context_filters(context)
            
            # Retrieve documents
            results = await self.retrieve(enhanced_query, top_k, filters)
            
            # Post-process results based on context
            results = self._post_process_results(results, context)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve documents with context: {e}")
            return []
    
    def _build_enhanced_query(self, query: str, context: Dict[str, Any]) -> str:
        """Build enhanced query with context information."""
        enhanced_parts = [query]
        
        # Add regulatory framework context
        if context.get('regulatory_framework'):
            enhanced_parts.append(f"regulatory framework: {context['regulatory_framework']}")
        
        # Add industry context
        if context.get('industry'):
            enhanced_parts.append(f"industry: {context['industry']}")
        
        # Add document type context
        if context.get('document_types'):
            doc_types = ", ".join(context['document_types'])
            enhanced_parts.append(f"document types: {doc_types}")
        
        # Add user role context
        if context.get('user_role'):
            enhanced_parts.append(f"user role: {context['user_role']}")
        
        return " ".join(enhanced_parts)
    
    def _build_context_filters(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Build filters based on context."""
        filters = {}
        
        # Filter by regulatory framework
        if context.get('regulatory_framework'):
            filters['regulatory_framework'] = context['regulatory_framework']
        
        # Filter by industry
        if context.get('industry'):
            filters['industry'] = context['industry']
        
        # Filter by document type
        if context.get('document_types'):
            filters['document_type'] = {"$in": context['document_types']}
        
        # Filter by date range
        if context.get('date_range'):
            date_range = context['date_range']
            if 'start_date' in date_range:
                filters['created_at'] = {"$gte": date_range['start_date']}
            if 'end_date' in date_range:
                filters['created_at'] = {"$lte": date_range['end_date']}
        
        return filters
    
    def _post_process_results(self, results: List[SearchResult], 
                             context: Dict[str, Any]) -> List[SearchResult]:
        """Post-process results based on context."""
        # Apply context-based scoring adjustments
        for result in results:
            # Boost score for exact framework matches
            if (context.get('regulatory_framework') and 
                result.document.regulatory_framework == context['regulatory_framework']):
                result.score *= 1.2
            
            # Boost score for exact industry matches
            if (context.get('industry') and 
                result.document.industry == context['industry']):
                result.score *= 1.1
        
        # Re-sort by adjusted scores
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results


class HybridRetriever(DocumentRetriever):
    """Hybrid retrieval combining semantic and keyword search."""
    
    def __init__(self, vector_store: VectorStore, embedding_model: EmbeddingModel,
                 keyword_weight: float = 0.3, semantic_weight: float = 0.7):
        """Initialize hybrid retriever.
        
        Args:
            vector_store: Vector store for semantic search
            embedding_model: Model for generating embeddings
            keyword_weight: Weight for keyword search results
            semantic_weight: Weight for semantic search results
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.keyword_weight = keyword_weight
        self.semantic_weight = semantic_weight
        self.logger = logging.getLogger(__name__)
    
    async def retrieve(self, query: str, top_k: int = 10, 
                      filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Retrieve documents using hybrid approach."""
        try:
            # Get semantic results
            semantic_results = await self._semantic_search(query, top_k, filters)
            
            # Get keyword results
            keyword_results = await self._keyword_search(query, top_k, filters)
            
            # Combine and rank results
            combined_results = self._combine_results(semantic_results, keyword_results, top_k)
            
            self.logger.info(f"Retrieved {len(combined_results)} documents using hybrid approach")
            return combined_results
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve documents with hybrid approach: {e}")
            return []
    
    async def retrieve_with_context(self, query: str, context: Dict[str, Any], 
                                   top_k: int = 10) -> List[SearchResult]:
        """Retrieve documents with context using hybrid approach."""
        try:
            # Build enhanced query
            enhanced_query = self._build_enhanced_query(query, context)
            
            # Apply context filters
            filters = self._build_context_filters(context)
            
            # Retrieve with hybrid approach
            results = await self.retrieve(enhanced_query, top_k, filters)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve documents with context: {e}")
            return []
    
    async def _semantic_search(self, query: str, top_k: int, 
                              filters: Optional[Dict[str, Any]]) -> List[SearchResult]:
        """Perform semantic search."""
        try:
            results = await self.vector_store.search(query, filters, top_k)
            return results
        except Exception as e:
            self.logger.error(f"Semantic search failed: {e}")
            return []
    
    async def _keyword_search(self, query: str, top_k: int, 
                             filters: Optional[Dict[str, Any]]) -> List[SearchResult]:
        """Perform keyword search."""
        try:
            # Simple keyword search implementation
            # This would typically use a full-text search engine like Elasticsearch
            # For now, we'll use a basic implementation
            
            # Extract keywords from query
            keywords = query.lower().split()
            
            # Search for documents containing these keywords
            # This is a simplified implementation
            results = await self.vector_store.search(query, filters, top_k)
            
            # Apply keyword-based scoring
            for result in results:
                content_lower = result.document.content.lower()
                keyword_matches = sum(1 for keyword in keywords if keyword in content_lower)
                keyword_score = keyword_matches / len(keywords) if keywords else 0
                result.score = keyword_score
            
            return results
            
        except Exception as e:
            self.logger.error(f"Keyword search failed: {e}")
            return []
    
    def _combine_results(self, semantic_results: List[SearchResult], 
                        keyword_results: List[SearchResult], top_k: int) -> List[SearchResult]:
        """Combine and rank results from both approaches."""
        # Create a dictionary to store combined scores
        combined_scores = {}
        
        # Add semantic results
        for result in semantic_results:
            doc_id = result.document.id
            combined_scores[doc_id] = {
                'result': result,
                'semantic_score': result.score,
                'keyword_score': 0.0
            }
        
        # Add keyword results
        for result in keyword_results:
            doc_id = result.document.id
            if doc_id in combined_scores:
                combined_scores[doc_id]['keyword_score'] = result.score
            else:
                combined_scores[doc_id] = {
                    'result': result,
                    'semantic_score': 0.0,
                    'keyword_score': result.score
                }
        
        # Calculate combined scores
        combined_results = []
        for doc_id, scores in combined_scores.items():
            combined_score = (self.semantic_weight * scores['semantic_score'] + 
                            self.keyword_weight * scores['keyword_score'])
            
            result = scores['result']
            result.score = combined_score
            combined_results.append(result)
        
        # Sort by combined score and return top_k
        combined_results.sort(key=lambda x: x.score, reverse=True)
        return combined_results[:top_k]
    
    def _build_enhanced_query(self, query: str, context: Dict[str, Any]) -> str:
        """Build enhanced query with context."""
        enhanced_parts = [query]
        
        if context.get('regulatory_framework'):
            enhanced_parts.append(f"regulatory framework: {context['regulatory_framework']}")
        
        if context.get('industry'):
            enhanced_parts.append(f"industry: {context['industry']}")
        
        return " ".join(enhanced_parts)
    
    def _build_context_filters(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Build filters based on context."""
        filters = {}
        
        if context.get('regulatory_framework'):
            filters['regulatory_framework'] = context['regulatory_framework']
        
        if context.get('industry'):
            filters['industry'] = context['industry']
        
        return filters


class ContextualRetriever(DocumentRetriever):
    """Contextual retrieval that considers conversation history and user context."""
    
    def __init__(self, vector_store: VectorStore, embedding_model: EmbeddingModel,
                 max_context_length: int = 1000):
        """Initialize contextual retriever.
        
        Args:
            vector_store: Vector store for document storage
            embedding_model: Model for generating embeddings
            max_context_length: Maximum length of context to consider
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.max_context_length = max_context_length
        self.logger = logging.getLogger(__name__)
    
    async def retrieve(self, query: str, top_k: int = 10, 
                      filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Retrieve documents with basic query."""
        try:
            results = await self.vector_store.search(query, filters, top_k)
            return results
        except Exception as e:
            self.logger.error(f"Failed to retrieve documents: {e}")
            return []
    
    async def retrieve_with_context(self, query: str, context: Dict[str, Any], 
                                   top_k: int = 10) -> List[SearchResult]:
        """Retrieve documents with full context consideration."""
        try:
            # Build contextual query
            contextual_query = self._build_contextual_query(query, context)
            
            # Apply contextual filters
            contextual_filters = self._build_contextual_filters(context)
            
            # Retrieve documents
            results = await self.vector_store.search(contextual_query, contextual_filters, top_k)
            
            # Apply contextual ranking
            results = self._apply_contextual_ranking(results, context)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve documents with context: {e}")
            return []
    
    def _build_contextual_query(self, query: str, context: Dict[str, Any]) -> str:
        """Build query considering conversation history and context."""
        query_parts = [query]
        
        # Add conversation history
        if context.get('conversation_history'):
            history = context['conversation_history']
            # Take last few turns to avoid too long queries
            recent_history = history[-3:] if len(history) > 3 else history
            history_text = " ".join([turn.get('content', '') for turn in recent_history])
            if history_text:
                query_parts.append(f"conversation context: {history_text}")
        
        # Add user profile context
        if context.get('user_profile'):
            profile = context['user_profile']
            if profile.get('role'):
                query_parts.append(f"user role: {profile['role']}")
            if profile.get('expertise_level'):
                query_parts.append(f"expertise level: {profile['expertise_level']}")
        
        # Add regulatory context
        if context.get('regulatory_framework'):
            query_parts.append(f"regulatory framework: {context['regulatory_framework']}")
        
        # Add industry context
        if context.get('industry'):
            query_parts.append(f"industry: {context['industry']}")
        
        return " ".join(query_parts)
    
    def _build_contextual_filters(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Build filters considering context."""
        filters = {}
        
        # Filter by regulatory framework
        if context.get('regulatory_framework'):
            filters['regulatory_framework'] = context['regulatory_framework']
        
        # Filter by industry
        if context.get('industry'):
            filters['industry'] = context['industry']
        
        # Filter by document type based on user role
        if context.get('user_profile', {}).get('role'):
            role = context['user_profile']['role']
            if role == 'auditor':
                filters['document_type'] = {"$in": ['audit_standard', 'implementation_guide']}
            elif role == 'compliance_officer':
                filters['document_type'] = {"$in": ['regulation', 'guidance', 'best_practice']}
            elif role == 'legal_counsel':
                filters['document_type'] = {"$in": ['regulation', 'case_law', 'legal_guidance']}
        
        return filters
    
    def _apply_contextual_ranking(self, results: List[SearchResult], 
                                 context: Dict[str, Any]) -> List[SearchResult]:
        """Apply contextual ranking to results."""
        for result in results:
            # Boost score for user role relevance
            if context.get('user_profile', {}).get('role'):
                role = context['user_profile']['role']
                if self._is_role_relevant(result.document, role):
                    result.score *= 1.2
            
            # Boost score for expertise level relevance
            if context.get('user_profile', {}).get('expertise_level'):
                expertise = context['user_profile']['expertise_level']
                if self._is_expertise_relevant(result.document, expertise):
                    result.score *= 1.1
            
            # Boost score for conversation relevance
            if context.get('conversation_history'):
                if self._is_conversation_relevant(result.document, context['conversation_history']):
                    result.score *= 1.15
        
        # Re-sort by adjusted scores
        results.sort(key=lambda x: x.score, reverse=True)
        return results
    
    def _is_role_relevant(self, document: Document, role: str) -> bool:
        """Check if document is relevant to user role."""
        # Simple role-based relevance check
        role_keywords = {
            'auditor': ['audit', 'testing', 'verification', 'control'],
            'compliance_officer': ['compliance', 'policy', 'procedure', 'governance'],
            'legal_counsel': ['legal', 'regulation', 'law', 'liability']
        }
        
        if role in role_keywords:
            content_lower = document.content.lower()
            return any(keyword in content_lower for keyword in role_keywords[role])
        
        return False
    
    def _is_expertise_relevant(self, document: Document, expertise: str) -> bool:
        """Check if document is relevant to expertise level."""
        # Simple expertise-based relevance check
        if expertise == 'beginner':
            return 'introduction' in document.content.lower() or 'basic' in document.content.lower()
        elif expertise == 'expert':
            return 'advanced' in document.content.lower() or 'detailed' in document.content.lower()
        
        return True
    
    def _is_conversation_relevant(self, document: Document, history: List[Dict[str, Any]]) -> bool:
        """Check if document is relevant to conversation history."""
        # Simple conversation relevance check
        recent_topics = []
        for turn in history[-2:]:  # Last 2 turns
            content = turn.get('content', '')
            # Extract key terms from conversation
            words = content.lower().split()
            recent_topics.extend([word for word in words if len(word) > 4])
        
        if recent_topics:
            content_lower = document.content.lower()
            return any(topic in content_lower for topic in recent_topics)
        
        return False
