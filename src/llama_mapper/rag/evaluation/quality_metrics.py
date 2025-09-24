"""
Quality metrics for RAG system evaluation.

Provides comprehensive evaluation metrics for assessing RAG system performance
in compliance scenarios.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import json

from ..core.vector_store import Document, SearchResult
from ..integration.model_enhancement import ExpertResponse

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """Quality metrics for RAG evaluation."""
    
    # Retrieval metrics
    retrieval_precision: float = 0.0
    retrieval_recall: float = 0.0
    retrieval_f1: float = 0.0
    retrieval_ndcg: float = 0.0
    
    # Response quality metrics
    response_relevance: float = 0.0
    response_accuracy: float = 0.0
    response_completeness: float = 0.0
    response_coherence: float = 0.0
    
    # Citation metrics
    citation_accuracy: float = 0.0
    citation_coverage: float = 0.0
    citation_relevance: float = 0.0
    
    # Compliance-specific metrics
    regulatory_accuracy: float = 0.0
    risk_assessment_accuracy: float = 0.0
    recommendation_quality: float = 0.0
    
    # Overall metrics
    overall_score: float = 0.0
    confidence_score: float = 0.0
    
    # Metadata
    evaluation_timestamp: datetime = field(default_factory=datetime.utcnow)
    evaluation_id: str = ""
    query_id: str = ""
    user_feedback: Optional[Dict[str, Any]] = None


class RAGQualityEvaluator:
    """Comprehensive quality evaluator for RAG systems."""
    
    def __init__(self):
        """Initialize RAG quality evaluator."""
        self.logger = logging.getLogger(__name__)
    
    async def evaluate_retrieval_quality(self, query: str, 
                                        retrieved_docs: List[SearchResult],
                                        ground_truth_docs: List[Document],
                                        top_k: int = 10) -> Dict[str, float]:
        """Evaluate retrieval quality metrics.
        
        Args:
            query: Search query
            retrieved_docs: Retrieved documents
            ground_truth_docs: Ground truth relevant documents
            top_k: Number of top documents to consider
            
        Returns:
            Dictionary of retrieval quality metrics
        """
        try:
            # Get top-k retrieved documents
            top_retrieved = retrieved_docs[:top_k]
            
            # Extract document IDs
            retrieved_ids = {doc.document.id for doc in top_retrieved}
            ground_truth_ids = {doc.id for doc in ground_truth_docs}
            
            # Calculate precision, recall, F1
            if retrieved_ids:
                precision = len(retrieved_ids & ground_truth_ids) / len(retrieved_ids)
            else:
                precision = 0.0
            
            if ground_truth_ids:
                recall = len(retrieved_ids & ground_truth_ids) / len(ground_truth_ids)
            else:
                recall = 0.0
            
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0
            
            # Calculate NDCG
            ndcg = self._calculate_ndcg(top_retrieved, ground_truth_ids)
            
            return {
                "retrieval_precision": precision,
                "retrieval_recall": recall,
                "retrieval_f1": f1,
                "retrieval_ndcg": ndcg
            }
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate retrieval quality: {e}")
            return {
                "retrieval_precision": 0.0,
                "retrieval_recall": 0.0,
                "retrieval_f1": 0.0,
                "retrieval_ndcg": 0.0
            }
    
    async def evaluate_response_quality(self, query: str, 
                                      response: ExpertResponse,
                                      ground_truth: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate response quality metrics.
        
        Args:
            query: Search query
            response: Generated response
            ground_truth: Ground truth response
            
        Returns:
            Dictionary of response quality metrics
        """
        try:
            # Evaluate relevance
            relevance = await self._evaluate_relevance(query, response, ground_truth)
            
            # Evaluate accuracy
            accuracy = await self._evaluate_accuracy(response, ground_truth)
            
            # Evaluate completeness
            completeness = await self._evaluate_completeness(response, ground_truth)
            
            # Evaluate coherence
            coherence = await self._evaluate_coherence(response)
            
            return {
                "response_relevance": relevance,
                "response_accuracy": accuracy,
                "response_completeness": completeness,
                "response_coherence": coherence
            }
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate response quality: {e}")
            return {
                "response_relevance": 0.0,
                "response_accuracy": 0.0,
                "response_completeness": 0.0,
                "response_coherence": 0.0
            }
    
    async def evaluate_citation_quality(self, response: ExpertResponse,
                                       retrieved_docs: List[SearchResult]) -> Dict[str, float]:
        """Evaluate citation quality metrics.
        
        Args:
            response: Generated response with citations
            retrieved_docs: Retrieved documents used for citations
            
        Returns:
            Dictionary of citation quality metrics
        """
        try:
            # Evaluate citation accuracy
            citation_accuracy = await self._evaluate_citation_accuracy(
                response.regulatory_citations, retrieved_docs
            )
            
            # Evaluate citation coverage
            citation_coverage = await self._evaluate_citation_coverage(
                response.regulatory_citations, retrieved_docs
            )
            
            # Evaluate citation relevance
            citation_relevance = await self._evaluate_citation_relevance(
                response.regulatory_citations, retrieved_docs
            )
            
            return {
                "citation_accuracy": citation_accuracy,
                "citation_coverage": citation_coverage,
                "citation_relevance": citation_relevance
            }
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate citation quality: {e}")
            return {
                "citation_accuracy": 0.0,
                "citation_coverage": 0.0,
                "citation_relevance": 0.0
            }
    
    async def evaluate_compliance_quality(self, response: ExpertResponse,
                                         ground_truth: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate compliance-specific quality metrics.
        
        Args:
            response: Generated compliance response
            ground_truth: Ground truth compliance information
            
        Returns:
            Dictionary of compliance quality metrics
        """
        try:
            # Evaluate regulatory accuracy
            regulatory_accuracy = await self._evaluate_regulatory_accuracy(
                response, ground_truth
            )
            
            # Evaluate risk assessment accuracy
            risk_accuracy = await self._evaluate_risk_assessment_accuracy(
                response, ground_truth
            )
            
            # Evaluate recommendation quality
            recommendation_quality = await self._evaluate_recommendation_quality(
                response, ground_truth
            )
            
            return {
                "regulatory_accuracy": regulatory_accuracy,
                "risk_assessment_accuracy": risk_accuracy,
                "recommendation_quality": recommendation_quality
            }
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate compliance quality: {e}")
            return {
                "regulatory_accuracy": 0.0,
                "risk_assessment_accuracy": 0.0,
                "recommendation_quality": 0.0
            }
    
    async def evaluate_comprehensive(self, query: str, 
                                   retrieved_docs: List[SearchResult],
                                   response: ExpertResponse,
                                   ground_truth: Dict[str, Any]) -> QualityMetrics:
        """Perform comprehensive evaluation of RAG system.
        
        Args:
            query: Search query
            retrieved_docs: Retrieved documents
            response: Generated response
            ground_truth: Ground truth information
            
        Returns:
            Comprehensive quality metrics
        """
        try:
            # Evaluate retrieval quality
            retrieval_metrics = await self.evaluate_retrieval_quality(
                query, retrieved_docs, ground_truth.get("relevant_docs", [])
            )
            
            # Evaluate response quality
            response_metrics = await self.evaluate_response_quality(
                query, response, ground_truth
            )
            
            # Evaluate citation quality
            citation_metrics = await self.evaluate_citation_quality(
                response, retrieved_docs
            )
            
            # Evaluate compliance quality
            compliance_metrics = await self.evaluate_compliance_quality(
                response, ground_truth
            )
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(
                retrieval_metrics, response_metrics, citation_metrics, compliance_metrics
            )
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(
                retrieval_metrics, response_metrics, citation_metrics, compliance_metrics
            )
            
            return QualityMetrics(
                **retrieval_metrics,
                **response_metrics,
                **citation_metrics,
                **compliance_metrics,
                overall_score=overall_score,
                confidence_score=confidence_score,
                evaluation_id=f"eval_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                query_id=ground_truth.get("query_id", "")
            )
            
        except Exception as e:
            self.logger.error(f"Failed to perform comprehensive evaluation: {e}")
            return QualityMetrics()
    
    def _calculate_ndcg(self, retrieved_docs: List[SearchResult], 
                       ground_truth_ids: set) -> float:
        """Calculate Normalized Discounted Cumulative Gain."""
        if not retrieved_docs:
            return 0.0
        
        # Calculate DCG
        dcg = 0.0
        for i, doc in enumerate(retrieved_docs):
            if doc.document.id in ground_truth_ids:
                dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(1) = 0
        
        # Calculate IDCG (ideal DCG)
        idcg = 0.0
        for i in range(min(len(ground_truth_ids), len(retrieved_docs))):
            idcg += 1.0 / np.log2(i + 2)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    async def _evaluate_relevance(self, query: str, response: ExpertResponse, 
                                 ground_truth: Dict[str, Any]) -> float:
        """Evaluate response relevance to query."""
        # Simple relevance evaluation based on keyword overlap
        query_words = set(query.lower().split())
        response_words = set(response.analysis.lower().split())
        
        if not query_words:
            return 0.0
        
        overlap = len(query_words & response_words)
        return overlap / len(query_words)
    
    async def _evaluate_accuracy(self, response: ExpertResponse, 
                                ground_truth: Dict[str, Any]) -> float:
        """Evaluate response accuracy."""
        # Compare key elements with ground truth
        accuracy_score = 0.0
        total_checks = 0
        
        # Check regulatory framework accuracy
        if "expected_framework" in ground_truth:
            total_checks += 1
            if response.jurisdictional_scope and ground_truth["expected_framework"] in response.jurisdictional_scope:
                accuracy_score += 1.0
        
        # Check risk assessment accuracy
        if "expected_risk_level" in ground_truth:
            total_checks += 1
            if response.risk_assessment and ground_truth["expected_risk_level"] in str(response.risk_assessment):
                accuracy_score += 1.0
        
        # Check recommendation alignment
        if "expected_recommendations" in ground_truth:
            total_checks += 1
            expected_recs = set(ground_truth["expected_recommendations"])
            actual_recs = set(response.recommendations)
            if expected_recs & actual_recs:  # If there's any overlap
                accuracy_score += 0.5  # Partial credit for overlap
        
        return accuracy_score / total_checks if total_checks > 0 else 0.0
    
    async def _evaluate_completeness(self, response: ExpertResponse, 
                                    ground_truth: Dict[str, Any]) -> float:
        """Evaluate response completeness."""
        completeness_score = 0.0
        total_checks = 0
        
        # Check if analysis is present
        total_checks += 1
        if response.analysis and len(response.analysis) > 50:
            completeness_score += 1.0
        
        # Check if recommendations are present
        total_checks += 1
        if response.recommendations and len(response.recommendations) > 0:
            completeness_score += 1.0
        
        # Check if risk assessment is present
        total_checks += 1
        if response.risk_assessment and len(response.risk_assessment) > 0:
            completeness_score += 1.0
        
        # Check if citations are present
        total_checks += 1
        if response.regulatory_citations and len(response.regulatory_citations) > 0:
            completeness_score += 1.0
        
        # Check if next actions are present
        total_checks += 1
        if response.next_actions and len(response.next_actions) > 0:
            completeness_score += 1.0
        
        return completeness_score / total_checks if total_checks > 0 else 0.0
    
    async def _evaluate_coherence(self, response: ExpertResponse) -> float:
        """Evaluate response coherence."""
        # Simple coherence check based on response structure
        coherence_score = 0.0
        total_checks = 0
        
        # Check if analysis is well-structured
        total_checks += 1
        if response.analysis and "→" in response.analysis:  # Check for structured format
            coherence_score += 1.0
        
        # Check if recommendations are actionable
        total_checks += 1
        if response.recommendations:
            actionable_recs = [rec for rec in response.recommendations 
                             if any(word in rec.lower() for word in ["implement", "conduct", "establish", "develop"])]
            if len(actionable_recs) > 0:
                coherence_score += 1.0
        
        # Check if next actions have clear ownership
        total_checks += 1
        if response.next_actions:
            clear_ownership = [action for action in response.next_actions 
                             if "owner" in action and action["owner"]]
            if len(clear_ownership) > 0:
                coherence_score += 1.0
        
        return coherence_score / total_checks if total_checks > 0 else 0.0
    
    async def _evaluate_citation_accuracy(self, citations: List[Dict[str, str]], 
                                        retrieved_docs: List[SearchResult]) -> float:
        """Evaluate citation accuracy."""
        if not citations or not retrieved_docs:
            return 0.0
        
        # Check if citations reference actual retrieved documents
        retrieved_sources = {doc.document.source for doc in retrieved_docs}
        cited_sources = {citation.get("title", "") for citation in citations}
        
        if not retrieved_sources or not cited_sources:
            return 0.0
        
        overlap = len(retrieved_sources & cited_sources)
        return overlap / len(cited_sources)
    
    async def _evaluate_citation_coverage(self, citations: List[Dict[str, str]], 
                                        retrieved_docs: List[SearchResult]) -> float:
        """Evaluate citation coverage."""
        if not retrieved_docs:
            return 0.0
        
        # Check how many retrieved documents are cited
        retrieved_sources = {doc.document.source for doc in retrieved_docs}
        cited_sources = {citation.get("title", "") for citation in citations}
        
        if not retrieved_sources:
            return 0.0
        
        overlap = len(retrieved_sources & cited_sources)
        return overlap / len(retrieved_sources)
    
    async def _evaluate_citation_relevance(self, citations: List[Dict[str, str]], 
                                          retrieved_docs: List[SearchResult]) -> float:
        """Evaluate citation relevance."""
        if not citations:
            return 0.0
        
        # Check if citations have required fields
        relevant_citations = 0
        for citation in citations:
            if citation.get("title") and citation.get("citation"):
                relevant_citations += 1
        
        return relevant_citations / len(citations)
    
    async def _evaluate_regulatory_accuracy(self, response: ExpertResponse, 
                                           ground_truth: Dict[str, Any]) -> float:
        """Evaluate regulatory accuracy."""
        if "expected_regulatory_framework" not in ground_truth:
            return 1.0  # No ground truth to compare against
        
        expected_framework = ground_truth["expected_regulatory_framework"]
        if response.jurisdictional_scope and expected_framework in response.jurisdictional_scope:
            return 1.0
        
        return 0.0
    
    async def _evaluate_risk_assessment_accuracy(self, response: ExpertResponse, 
                                                ground_truth: Dict[str, Any]) -> float:
        """Evaluate risk assessment accuracy."""
        if "expected_risk_level" not in ground_truth:
            return 1.0  # No ground truth to compare against
        
        expected_risk = ground_truth["expected_risk_level"]
        if response.risk_assessment and expected_risk in str(response.risk_assessment):
            return 1.0
        
        return 0.0
    
    async def _evaluate_recommendation_quality(self, response: ExpertResponse, 
                                              ground_truth: Dict[str, Any]) -> float:
        """Evaluate recommendation quality."""
        if not response.recommendations:
            return 0.0
        
        # Check if recommendations are actionable
        actionable_count = 0
        for rec in response.recommendations:
            if any(word in rec.lower() for word in ["implement", "conduct", "establish", "develop", "create"]):
                actionable_count += 1
        
        return actionable_count / len(response.recommendations)
    
    def _calculate_overall_score(self, retrieval_metrics: Dict[str, float],
                                response_metrics: Dict[str, float],
                                citation_metrics: Dict[str, float],
                                compliance_metrics: Dict[str, float]) -> float:
        """Calculate overall quality score."""
        # Weighted average of all metrics
        weights = {
            "retrieval": 0.25,
            "response": 0.35,
            "citation": 0.20,
            "compliance": 0.20
        }
        
        retrieval_score = np.mean(list(retrieval_metrics.values()))
        response_score = np.mean(list(response_metrics.values()))
        citation_score = np.mean(list(citation_metrics.values()))
        compliance_score = np.mean(list(compliance_metrics.values()))
        
        overall_score = (weights["retrieval"] * retrieval_score +
                        weights["response"] * response_score +
                        weights["citation"] * citation_score +
                        weights["compliance"] * compliance_score)
        
        return overall_score
    
    def _calculate_confidence_score(self, retrieval_metrics: Dict[str, float],
                                  response_metrics: Dict[str, float],
                                  citation_metrics: Dict[str, float],
                                  compliance_metrics: Dict[str, float]) -> float:
        """Calculate confidence score based on metric consistency."""
        # Calculate variance across metrics to assess confidence
        all_scores = (list(retrieval_metrics.values()) +
                     list(response_metrics.values()) +
                     list(citation_metrics.values()) +
                     list(compliance_metrics.values()))
        
        if not all_scores:
            return 0.0
        
        # Lower variance = higher confidence
        variance = np.var(all_scores)
        confidence = max(0.0, 1.0 - variance)
        
        return confidence
