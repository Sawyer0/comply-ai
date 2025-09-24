"""
Retrieval quality tests for RAG system.

Tests the quality and accuracy of document retrieval.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import statistics
from datetime import datetime

from ..core.retriever import DocumentRetriever
from ..core.vector_store import Document, SearchResult

logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    """Represents a test case for retrieval testing."""
    query: str
    expected_documents: List[str]  # Document IDs
    expected_frameworks: List[str]
    expected_industries: List[str]
    min_relevance_score: float = 0.7
    max_results: int = 10


@dataclass
class TestResult:
    """Represents the result of a retrieval test."""
    test_case: TestCase
    retrieved_documents: List[SearchResult]
    precision: float
    recall: float
    f1_score: float
    avg_relevance_score: float
    framework_coverage: float
    industry_coverage: float
    passed: bool
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class RetrievalQualityTests:
    """Tests for retrieval quality and accuracy."""
    
    def __init__(self, retriever: DocumentRetriever):
        """Initialize retrieval quality tests.
        
        Args:
            retriever: Document retriever to test
        """
        self.retriever = retriever
        self.logger = logging.getLogger(__name__)
    
    async def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run comprehensive retrieval quality tests."""
        test_results = {}
        
        # Test basic retrieval
        basic_tests = await self._run_basic_retrieval_tests()
        test_results["basic_retrieval"] = basic_tests
        
        # Test framework-specific retrieval
        framework_tests = await self._run_framework_specific_tests()
        test_results["framework_specific"] = framework_tests
        
        # Test industry-specific retrieval
        industry_tests = await self._run_industry_specific_tests()
        test_results["industry_specific"] = industry_tests
        
        # Test complex queries
        complex_tests = await self._run_complex_query_tests()
        test_results["complex_queries"] = complex_tests
        
        # Calculate overall metrics
        overall_metrics = self._calculate_overall_metrics(test_results)
        test_results["overall_metrics"] = overall_metrics
        
        return test_results
    
    async def _run_basic_retrieval_tests(self) -> List[TestResult]:
        """Run basic retrieval tests."""
        test_cases = [
            TestCase(
                query="GDPR compliance requirements",
                expected_documents=["gdpr_art_5", "gdpr_art_6", "gdpr_art_32"],
                expected_frameworks=["GDPR"],
                expected_industries=["technology", "healthcare"]
            ),
            TestCase(
                query="HIPAA security controls",
                expected_documents=["hipaa_164_308", "hipaa_164_312"],
                expected_frameworks=["HIPAA"],
                expected_industries=["healthcare"]
            ),
            TestCase(
                query="SOX internal controls",
                expected_documents=["sox_302", "sox_404"],
                expected_frameworks=["SOX"],
                expected_industries=["financial_services"]
            )
        ]
        
        results = []
        for test_case in test_cases:
            result = await self._run_single_test(test_case)
            results.append(result)
        
        return results
    
    async def _run_framework_specific_tests(self) -> List[TestResult]:
        """Run framework-specific retrieval tests."""
        test_cases = [
            TestCase(
                query="data protection impact assessment",
                expected_documents=["gdpr_art_35", "gdpr_art_36"],
                expected_frameworks=["GDPR"],
                expected_industries=["technology"]
            ),
            TestCase(
                query="audit trail requirements",
                expected_documents=["sox_302", "sox_404", "iso_27001_12_4_1"],
                expected_frameworks=["SOX", "ISO27001"],
                expected_industries=["financial_services", "technology"]
            )
        ]
        
        results = []
        for test_case in test_cases:
            result = await self._run_single_test(test_case)
            results.append(result)
        
        return results
    
    async def _run_industry_specific_tests(self) -> List[TestResult]:
        """Run industry-specific retrieval tests."""
        test_cases = [
            TestCase(
                query="banking compliance monitoring",
                expected_documents=["basel_iii", "dodd_frank", "mifid_ii"],
                expected_frameworks=["Basel_III", "Dodd_Frank", "MiFID_II"],
                expected_industries=["financial_services"]
            ),
            TestCase(
                query="healthcare data security",
                expected_documents=["hipaa_164_308", "hipaa_164_312", "hitech"],
                expected_frameworks=["HIPAA", "HITECH"],
                expected_industries=["healthcare"]
            )
        ]
        
        results = []
        for test_case in test_cases:
            result = await self._run_single_test(test_case)
            results.append(result)
        
        return results
    
    async def _run_complex_query_tests(self) -> List[TestResult]:
        """Run complex query tests."""
        test_cases = [
            TestCase(
                query="multi-jurisdictional data privacy compliance for healthcare technology companies",
                expected_documents=["gdpr_art_5", "hipaa_164_308", "ccpa_1798_100"],
                expected_frameworks=["GDPR", "HIPAA", "CCPA"],
                expected_industries=["healthcare", "technology"]
            ),
            TestCase(
                query="enterprise risk management and compliance program design",
                expected_documents=["coso_erm", "iso_31000", "sox_302"],
                expected_frameworks=["COSO", "ISO31000", "SOX"],
                expected_industries=["financial_services", "technology"]
            )
        ]
        
        results = []
        for test_case in test_cases:
            result = await self._run_single_test(test_case)
            results.append(result)
        
        return results
    
    async def _run_single_test(self, test_case: TestCase) -> TestResult:
        """Run a single retrieval test."""
        try:
            # Retrieve documents
            search_results = await self.retriever.retrieve(
                query=test_case.query,
                top_k=test_case.max_results
            )
            
            # Calculate metrics
            precision = self._calculate_precision(search_results, test_case.expected_documents)
            recall = self._calculate_recall(search_results, test_case.expected_documents)
            f1_score = self._calculate_f1_score(precision, recall)
            avg_relevance_score = self._calculate_avg_relevance_score(search_results)
            framework_coverage = self._calculate_framework_coverage(search_results, test_case.expected_frameworks)
            industry_coverage = self._calculate_industry_coverage(search_results, test_case.expected_industries)
            
            # Determine if test passed
            passed = (precision >= 0.7 and recall >= 0.7 and 
                     avg_relevance_score >= test_case.min_relevance_score)
            
            return TestResult(
                test_case=test_case,
                retrieved_documents=search_results,
                precision=precision,
                recall=recall,
                f1_score=f1_score,
                avg_relevance_score=avg_relevance_score,
                framework_coverage=framework_coverage,
                industry_coverage=industry_coverage,
                passed=passed
            )
            
        except Exception as e:
            self.logger.error(f"Test failed for query '{test_case.query}': {e}")
            return TestResult(
                test_case=test_case,
                retrieved_documents=[],
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                avg_relevance_score=0.0,
                framework_coverage=0.0,
                industry_coverage=0.0,
                passed=False,
                errors=[str(e)]
            )
    
    def _calculate_precision(self, search_results: List[SearchResult], 
                           expected_documents: List[str]) -> float:
        """Calculate precision for search results."""
        if not search_results:
            return 0.0
        
        retrieved_ids = [result.document.id for result in search_results]
        relevant_retrieved = set(retrieved_ids) & set(expected_documents)
        
        return len(relevant_retrieved) / len(retrieved_ids)
    
    def _calculate_recall(self, search_results: List[SearchResult], 
                         expected_documents: List[str]) -> float:
        """Calculate recall for search results."""
        if not expected_documents:
            return 1.0
        
        retrieved_ids = [result.document.id for result in search_results]
        relevant_retrieved = set(retrieved_ids) & set(expected_documents)
        
        return len(relevant_retrieved) / len(expected_documents)
    
    def _calculate_f1_score(self, precision: float, recall: float) -> float:
        """Calculate F1 score from precision and recall."""
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def _calculate_avg_relevance_score(self, search_results: List[SearchResult]) -> float:
        """Calculate average relevance score."""
        if not search_results:
            return 0.0
        
        scores = [result.score for result in search_results]
        return statistics.mean(scores)
    
    def _calculate_framework_coverage(self, search_results: List[SearchResult], 
                                    expected_frameworks: List[str]) -> float:
        """Calculate framework coverage."""
        if not expected_frameworks:
            return 1.0
        
        retrieved_frameworks = set()
        for result in search_results:
            if result.document.regulatory_framework:
                retrieved_frameworks.add(result.document.regulatory_framework)
        
        expected_set = set(expected_frameworks)
        covered_frameworks = retrieved_frameworks & expected_set
        
        return len(covered_frameworks) / len(expected_set)
    
    def _calculate_industry_coverage(self, search_results: List[SearchResult], 
                                   expected_industries: List[str]) -> float:
        """Calculate industry coverage."""
        if not expected_industries:
            return 1.0
        
        retrieved_industries = set()
        for result in search_results:
            if result.document.industry:
                retrieved_industries.add(result.document.industry)
        
        expected_set = set(expected_industries)
        covered_industries = retrieved_industries & expected_set
        
        return len(covered_industries) / len(expected_set)
    
    def _calculate_overall_metrics(self, test_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate overall test metrics."""
        all_results = []
        for category, results in test_results.items():
            if isinstance(results, list):
                all_results.extend(results)
        
        if not all_results:
            return {}
        
        # Calculate overall metrics
        precision_scores = [result.precision for result in all_results]
        recall_scores = [result.recall for result in all_results]
        f1_scores = [result.f1_score for result in all_results]
        relevance_scores = [result.avg_relevance_score for result in all_results]
        framework_coverage = [result.framework_coverage for result in all_results]
        industry_coverage = [result.industry_coverage for result in all_results]
        
        return {
            "overall_precision": statistics.mean(precision_scores),
            "overall_recall": statistics.mean(recall_scores),
            "overall_f1_score": statistics.mean(f1_scores),
            "overall_relevance": statistics.mean(relevance_scores),
            "overall_framework_coverage": statistics.mean(framework_coverage),
            "overall_industry_coverage": statistics.mean(industry_coverage),
            "test_pass_rate": sum(1 for result in all_results if result.passed) / len(all_results),
            "total_tests": len(all_results)
        }


class RetrievalAccuracyTests:
    """Tests for retrieval accuracy and consistency."""
    
    def __init__(self, retriever: DocumentRetriever):
        """Initialize retrieval accuracy tests."""
        self.retriever = retriever
        self.logger = logging.getLogger(__name__)
    
    async def test_retrieval_consistency(self, query: str, num_runs: int = 5) -> Dict[str, Any]:
        """Test retrieval consistency across multiple runs."""
        try:
            results = []
            
            for i in range(num_runs):
                search_results = await self.retriever.retrieve(query, top_k=10)
                results.append(search_results)
            
            # Calculate consistency metrics
            consistency_metrics = self._calculate_consistency_metrics(results)
            
            return {
                "query": query,
                "num_runs": num_runs,
                "consistency_metrics": consistency_metrics,
                "results": results
            }
            
        except Exception as e:
            self.logger.error(f"Consistency test failed: {e}")
            return {"error": str(e)}
    
    async def test_retrieval_robustness(self, queries: List[str]) -> Dict[str, Any]:
        """Test retrieval robustness across different query types."""
        try:
            results = {}
            
            for query in queries:
                search_results = await self.retriever.retrieve(query, top_k=10)
                
                # Calculate robustness metrics
                robustness_metrics = self._calculate_robustness_metrics(search_results)
                results[query] = {
                    "search_results": search_results,
                    "robustness_metrics": robustness_metrics
                }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Robustness test failed: {e}")
            return {"error": str(e)}
    
    def _calculate_consistency_metrics(self, results: List[List[SearchResult]]) -> Dict[str, float]:
        """Calculate consistency metrics across multiple runs."""
        if not results or len(results) < 2:
            return {}
        
        # Calculate document ID consistency
        all_doc_ids = [set(result.document.id for result in run) for run in results]
        doc_id_consistency = self._calculate_set_consistency(all_doc_ids)
        
        # Calculate score consistency
        all_scores = [result.score for run in results for result in run]
        score_consistency = 1.0 - statistics.stdev(all_scores) if all_scores else 0.0
        
        # Calculate rank consistency
        rank_consistency = self._calculate_rank_consistency(results)
        
        return {
            "document_id_consistency": doc_id_consistency,
            "score_consistency": score_consistency,
            "rank_consistency": rank_consistency
        }
    
    def _calculate_robustness_metrics(self, search_results: List[SearchResult]) -> Dict[str, float]:
        """Calculate robustness metrics for search results."""
        if not search_results:
            return {}
        
        # Calculate score distribution
        scores = [result.score for result in search_results]
        score_variance = statistics.variance(scores) if len(scores) > 1 else 0.0
        
        # Calculate diversity
        diversity = self._calculate_diversity(search_results)
        
        # Calculate coverage
        coverage = self._calculate_coverage(search_results)
        
        return {
            "score_variance": score_variance,
            "diversity": diversity,
            "coverage": coverage
        }
    
    def _calculate_set_consistency(self, sets: List[set]) -> float:
        """Calculate consistency between sets."""
        if not sets:
            return 0.0
        
        # Calculate Jaccard similarity between all pairs
        similarities = []
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                intersection = len(sets[i] & sets[j])
                union = len(sets[i] | sets[j])
                similarity = intersection / union if union > 0 else 0.0
                similarities.append(similarity)
        
        return statistics.mean(similarities) if similarities else 0.0
    
    def _calculate_rank_consistency(self, results: List[List[SearchResult]]) -> float:
        """Calculate rank consistency across multiple runs."""
        if not results or len(results) < 2:
            return 0.0
        
        # Calculate rank correlation between first run and others
        first_run = results[0]
        correlations = []
        
        for run in results[1:]:
            correlation = self._calculate_rank_correlation(first_run, run)
            correlations.append(correlation)
        
        return statistics.mean(correlations) if correlations else 0.0
    
    def _calculate_rank_correlation(self, run1: List[SearchResult], run2: List[SearchResult]) -> float:
        """Calculate rank correlation between two runs."""
        # Simplified rank correlation calculation
        # In a real implementation, this would use Spearman's rank correlation
        
        if not run1 or not run2:
            return 0.0
        
        # Get common documents
        ids1 = {result.document.id for result in run1}
        ids2 = {result.document.id for result in run2}
        common_ids = ids1 & ids2
        
        if not common_ids:
            return 0.0
        
        # Calculate rank positions for common documents
        ranks1 = {result.document.id: i for i, result in enumerate(run1) if result.document.id in common_ids}
        ranks2 = {result.document.id: i for i, result in enumerate(run2) if result.document.id in common_ids}
        
        # Calculate correlation
        rank_diffs = []
        for doc_id in common_ids:
            rank1 = ranks1[doc_id]
            rank2 = ranks2[doc_id]
            rank_diffs.append(abs(rank1 - rank2))
        
        # Convert to correlation (lower differences = higher correlation)
        max_possible_diff = len(common_ids) - 1
        avg_diff = statistics.mean(rank_diffs)
        correlation = 1.0 - (avg_diff / max_possible_diff)
        
        return max(0.0, correlation)
    
    def _calculate_diversity(self, search_results: List[SearchResult]) -> float:
        """Calculate diversity of search results."""
        if not search_results:
            return 0.0
        
        # Calculate diversity based on unique sources
        sources = set()
        for result in search_results:
            if hasattr(result.document, 'source'):
                sources.add(result.document.source)
        
        return len(sources) / len(search_results)
    
    def _calculate_coverage(self, search_results: List[SearchResult]) -> float:
        """Calculate coverage of search results."""
        if not search_results:
            return 0.0
        
        # Calculate coverage based on unique frameworks
        frameworks = set()
        for result in search_results:
            if hasattr(result.document, 'regulatory_framework') and result.document.regulatory_framework:
                frameworks.add(result.document.regulatory_framework)
        
        # Assuming 10 major frameworks
        return len(frameworks) / 10.0
