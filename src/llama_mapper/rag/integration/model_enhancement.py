"""
RAG model enhancement for compliance AI.

Integrates RAG capabilities with existing context management system
to provide enhanced compliance analysis with regulatory knowledge retrieval.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from ..core.vector_store import Document, SearchResult
from ..core.retriever import DocumentRetriever
from ..core.ranker import DocumentRanker
from ...context.enhanced_context_manager import (
    EnhancedContext, ContextManager, ContextAwarePromptBuilder
)

logger = logging.getLogger(__name__)


@dataclass
class RAGEnhancedContext:
    """RAG enhancement to existing EnhancedContext."""
    base_context: EnhancedContext
    retrieved_documents: List[Document]
    knowledge_coverage: Dict[str, float]
    retrieval_confidence: float
    regulatory_citations: List[str]
    processing_time_ms: int = 0
    
    def __post_init__(self):
        if self.knowledge_coverage is None:
            self.knowledge_coverage = {}


class RAGContextEnhancer:
    """
    Enhances existing context management with RAG capabilities.
    
    Integrates with the existing ContextManager and ContextAwarePromptBuilder
    to add regulatory knowledge retrieval without replacing the existing system.
    """
    
    def __init__(self, vector_store, retriever: DocumentRetriever, 
                 context_manager: ContextManager,
                 ranker: Optional[DocumentRanker] = None):
        """Initialize RAG context enhancer.
        
        Args:
            vector_store: Vector store for documents
            retriever: Document retriever
            context_manager: Existing context manager to enhance
            ranker: Optional document ranker
        """
        self.vector_store = vector_store
        self.retriever = retriever
        self.context_manager = context_manager
        self.ranker = ranker
        self.logger = logging.getLogger(__name__)
    
    async def enhance_context(self, enhanced_context: EnhancedContext, query: str) -> RAGEnhancedContext:
        """Enhance existing context with RAG capabilities."""
        start_time = datetime.utcnow()
        
        try:
            # Build retrieval context from enhanced context
            retrieval_context = self._build_retrieval_context_from_enhanced(enhanced_context)
            
            # Retrieve relevant documents
            search_results = await self.retriever.retrieve_with_context(
                query=query,
                context=retrieval_context,
                top_k=10
            )
            
            # Re-rank if ranker is available
            if self.ranker and search_results:
                search_results = await self.ranker.rerank(
                    query=query,
                    documents=search_results,
                    top_k=10
                )
            
            # Extract documents from search results
            retrieved_documents = [result.document for result in search_results]
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(search_results)
            
            # Calculate knowledge coverage
            knowledge_coverage = self._calculate_knowledge_coverage(retrieved_documents)
            
            # Extract regulatory citations from retrieved documents
            regulatory_citations = self._extract_regulatory_citations(retrieved_documents)
            
            processing_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            return RAGEnhancedContext(
                base_context=enhanced_context,
                retrieved_documents=retrieved_documents,
                knowledge_coverage=knowledge_coverage,
                retrieval_confidence=confidence_score,
                regulatory_citations=regulatory_citations,
                processing_time_ms=processing_time_ms
            )
            
        except Exception as e:
            self.logger.error(f"Failed to enhance context with RAG: {e}")
            return RAGEnhancedContext(
                base_context=enhanced_context,
                retrieved_documents=[],
                knowledge_coverage={},
                retrieval_confidence=0.0,
                regulatory_citations=[],
                processing_time_ms=0
            )
    
    async def generate_expert_response(self, rag_enhanced_context: RAGEnhancedContext, 
                                     model_type: str = "compliance_expert") -> str:
        """Generate expert response using RAG context."""
        try:
            if model_type == "compliance_expert":
                return self._generate_compliance_expert_response(rag_enhanced_context)
            elif model_type == "audit_expert":
                return self._generate_audit_expert_response(rag_enhanced_context)
            elif model_type == "risk_expert":
                return self._generate_risk_expert_response(rag_enhanced_context)
            else:
                return self._generate_general_expert_response(rag_enhanced_context)
            
        except Exception as e:
            self.logger.error(f"Failed to generate expert response: {e}")
            return f"I apologize, but I encountered an error: {str(e)}"
    
    def _build_retrieval_context_from_enhanced(self, enhanced_context: EnhancedContext) -> Dict[str, Any]:
        """Build retrieval context from enhanced context."""
        retrieval_context = {}
        
        # Extract regulatory framework from business context
        if enhanced_context.business and enhanced_context.business.compliance_requirements:
            retrieval_context["regulatory_framework"] = enhanced_context.business.compliance_requirements[0]
        
        # Extract industry from business context
        if enhanced_context.business:
            retrieval_context["industry"] = enhanced_context.business.industry
        
        # Extract user role from application context
        if enhanced_context.application and enhanced_context.application.user_role:
            retrieval_context["user_role"] = enhanced_context.application.user_role
        
        # Extract enforcement level from policy context
        if enhanced_context.policy:
            retrieval_context["enforcement_level"] = enhanced_context.policy.enforcement_level
            retrieval_context["applicable_frameworks"] = enhanced_context.policy.applicable_frameworks
        
        return retrieval_context
    
    def _extract_regulatory_citations(self, documents: List[Document]) -> List[str]:
        """Extract regulatory citations from retrieved documents."""
        citations = []
        for doc in documents:
            if doc.regulatory_framework and doc.source:
                citations.append(f"{doc.regulatory_framework} - {doc.source}")
        return citations
    
    def _build_retrieval_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Build context for document retrieval."""
        retrieval_context = {}
        
        if "regulatory_framework" in context:
            retrieval_context["regulatory_framework"] = context["regulatory_framework"]
        
        if "industry" in context:
            retrieval_context["industry"] = context["industry"]
        
        if "document_types" in context:
            retrieval_context["document_types"] = context["document_types"]
        
        if "user_role" in context:
            retrieval_context["user_role"] = context["user_role"]
        
        if "expertise_level" in context:
            retrieval_context["expertise_level"] = context["expertise_level"]
        
        return retrieval_context
    
    def _build_enhanced_prompt(self, query: str, documents: List[Document], 
                              context: Dict[str, Any]) -> str:
        """Build enhanced prompt with retrieved knowledge."""
        prompt_parts = []
        
        # Add system prompt
        system_prompt = self._get_system_prompt(context)
        prompt_parts.append(system_prompt)
        
        # Add regulatory context
        if context.get("regulatory_framework"):
            prompt_parts.append(f"Regulatory Framework: {context['regulatory_framework']}")
        
        # Add industry context
        if context.get("industry"):
            prompt_parts.append(f"Industry: {context['industry']}")
        
        # Add user context
        if context.get("user_role"):
            prompt_parts.append(f"User Role: {context['user_role']}")
        
        # Add retrieved knowledge
        if documents:
            prompt_parts.append("Relevant Knowledge:")
            for i, doc in enumerate(documents[:5]):
                prompt_parts.append(f"Document {i+1}: {doc.content[:500]}...")
        
        # Add user query
        prompt_parts.append(f"User Query: {query}")
        
        # Add instructions
        instructions = self._get_response_instructions(context)
        prompt_parts.append(instructions)
        
        return "\n\n".join(prompt_parts)
    
    def _get_system_prompt(self, context: Dict[str, Any]) -> str:
        """Get system prompt based on context."""
        base_prompt = "You are a senior compliance expert with 15+ years of experience."
        
        user_role = context.get("user_role", "").lower()
        if "auditor" in user_role:
            base_prompt += " You specialize in audit methodologies and compliance testing."
        elif "compliance" in user_role:
            base_prompt += " You specialize in compliance program design and implementation."
        elif "legal" in user_role:
            base_prompt += " You specialize in regulatory interpretation and legal compliance."
        
        industry = context.get("industry", "").lower()
        if "financial" in industry:
            base_prompt += " You have deep expertise in financial services compliance."
        elif "healthcare" in industry:
            base_prompt += " You have deep expertise in healthcare compliance."
        elif "technology" in industry:
            base_prompt += " You have deep expertise in technology compliance."
        
        return base_prompt
    
    def _get_response_instructions(self, context: Dict[str, Any]) -> str:
        """Get response instructions based on context."""
        instructions = [
            "Provide expert-level compliance analysis with:",
            "1. Clear, actionable recommendations",
            "2. Regulatory citations and references",
            "3. Risk assessment and mitigation strategies",
            "4. Implementation guidance",
            "5. Cost and resource considerations"
        ]
        
        analysis_type = context.get("analysis_type", "").lower()
        if "gap" in analysis_type:
            instructions.append("6. Specific gap identification and prioritization")
        elif "audit" in analysis_type:
            instructions.append("6. Audit preparation and evidence collection guidance")
        elif "risk" in analysis_type:
            instructions.append("6. Comprehensive risk assessment and scoring")
        
        return "\n".join(instructions)
    
    def _calculate_confidence_score(self, search_results: List[SearchResult]) -> float:
        """Calculate confidence score for search results."""
        if not search_results:
            return 0.0
        
        scores = [result.score for result in search_results]
        avg_score = sum(scores) / len(scores)
        
        confidence = avg_score
        
        if len(search_results) >= 5:
            confidence *= 1.1
        
        regulatory_matches = sum(1 for result in search_results 
                               if result.document.regulatory_framework)
        if regulatory_matches > 0:
            confidence *= 1.05
        
        return min(confidence, 1.0)
    
    def _calculate_knowledge_coverage(self, documents: List[Document]) -> Dict[str, float]:
        """Calculate knowledge coverage metrics."""
        if not documents:
            return {}
        
        frameworks = set(doc.regulatory_framework for doc in documents if doc.regulatory_framework)
        framework_coverage = len(frameworks) / 10.0
        
        industries = set(doc.industry for doc in documents if doc.industry)
        industry_coverage = len(industries) / 5.0
        
        doc_types = set(doc.document_type for doc in documents)
        type_coverage = len(doc_types) / 8.0
        
        return {
            "regulatory_frameworks": min(framework_coverage, 1.0),
            "industries": min(industry_coverage, 1.0),
            "document_types": min(type_coverage, 1.0),
            "overall_coverage": (framework_coverage + industry_coverage + type_coverage) / 3.0
        }
    
    def _generate_compliance_expert_response(self, rag_enhanced_context: RAGEnhancedContext) -> str:
        """Generate compliance expert response."""
        response_parts = []
        
        if rag_enhanced_context.retrieved_documents:
            response_parts.append("Based on the relevant regulatory knowledge:")
            
            for i, doc in enumerate(rag_enhanced_context.retrieved_documents[:3]):
                response_parts.append(f"- {doc.content[:200]}...")
        
        response_parts.append("\nExpert Analysis:")
        response_parts.append("As a senior compliance expert, I recommend:")
        response_parts.append("1. Conduct a comprehensive compliance assessment")
        response_parts.append("2. Implement appropriate controls and monitoring")
        response_parts.append("3. Establish regular review and update procedures")
        
        # Add regulatory citations
        if rag_enhanced_context.regulatory_citations:
            response_parts.append(f"\nRegulatory Citations: {', '.join(rag_enhanced_context.regulatory_citations)}")
        
        # Add industry context from enhanced context
        if rag_enhanced_context.base_context.business:
            response_parts.append(f"Industry: {rag_enhanced_context.base_context.business.industry}")
        
        return "\n".join(response_parts)
    
    def _generate_audit_expert_response(self, rag_enhanced_context: RAGEnhancedContext) -> str:
        """Generate audit expert response."""
        response_parts = []
        
        response_parts.append("Audit Expert Analysis:")
        response_parts.append("Based on my audit experience:")
        response_parts.append("1. Risk assessment and control evaluation")
        response_parts.append("2. Evidence collection and documentation")
        response_parts.append("3. Testing procedures and sampling methods")
        response_parts.append("4. Findings management and remediation tracking")
        
        return "\n".join(response_parts)
    
    def _generate_risk_expert_response(self, rag_enhanced_context: RAGEnhancedContext) -> str:
        """Generate risk expert response."""
        response_parts = []
        
        response_parts.append("Risk Expert Analysis:")
        response_parts.append("From a risk management perspective:")
        response_parts.append("1. Identify and assess key risk factors")
        response_parts.append("2. Evaluate impact and likelihood")
        response_parts.append("3. Develop mitigation strategies")
        response_parts.append("4. Implement monitoring and reporting")
        
        return "\n".join(response_parts)
    
    def _generate_general_expert_response(self, rag_enhanced_context: RAGEnhancedContext) -> str:
        """Generate general expert response."""
        response_parts = []
        
        response_parts.append("Expert Analysis:")
        response_parts.append("Based on my expertise in compliance and regulatory matters:")
        response_parts.append("1. Comprehensive assessment of the situation")
        response_parts.append("2. Strategic recommendations for improvement")
        response_parts.append("3. Implementation guidance and best practices")
        response_parts.append("4. Ongoing monitoring and maintenance")
        
        return "\n".join(response_parts)


# Backward compatibility aliases
RAGContext = RAGEnhancedContext
RAGModelEnhancer = RAGContextEnhancer