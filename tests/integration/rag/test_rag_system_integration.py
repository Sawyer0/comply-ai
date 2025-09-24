"""
Integration tests for complete RAG system.

Tests the full RAG pipeline including document ingestion,
retrieval, enhancement, and guardrails.
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
from typing import List, Dict, Any

from src.llama_mapper.rag.core.vector_store import ChromaDBVectorStore, Document
from src.llama_mapper.rag.core.embeddings import SentenceTransformerEmbeddings
from src.llama_mapper.rag.core.retriever import SemanticRetriever
from src.llama_mapper.rag.integration.model_enhancement import RAGModelEnhancer
from src.llama_mapper.rag.guardrails.compliance_guardrails import ComplianceGuardrails
from src.llama_mapper.rag.evaluation.quality_metrics import RAGQualityEvaluator


class TestRAGSystemIntegration:
    """Test complete RAG system integration."""
    
    @pytest.fixture
    async def temp_db_dir(self):
        """Create temporary database directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    async def rag_system_components(self, temp_db_dir):
        """Set up complete RAG system components."""
        # Create vector store
        vector_store = ChromaDBVectorStore(
            collection_name="test_compliance",
            persist_directory=temp_db_dir
        )
        
        # Create embedding model (mocked for testing)
        embedding_model = Mock()
        embedding_model.embed_text = Mock(return_value=[0.1, 0.2, 0.3])
        embedding_model.embed_batch = Mock(return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        embedding_model.health_check = Mock(return_value=True)
        
        # Create retriever
        retriever = SemanticRetriever(vector_store, embedding_model)
        
        # Create RAG enhancer
        rag_enhancer = RAGModelEnhancer(vector_store, retriever)
        
        # Create guardrails
        guardrails = ComplianceGuardrails()
        
        # Create quality evaluator
        quality_evaluator = RAGQualityEvaluator()
        
        return {
            "vector_store": vector_store,
            "embedding_model": embedding_model,
            "retriever": retriever,
            "rag_enhancer": rag_enhancer,
            "guardrails": guardrails,
            "quality_evaluator": quality_evaluator
        }
    
    @pytest.fixture
    def sample_documents(self):
        """Create sample compliance documents."""
        return [
            Document(
                content="GDPR Article 6 requires a lawful basis for processing personal data. The lawful bases include consent, contract, legal obligation, vital interests, public task, and legitimate interests.",
                source="gdpr_article_6.txt",
                document_type="regulation",
                regulatory_framework="GDPR",
                industry="technology"
            ),
            Document(
                content="HIPAA requires covered entities to implement safeguards to protect the privacy and security of protected health information (PHI).",
                source="hipaa_overview.txt",
                document_type="regulation",
                regulatory_framework="HIPAA",
                industry="healthcare"
            ),
            Document(
                content="SOX Section 404 requires management to assess the effectiveness of internal controls over financial reporting.",
                source="sox_section_404.txt",
                document_type="regulation",
                regulatory_framework="SOX",
                industry="financial_services"
            )
        ]
    
    @patch('src.llama_mapper.rag.core.vector_store.chromadb')
    @pytest.mark.asyncio
    async def test_end_to_end_rag_pipeline(self, mock_chromadb, rag_system_components, sample_documents):
        """Test complete end-to-end RAG pipeline."""
        # Mock ChromaDB
        mock_client = Mock()
        mock_collection = Mock()
        mock_chromadb.PersistentClient.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection
        
        # Mock search results
        mock_collection.query.return_value = {
            'documents': [['GDPR Article 6 requires lawful basis for processing']],
            'metadatas': [{'source': 'gdpr_article_6.txt', 'regulatory_framework': 'GDPR'}],
            'distances': [[0.1]],
            'ids': [['doc-1']]
        }
        
        components = rag_system_components
        
        # Step 1: Add documents to vector store
        await components["vector_store"].add_documents(sample_documents)
        
        # Step 2: Query the RAG system
        context = {
            "regulatory_framework": "GDPR",
            "industry": "technology"
        }
        
        rag_context = await components["rag_enhancer"].enhance_query(
            "What are the GDPR requirements for data processing?",
            context
        )
        
        # Verify RAG context
        assert rag_context.query == "What are the GDPR requirements for data processing?"
        assert rag_context.regulatory_framework == "GDPR"
        assert rag_context.industry == "technology"
        
        # Step 3: Generate expert response
        expert_response = await components["rag_enhancer"].generate_expert_response(rag_context)
        
        # Verify expert response structure
        assert expert_response.analysis is not None
        assert len(expert_response.recommendations) > 0
        assert expert_response.risk_assessment is not None
        assert expert_response.confidence_score > 0.0
        
        # Step 4: Evaluate guardrails
        guardrail_result = await components["guardrails"].evaluate_response(
            expert_response, rag_context
        )
        
        # Verify guardrails evaluation
        assert guardrail_result is not None
        assert isinstance(guardrail_result.passed, bool)
        assert guardrail_result.confidence_score >= 0.0
    
    @patch('src.llama_mapper.rag.core.vector_store.chromadb')
    @pytest.mark.asyncio
    async def test_document_retrieval_integration(self, mock_chromadb, rag_system_components, sample_documents):
        """Test document retrieval integration."""
        # Mock ChromaDB
        mock_client = Mock()
        mock_collection = Mock()
        mock_chromadb.PersistentClient.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection
        
        # Mock search results with multiple documents
        mock_collection.query.return_value = {
            'documents': [['GDPR Article 6', 'HIPAA safeguards']],
            'metadatas': [
                {'source': 'gdpr_article_6.txt', 'regulatory_framework': 'GDPR'},
                {'source': 'hipaa_overview.txt', 'regulatory_framework': 'HIPAA'}
            ],
            'distances': [[0.1, 0.3]],
            'ids': [['doc-1', 'doc-2']]
        }
        
        components = rag_system_components
        
        # Add documents
        await components["vector_store"].add_documents(sample_documents)
        
        # Test retrieval
        results = await components["retriever"].retrieve(
            "data protection requirements",
            top_k=5
        )
        
        # Verify retrieval results
        assert len(results) == 2
        assert results[0].score > results[1].score  # Results should be ordered by score
        assert all(hasattr(result, 'document') for result in results)
        assert all(hasattr(result, 'score') for result in results)
        assert all(hasattr(result, 'rank') for result in results)
    
    @patch('src.llama_mapper.rag.core.vector_store.chromadb')
    @pytest.mark.asyncio
    async def test_contextual_retrieval(self, mock_chromadb, rag_system_components):
        """Test contextual retrieval with filters."""
        # Mock ChromaDB
        mock_client = Mock()
        mock_collection = Mock()
        mock_chromadb.PersistentClient.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection
        
        # Mock filtered search results
        mock_collection.query.return_value = {
            'documents': [['GDPR Article 6 lawful basis']],
            'metadatas': [{'source': 'gdpr_article_6.txt', 'regulatory_framework': 'GDPR'}],
            'distances': [[0.1]],
            'ids': [['doc-1']]
        }
        
        components = rag_system_components
        
        # Test contextual retrieval
        context = {
            "regulatory_framework": "GDPR",
            "industry": "technology",
            "document_types": ["regulation"]
        }
        
        results = await components["retriever"].retrieve_with_context(
            "data processing requirements",
            context,
            top_k=5
        )
        
        # Verify contextual filtering
        assert len(results) == 1
        assert results[0].document.metadata.get('regulatory_framework') == 'GDPR'
    
    @pytest.mark.asyncio
    async def test_guardrails_integration(self, rag_system_components):
        """Test guardrails integration with RAG system."""
        from src.llama_mapper.rag.integration.model_enhancement import ExpertResponse, RAGContext
        
        components = rag_system_components
        
        # Create test expert response
        expert_response = ExpertResponse(
            analysis="Issue: Data processing\nRule: GDPR Article 6\nAnalysis: Requires lawful basis\nConclusion: Implement consent",
            recommendations=["Implement consent management", "Conduct DPIA"],
            risk_assessment={"regulatory_risk": "High", "implementation_risk": "Medium"},
            regulatory_citations=[
                {"title": "GDPR Article 6", "citation": "GDPR Art. 6", "date": "2024-01-01"}
            ],
            confidence_score=0.85,
            evidence_required=["Consent records", "DPIA documentation"],
            next_actions=[
                {"owner": "Compliance Officer", "due_by": "7 days", "evidence": "Consent audit"}
            ],
            jurisdictional_scope=["EU"],
            effective_dates=["2024-01-01"],
            implementation_complexity="High",
            cost_impact="Medium"
        )
        
        # Create test RAG context
        rag_context = RAGContext(
            query="GDPR data processing requirements",
            retrieved_documents=[
                Document(content="GDPR Article 6", source="gdpr.txt", regulatory_framework="GDPR")
            ],
            regulatory_framework="GDPR",
            industry="technology"
        )
        
        # Test guardrails evaluation
        result = await components["guardrails"].evaluate_response(expert_response, rag_context)
        
        # Verify guardrails work correctly
        assert result.passed is True
        assert result.confidence_score > 0.0
        assert len(result.violations) == 0
    
    @pytest.mark.asyncio
    async def test_quality_evaluation_integration(self, rag_system_components):
        """Test quality evaluation integration."""
        components = rag_system_components
        
        # This would be a more comprehensive test in a real implementation
        # For now, verify the quality evaluator is properly instantiated
        assert components["quality_evaluator"] is not None
    
    @patch('src.llama_mapper.rag.core.vector_store.chromadb')
    @pytest.mark.asyncio
    async def test_error_handling_integration(self, mock_chromadb, rag_system_components):
        """Test error handling across RAG system components."""
        # Mock ChromaDB with exception
        mock_chromadb.PersistentClient.side_effect = Exception("Database connection failed")
        
        components = rag_system_components
        
        # Test that system handles database errors gracefully
        with pytest.raises(Exception):
            await components["vector_store"]._get_client()
    
    @patch('src.llama_mapper.rag.core.vector_store.chromadb')
    @pytest.mark.asyncio
    async def test_health_check_integration(self, mock_chromadb, rag_system_components):
        """Test health check integration across components."""
        # Mock healthy ChromaDB
        mock_client = Mock()
        mock_collection = Mock()
        mock_chromadb.PersistentClient.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_collection.count.return_value = 10
        
        components = rag_system_components
        
        # Test health checks
        vector_store_healthy = await components["vector_store"].health_check()
        embedding_model_healthy = await components["embedding_model"].health_check()
        
        assert vector_store_healthy is True
        assert embedding_model_healthy is True
    
    @patch('src.llama_mapper.rag.core.vector_store.chromadb')
    @pytest.mark.asyncio
    async def test_multi_framework_processing(self, mock_chromadb, rag_system_components, sample_documents):
        """Test processing documents from multiple regulatory frameworks."""
        # Mock ChromaDB
        mock_client = Mock()
        mock_collection = Mock()
        mock_chromadb.PersistentClient.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection
        
        # Mock search results with multiple frameworks
        mock_collection.query.return_value = {
            'documents': [['GDPR Article 6', 'HIPAA safeguards', 'SOX Section 404']],
            'metadatas': [
                {'regulatory_framework': 'GDPR'},
                {'regulatory_framework': 'HIPAA'},
                {'regulatory_framework': 'SOX'}
            ],
            'distances': [[0.1, 0.2, 0.3]],
            'ids': [['doc-1', 'doc-2', 'doc-3']]
        }
        
        components = rag_system_components
        
        # Add documents from multiple frameworks
        await components["vector_store"].add_documents(sample_documents)
        
        # Test cross-framework query
        rag_context = await components["rag_enhancer"].enhance_query(
            "What are the compliance requirements for data protection?",
            {"industry": "technology"}
        )
        
        # Verify the system can handle multiple frameworks
        assert rag_context.query is not None
        assert rag_context.industry == "technology"
