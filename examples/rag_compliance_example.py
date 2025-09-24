"""
Comprehensive example of RAG-enhanced compliance AI system.

This example demonstrates the complete RAG system implementation including:
- Document ingestion and processing
- RAG-enhanced query processing
- Expert-level compliance guidance
- Guardrails and quality evaluation
- Fine-tuning pipeline
"""

import asyncio
import logging
import json
from pathlib import Path
from typing import List, Dict, Any

# Import RAG components
from src.llama_mapper.rag.core.vector_store import ChromaDBVectorStore, Document
from src.llama_mapper.rag.core.embeddings import SentenceTransformerEmbeddings
from src.llama_mapper.rag.core.retriever import SemanticRetriever
from src.llama_mapper.rag.integration.model_enhancement import RAGModelEnhancer
from src.llama_mapper.rag.guardrails.compliance_guardrails import ComplianceGuardrails
from src.llama_mapper.rag.evaluation.quality_metrics import RAGQualityEvaluator
from src.llama_mapper.rag.training.fine_tuning_pipeline import (
    ComplianceFineTuningPipeline, 
    FineTuningConfig,
    ComplianceModelTrainer
)
from src.llama_mapper.rag.knowledge_base.document_processor import (
    DocumentProcessor, 
    RegulatoryDocumentProcessor
)
from src.llama_mapper.rag.knowledge_base.chunking import FixedSizeChunker
from src.llama_mapper.rag.knowledge_base.metadata_extractor import RegulatoryMetadataExtractor
from src.llama_mapper.rag.knowledge_base.schema_validator import RegulatorySchemaValidator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComplianceRAGSystem:
    """Complete RAG system for compliance AI."""
    
    def __init__(self):
        """Initialize the RAG system."""
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.vector_store = None
        self.embedding_model = None
        self.retriever = None
        self.rag_enhancer = None
        self.guardrails = None
        self.quality_evaluator = None
        
        # Document processing components
        self.document_processor = None
        
    async def initialize(self):
        """Initialize all RAG system components."""
        try:
            self.logger.info("Initializing RAG system...")
            
            # Initialize vector store
            self.vector_store = ChromaDBVectorStore(
                collection_name="compliance_knowledge",
                persist_directory="./chroma_db"
            )
            
            # Initialize embedding model
            self.embedding_model = SentenceTransformerEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            # Initialize retriever
            self.retriever = SemanticRetriever(
                vector_store=self.vector_store,
                embedding_model=self.embedding_model
            )
            
            # Initialize RAG enhancer
            self.rag_enhancer = RAGModelEnhancer(
                vector_store=self.vector_store,
                retriever=self.retriever
            )
            
            # Initialize guardrails
            self.guardrails = ComplianceGuardrails()
            
            # Initialize quality evaluator
            self.quality_evaluator = RAGQualityEvaluator()
            
            # Initialize document processor
            chunker = FixedSizeChunker(chunk_size=1000, overlap=200)
            metadata_extractor = RegulatoryMetadataExtractor()
            schema_validator = RegulatorySchemaValidator()
            self.document_processor = RegulatoryDocumentProcessor(
                chunker=chunker,
                metadata_extractor=metadata_extractor,
                schema_validator=schema_validator
            )
            
            self.logger.info("RAG system initialization complete")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize RAG system: {e}")
            raise
    
    async def ingest_regulatory_documents(self, documents: List[Dict[str, Any]]):
        """Ingest regulatory documents into the knowledge base."""
        try:
            self.logger.info(f"Ingesting {len(documents)} regulatory documents...")
            
            all_processed_docs = []
            
            for doc_info in documents:
                # Process regulatory document
                result = await self.document_processor.process_regulatory_document(
                    file_path=doc_info["file_path"],
                    regulatory_framework=doc_info["regulatory_framework"],
                    document_type=doc_info["document_type"],
                    industry=doc_info.get("industry")
                )
                
                if result.success:
                    all_processed_docs.extend(result.documents)
                    self.logger.info(f"Successfully processed {doc_info['file_path']} into {len(result.documents)} chunks")
                else:
                    self.logger.error(f"Failed to process {doc_info['file_path']}: {result.errors}")
            
            # Add all documents to vector store
            if all_processed_docs:
                await self.vector_store.add_documents(all_processed_docs)
                self.logger.info(f"Added {len(all_processed_docs)} document chunks to vector store")
            
        except Exception as e:
            self.logger.error(f"Document ingestion failed: {e}")
            raise
    
    async def query_compliance_guidance(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Query the RAG system for compliance guidance."""
        try:
            self.logger.info(f"Processing compliance query: {query}")
            
            # Enhance query with RAG
            rag_context = await self.rag_enhancer.enhance_query(query, context or {})
            
            # Generate expert response
            expert_response = await self.rag_enhancer.generate_expert_response(rag_context)
            
            # Evaluate guardrails
            guardrail_result = await self.guardrails.evaluate_response(expert_response, rag_context)
            
            # Build response
            response = {
                "query": query,
                "analysis": expert_response.analysis,
                "recommendations": expert_response.recommendations,
                "risk_assessment": expert_response.risk_assessment,
                "regulatory_citations": expert_response.regulatory_citations,
                "evidence_required": expert_response.evidence_required,
                "next_actions": expert_response.next_actions,
                "jurisdictional_scope": expert_response.jurisdictional_scope,
                "effective_dates": expert_response.effective_dates,
                "implementation_complexity": expert_response.implementation_complexity,
                "cost_impact": expert_response.cost_impact,
                "confidence_score": expert_response.confidence_score,
                "guardrail_result": guardrail_result.to_dict(),
                "knowledge_coverage": rag_context.knowledge_coverage,
                "processing_time_ms": rag_context.processing_time_ms
            }
            
            self.logger.info(f"Generated compliance guidance with confidence: {expert_response.confidence_score:.2f}")
            
            return response
            
        except Exception as e:
            self.logger.error(f"Compliance query failed: {e}")
            raise
    
    async def evaluate_system_quality(self, test_queries: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate system quality on test queries."""
        try:
            self.logger.info(f"Evaluating system quality on {len(test_queries)} test queries...")
            
            quality_scores = []
            
            for test_case in test_queries:
                # Process query
                response = await self.query_compliance_guidance(
                    test_case["query"], 
                    test_case.get("context", {})
                )
                
                # Evaluate quality
                quality_metrics = await self.quality_evaluator.evaluate_comprehensive(
                    query=test_case["query"],
                    retrieved_docs=[],  # Would be populated from actual retrieval
                    response=response,
                    ground_truth=test_case.get("ground_truth", {})
                )
                
                quality_scores.append(quality_metrics.overall_score)
            
            avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
            
            return {
                "average_quality": avg_quality,
                "quality_scores": quality_scores,
                "num_queries": len(test_queries)
            }
            
        except Exception as e:
            self.logger.error(f"Quality evaluation failed: {e}")
            raise
    
    async def train_compliance_model(self, config: FineTuningConfig = None):
        """Train a compliance model using fine-tuning."""
        try:
            self.logger.info("Starting compliance model training...")
            
            # Use default config if none provided
            if config is None:
                config = FineTuningConfig(
                    model_name="microsoft/DialoGPT-medium",
                    use_lora=True,
                    num_epochs=3,
                    batch_size=4,
                    learning_rate=2e-4,
                    output_dir="./fine_tuned_models"
                )
            
            # Create trainer
            trainer = ComplianceModelTrainer(config)
            
            # Train model
            result = await trainer.train_compliance_model()
            
            if result.success:
                self.logger.info(f"Model training completed successfully. Model saved to: {result.model_path}")
                self.logger.info(f"Training time: {result.training_time:.2f} seconds")
                self.logger.info(f"Final loss: {result.training_loss:.4f}")
            else:
                self.logger.error("Model training failed")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Model training failed: {e}")
            raise


async def main():
    """Main example demonstrating the complete RAG system."""
    try:
        # Initialize RAG system
        rag_system = ComplianceRAGSystem()
        await rag_system.initialize()
        
        # Example regulatory documents
        regulatory_documents = [
            {
                "file_path": "examples/sample_documents/gdpr_regulation.txt",
                "regulatory_framework": "GDPR",
                "document_type": "regulation",
                "industry": "technology"
            },
            {
                "file_path": "examples/sample_documents/hipaa_guidance.txt",
                "regulatory_framework": "HIPAA",
                "document_type": "guidance",
                "industry": "healthcare"
            },
            {
                "file_path": "examples/sample_documents/sox_requirements.txt",
                "regulatory_framework": "SOX",
                "document_type": "regulation",
                "industry": "financial_services"
            }
        ]
        
        # Ingest documents
        print("📚 Ingesting regulatory documents...")
        await rag_system.ingest_regulatory_documents(regulatory_documents)
        
        # Example compliance queries
        compliance_queries = [
            {
                "query": "What are the data protection requirements for processing personal data under GDPR?",
                "context": {
                    "regulatory_framework": "GDPR",
                    "industry": "technology"
                }
            },
            {
                "query": "How should we handle patient data breaches under HIPAA?",
                "context": {
                    "regulatory_framework": "HIPAA",
                    "industry": "healthcare"
                }
            },
            {
                "query": "What internal controls are required for financial reporting under SOX?",
                "context": {
                    "regulatory_framework": "SOX",
                    "industry": "financial_services"
                }
            }
        ]
        
        # Process compliance queries
        print("\n🔍 Processing compliance queries...")
        for i, query_info in enumerate(compliance_queries, 1):
            print(f"\n--- Query {i} ---")
            print(f"Query: {query_info['query']}")
            
            response = await rag_system.query_compliance_guidance(
                query_info["query"], 
                query_info["context"]
            )
            
            print(f"Analysis: {response['analysis'][:200]}...")
            print(f"Recommendations: {response['recommendations'][:3]}")
            print(f"Risk Assessment: {response['risk_assessment']}")
            print(f"Confidence Score: {response['confidence_score']:.2f}")
            print(f"Guardrail Result: {'PASSED' if response['guardrail_result']['passed'] else 'FAILED'}")
        
        # Example fine-tuning
        print("\n🤖 Training compliance model...")
        training_result = await rag_system.train_compliance_model()
        
        if training_result.success:
            print(f"✅ Model training completed successfully!")
            print(f"Model saved to: {training_result.model_path}")
            print(f"Training time: {training_result.training_time:.2f} seconds")
        else:
            print("❌ Model training failed")
        
        # Example quality evaluation
        print("\n📊 Evaluating system quality...")
        test_queries = [
            {
                "query": "What are the key compliance requirements for data processing?",
                "context": {"regulatory_framework": "GDPR"},
                "ground_truth": {"expected_framework": "GDPR", "expected_risk_level": "high"}
            }
        ]
        
        quality_results = await rag_system.evaluate_system_quality(test_queries)
        print(f"Average Quality Score: {quality_results['average_quality']:.2f}")
        
        print("\n🎉 RAG system demonstration completed successfully!")
        
    except Exception as e:
        logger.error(f"RAG system demonstration failed: {e}")
        raise


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())
