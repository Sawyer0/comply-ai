"""
RAG Engine

This module provides the main RAG (Retrieval-Augmented Generation) engine
that integrates knowledge base, retrieval, and generation capabilities.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone

from .knowledge_base import KnowledgeBase, Document, DocumentChunk
from ..ml.model_server import ModelServer
from ..ml.embeddings import EmbeddingsManager

logger = logging.getLogger(__name__)


class RAGEngine:
    """
    Complete RAG engine for Analysis Service.

    Integrates:
    - Knowledge base management
    - Document retrieval
    - Context generation
    - Answer generation with citations
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rag_config = config.get("rag", {})

        # Configuration
        self.max_context_length = self.rag_config.get("max_context_length", 2048)
        self.retrieval_top_k = self.rag_config.get("retrieval_top_k", 5)
        self.similarity_threshold = self.rag_config.get("similarity_threshold", 0.7)
        self.enable_citations = self.rag_config.get("enable_citations", True)

        # Components
        self.knowledge_base = KnowledgeBase(config)
        self.model_server = ModelServer(config)
        self.embeddings_manager = EmbeddingsManager(config)

        # Performance tracking
        self.query_count = 0
        self.total_query_time = 0.0
        self.retrieval_stats = {"hits": 0, "misses": 0}

        logger.info(
            "RAG Engine initialized",
            max_context_length=self.max_context_length,
            retrieval_top_k=self.retrieval_top_k,
        )

    async def initialize(self):
        """Initialize RAG engine and all components."""
        try:
            # Initialize components
            await self.model_server.initialize()
            await self.embeddings_manager.initialize()

            # Load default knowledge base
            await self._load_default_knowledge()

            logger.info("RAG Engine initialized successfully")

        except Exception as e:
            logger.error("Failed to initialize RAG Engine", error=str(e))
            raise

    async def query(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None,
        document_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a query using RAG.

        Args:
            question: User question
            context: Additional context for the query
            document_type: Filter by document type

        Returns:
            RAG response with answer and citations
        """
        start_time = datetime.now(timezone.utc)

        try:
            # Step 1: Retrieve relevant documents
            retrieved_docs = await self._retrieve_documents(question, document_type)

            if not retrieved_docs:
                self.retrieval_stats["misses"] += 1
                return await self._generate_fallback_response(question, context)

            self.retrieval_stats["hits"] += 1

            # Step 2: Build context from retrieved documents
            context_text, citations = await self._build_context(
                retrieved_docs, question
            )

            # Step 3: Generate answer using context
            answer = await self._generate_answer(question, context_text, context)

            # Step 4: Create response
            response = {
                "answer": answer,
                "question": question,
                "context_used": (
                    context_text[:500] + "..."
                    if len(context_text) > 500
                    else context_text
                ),
                "citations": citations if self.enable_citations else [],
                "retrieved_documents": len(retrieved_docs),
                "confidence": self._calculate_response_confidence(
                    retrieved_docs, answer
                ),
                "metadata": {
                    "query_time": (
                        datetime.now(timezone.utc) - start_time
                    ).total_seconds(),
                    "retrieval_method": "semantic_search",
                    "generation_method": "rag",
                    "timestamp": start_time.isoformat(),
                },
            }

            # Track performance
            query_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            self.query_count += 1
            self.total_query_time += query_time

            logger.info(
                "RAG query processed",
                question_length=len(question),
                retrieved_docs=len(retrieved_docs),
                query_time=query_time,
            )

            return response

        except Exception as e:
            logger.error("RAG query failed", error=str(e))
            return await self._generate_error_response(question, str(e))

    async def add_knowledge(
        self,
        title: str,
        content: str,
        document_type: str = "regulatory",
        source: str = "manual",
        tags: List[str] = None,
        metadata: Dict[str, Any] = None,
    ) -> str:
        """
        Add knowledge to the RAG system.

        Args:
            title: Document title
            content: Document content
            document_type: Type of document
            source: Source of the document
            tags: Tags for categorization
            metadata: Additional metadata

        Returns:
            Document ID
        """
        try:
            # Add to knowledge base
            doc_id = await self.knowledge_base.add_document(
                title=title,
                content=content,
                document_type=document_type,
                source=source,
                tags=tags or [],
                metadata=metadata or {},
            )

            # Generate embeddings for the document chunks
            await self._generate_document_embeddings(doc_id)

            logger.info(
                "Knowledge added to RAG system",
                doc_id=doc_id,
                title=title,
                document_type=document_type,
            )

            return doc_id

        except Exception as e:
            logger.error("Failed to add knowledge to RAG system", error=str(e))
            raise

    async def update_knowledge(self, doc_id: str, **updates) -> bool:
        """Update existing knowledge."""
        try:
            success = await self.knowledge_base.update_document(doc_id, **updates)

            if success and "content" in updates:
                # Regenerate embeddings if content changed
                await self._generate_document_embeddings(doc_id)

            return success

        except Exception as e:
            logger.error("Failed to update knowledge", doc_id=doc_id, error=str(e))
            return False

    async def delete_knowledge(self, doc_id: str) -> bool:
        """Delete knowledge from the RAG system."""
        try:
            return await self.knowledge_base.delete_document(doc_id)

        except Exception as e:
            logger.error("Failed to delete knowledge", doc_id=doc_id, error=str(e))
            return False

    async def search_knowledge(
        self, query: str, document_type: str = None, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search knowledge base.

        Args:
            query: Search query
            document_type: Filter by document type
            limit: Maximum results

        Returns:
            List of matching documents with metadata
        """
        try:
            documents = await self.knowledge_base.search_documents(
                query=query, document_type=document_type, limit=limit
            )

            # Convert to dict format with relevance scores
            results = []
            for doc in documents:
                results.append(
                    {
                        "id": doc.id,
                        "title": doc.title,
                        "content_preview": (
                            doc.content[:200] + "..."
                            if len(doc.content) > 200
                            else doc.content
                        ),
                        "document_type": doc.document_type,
                        "source": doc.source,
                        "tags": doc.tags,
                        "created_at": doc.created_at.isoformat(),
                        "updated_at": doc.updated_at.isoformat(),
                    }
                )

            return results

        except Exception as e:
            logger.error("Knowledge search failed", error=str(e))
            return []

    async def _retrieve_documents(
        self, question: str, document_type: Optional[str] = None
    ) -> List[DocumentChunk]:
        """Retrieve relevant documents for the question."""
        try:
            # Use semantic search with embeddings
            question_embedding = await self.embeddings_manager.embed_text(question)

            # Get all chunks (in production, would use vector database)
            all_chunks = list(self.knowledge_base.chunks.values())

            # Filter by document type if specified
            if document_type:
                filtered_chunks = []
                for chunk in all_chunks:
                    doc = await self.knowledge_base.get_document(chunk.document_id)
                    if doc and doc.document_type == document_type:
                        filtered_chunks.append(chunk)
                all_chunks = filtered_chunks

            # Calculate similarities (simplified - in production would use vector search)
            chunk_similarities = []
            for chunk in all_chunks:
                if chunk.embedding:
                    # Would use proper vector similarity in production
                    similarity = self._calculate_text_similarity(
                        question, chunk.content
                    )
                    if similarity >= self.similarity_threshold:
                        chunk_similarities.append((chunk, similarity))

            # Sort by similarity and return top_k
            chunk_similarities.sort(key=lambda x: x[1], reverse=True)
            retrieved_chunks = [
                chunk for chunk, _ in chunk_similarities[: self.retrieval_top_k]
            ]

            return retrieved_chunks

        except Exception as e:
            logger.error("Document retrieval failed", error=str(e))
            return []

    async def _build_context(
        self, retrieved_docs: List[DocumentChunk], question: str
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Build context from retrieved documents."""
        try:
            context_parts = []
            citations = []
            total_length = 0

            for i, chunk in enumerate(retrieved_docs):
                # Get parent document for citation
                document = await self.knowledge_base.get_document(chunk.document_id)

                if (
                    document
                    and total_length + len(chunk.content) <= self.max_context_length
                ):
                    context_parts.append(f"[{i+1}] {chunk.content}")

                    citations.append(
                        {
                            "id": i + 1,
                            "document_id": document.id,
                            "document_title": document.title,
                            "document_type": document.document_type,
                            "source": document.source,
                            "chunk_id": chunk.id,
                            "content_preview": (
                                chunk.content[:100] + "..."
                                if len(chunk.content) > 100
                                else chunk.content
                            ),
                        }
                    )

                    total_length += len(chunk.content)
                else:
                    break

            context_text = "\n\n".join(context_parts)
            return context_text, citations

        except Exception as e:
            logger.error("Context building failed", error=str(e))
            return "", []

    async def _generate_answer(
        self,
        question: str,
        context: str,
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate answer using the model server."""
        try:
            # Build prompt
            prompt = self._build_rag_prompt(question, context, additional_context)

            # Generate answer
            result = await self.model_server.generate_analysis(
                prompt=prompt,
                max_tokens=512,
                temperature=0.7,
                stop=["[END]", "\n\nQuestion:", "\n\nContext:"],
            )

            answer = result.get("generated_text", "").strip()

            # Clean up answer
            answer = self._clean_generated_answer(answer)

            return answer

        except Exception as e:
            logger.error("Answer generation failed", error=str(e))
            return "I apologize, but I encountered an error while generating the answer. Please try again."

    def _build_rag_prompt(
        self,
        question: str,
        context: str,
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build RAG prompt for the model."""
        prompt_parts = [
            "You are a helpful assistant that answers questions based on the provided context.",
            "Use only the information from the context to answer the question.",
            "If the context doesn't contain enough information, say so clearly.",
            "",
            "Context:",
            context,
            "",
            f"Question: {question}",
            "",
            "Answer:",
        ]

        # Add additional context if provided
        if additional_context:
            context_info = []
            for key, value in additional_context.items():
                context_info.append(f"{key}: {value}")

            if context_info:
                prompt_parts.insert(
                    -3, f"Additional Context: {'; '.join(context_info)}"
                )
                prompt_parts.insert(-3, "")

        return "\n".join(prompt_parts)

    def _clean_generated_answer(self, answer: str) -> str:
        """Clean up generated answer."""
        # Remove common artifacts
        answer = answer.replace("[END]", "").strip()

        # Remove incomplete sentences at the end
        sentences = answer.split(". ")
        if len(sentences) > 1 and not sentences[-1].endswith((".", "!", "?")):
            answer = ". ".join(sentences[:-1]) + "."

        return answer

    async def _generate_fallback_response(
        self, question: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate fallback response when no relevant documents found."""
        fallback_answer = (
            "I don't have specific information in my knowledge base to answer this question. "
            "For accurate compliance and regulatory guidance, please consult the relevant "
            "official documentation or speak with a compliance expert."
        )

        return {
            "answer": fallback_answer,
            "question": question,
            "context_used": "",
            "citations": [],
            "retrieved_documents": 0,
            "confidence": 0.1,
            "metadata": {
                "query_time": 0.0,
                "retrieval_method": "fallback",
                "generation_method": "template",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        }

    async def _generate_error_response(
        self, question: str, error: str
    ) -> Dict[str, Any]:
        """Generate error response."""
        return {
            "answer": "I encountered an error while processing your question. Please try again.",
            "question": question,
            "context_used": "",
            "citations": [],
            "retrieved_documents": 0,
            "confidence": 0.0,
            "error": error,
            "metadata": {
                "query_time": 0.0,
                "retrieval_method": "error",
                "generation_method": "error",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        }

    def _calculate_response_confidence(
        self, retrieved_docs: List[DocumentChunk], answer: str
    ) -> float:
        """Calculate confidence in the response."""
        try:
            # Base confidence from number of retrieved documents
            doc_confidence = min(1.0, len(retrieved_docs) / self.retrieval_top_k)

            # Confidence from answer length (reasonable answers should have some length)
            length_confidence = min(1.0, len(answer) / 100)  # Normalize to 100 chars

            # Combined confidence
            confidence = (doc_confidence + length_confidence) / 2

            return confidence

        except Exception:
            return 0.5

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity calculation (fallback)."""
        try:
            # Simple word overlap similarity
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())

            if not words1 or not words2:
                return 0.0

            intersection = words1.intersection(words2)
            union = words1.union(words2)

            return len(intersection) / len(union) if union else 0.0

        except Exception:
            return 0.0

    async def _generate_document_embeddings(self, doc_id: str):
        """Generate embeddings for document chunks."""
        try:
            chunks = await self.knowledge_base.get_document_chunks(doc_id)

            if chunks:
                # Generate embeddings for all chunks
                chunk_texts = [chunk.content for chunk in chunks]
                embeddings = await self.embeddings_manager.embed_texts(chunk_texts)

                # Store embeddings in chunks
                for chunk, embedding in zip(chunks, embeddings):
                    chunk.embedding = embedding.tolist()

                logger.debug(
                    "Generated embeddings for document",
                    doc_id=doc_id,
                    chunks=len(chunks),
                )

        except Exception as e:
            logger.error(
                "Failed to generate document embeddings", doc_id=doc_id, error=str(e)
            )

    async def _load_default_knowledge(self):
        """Load default regulatory knowledge."""
        try:
            # Add some default regulatory knowledge
            default_docs = [
                {
                    "title": "SOC 2 Type II Overview",
                    "content": """SOC 2 Type II is an auditing procedure that ensures your service providers securely manage your data to protect the interests of your organization and the privacy of its clients. SOC 2 compliance is determined by the American Institute of CPAs (AICPA) auditing standards and is broken into two types: Type I and Type II. Type II reports are more comprehensive and examine the operational effectiveness of controls over a period of time.""",
                    "document_type": "regulatory",
                    "source": "AICPA",
                    "tags": ["soc2", "compliance", "audit"],
                },
                {
                    "title": "GDPR Data Protection Principles",
                    "content": """The General Data Protection Regulation (GDPR) sets out seven key principles: 1) Lawfulness, fairness and transparency, 2) Purpose limitation, 3) Data minimisation, 4) Accuracy, 5) Storage limitation, 6) Integrity and confidentiality (security), 7) Accountability. Organizations must be able to demonstrate compliance with these principles.""",
                    "document_type": "regulatory",
                    "source": "EU GDPR",
                    "tags": ["gdpr", "privacy", "data-protection"],
                },
                {
                    "title": "HIPAA Security Rule Requirements",
                    "content": """The HIPAA Security Rule requires covered entities to maintain reasonable and appropriate administrative, technical, and physical safeguards for protecting e-PHI. Administrative safeguards include security management processes, assigned security responsibilities, workforce training, and access management procedures.""",
                    "document_type": "regulatory",
                    "source": "HHS",
                    "tags": ["hipaa", "healthcare", "security"],
                },
            ]

            for doc_info in default_docs:
                await self.add_knowledge(**doc_info)

            logger.info(
                "Default regulatory knowledge loaded", documents=len(default_docs)
            )

        except Exception as e:
            logger.error("Failed to load default knowledge", error=str(e))

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get RAG engine performance metrics."""
        avg_query_time = (
            self.total_query_time / self.query_count if self.query_count > 0 else 0
        )
        retrieval_hit_rate = (
            (
                self.retrieval_stats["hits"]
                / (self.retrieval_stats["hits"] + self.retrieval_stats["misses"])
            )
            if (self.retrieval_stats["hits"] + self.retrieval_stats["misses"]) > 0
            else 0
        )

        return {
            "query_count": self.query_count,
            "average_query_time": avg_query_time,
            "retrieval_hit_rate": retrieval_hit_rate,
            "knowledge_base_stats": asyncio.create_task(
                self.knowledge_base.get_statistics()
            ),
            "model_server_stats": self.model_server.get_performance_metrics(),
            "embeddings_stats": self.embeddings_manager.get_performance_metrics(),
            "configuration": {
                "max_context_length": self.max_context_length,
                "retrieval_top_k": self.retrieval_top_k,
                "similarity_threshold": self.similarity_threshold,
                "enable_citations": self.enable_citations,
            },
        }

    async def shutdown(self):
        """Gracefully shutdown RAG engine."""
        try:
            logger.info("Shutting down RAG Engine...")

            # Shutdown components
            await self.model_server.shutdown()
            await self.embeddings_manager.shutdown()

            logger.info("RAG Engine shutdown complete")

        except Exception as e:
            logger.error("Error during RAG Engine shutdown", error=str(e))
