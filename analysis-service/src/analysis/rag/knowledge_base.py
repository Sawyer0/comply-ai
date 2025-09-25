"""
Knowledge Base Management

This module provides knowledge base management for the RAG system,
including document storage, indexing, and retrieval optimization.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
import json
import hashlib
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Document representation in the knowledge base."""

    id: str
    title: str
    content: str
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    document_type: str
    source: str
    tags: List[str]
    embedding: Optional[List[float]] = None


@dataclass
class DocumentChunk:
    """Document chunk for efficient retrieval."""

    id: str
    document_id: str
    content: str
    chunk_index: int
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None


class KnowledgeBase:
    """
    Knowledge base management system for RAG.

    Manages:
    - Document storage and indexing
    - Document chunking and embedding
    - Metadata management
    - Search and retrieval optimization
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.kb_config = config.get("knowledge_base", {})

        # Configuration
        self.chunk_size = self.kb_config.get("chunk_size", 512)
        self.chunk_overlap = self.kb_config.get("chunk_overlap", 50)
        self.max_documents = self.kb_config.get("max_documents", 10000)

        # Storage
        self.documents = {}  # In production, would use proper database
        self.chunks = {}
        self.document_index = {}  # For fast lookup
        self.chunk_index = {}

        # Statistics
        self.document_count = 0
        self.chunk_count = 0
        self.total_content_size = 0

        logger.info(
            "Knowledge Base initialized",
            chunk_size=self.chunk_size,
            max_documents=self.max_documents,
        )

    async def add_document(
        self,
        title: str,
        content: str,
        document_type: str = "general",
        source: str = "unknown",
        tags: List[str] = None,
        metadata: Dict[str, Any] = None,
    ) -> str:
        """
        Add a document to the knowledge base.

        Args:
            title: Document title
            content: Document content
            document_type: Type of document (regulatory, policy, etc.)
            source: Source of the document
            tags: List of tags for categorization
            metadata: Additional metadata

        Returns:
            Document ID
        """
        try:
            # Generate document ID
            doc_id = self._generate_document_id(title, content)

            # Check if document already exists
            if doc_id in self.documents:
                logger.warning("Document already exists", doc_id=doc_id)
                return doc_id

            # Create document
            document = Document(
                id=doc_id,
                title=title,
                content=content,
                metadata=metadata or {},
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                document_type=document_type,
                source=source,
                tags=tags or [],
            )

            # Store document
            self.documents[doc_id] = document
            self.document_index[title.lower()] = doc_id

            # Create chunks
            chunks = await self._create_document_chunks(document)
            for chunk in chunks:
                self.chunks[chunk.id] = chunk
                self.chunk_index[chunk.id] = doc_id

            # Update statistics
            self.document_count += 1
            self.chunk_count += len(chunks)
            self.total_content_size += len(content)

            logger.info(
                "Document added to knowledge base",
                doc_id=doc_id,
                title=title,
                chunks_created=len(chunks),
            )

            return doc_id

        except Exception as e:
            logger.error("Failed to add document to knowledge base", error=str(e))
            raise

    async def get_document(self, doc_id: str) -> Optional[Document]:
        """Get document by ID."""
        return self.documents.get(doc_id)

    async def search_documents(
        self,
        query: str,
        document_type: str = None,
        tags: List[str] = None,
        limit: int = 10,
    ) -> List[Document]:
        """
        Search documents by query and filters.

        Args:
            query: Search query
            document_type: Filter by document type
            tags: Filter by tags
            limit: Maximum number of results

        Returns:
            List of matching documents
        """
        try:
            matching_docs = []
            query_lower = query.lower()

            for document in self.documents.values():
                # Check type filter
                if document_type and document.document_type != document_type:
                    continue

                # Check tags filter
                if tags and not any(tag in document.tags for tag in tags):
                    continue

                # Check content match
                if (
                    query_lower in document.title.lower()
                    or query_lower in document.content.lower()
                ):
                    matching_docs.append(document)

            # Sort by relevance (simplified)
            matching_docs.sort(key=lambda d: self._calculate_relevance_score(d, query))

            return matching_docs[:limit]

        except Exception as e:
            logger.error("Document search failed", error=str(e))
            return []

    async def get_document_chunks(self, doc_id: str) -> List[DocumentChunk]:
        """Get all chunks for a document."""
        return [chunk for chunk in self.chunks.values() if chunk.document_id == doc_id]

    async def search_chunks(
        self, query: str, document_type: str = None, limit: int = 20
    ) -> List[DocumentChunk]:
        """
        Search document chunks by query.

        Args:
            query: Search query
            document_type: Filter by document type
            limit: Maximum number of results

        Returns:
            List of matching chunks
        """
        try:
            matching_chunks = []
            query_lower = query.lower()

            for chunk in self.chunks.values():
                # Get parent document for filtering
                document = self.documents.get(chunk.document_id)
                if not document:
                    continue

                # Check type filter
                if document_type and document.document_type != document_type:
                    continue

                # Check content match
                if query_lower in chunk.content.lower():
                    matching_chunks.append(chunk)

            # Sort by relevance
            matching_chunks.sort(
                key=lambda c: self._calculate_chunk_relevance(c, query)
            )

            return matching_chunks[:limit]

        except Exception as e:
            logger.error("Chunk search failed", error=str(e))
            return []

    async def update_document(self, doc_id: str, **updates) -> bool:
        """Update document fields."""
        try:
            if doc_id not in self.documents:
                return False

            document = self.documents[doc_id]

            # Update fields
            for field, value in updates.items():
                if hasattr(document, field):
                    setattr(document, field, value)

            document.updated_at = datetime.now(timezone.utc)

            # If content changed, recreate chunks
            if "content" in updates:
                # Remove old chunks
                old_chunks = await self.get_document_chunks(doc_id)
                for chunk in old_chunks:
                    del self.chunks[chunk.id]
                    del self.chunk_index[chunk.id]

                # Create new chunks
                new_chunks = await self._create_document_chunks(document)
                for chunk in new_chunks:
                    self.chunks[chunk.id] = chunk
                    self.chunk_index[chunk.id] = doc_id

                # Update statistics
                self.chunk_count = self.chunk_count - len(old_chunks) + len(new_chunks)

            logger.info("Document updated", doc_id=doc_id, fields=list(updates.keys()))
            return True

        except Exception as e:
            logger.error("Failed to update document", doc_id=doc_id, error=str(e))
            return False

    async def delete_document(self, doc_id: str) -> bool:
        """Delete document and its chunks."""
        try:
            if doc_id not in self.documents:
                return False

            document = self.documents[doc_id]

            # Remove chunks
            chunks = await self.get_document_chunks(doc_id)
            for chunk in chunks:
                del self.chunks[chunk.id]
                del self.chunk_index[chunk.id]

            # Remove document
            del self.documents[doc_id]

            # Remove from index
            title_key = document.title.lower()
            if title_key in self.document_index:
                del self.document_index[title_key]

            # Update statistics
            self.document_count -= 1
            self.chunk_count -= len(chunks)
            self.total_content_size -= len(document.content)

            logger.info("Document deleted", doc_id=doc_id, chunks_removed=len(chunks))
            return True

        except Exception as e:
            logger.error("Failed to delete document", doc_id=doc_id, error=str(e))
            return False

    async def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        document_types = {}
        sources = {}

        for document in self.documents.values():
            document_types[document.document_type] = (
                document_types.get(document.document_type, 0) + 1
            )
            sources[document.source] = sources.get(document.source, 0) + 1

        return {
            "document_count": self.document_count,
            "chunk_count": self.chunk_count,
            "total_content_size": self.total_content_size,
            "average_document_size": (
                self.total_content_size / self.document_count
                if self.document_count > 0
                else 0
            ),
            "average_chunks_per_document": (
                self.chunk_count / self.document_count if self.document_count > 0 else 0
            ),
            "document_types": document_types,
            "sources": sources,
            "configuration": {
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "max_documents": self.max_documents,
            },
        }

    async def _create_document_chunks(self, document: Document) -> List[DocumentChunk]:
        """Create chunks from document content."""
        try:
            chunks = []
            content = document.content

            # Simple text chunking
            start = 0
            chunk_index = 0

            while start < len(content):
                # Calculate chunk end
                end = min(start + self.chunk_size, len(content))

                # Try to break at word boundary
                if end < len(content):
                    # Look for last space within overlap range
                    for i in range(end, max(end - self.chunk_overlap, start), -1):
                        if content[i].isspace():
                            end = i
                            break

                # Extract chunk content
                chunk_content = content[start:end].strip()

                if chunk_content:  # Only create non-empty chunks
                    chunk_id = f"{document.id}_chunk_{chunk_index}"

                    chunk = DocumentChunk(
                        id=chunk_id,
                        document_id=document.id,
                        content=chunk_content,
                        chunk_index=chunk_index,
                        metadata={
                            "document_title": document.title,
                            "document_type": document.document_type,
                            "source": document.source,
                            "tags": document.tags,
                            "start_position": start,
                            "end_position": end,
                        },
                    )

                    chunks.append(chunk)
                    chunk_index += 1

                # Move to next chunk with overlap
                start = max(end - self.chunk_overlap, start + 1)

                # Prevent infinite loop
                if start >= len(content):
                    break

            return chunks

        except Exception as e:
            logger.error(
                "Failed to create document chunks", doc_id=document.id, error=str(e)
            )
            return []

    def _generate_document_id(self, title: str, content: str) -> str:
        """Generate unique document ID."""
        content_hash = hashlib.sha256(f"{title}:{content}".encode()).hexdigest()
        return f"doc_{content_hash[:16]}"

    def _calculate_relevance_score(self, document: Document, query: str) -> float:
        """Calculate relevance score for document."""
        score = 0.0
        query_lower = query.lower()

        # Title match (higher weight)
        if query_lower in document.title.lower():
            score += 2.0

        # Content match
        content_lower = document.content.lower()
        query_words = query_lower.split()

        for word in query_words:
            if word in content_lower:
                score += 1.0

        # Tag match
        for tag in document.tags:
            if query_lower in tag.lower():
                score += 1.5

        return score

    def _calculate_chunk_relevance(self, chunk: DocumentChunk, query: str) -> float:
        """Calculate relevance score for chunk."""
        score = 0.0
        query_lower = query.lower()
        content_lower = chunk.content.lower()

        # Count query word occurrences
        query_words = query_lower.split()
        for word in query_words:
            score += content_lower.count(word)

        # Boost score for exact phrase match
        if query_lower in content_lower:
            score += 5.0

        return score

    async def clear_knowledge_base(self):
        """Clear all documents and chunks."""
        self.documents.clear()
        self.chunks.clear()
        self.document_index.clear()
        self.chunk_index.clear()

        self.document_count = 0
        self.chunk_count = 0
        self.total_content_size = 0

        logger.info("Knowledge base cleared")

    async def export_knowledge_base(self) -> Dict[str, Any]:
        """Export knowledge base data."""
        try:
            export_data = {
                "documents": {},
                "chunks": {},
                "statistics": await self.get_statistics(),
                "exported_at": datetime.now(timezone.utc).isoformat(),
            }

            # Export documents
            for doc_id, document in self.documents.items():
                doc_dict = asdict(document)
                doc_dict["created_at"] = document.created_at.isoformat()
                doc_dict["updated_at"] = document.updated_at.isoformat()
                export_data["documents"][doc_id] = doc_dict

            # Export chunks
            for chunk_id, chunk in self.chunks.items():
                export_data["chunks"][chunk_id] = asdict(chunk)

            return export_data

        except Exception as e:
            logger.error("Failed to export knowledge base", error=str(e))
            raise

    async def import_knowledge_base(self, import_data: Dict[str, Any]):
        """Import knowledge base data."""
        try:
            # Clear existing data
            await self.clear_knowledge_base()

            # Import documents
            for doc_id, doc_data in import_data.get("documents", {}).items():
                document = Document(
                    id=doc_data["id"],
                    title=doc_data["title"],
                    content=doc_data["content"],
                    metadata=doc_data["metadata"],
                    created_at=datetime.fromisoformat(doc_data["created_at"]),
                    updated_at=datetime.fromisoformat(doc_data["updated_at"]),
                    document_type=doc_data["document_type"],
                    source=doc_data["source"],
                    tags=doc_data["tags"],
                    embedding=doc_data.get("embedding"),
                )
                self.documents[doc_id] = document
                self.document_index[document.title.lower()] = doc_id

            # Import chunks
            for chunk_id, chunk_data in import_data.get("chunks", {}).items():
                chunk = DocumentChunk(
                    id=chunk_data["id"],
                    document_id=chunk_data["document_id"],
                    content=chunk_data["content"],
                    chunk_index=chunk_data["chunk_index"],
                    metadata=chunk_data["metadata"],
                    embedding=chunk_data.get("embedding"),
                )
                self.chunks[chunk_id] = chunk
                self.chunk_index[chunk_id] = chunk.document_id

            # Update statistics
            self.document_count = len(self.documents)
            self.chunk_count = len(self.chunks)
            self.total_content_size = sum(
                len(doc.content) for doc in self.documents.values()
            )

            logger.info(
                "Knowledge base imported",
                documents=self.document_count,
                chunks=self.chunk_count,
            )

        except Exception as e:
            logger.error("Failed to import knowledge base", error=str(e))
            raise
