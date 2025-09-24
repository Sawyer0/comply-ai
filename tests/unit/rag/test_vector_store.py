"""
Unit tests for vector store implementations.

Tests all vector store functionality including document management,
search, and health checking.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
from typing import List, Dict, Any

from src.llama_mapper.rag.core.vector_store import (
    Document, SearchResult, VectorStore, ChromaDBVectorStore
)


class TestDocument:
    """Test Document dataclass."""
    
    def test_document_creation(self):
        """Test document creation with default values."""
        doc = Document(content="test content", source="test source")
        
        assert doc.content == "test content"
        assert doc.source == "test source"
        assert doc.id is not None
        assert isinstance(doc.created_at, datetime)
        assert doc.metadata == {}
    
    def test_document_to_dict(self):
        """Test document serialization to dictionary."""
        doc = Document(
            content="test content",
            source="test source",
            document_type="regulation",
            regulatory_framework="GDPR"
        )
        
        doc_dict = doc.to_dict()
        
        assert doc_dict["content"] == "test content"
        assert doc_dict["source"] == "test source"
        assert doc_dict["document_type"] == "regulation"
        assert doc_dict["regulatory_framework"] == "GDPR"
        assert "created_at" in doc_dict
    
    def test_document_from_dict(self):
        """Test document creation from dictionary."""
        doc_dict = {
            "id": "test-id",
            "content": "test content",
            "source": "test source",
            "document_type": "regulation",
            "regulatory_framework": "GDPR",
            "created_at": datetime.utcnow().isoformat()
        }
        
        doc = Document.from_dict(doc_dict)
        
        assert doc.id == "test-id"
        assert doc.content == "test content"
        assert doc.source == "test source"
        assert doc.document_type == "regulation"
        assert doc.regulatory_framework == "GDPR"


class TestSearchResult:
    """Test SearchResult dataclass."""
    
    def test_search_result_creation(self):
        """Test search result creation."""
        doc = Document(content="test", source="test")
        result = SearchResult(document=doc, score=0.8, rank=1)
        
        assert result.document == doc
        assert result.score == 0.8
        assert result.rank == 1
    
    def test_search_result_to_dict(self):
        """Test search result serialization."""
        doc = Document(content="test", source="test")
        result = SearchResult(document=doc, score=0.8, rank=1)
        
        result_dict = result.to_dict()
        
        assert result_dict["score"] == 0.8
        assert result_dict["rank"] == 1
        assert "document" in result_dict


class TestChromaDBVectorStore:
    """Test ChromaDB vector store implementation."""
    
    @pytest.fixture
    def vector_store(self):
        """Create vector store instance for testing."""
        return ChromaDBVectorStore(
            collection_name="test_collection",
            persist_directory="./test_chroma_db"
        )
    
    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            Document(
                content="GDPR Article 5 requires data minimization",
                source="gdpr.txt",
                document_type="regulation",
                regulatory_framework="GDPR"
            ),
            Document(
                content="HIPAA requires protection of PHI",
                source="hipaa.txt",
                document_type="regulation",
                regulatory_framework="HIPAA"
            )
        ]
    
    @patch('src.llama_mapper.rag.core.vector_store.chromadb')
    @pytest.mark.asyncio
    async def test_add_documents_success(self, mock_chromadb, vector_store, sample_documents):
        """Test successful document addition."""
        # Mock ChromaDB
        mock_client = Mock()
        mock_collection = Mock()
        mock_chromadb.PersistentClient.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection
        
        # Test add documents
        result = await vector_store.add_documents(sample_documents)
        
        assert result is True
        mock_collection.add.assert_called_once()
    
    @patch('src.llama_mapper.rag.core.vector_store.chromadb')
    @pytest.mark.asyncio
    async def test_add_documents_empty_list(self, mock_chromadb, vector_store):
        """Test adding empty document list."""
        # Mock ChromaDB
        mock_client = Mock()
        mock_collection = Mock()
        mock_chromadb.PersistentClient.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection
        
        # Test add empty documents
        result = await vector_store.add_documents([])
        
        assert result is True
        mock_collection.add.assert_not_called()
    
    @patch('src.llama_mapper.rag.core.vector_store.chromadb')
    @pytest.mark.asyncio
    async def test_search_documents(self, mock_chromadb, vector_store):
        """Test document search functionality."""
        # Mock ChromaDB
        mock_client = Mock()
        mock_collection = Mock()
        mock_chromadb.PersistentClient.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection
        
        # Mock search results
        mock_collection.query.return_value = {
            'documents': [['Test document content']],
            'metadatas': [{'source': 'test.txt', 'document_type': 'regulation'}],
            'distances': [[0.2]],
            'ids': [['doc-1']]
        }
        
        # Test search
        results = await vector_store.search("test query", top_k=5)
        
        assert len(results) == 1
        assert results[0].document.content == 'Test document content'
        assert results[0].score == 0.8  # 1 - 0.2
        assert results[0].rank == 1
        mock_collection.query.assert_called_once()
    
    @patch('src.llama_mapper.rag.core.vector_store.chromadb')
    @pytest.mark.asyncio
    async def test_search_no_results(self, mock_chromadb, vector_store):
        """Test search with no results."""
        # Mock ChromaDB
        mock_client = Mock()
        mock_collection = Mock()
        mock_chromadb.PersistentClient.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection
        
        # Mock empty search results
        mock_collection.query.return_value = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]],
            'ids': [[]]
        }
        
        # Test search
        results = await vector_store.search("test query")
        
        assert len(results) == 0
    
    @patch('src.llama_mapper.rag.core.vector_store.chromadb')
    @pytest.mark.asyncio
    async def test_update_document(self, mock_chromadb, vector_store):
        """Test document update functionality."""
        # Mock ChromaDB
        mock_client = Mock()
        mock_collection = Mock()
        mock_chromadb.PersistentClient.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection
        
        # Test update
        doc = Document(content="updated content", source="test.txt")
        result = await vector_store.update_document("doc-1", doc)
        
        assert result is True
        mock_collection.update.assert_called_once()
    
    @patch('src.llama_mapper.rag.core.vector_store.chromadb')
    @pytest.mark.asyncio
    async def test_delete_document(self, mock_chromadb, vector_store):
        """Test document deletion."""
        # Mock ChromaDB
        mock_client = Mock()
        mock_collection = Mock()
        mock_chromadb.PersistentClient.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection
        
        # Test delete
        result = await vector_store.delete_document("doc-1")
        
        assert result is True
        mock_collection.delete.assert_called_once_with(ids=["doc-1"])
    
    @patch('src.llama_mapper.rag.core.vector_store.chromadb')
    @pytest.mark.asyncio
    async def test_get_document(self, mock_chromadb, vector_store):
        """Test getting specific document."""
        # Mock ChromaDB
        mock_client = Mock()
        mock_collection = Mock()
        mock_chromadb.PersistentClient.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection
        
        # Mock get results
        mock_collection.get.return_value = {
            'documents': ['Test document content'],
            'metadatas': [{'source': 'test.txt', 'document_type': 'regulation'}]
        }
        
        # Test get
        doc = await vector_store.get_document("doc-1")
        
        assert doc is not None
        assert doc.content == 'Test document content'
        assert doc.source == 'test.txt'
        mock_collection.get.assert_called_once_with(ids=["doc-1"])
    
    @patch('src.llama_mapper.rag.core.vector_store.chromadb')
    @pytest.mark.asyncio
    async def test_get_document_not_found(self, mock_chromadb, vector_store):
        """Test getting non-existent document."""
        # Mock ChromaDB
        mock_client = Mock()
        mock_collection = Mock()
        mock_chromadb.PersistentClient.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection
        
        # Mock empty get results
        mock_collection.get.return_value = {
            'documents': [],
            'metadatas': []
        }
        
        # Test get
        doc = await vector_store.get_document("non-existent")
        
        assert doc is None
    
    @patch('src.llama_mapper.rag.core.vector_store.chromadb')
    @pytest.mark.asyncio
    async def test_collection_stats(self, mock_chromadb, vector_store):
        """Test getting collection statistics."""
        # Mock ChromaDB
        mock_client = Mock()
        mock_collection = Mock()
        mock_chromadb.PersistentClient.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_collection.count.return_value = 42
        
        # Test stats
        stats = await vector_store.get_collection_stats()
        
        assert stats["total_documents"] == 42
        assert stats["collection_name"] == "test_collection"
        assert "persist_directory" in stats
    
    @patch('src.llama_mapper.rag.core.vector_store.chromadb')
    @pytest.mark.asyncio
    async def test_health_check_success(self, mock_chromadb, vector_store):
        """Test successful health check."""
        # Mock ChromaDB
        mock_client = Mock()
        mock_collection = Mock()
        mock_chromadb.PersistentClient.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_collection.count.return_value = 0
        
        # Test health check
        is_healthy = await vector_store.health_check()
        
        assert is_healthy is True
    
    @patch('src.llama_mapper.rag.core.vector_store.chromadb')
    @pytest.mark.asyncio
    async def test_health_check_failure(self, mock_chromadb, vector_store):
        """Test failed health check."""
        # Mock ChromaDB with exception
        mock_client = Mock()
        mock_collection = Mock()
        mock_chromadb.PersistentClient.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_collection.count.side_effect = Exception("Connection failed")
        
        # Test health check
        is_healthy = await vector_store.health_check()
        
        assert is_healthy is False
    
    @patch('src.llama_mapper.rag.core.vector_store.chromadb')
    @pytest.mark.asyncio
    async def test_chromadb_import_error(self, mock_chromadb, vector_store):
        """Test handling of ChromaDB import error."""
        # Mock import error
        mock_chromadb.side_effect = ImportError("ChromaDB not installed")
        
        # Test that it raises ImportError
        with pytest.raises(ImportError, match="ChromaDB is required"):
            await vector_store._get_client()
    
    @patch('src.llama_mapper.rag.core.vector_store.chromadb')
    @pytest.mark.asyncio
    async def test_search_with_filters(self, mock_chromadb, vector_store):
        """Test search with metadata filters."""
        # Mock ChromaDB
        mock_client = Mock()
        mock_collection = Mock()
        mock_chromadb.PersistentClient.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection
        
        # Mock search results
        mock_collection.query.return_value = {
            'documents': [['Test document content']],
            'metadatas': [{'source': 'test.txt', 'regulatory_framework': 'GDPR'}],
            'distances': [[0.1]],
            'ids': [['doc-1']]
        }
        
        # Test search with filters
        filters = {'regulatory_framework': 'GDPR'}
        results = await vector_store.search("test query", filters=filters)
        
        assert len(results) == 1
        mock_collection.query.assert_called_once()
        # Verify filters were passed
        call_args = mock_collection.query.call_args
        assert call_args[1]['where'] == filters
