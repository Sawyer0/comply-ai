"""
Unit tests for embedding models.

Tests all embedding model implementations including Sentence Transformers,
OpenAI, and Hugging Face models.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import numpy as np
from typing import List

from src.llama_mapper.rag.core.embeddings import (
    EmbeddingModel, SentenceTransformerEmbeddings, 
    OpenAIEmbeddings, HuggingFaceEmbeddings, EmbeddingFactory
)


class TestSentenceTransformerEmbeddings:
    """Test Sentence Transformers embedding implementation."""
    
    @pytest.fixture
    def embedding_model(self):
        """Create embedding model instance for testing."""
        return SentenceTransformerEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
            batch_size=32
        )
    
    @patch('src.llama_mapper.rag.core.embeddings.SentenceTransformer')
    @pytest.mark.asyncio
    async def test_embed_text_success(self, mock_st_class, embedding_model):
        """Test successful text embedding."""
        # Mock SentenceTransformer
        mock_model = Mock()
        mock_st_class.return_value = mock_model
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        
        # Test embedding
        result = await embedding_model.embed_text("test text")
        
        assert isinstance(result, list)
        assert len(result) == 3
        assert result == [0.1, 0.2, 0.3]
        mock_model.encode.assert_called_once_with(["test text"], convert_to_tensor=False)
    
    @patch('src.llama_mapper.rag.core.embeddings.SentenceTransformer')
    @pytest.mark.asyncio
    async def test_embed_batch_success(self, mock_st_class, embedding_model):
        """Test successful batch embedding."""
        # Mock SentenceTransformer
        mock_model = Mock()
        mock_st_class.return_value = mock_model
        mock_model.encode.return_value = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]
        ])
        
        # Test batch embedding
        texts = ["text 1", "text 2"]
        result = await embedding_model.embed_batch(texts)
        
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]
        assert result[1] == [0.4, 0.5, 0.6]
    
    @pytest.mark.asyncio
    async def test_embed_batch_empty(self, embedding_model):
        """Test batch embedding with empty list."""
        result = await embedding_model.embed_batch([])
        assert result == []
    
    def test_get_dimension(self, embedding_model):
        """Test getting embedding dimension."""
        dimension = embedding_model.get_dimension()
        assert dimension == 384  # Default for MiniLM
    
    @patch('src.llama_mapper.rag.core.embeddings.SentenceTransformer')
    @pytest.mark.asyncio
    async def test_health_check_success(self, mock_st_class, embedding_model):
        """Test successful health check."""
        # Mock SentenceTransformer
        mock_model = Mock()
        mock_st_class.return_value = mock_model
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        
        # Test health check
        is_healthy = await embedding_model.health_check()
        
        assert is_healthy is True
    
    @patch('src.llama_mapper.rag.core.embeddings.SentenceTransformer')
    @pytest.mark.asyncio
    async def test_health_check_failure(self, mock_st_class, embedding_model):
        """Test failed health check."""
        # Mock SentenceTransformer with exception
        mock_model = Mock()
        mock_st_class.return_value = mock_model
        mock_model.encode.side_effect = Exception("Model error")
        
        # Test health check
        is_healthy = await embedding_model.health_check()
        
        assert is_healthy is False
    
    @pytest.mark.asyncio
    async def test_import_error_handling(self, embedding_model):
        """Test handling of import error."""
        with patch('src.llama_mapper.rag.core.embeddings.SentenceTransformer', side_effect=ImportError):
            with pytest.raises(ImportError, match="sentence-transformers is required"):
                await embedding_model._get_model()


class TestOpenAIEmbeddings:
    """Test OpenAI embedding implementation."""
    
    @pytest.fixture
    def embedding_model(self):
        """Create OpenAI embedding model for testing."""
        return OpenAIEmbeddings(
            api_key="test-api-key",
            model="text-embedding-ada-002"
        )
    
    @patch('src.llama_mapper.rag.core.embeddings.openai')
    @pytest.mark.asyncio
    async def test_embed_text_success(self, mock_openai, embedding_model):
        """Test successful OpenAI text embedding."""
        # Mock OpenAI response
        mock_openai.Embedding.acreate.return_value = {
            'data': [{'embedding': [0.1, 0.2, 0.3]}]
        }
        
        # Test embedding
        result = await embedding_model.embed_text("test text")
        
        assert isinstance(result, list)
        assert len(result) == 3
        assert result == [0.1, 0.2, 0.3]
        mock_openai.Embedding.acreate.assert_called_once()
    
    @patch('src.llama_mapper.rag.core.embeddings.openai')
    @pytest.mark.asyncio
    async def test_embed_batch_success(self, mock_openai, embedding_model):
        """Test successful OpenAI batch embedding."""
        # Mock OpenAI response
        mock_openai.Embedding.acreate.return_value = {
            'data': [
                {'embedding': [0.1, 0.2, 0.3]},
                {'embedding': [0.4, 0.5, 0.6]}
            ]
        }
        
        # Test batch embedding
        texts = ["text 1", "text 2"]
        result = await embedding_model.embed_batch(texts)
        
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]
        assert result[1] == [0.4, 0.5, 0.6]
    
    def test_get_dimension(self, embedding_model):
        """Test getting OpenAI embedding dimension."""
        dimension = embedding_model.get_dimension()
        assert dimension == 1536  # OpenAI ada-002 dimension
    
    @patch('src.llama_mapper.rag.core.embeddings.openai')
    @pytest.mark.asyncio
    async def test_health_check_success(self, mock_openai, embedding_model):
        """Test successful OpenAI health check."""
        # Mock OpenAI response
        mock_openai.Embedding.acreate.return_value = {
            'data': [{'embedding': [0.1, 0.2, 0.3]}]
        }
        
        # Test health check
        is_healthy = await embedding_model.health_check()
        
        assert is_healthy is True
    
    @patch('src.llama_mapper.rag.core.embeddings.openai')
    @pytest.mark.asyncio
    async def test_health_check_failure(self, mock_openai, embedding_model):
        """Test failed OpenAI health check."""
        # Mock OpenAI exception
        mock_openai.Embedding.acreate.side_effect = Exception("API error")
        
        # Test health check
        is_healthy = await embedding_model.health_check()
        
        assert is_healthy is False


class TestHuggingFaceEmbeddings:
    """Test Hugging Face embedding implementation."""
    
    @pytest.fixture
    def embedding_model(self):
        """Create Hugging Face embedding model for testing."""
        return HuggingFaceEmbeddings(
            model_name="bert-base-uncased",
            device="cpu",
            batch_size=32
        )
    
    @patch('src.llama_mapper.rag.core.embeddings.AutoModel')
    @patch('src.llama_mapper.rag.core.embeddings.AutoTokenizer')
    @patch('src.llama_mapper.rag.core.embeddings.torch')
    @pytest.mark.asyncio
    async def test_embed_text_success(self, mock_torch, mock_tokenizer_class, 
                                     mock_model_class, embedding_model):
        """Test successful Hugging Face text embedding."""
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_tokenizer.return_value = {"input_ids": Mock(), "attention_mask": Mock()}
        
        # Mock model
        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_outputs = Mock()
        mock_outputs.last_hidden_state = Mock()
        mock_outputs.last_hidden_state.mean.return_value.squeeze.return_value.cpu.return_value.numpy.return_value.tolist.return_value = [0.1, 0.2, 0.3]
        mock_model.return_value = mock_outputs
        
        # Mock torch
        mock_torch.no_grad.return_value.__enter__ = Mock()
        mock_torch.no_grad.return_value.__exit__ = Mock()
        mock_torch.cuda.is_available.return_value = False
        
        # Test embedding
        result = await embedding_model.embed_text("test text")
        
        assert isinstance(result, list)
        assert len(result) == 3
        assert result == [0.1, 0.2, 0.3]
    
    def test_get_dimension(self, embedding_model):
        """Test getting Hugging Face embedding dimension."""
        dimension = embedding_model.get_dimension()
        assert dimension == 768  # Default BERT dimension


class TestEmbeddingFactory:
    """Test embedding model factory."""
    
    def test_create_sentence_transformer(self):
        """Test creating sentence transformer model."""
        model = EmbeddingFactory.create_embedding_model(
            "sentence_transformer",
            model_name="test-model"
        )
        
        assert isinstance(model, SentenceTransformerEmbeddings)
        assert model.model_name == "test-model"
    
    def test_create_openai(self):
        """Test creating OpenAI model."""
        model = EmbeddingFactory.create_embedding_model(
            "openai",
            api_key="test-key"
        )
        
        assert isinstance(model, OpenAIEmbeddings)
        assert model.api_key == "test-key"
    
    def test_create_huggingface(self):
        """Test creating Hugging Face model."""
        model = EmbeddingFactory.create_embedding_model(
            "huggingface",
            model_name="test-model"
        )
        
        assert isinstance(model, HuggingFaceEmbeddings)
        assert model.model_name == "test-model"
    
    def test_unknown_model_type(self):
        """Test handling unknown model type."""
        with pytest.raises(ValueError, match="Unknown embedding model type"):
            EmbeddingFactory.create_embedding_model("unknown_type")
