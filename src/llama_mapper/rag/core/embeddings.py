"""
Embedding models for RAG system.

Provides interfaces and implementations for generating document embeddings.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import logging
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingModel(ABC):
    """Abstract interface for embedding models."""
    
    @abstractmethod
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        pass
    
    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Get embedding dimension.
        
        Returns:
            Embedding dimension
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if embedding model is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        pass


class SentenceTransformerEmbeddings(EmbeddingModel):
    """Sentence Transformers implementation of embedding model."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", 
                 device: str = "cpu", batch_size: int = 32):
        """Initialize Sentence Transformers embedding model.
        
        Args:
            model_name: Name of the sentence transformer model
            device: Device to run model on ('cpu' or 'cuda')
            batch_size: Batch size for processing
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self.logger = logging.getLogger(__name__)
    
    async def _get_model(self):
        """Get or load the sentence transformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                
                self._model = SentenceTransformer(self.model_name, device=self.device)
                self.logger.info(f"Loaded SentenceTransformer model: {self.model_name}")
            except ImportError:
                raise ImportError("sentence-transformers is required. Install with: pip install sentence-transformers")
            except Exception as e:
                self.logger.error(f"Failed to load SentenceTransformer model: {e}")
                raise
        
        return self._model
    
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        try:
            model = await self._get_model()
            
            # Generate embedding
            embedding = model.encode([text], convert_to_tensor=False)
            
            return embedding[0].tolist()
            
        except Exception as e:
            self.logger.error(f"Failed to embed text: {e}")
            raise
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        try:
            model = await self._get_model()
            
            if not texts:
                return []
            
            # Process in batches
            embeddings = []
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                batch_embeddings = model.encode(batch, convert_to_tensor=False)
                embeddings.extend(batch_embeddings.tolist())
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Failed to embed batch: {e}")
            raise
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        try:
            # Most sentence transformer models have 384 dimensions
            # This is a fallback - should be overridden by actual model
            return 384
        except Exception:
            return 384
    
    async def health_check(self) -> bool:
        """Check if embedding model is healthy."""
        try:
            model = await self._get_model()
            # Simple health check by encoding a test string
            test_embedding = model.encode(["test"], convert_to_tensor=False)
            return len(test_embedding) > 0
        except Exception as e:
            self.logger.error(f"Embedding model health check failed: {e}")
            return False


class OpenAIEmbeddings(EmbeddingModel):
    """OpenAI embeddings implementation."""
    
    def __init__(self, api_key: str, model: str = "text-embedding-ada-002"):
        """Initialize OpenAI embeddings.
        
        Args:
            api_key: OpenAI API key
            model: OpenAI embedding model name
        """
        self.api_key = api_key
        self.model = model
        self.logger = logging.getLogger(__name__)
    
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding using OpenAI API."""
        try:
            import openai
            
            openai.api_key = self.api_key
            
            # Use the new OpenAI client API
            client = openai.AsyncOpenAI(api_key=self.api_key)
            response = await client.embeddings.create(
                input=text,
                model=self.model
            )
            
            return response['data'][0]['embedding']
            
        except Exception as e:
            self.logger.error(f"Failed to embed text with OpenAI: {e}")
            raise
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts using OpenAI API."""
        try:
            import openai
            
            # Use the new OpenAI client API
            client = openai.AsyncOpenAI(api_key=self.api_key)
            response = await client.embeddings.create(
                input=texts,
                model=self.model
            )
            
            return [item['embedding'] for item in response['data']]
            
        except Exception as e:
            self.logger.error(f"Failed to embed batch with OpenAI: {e}")
            raise
    
    def get_dimension(self) -> int:
        """Get embedding dimension for OpenAI ada-002 model."""
        return 1536  # OpenAI ada-002 has 1536 dimensions
    
    async def health_check(self) -> bool:
        """Check if OpenAI embeddings are healthy."""
        try:
            # Simple health check by embedding a test string
            test_embedding = await self.embed_text("test")
            return len(test_embedding) > 0
        except Exception as e:
            self.logger.error(f"OpenAI embeddings health check failed: {e}")
            return False


class HuggingFaceEmbeddings(EmbeddingModel):
    """Hugging Face transformers implementation of embedding model."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: str = "cpu", batch_size: int = 32):
        """Initialize Hugging Face embedding model.
        
        Args:
            model_name: Name of the Hugging Face model
            device: Device to run model on
            batch_size: Batch size for processing
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._tokenizer = None
        self.logger = logging.getLogger(__name__)
    
    async def _get_model(self):
        """Get or load the Hugging Face model."""
        if self._model is None:
            try:
                from transformers import AutoModel, AutoTokenizer
                import torch
                
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModel.from_pretrained(self.model_name)
                
                if self.device == "cuda" and torch.cuda.is_available():
                    self._model = self._model.cuda()
                
                self._model.eval()
                self.logger.info(f"Loaded Hugging Face model: {self.model_name}")
            except ImportError:
                raise ImportError("transformers is required. Install with: pip install transformers torch")
            except Exception as e:
                self.logger.error(f"Failed to load Hugging Face model: {e}")
                raise
        
        return self._model, self._tokenizer
    
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        try:
            model, tokenizer = await self._get_model()
            
            # Tokenize and encode
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            
            if self.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate embedding
            try:
                import torch
                with torch.no_grad():
                    outputs = model(**inputs)
                    # Use mean pooling of last hidden states
                    embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
                
                return embedding.cpu().numpy().tolist()
            except ImportError:
                raise ImportError("torch is required for Hugging Face embeddings")
            
        except Exception as e:
            self.logger.error(f"Failed to embed text with Hugging Face: {e}")
            raise
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        try:
            model, tokenizer = await self._get_model()
            
            if not texts:
                return []
            
            embeddings = []
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                
                # Tokenize batch
                inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True)
                
                if self.device == "cuda":
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                # Generate embeddings
                try:
                    import torch
                    with torch.no_grad():
                        outputs = model(**inputs)
                        # Use mean pooling of last hidden states
                        batch_embeddings = outputs.last_hidden_state.mean(dim=1)
                        embeddings.extend(batch_embeddings.cpu().numpy().tolist())
                except ImportError:
                    raise ImportError("torch is required for Hugging Face embeddings")
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Failed to embed batch with Hugging Face: {e}")
            raise
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        try:
            # This should be determined by the actual model
            # For most transformer models, this is 768 or 384
            return 768  # Common dimension for BERT-like models
        except Exception:
            return 768
    
    async def health_check(self) -> bool:
        """Check if Hugging Face model is healthy."""
        try:
            model, tokenizer = await self._get_model()
            # Simple health check by encoding a test string
            test_embedding = await self.embed_text("test")
            return len(test_embedding) > 0
        except Exception as e:
            self.logger.error(f"Hugging Face model health check failed: {e}")
            return False


class EmbeddingFactory:
    """Factory for creating embedding models."""
    
    @staticmethod
    def create_embedding_model(model_type: str, **kwargs) -> EmbeddingModel:
        """Create embedding model based on type.
        
        Args:
            model_type: Type of embedding model ('sentence_transformer', 'openai', 'huggingface')
            **kwargs: Additional arguments for model initialization
            
        Returns:
            Embedding model instance
        """
        if model_type == "sentence_transformer":
            return SentenceTransformerEmbeddings(**kwargs)
        elif model_type == "openai":
            return OpenAIEmbeddings(**kwargs)
        elif model_type == "huggingface":
            return HuggingFaceEmbeddings(**kwargs)
        else:
            raise ValueError(f"Unknown embedding model type: {model_type}")
