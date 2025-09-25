"""
Embeddings Management System

This module provides embeddings management for the Analysis Service,
supporting various embedding models and efficient vector operations.
"""

import logging
import asyncio
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timezone
import json
import hashlib

logger = logging.getLogger(__name__)


class EmbeddingsManager:
    """
    Embeddings management system for Analysis Service.

    Supports:
    - Multiple embedding models
    - Vector similarity search
    - Embedding caching and optimization
    - Batch processing
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.embeddings_config = config.get("embeddings", {})

        # Model configuration
        self.model_name = self.embeddings_config.get(
            "model_name", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.embedding_dim = self.embeddings_config.get("embedding_dim", 384)
        self.batch_size = self.embeddings_config.get("batch_size", 32)

        # Caching
        self.cache_enabled = self.embeddings_config.get("cache_enabled", True)
        self.embedding_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

        # Model instance
        self.model = None
        self.model_loaded = False

        # Performance tracking
        self.embedding_count = 0
        self.total_embedding_time = 0.0

        logger.info(
            "Embeddings Manager initialized",
            model=self.model_name,
            dimension=self.embedding_dim,
        )

    async def initialize(self):
        """Initialize embeddings model."""
        try:
            await self._load_embedding_model()
            self.model_loaded = True

            logger.info(
                "Embeddings Manager initialized successfully", model=self.model_name
            )

        except Exception as e:
            logger.error("Failed to initialize Embeddings Manager", error=str(e))
            raise

    async def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as numpy array
        """
        return await self.embed_texts([text])[0]

    async def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not self.model_loaded:
            await self.initialize()

        start_time = datetime.now(timezone.utc)

        try:
            # Check cache first
            embeddings = []
            uncached_texts = []
            uncached_indices = []

            for i, text in enumerate(texts):
                if self.cache_enabled:
                    cached_embedding = self._get_cached_embedding(text)
                    if cached_embedding is not None:
                        embeddings.append(cached_embedding)
                        self.cache_hits += 1
                        continue

                # Text not in cache
                embeddings.append(None)  # Placeholder
                uncached_texts.append(text)
                uncached_indices.append(i)
                self.cache_misses += 1

            # Generate embeddings for uncached texts
            if uncached_texts:
                new_embeddings = await self._generate_embeddings(uncached_texts)

                # Insert new embeddings and cache them
                for idx, embedding in zip(uncached_indices, new_embeddings):
                    embeddings[idx] = embedding
                    if self.cache_enabled:
                        self._cache_embedding(texts[idx], embedding)

            # Track performance
            elapsed_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            self.embedding_count += len(texts)
            self.total_embedding_time += elapsed_time

            logger.debug(
                "Embeddings generated",
                text_count=len(texts),
                cache_hits=self.cache_hits,
                cache_misses=self.cache_misses,
                elapsed_time=elapsed_time,
            )

            return embeddings

        except Exception as e:
            logger.error("Embedding generation failed", error=str(e))
            raise

    async def similarity_search(
        self, query_text: str, candidate_texts: List[str], top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Perform similarity search between query and candidate texts.

        Args:
            query_text: Query text
            candidate_texts: List of candidate texts
            top_k: Number of top results to return

        Returns:
            List of (text, similarity_score) tuples, sorted by similarity
        """
        try:
            # Generate embeddings
            query_embedding = await self.embed_text(query_text)
            candidate_embeddings = await self.embed_texts(candidate_texts)

            # Calculate similarities
            similarities = []
            for i, candidate_embedding in enumerate(candidate_embeddings):
                similarity = self._cosine_similarity(
                    query_embedding, candidate_embedding
                )
                similarities.append((candidate_texts[i], float(similarity)))

            # Sort by similarity and return top_k
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]

        except Exception as e:
            logger.error("Similarity search failed", error=str(e))
            raise

    async def batch_similarity_matrix(self, texts: List[str]) -> np.ndarray:
        """
        Generate similarity matrix for a batch of texts.

        Args:
            texts: List of texts

        Returns:
            Similarity matrix as numpy array
        """
        try:
            # Generate embeddings
            embeddings = await self.embed_texts(texts)

            # Calculate similarity matrix
            n = len(embeddings)
            similarity_matrix = np.zeros((n, n))

            for i in range(n):
                for j in range(i, n):
                    similarity = self._cosine_similarity(embeddings[i], embeddings[j])
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity  # Symmetric matrix

            return similarity_matrix

        except Exception as e:
            logger.error("Batch similarity matrix calculation failed", error=str(e))
            raise

    async def find_similar_patterns(
        self, pattern_descriptions: List[str], threshold: float = 0.7
    ) -> List[List[int]]:
        """
        Find similar patterns based on their descriptions.

        Args:
            pattern_descriptions: List of pattern descriptions
            threshold: Similarity threshold for grouping

        Returns:
            List of groups, where each group is a list of pattern indices
        """
        try:
            # Generate similarity matrix
            similarity_matrix = await self.batch_similarity_matrix(pattern_descriptions)

            # Group similar patterns
            n = len(pattern_descriptions)
            visited = [False] * n
            groups = []

            for i in range(n):
                if visited[i]:
                    continue

                # Start new group
                group = [i]
                visited[i] = True

                # Find similar patterns
                for j in range(i + 1, n):
                    if not visited[j] and similarity_matrix[i, j] >= threshold:
                        group.append(j)
                        visited[j] = True

                groups.append(group)

            logger.debug(
                "Pattern grouping completed",
                total_patterns=n,
                groups_found=len(groups),
                threshold=threshold,
            )

            return groups

        except Exception as e:
            logger.error("Pattern similarity analysis failed", error=str(e))
            raise

    async def _load_embedding_model(self):
        """Load the embedding model."""
        try:
            # Try to load sentence-transformers model
            try:
                from sentence_transformers import SentenceTransformer

                self.model = SentenceTransformer(self.model_name)
                logger.info("Loaded SentenceTransformer model", model=self.model_name)
                return
            except ImportError:
                logger.warning("sentence-transformers not available, using fallback")

            # Fallback to basic embedding
            self.model = self._create_fallback_embedder()
            logger.info("Using fallback embedding model")

        except Exception as e:
            logger.error("Failed to load embedding model", error=str(e))
            raise

    async def _generate_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings using the loaded model."""
        try:
            if hasattr(self.model, "encode"):
                # SentenceTransformer model
                embeddings = self.model.encode(texts, batch_size=self.batch_size)
                return [np.array(emb) for emb in embeddings]
            else:
                # Fallback model
                return [self._generate_fallback_embedding(text) for text in texts]

        except Exception as e:
            logger.error("Embedding generation failed", error=str(e))
            raise

    def _create_fallback_embedder(self):
        """Create a simple fallback embedder."""
        return {"type": "fallback", "dimension": self.embedding_dim}

    def _generate_fallback_embedding(self, text: str) -> np.ndarray:
        """Generate simple hash-based embedding as fallback."""
        try:
            # Create deterministic embedding from text hash
            text_hash = hashlib.md5(text.encode()).hexdigest()

            # Convert hash to numbers and normalize
            embedding = []
            for i in range(0, min(len(text_hash), self.embedding_dim * 2), 2):
                hex_pair = text_hash[i : i + 2]
                value = int(hex_pair, 16) / 255.0  # Normalize to [0, 1]
                embedding.append(value)

            # Pad or truncate to desired dimension
            while len(embedding) < self.embedding_dim:
                embedding.append(0.0)
            embedding = embedding[: self.embedding_dim]

            # Normalize to unit vector
            embedding = np.array(embedding)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            return embedding

        except Exception as e:
            logger.error("Fallback embedding generation failed", error=str(e))
            # Return random normalized vector as last resort
            embedding = np.random.random(self.embedding_dim)
            return embedding / np.linalg.norm(embedding)

    def _get_cached_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding from cache."""
        text_hash = self._hash_text(text)
        return self.embedding_cache.get(text_hash)

    def _cache_embedding(self, text: str, embedding: np.ndarray):
        """Cache embedding."""
        text_hash = self._hash_text(text)
        self.embedding_cache[text_hash] = embedding

        # Simple cache size management
        if len(self.embedding_cache) > 10000:  # Max cache size
            # Remove oldest entries (simplified LRU)
            keys_to_remove = list(self.embedding_cache.keys())[:1000]
            for key in keys_to_remove:
                del self.embedding_cache[key]

    def _hash_text(self, text: str) -> str:
        """Generate hash for text caching."""
        return hashlib.sha256(text.encode()).hexdigest()

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return dot_product / (norm1 * norm2)

        except Exception as e:
            logger.error("Cosine similarity calculation failed", error=str(e))
            return 0.0

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get embeddings performance metrics."""
        avg_time = (
            self.total_embedding_time / self.embedding_count
            if self.embedding_count > 0
            else 0
        )
        cache_hit_rate = (
            self.cache_hits / (self.cache_hits + self.cache_misses)
            if (self.cache_hits + self.cache_misses) > 0
            else 0
        )

        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "total_embeddings": self.embedding_count,
            "average_time_per_embedding": avg_time,
            "cache_hit_rate": cache_hit_rate,
            "cache_size": len(self.embedding_cache),
            "model_loaded": self.model_loaded,
        }

    async def clear_cache(self):
        """Clear embedding cache."""
        self.embedding_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        logger.info("Embedding cache cleared")

    async def shutdown(self):
        """Gracefully shutdown embeddings manager."""
        try:
            logger.info("Shutting down Embeddings Manager...")

            # Clear cache
            await self.clear_cache()

            # Clean up model
            self.model = None
            self.model_loaded = False

            logger.info("Embeddings Manager shutdown complete")

        except Exception as e:
            logger.error("Error during Embeddings Manager shutdown", error=str(e))
