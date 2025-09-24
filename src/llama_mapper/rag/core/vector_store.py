"""
Vector store interface and implementations for RAG system.

Provides abstract interface and concrete implementations for vector database operations.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import logging

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Represents a document in the vector store."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    source: str = ""
    document_type: str = ""
    regulatory_framework: Optional[str] = None
    industry: Optional[str] = None
    last_updated: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "embedding": self.embedding,
            "source": self.source,
            "document_type": self.document_type,
            "regulatory_framework": self.regulatory_framework,
            "industry": self.industry,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "created_at": self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        """Create document from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            content=data.get("content", ""),
            metadata=data.get("metadata", {}),
            embedding=data.get("embedding"),
            source=data.get("source", ""),
            document_type=data.get("document_type", ""),
            regulatory_framework=data.get("regulatory_framework"),
            industry=data.get("industry"),
            last_updated=datetime.fromisoformat(data["last_updated"]) if data.get("last_updated") else None,
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.utcnow()
        )


@dataclass
class SearchResult:
    """Represents a search result with relevance score."""
    
    document: Document
    score: float
    rank: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert search result to dictionary."""
        return {
            "document": self.document.to_dict(),
            "score": self.score,
            "rank": self.rank
        }


class VectorStore(ABC):
    """Abstract interface for vector database operations."""
    
    @abstractmethod
    async def add_documents(self, documents: List[Document]) -> bool:
        """Add documents to the vector store.
        
        Args:
            documents: List of documents to add
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def search(self, query: str, filters: Optional[Dict[str, Any]] = None, 
                    top_k: int = 10) -> List[SearchResult]:
        """Search for relevant documents.
        
        Args:
            query: Search query
            filters: Optional filters for metadata
            top_k: Number of results to return
            
        Returns:
            List of search results ordered by relevance
        """
        pass
    
    @abstractmethod
    async def update_document(self, doc_id: str, document: Document) -> bool:
        """Update an existing document.
        
        Args:
            doc_id: Document ID to update
            document: Updated document
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document.
        
        Args:
            doc_id: Document ID to delete
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def get_document(self, doc_id: str) -> Optional[Document]:
        """Retrieve a specific document.
        
        Args:
            doc_id: Document ID to retrieve
            
        Returns:
            Document if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics.
        
        Returns:
            Dictionary with collection statistics
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the vector store is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        pass


class ChromaDBVectorStore(VectorStore):
    """ChromaDB implementation of vector store."""
    
    def __init__(self, collection_name: str = "compliance_knowledge", 
                 persist_directory: str = "./chroma_db"):
        """Initialize ChromaDB vector store.
        
        Args:
            collection_name: Name of the collection
            persist_directory: Directory to persist data
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self._client = None
        self._collection = None
        self.logger = logging.getLogger(__name__)
    
    async def _get_client(self):
        """Get or create ChromaDB client."""
        if self._client is None:
            try:
                import chromadb
                from chromadb.config import Settings
                
                self._client = chromadb.PersistentClient(
                    path=self.persist_directory,
                    settings=Settings(anonymized_telemetry=False)
                )
                self._collection = self._client.get_or_create_collection(
                    name=self.collection_name
                )
                self.logger.info(f"Connected to ChromaDB collection: {self.collection_name}")
            except ImportError:
                raise ImportError("ChromaDB is required. Install with: pip install chromadb")
            except Exception as e:
                self.logger.error(f"Failed to connect to ChromaDB: {e}")
                raise
        
        return self._client, self._collection
    
    async def add_documents(self, documents: List[Document]) -> bool:
        """Add documents to ChromaDB."""
        try:
            client, collection = await self._get_client()
            
            if not documents:
                return True
            
            # Prepare data for ChromaDB
            ids = [doc.id for doc in documents]
            contents = [doc.content for doc in documents]
            embeddings = [doc.embedding for doc in documents if doc.embedding]
            metadatas = [doc.metadata for doc in documents]
            
            # Add documents to collection
            if embeddings:
                collection.add(
                    ids=ids,
                    documents=contents,
                    embeddings=embeddings,
                    metadatas=metadatas
                )
            else:
                collection.add(
                    ids=ids,
                    documents=contents,
                    metadatas=metadatas
                )
            
            self.logger.info(f"Added {len(documents)} documents to ChromaDB")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add documents to ChromaDB: {e}")
            return False
    
    async def search(self, query: str, filters: Optional[Dict[str, Any]] = None, 
                    top_k: int = 10) -> List[SearchResult]:
        """Search documents in ChromaDB."""
        try:
            client, collection = await self._get_client()
            
            # Prepare where clause for filters
            where_clause = {}
            if filters:
                where_clause = filters
            
            # Search collection
            results = collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where_clause if where_clause else None
            )
            
            # Convert results to SearchResult objects
            search_results = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0] if results['metadatas'] else [{}] * len(results['documents'][0]),
                    results['distances'][0] if results['distances'] else [0.0] * len(results['documents'][0])
                )):
                    # Convert distance to similarity score (1 - distance)
                    score = 1.0 - distance if distance is not None else 0.0
                    
                    document = Document(
                        id=results['ids'][0][i] if results['ids'] and results['ids'][0] else str(uuid.uuid4()),
                        content=doc,
                        metadata=metadata,
                        source=metadata.get('source', ''),
                        document_type=metadata.get('document_type', ''),
                        regulatory_framework=metadata.get('regulatory_framework'),
                        industry=metadata.get('industry')
                    )
                    
                    search_results.append(SearchResult(
                        document=document,
                        score=score,
                        rank=i + 1
                    ))
            
            self.logger.info(f"Found {len(search_results)} documents for query: {query}")
            return search_results
            
        except Exception as e:
            self.logger.error(f"Failed to search ChromaDB: {e}")
            return []
    
    async def update_document(self, doc_id: str, document: Document) -> bool:
        """Update document in ChromaDB."""
        try:
            client, collection = await self._get_client()
            
            # Update document
            collection.update(
                ids=[doc_id],
                documents=[document.content],
                embeddings=[document.embedding] if document.embedding else None,
                metadatas=[document.metadata]
            )
            
            self.logger.info(f"Updated document {doc_id} in ChromaDB")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update document {doc_id} in ChromaDB: {e}")
            return False
    
    async def delete_document(self, doc_id: str) -> bool:
        """Delete document from ChromaDB."""
        try:
            client, collection = await self._get_client()
            
            collection.delete(ids=[doc_id])
            
            self.logger.info(f"Deleted document {doc_id} from ChromaDB")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete document {doc_id} from ChromaDB: {e}")
            return False
    
    async def get_document(self, doc_id: str) -> Optional[Document]:
        """Get document from ChromaDB."""
        try:
            client, collection = await self._get_client()
            
            results = collection.get(ids=[doc_id])
            
            if not results['documents'] or not results['documents'][0]:
                return None
            
            doc = results['documents'][0]
            metadata = results['metadatas'][0] if results['metadatas'] else {}
            
            return Document(
                id=doc_id,
                content=doc,
                metadata=metadata,
                source=metadata.get('source', ''),
                document_type=metadata.get('document_type', ''),
                regulatory_framework=metadata.get('regulatory_framework'),
                industry=metadata.get('industry')
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get document {doc_id} from ChromaDB: {e}")
            return None
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get ChromaDB collection statistics."""
        try:
            client, collection = await self._get_client()
            
            count = collection.count()
            
            return {
                "total_documents": count,
                "collection_name": self.collection_name,
                "persist_directory": self.persist_directory
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get collection stats: {e}")
            return {}
    
    async def health_check(self) -> bool:
        """Check ChromaDB health."""
        try:
            client, collection = await self._get_client()
            # Simple health check by getting collection stats
            collection.count()
            return True
        except Exception as e:
            self.logger.error(f"ChromaDB health check failed: {e}")
            return False


class PineconeVectorStore(VectorStore):
    """Pinecone implementation of vector store."""
    
    def __init__(self, api_key: str, environment: str, index_name: str = "compliance-knowledge"):
        """Initialize Pinecone vector store.
        
        Args:
            api_key: Pinecone API key
            environment: Pinecone environment
            index_name: Name of the Pinecone index
        """
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self._index = None
        self.logger = logging.getLogger(__name__)
    
    async def _get_index(self):
        """Get or create Pinecone index."""
        if self._index is None:
            try:
                import pinecone
                
                pinecone.init(api_key=self.api_key, environment=self.environment)
                self._index = pinecone.Index(self.index_name)
                self.logger.info(f"Connected to Pinecone index: {self.index_name}")
            except ImportError:
                raise ImportError("Pinecone is required. Install with: pip install pinecone-client")
            except Exception as e:
                self.logger.error(f"Failed to connect to Pinecone: {e}")
                raise
        
        return self._index
    
    async def add_documents(self, documents: List[Document]) -> bool:
        """Add documents to Pinecone."""
        try:
            index = await self._get_index()
            
            if not documents:
                return True
            
            # Prepare vectors for Pinecone
            vectors = []
            for doc in documents:
                if doc.embedding:
                    vectors.append({
                        "id": doc.id,
                        "values": doc.embedding,
                        "metadata": {
                            **doc.metadata,
                            "content": doc.content,
                            "source": doc.source,
                            "document_type": doc.document_type,
                            "regulatory_framework": doc.regulatory_framework,
                            "industry": doc.industry
                        }
                    })
            
            if vectors:
                index.upsert(vectors=vectors)
                self.logger.info(f"Added {len(vectors)} documents to Pinecone")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add documents to Pinecone: {e}")
            return False
    
    async def search(self, query: str, filters: Optional[Dict[str, Any]] = None, 
                    top_k: int = 10) -> List[SearchResult]:
        """Search documents in Pinecone."""
        try:
            index = await self._get_index()
            
            # Prepare filter for Pinecone
            filter_dict = {}
            if filters:
                filter_dict = filters
            
            # Search index
            results = index.query(
                vector=query,  # This would need to be embedded first
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict if filter_dict else None
            )
            
            # Convert results to SearchResult objects
            search_results = []
            if results.matches:
                for i, match in enumerate(results.matches):
                    metadata = match.metadata or {}
                    
                    document = Document(
                        id=match.id,
                        content=metadata.get('content', ''),
                        metadata=metadata,
                        source=metadata.get('source', ''),
                        document_type=metadata.get('document_type', ''),
                        regulatory_framework=metadata.get('regulatory_framework'),
                        industry=metadata.get('industry')
                    )
                    
                    search_results.append(SearchResult(
                        document=document,
                        score=match.score,
                        rank=i + 1
                    ))
            
            self.logger.info(f"Found {len(search_results)} documents for query: {query}")
            return search_results
            
        except Exception as e:
            self.logger.error(f"Failed to search Pinecone: {e}")
            return []
    
    async def update_document(self, doc_id: str, document: Document) -> bool:
        """Update document in Pinecone."""
        try:
            index = await self._get_index()
            
            if document.embedding:
                index.upsert(vectors=[{
                    "id": doc_id,
                    "values": document.embedding,
                    "metadata": {
                        **document.metadata,
                        "content": document.content,
                        "source": document.source,
                        "document_type": document.document_type,
                        "regulatory_framework": document.regulatory_framework,
                        "industry": document.industry
                    }
                }])
            
            self.logger.info(f"Updated document {doc_id} in Pinecone")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update document {doc_id} in Pinecone: {e}")
            return False
    
    async def delete_document(self, doc_id: str) -> bool:
        """Delete document from Pinecone."""
        try:
            index = await self._get_index()
            
            index.delete(ids=[doc_id])
            
            self.logger.info(f"Deleted document {doc_id} from Pinecone")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete document {doc_id} from Pinecone: {e}")
            return False
    
    async def get_document(self, doc_id: str) -> Optional[Document]:
        """Get document from Pinecone."""
        try:
            index = await self._get_index()
            
            results = index.fetch(ids=[doc_id])
            
            if not results.vectors or doc_id not in results.vectors:
                return None
            
            vector_data = results.vectors[doc_id]
            metadata = vector_data.metadata or {}
            
            return Document(
                id=doc_id,
                content=metadata.get('content', ''),
                metadata=metadata,
                source=metadata.get('source', ''),
                document_type=metadata.get('document_type', ''),
                regulatory_framework=metadata.get('regulatory_framework'),
                industry=metadata.get('industry')
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get document {doc_id} from Pinecone: {e}")
            return None
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get Pinecone index statistics."""
        try:
            index = await self._get_index()
            
            stats = index.describe_index_stats()
            
            return {
                "total_documents": stats.total_vector_count,
                "index_name": self.index_name,
                "dimension": stats.dimension,
                "index_fullness": stats.index_fullness
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get index stats: {e}")
            return {}
    
    async def health_check(self) -> bool:
        """Check Pinecone health."""
        try:
            index = await self._get_index()
            # Simple health check by getting index stats
            index.describe_index_stats()
            return True
        except Exception as e:
            self.logger.error(f"Pinecone health check failed: {e}")
            return False
