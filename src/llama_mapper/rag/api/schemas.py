"""
Pydantic schemas for RAG API endpoints.

Defines request and response models for the RAG system.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum


class DocumentType(str, Enum):
    """Supported document types."""
    REGULATION = "regulation"
    GUIDANCE = "guidance"
    CASE_LAW = "case_law"
    AUDIT_STANDARD = "audit_standard"
    IMPLEMENTATION_GUIDE = "implementation_guide"
    BEST_PRACTICE = "best_practice"
    CASE_STUDY = "case_study"
    METHODOLOGY = "methodology"
    TEMPLATE = "template"
    CHECKLIST = "checklist"


class RegulatoryFramework(str, Enum):
    """Supported regulatory frameworks."""
    GDPR = "GDPR"
    SOX = "SOX"
    HIPAA = "HIPAA"
    ISO27001 = "ISO27001"
    PCI_DSS = "PCI_DSS"
    FDA_21CFR = "FDA_21CFR"
    AML_BSA = "AML_BSA"
    FERPA = "FERPA"
    CCPA = "CCPA"
    NIST = "NIST"
    SOC2 = "SOC2"
    COBIT = "COBIT"


class Industry(str, Enum):
    """Supported industries."""
    FINANCIAL_SERVICES = "financial_services"
    HEALTHCARE = "healthcare"
    PHARMACEUTICALS = "pharmaceuticals"
    TECHNOLOGY = "technology"
    MANUFACTURING = "manufacturing"
    RETAIL = "retail"
    EDUCATION = "education"
    GOVERNMENT = "government"
    NON_PROFIT = "non_profit"


class AnalysisType(str, Enum):
    """Supported analysis types."""
    GAP_ANALYSIS = "gap_analysis"
    AUDIT_PREP = "audit_prep"
    RISK_ASSESSMENT = "risk_assessment"
    COMPLIANCE_PROGRAM_DESIGN = "compliance_program_design"
    REMEDIATION_PLANNING = "remediation_planning"
    REGULATORY_INTERPRETATION = "regulatory_interpretation"
    STAKEHOLDER_ENGAGEMENT = "stakeholder_engagement"
    COST_OPTIMIZATION = "cost_optimization"


class ComplianceLevel(str, Enum):
    """Compliance levels."""
    MANDATORY = "mandatory"
    RECOMMENDED = "recommended"
    BEST_PRACTICE = "best_practice"
    OPTIONAL = "optional"


class RiskLevel(str, Enum):
    """Risk levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Document(BaseModel):
    """Document representation."""
    id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source: str
    document_type: DocumentType
    regulatory_framework: Optional[RegulatoryFramework] = None
    industry: Optional[Industry] = None
    last_updated: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class SearchResult(BaseModel):
    """Search result representation."""
    document: Document
    score: float = Field(ge=0.0, le=1.0)
    rank: int = Field(ge=1)
    relevance_explanation: Optional[str] = None


class RAGQueryRequest(BaseModel):
    """Request schema for RAG query."""
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    regulatory_framework: Optional[RegulatoryFramework] = Field(None, description="Filter by regulatory framework")
    industry: Optional[Industry] = Field(None, description="Filter by industry")
    document_types: Optional[List[DocumentType]] = Field(None, description="Filter by document types")
    max_results: int = Field(10, ge=1, le=50, description="Maximum number of results")
    include_metadata: bool = Field(True, description="Include document metadata")
    user_context: Optional[Dict[str, Any]] = Field(None, description="Additional user context")
    similarity_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum similarity threshold")
    rerank: bool = Field(True, description="Enable result re-ranking")
    diversity_filter: bool = Field(False, description="Enable diversity filtering")
    
    @validator('query')
    def validate_query(cls, v):  # pylint: disable=no-self-argument
        if not v.strip():
            raise ValueError('Query cannot be empty')
        return v.strip()


class RAGQueryResponse(BaseModel):
    """Response schema for RAG query."""
    query: str
    retrieved_documents: List[SearchResult]
    enhanced_context: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    knowledge_coverage: Dict[str, float] = Field(default_factory=dict)
    processing_time_ms: int = Field(ge=0)
    total_documents_searched: int = Field(ge=0)
    filters_applied: Dict[str, Any] = Field(default_factory=dict)
    retrieval_strategy: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ExpertAnalysisRequest(BaseModel):
    """Request schema for expert analysis."""
    compliance_scenario: str = Field(..., min_length=10, max_length=5000, description="Compliance scenario to analyze")
    industry: Industry = Field(..., description="Target industry")
    regulatory_framework: RegulatoryFramework = Field(..., description="Regulatory framework")
    analysis_type: AnalysisType = Field(..., description="Type of analysis requested")
    include_recommendations: bool = Field(True, description="Include actionable recommendations")
    include_cost_analysis: bool = Field(False, description="Include cost analysis")
    include_risk_assessment: bool = Field(True, description="Include risk assessment")
    include_implementation_plan: bool = Field(False, description="Include implementation plan")
    user_role: Optional[str] = Field(None, description="User role (auditor, compliance officer, etc.)")
    expertise_level: Optional[str] = Field(None, description="User expertise level")
    company_size: Optional[str] = Field(None, description="Company size")
    compliance_maturity: Optional[str] = Field(None, description="Current compliance maturity level")
    specific_requirements: Optional[List[str]] = Field(None, description="Specific compliance requirements")
    timeline_constraints: Optional[str] = Field(None, description="Implementation timeline constraints")
    budget_constraints: Optional[str] = Field(None, description="Budget constraints")
    
    @validator('compliance_scenario')
    def validate_scenario(cls, v):  # pylint: disable=no-self-argument
        if not v.strip():
            raise ValueError('Compliance scenario cannot be empty')
        return v.strip()


class ExpertAnalysisResponse(BaseModel):
    """Response schema for expert analysis."""
    analysis: str = Field(..., description="Expert analysis of the scenario")
    recommendations: List[str] = Field(default_factory=list, description="Actionable recommendations")
    risk_assessment: Optional[Dict[str, Any]] = Field(None, description="Risk assessment results")
    implementation_plan: Optional[Dict[str, Any]] = Field(None, description="Implementation plan")
    cost_analysis: Optional[Dict[str, Any]] = Field(None, description="Cost analysis")
    regulatory_citations: List[str] = Field(default_factory=list, description="Relevant regulatory citations")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Analysis confidence score")
    knowledge_sources: List[str] = Field(default_factory=list, description="Knowledge sources used")
    processing_time_ms: int = Field(ge=0, description="Processing time in milliseconds")
    analysis_metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional analysis metadata")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class DocumentIngestionRequest(BaseModel):
    """Request schema for document ingestion."""
    file_path: str = Field(..., description="Path to document file")
    document_type: DocumentType = Field(..., description="Type of document")
    regulatory_framework: Optional[RegulatoryFramework] = Field(None, description="Regulatory framework")
    industry: Optional[Industry] = Field(None, description="Industry")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    chunking_strategy: Optional[str] = Field("semantic", description="Chunking strategy to use")
    chunk_size: Optional[int] = Field(1000, ge=100, le=5000, description="Chunk size in characters")
    chunk_overlap: Optional[int] = Field(200, ge=0, le=1000, description="Chunk overlap in characters")
    extract_metadata: bool = Field(True, description="Extract metadata from document")
    validate_schema: bool = Field(True, description="Validate document schema")
    generate_embeddings: bool = Field(True, description="Generate embeddings for document")
    store_in_vector_db: bool = Field(True, description="Store document in vector database")
    
    @validator('file_path')
    def validate_file_path(cls, v):  # pylint: disable=no-self-argument
        if not v.strip():
            raise ValueError('File path cannot be empty')
        return v.strip()


class DocumentIngestionResponse(BaseModel):
    """Response schema for document ingestion."""
    success: bool = Field(..., description="Whether ingestion was successful")
    document_id: Optional[str] = Field(None, description="ID of ingested document")
    chunk_count: int = Field(0, ge=0, description="Number of chunks created")
    processing_time_ms: int = Field(ge=0, description="Processing time in milliseconds")
    errors: List[str] = Field(default_factory=list, description="Processing errors")
    warnings: List[str] = Field(default_factory=list, description="Processing warnings")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Extracted metadata")
    quality_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Document quality score")
    validation_results: Optional[Dict[str, Any]] = Field(None, description="Schema validation results")
    embedding_model: Optional[str] = Field(None, description="Embedding model used")
    vector_store: Optional[str] = Field(None, description="Vector store used")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class KnowledgeBaseStats(BaseModel):
    """Knowledge base statistics."""
    total_documents: int = Field(ge=0, description="Total number of documents")
    total_chunks: int = Field(ge=0, description="Total number of chunks")
    regulatory_frameworks: Dict[str, int] = Field(default_factory=dict, description="Documents by framework")
    industries: Dict[str, int] = Field(default_factory=dict, description="Documents by industry")
    document_types: Dict[str, int] = Field(default_factory=dict, description="Documents by type")
    last_updated: Optional[datetime] = Field(None, description="Last update timestamp")
    storage_size_mb: Optional[float] = Field(None, ge=0, description="Storage size in MB")
    embedding_model: Optional[str] = Field(None, description="Current embedding model")
    vector_store: Optional[str] = Field(None, description="Current vector store")
    quality_metrics: Optional[Dict[str, float]] = Field(None, description="Quality metrics")
    performance_metrics: Optional[Dict[str, float]] = Field(None, description="Performance metrics")


class RAGHealthCheck(BaseModel):
    """RAG system health check."""
    status: str = Field(..., description="System status")
    vector_store_healthy: bool = Field(..., description="Vector store health")
    embedding_model_healthy: bool = Field(..., description="Embedding model health")
    retrieval_system_healthy: bool = Field(..., description="Retrieval system health")
    knowledge_base_healthy: bool = Field(..., description="Knowledge base health")
    response_time_ms: int = Field(ge=0, description="Health check response time")
    error_messages: List[str] = Field(default_factory=list, description="Error messages")
    warnings: List[str] = Field(default_factory=list, description="Warning messages")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class BatchIngestionRequest(BaseModel):
    """Request schema for batch document ingestion."""
    file_paths: List[str] = Field(..., min_length=1, max_length=100, description="List of file paths")
    document_type: DocumentType = Field(..., description="Type of documents")
    regulatory_framework: Optional[RegulatoryFramework] = Field(None, description="Regulatory framework")
    industry: Optional[Industry] = Field(None, description="Industry")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    chunking_strategy: Optional[str] = Field("semantic", description="Chunking strategy to use")
    max_concurrent: int = Field(5, ge=1, le=20, description="Maximum concurrent processing")
    batch_size: int = Field(50, ge=1, le=200, description="Batch size for processing")
    
    @validator('file_paths')
    def validate_file_paths(cls, v):  # pylint: disable=no-self-argument
        if not v:
            raise ValueError('File paths cannot be empty')
        return [path.strip() for path in v if path.strip()]


class BatchIngestionResponse(BaseModel):
    """Response schema for batch document ingestion."""
    success: bool = Field(..., description="Whether batch ingestion was successful")
    total_files: int = Field(ge=0, description="Total number of files processed")
    successful_files: int = Field(ge=0, description="Number of successfully processed files")
    failed_files: int = Field(ge=0, description="Number of failed files")
    total_chunks: int = Field(ge=0, description="Total number of chunks created")
    processing_time_ms: int = Field(ge=0, description="Total processing time in milliseconds")
    errors: List[str] = Field(default_factory=list, description="Processing errors")
    warnings: List[str] = Field(default_factory=list, description="Processing warnings")
    results: List[DocumentIngestionResponse] = Field(default_factory=list, description="Individual results")
    summary: Dict[str, Any] = Field(default_factory=dict, description="Processing summary")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class KnowledgeBaseQueryRequest(BaseModel):
    """Request schema for knowledge base queries."""
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    filters: Optional[Dict[str, Any]] = Field(None, description="Additional filters")
    max_results: int = Field(10, ge=1, le=100, description="Maximum number of results")
    include_metadata: bool = Field(True, description="Include document metadata")
    sort_by: Optional[str] = Field("relevance", description="Sort results by")
    sort_order: Optional[str] = Field("desc", description="Sort order")
    
    @validator('query')
    def validate_query(cls, v):  # pylint: disable=no-self-argument
        if not v.strip():
            raise ValueError('Query cannot be empty')
        return v.strip()


class KnowledgeBaseQueryResponse(BaseModel):
    """Response schema for knowledge base queries."""
    query: str
    results: List[SearchResult]
    total_results: int = Field(ge=0, description="Total number of results found")
    processing_time_ms: int = Field(ge=0, description="Query processing time")
    filters_applied: Dict[str, Any] = Field(default_factory=dict, description="Filters applied")
    search_strategy: str = Field(..., description="Search strategy used")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class RAGConfiguration(BaseModel):
    """RAG system configuration."""
    vector_store_type: str = Field(..., description="Vector store type")
    embedding_model: str = Field(..., description="Embedding model")
    retrieval_strategy: str = Field(..., description="Retrieval strategy")
    reranking_enabled: bool = Field(True, description="Whether reranking is enabled")
    diversity_filtering: bool = Field(False, description="Whether diversity filtering is enabled")
    max_context_length: int = Field(4000, ge=1000, le=10000, description="Maximum context length")
    similarity_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Similarity threshold")
    top_k: int = Field(10, ge=1, le=50, description="Number of top results")
    batch_size: int = Field(32, ge=1, le=128, description="Batch size for processing")
    timeout_seconds: int = Field(30, ge=1, le=300, description="Request timeout")
    cache_enabled: bool = Field(True, description="Whether caching is enabled")
    cache_ttl_seconds: int = Field(3600, ge=60, le=86400, description="Cache TTL in seconds")
