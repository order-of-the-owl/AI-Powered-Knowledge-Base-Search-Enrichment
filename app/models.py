from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime


class DocumentUploadResponse(BaseModel):
    """Response after uploading a document"""
    filename: str
    status: str
    chunks_created: int
    message: str
    processing_time: Optional[float] = None
    file_size: Optional[int] = None


class SearchQuery(BaseModel):
    """User's search query with enhanced validation"""
    query: str = Field(..., min_length=1, max_length=1000, description="The search question")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of relevant chunks to retrieve")
    include_auto_enrichment: bool = Field(default=False, description="Whether to attempt auto-enrichment")


class ConfidenceLevel(str, Enum):
    """How confident the AI is in its answer"""
    HIGH = "high"      # Strong evidence in documents
    MEDIUM = "medium"  # Partial information available
    LOW = "low"        # Very limited or uncertain information


class MissingInfo(BaseModel):
    """Information that's missing from the knowledge base"""
    topic: str = Field(..., min_length=1, description="What information is missing")
    reason: str = Field(..., min_length=1, description="Why this information is needed")
    suggested_source: Optional[str] = Field(None, description="Where to find this information")
    confidence_impact: str = Field(default="medium", description="How much this would improve confidence")


class EnrichmentSuggestion(BaseModel):
    """Suggestions to improve the knowledge base"""
    suggestion_type: str = Field(..., description="Type: 'document', 'data_source', 'clarification', 'auto_fetch'")
    description: str = Field(..., min_length=1, description="What to add or improve")
    priority: str = Field(..., description="Priority: 'high', 'medium', 'low'")
    actionable: bool = Field(default=True, description="Whether this suggestion can be automatically acted upon")
    estimated_effort: Optional[str] = Field(None, description="Estimated effort to implement")


class SourceDocument(BaseModel):
    """Source document information"""
    filename: str
    chunk_id: int
    content_preview: str = Field(..., max_length=500, description="Preview of the content")
    relevance_score: Optional[float] = Field(None, description="Relevance score (0-1)")
    page_number: Optional[int] = Field(None, description="Page number if available")


class AutoEnrichmentResult(BaseModel):
    """Result from auto-enrichment attempt"""
    success: bool = Field(..., description="Whether auto-enrichment was successful")
    source: str = Field(..., description="Source of the enrichment (e.g., 'knowledge_base', 'external_api')")
    content: Optional[str] = Field(None, description="The enriched content if successful")
    confidence: Optional[float] = Field(None, description="Confidence score for the enrichment")
    error_message: Optional[str] = Field(None, description="Error message if enrichment failed")


class SearchResponse(BaseModel):
    """Complete response to a search query with enhanced structure"""
    query: str
    answer: str
    confidence: ConfidenceLevel
    sources: List[SourceDocument] = Field(default_factory=list, description="Source documents used")
    missing_info: List[MissingInfo] = Field(default_factory=list)
    enrichment_suggestions: List[EnrichmentSuggestion] = Field(default_factory=list)
    retrieved_chunks: int = Field(..., description="Number of relevant chunks found")
    processing_time: Optional[float] = Field(None, description="Time taken to process query in seconds")
    auto_enrichment: Optional[AutoEnrichmentResult] = Field(None, description="Auto-enrichment attempt result")
    reasoning: Optional[str] = Field(None, description="AI's reasoning for the confidence level")


class HealthCheck(BaseModel):
    """API health status"""
    status: str
    message: str