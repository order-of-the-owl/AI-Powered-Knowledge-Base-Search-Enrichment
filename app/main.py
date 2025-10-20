from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
from typing import List

from app.models import (
    DocumentUploadResponse,
    SearchQuery,
    SearchResponse,
    HealthCheck,
    AutoEnrichmentResult
)
from app.document_processor import get_document_processor
from app.vector_store import get_vector_store
from app.rag_pipeline import get_rag_pipeline
import time

# Initialize FastAPI app
app = FastAPI(
    title="AI Knowledge Base API",
    description="RAG-powered document search with enrichment suggestions",
    version="1.0.0"
)


# Initialize services (singletons)
document_processor = get_document_processor()
vector_store = get_vector_store()
rag_pipeline = get_rag_pipeline()


def simple_auto_enrichment(missing_info_list):
    """
    Simple auto-enrichment that provides basic information for common topics.
    In a real implementation, this would fetch from external APIs.
    """
    enrichment_results = []
    
    for missing_info in missing_info_list:
        topic = missing_info.topic.lower()
        
        # Simple keyword-based enrichment
        if any(word in topic for word in ['python', 'programming', 'code']):
            enrichment_results.append(AutoEnrichmentResult(
                success=True,
                source="knowledge_base",
                content=f"Python is a high-level programming language known for its simplicity and readability. {missing_info.topic} is a common topic in Python development.",
                confidence=0.7
            ))
        elif any(word in topic for word in ['ai', 'artificial', 'intelligence', 'machine', 'learning']):
            enrichment_results.append(AutoEnrichmentResult(
                success=True,
                source="knowledge_base",
                content=f"Artificial Intelligence (AI) refers to the simulation of human intelligence in machines. {missing_info.topic} is a key concept in AI research and applications.",
                confidence=0.6
            ))
        else:
            enrichment_results.append(AutoEnrichmentResult(
                success=False,
                source="none",
                error_message="No auto-enrichment available for this topic"
            ))
    
    return enrichment_results


@app.get("/", response_model=HealthCheck)
async def root():
    """
    Root endpoint - health check
    """
    return HealthCheck(
        status="healthy",
        message="AI Knowledge Base API is running"
    )


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """
    Detailed health check with system status
    """
    try:
        doc_count = vector_store.get_collection_count()
        return HealthCheck(
            status="healthy",
            message=f"System operational. Knowledge base contains {doc_count} document chunks."
        )
    except Exception as e:
        return HealthCheck(
            status="degraded",
            message=f"System running but encountered issue: {str(e)}"
        )


@app.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a document (PDF, DOCX, or TXT)
    
    The document will be:
    1. Saved temporarily
    2. Text extracted
    3. Chunked into smaller pieces
    4. Embedded and stored in vector database
    5. Temporary file deleted
    """
    try:
        # Validate file type
        allowed_extensions = ['.pdf', '.docx', '.doc', '.txt']
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_ext}. Allowed: {', '.join(allowed_extensions)}"
            )
        
        print(f"\n📤 Receiving file: {file.filename}")
        
        # Read file content
        file_content = await file.read()
        file_size = len(file_content)
        
        # Save file temporarily
        file_path = document_processor.save_uploaded_file(file_content, file.filename)
        
        try:
            # Process document (extract text and chunk)
            result = document_processor.process_document(file_path, file.filename)
            
            if result['status'] == 'error':
                raise HTTPException(status_code=400, detail=result['error'])
            
            # Add chunks to vector store
            chunks_added = vector_store.add_documents(
                texts=result['chunks'],
                metadatas=result['metadatas']
            )
            
            # Clean up temporary file
            document_processor.cleanup_file(file_path)
            
            return DocumentUploadResponse(
                filename=file.filename,
                status="success",
                chunks_created=chunks_added,
                message=f"Successfully processed {file.filename} into {chunks_added} chunks",
                processing_time=result.get('processing_time'),
                file_size=file_size
            )
            
        except Exception as e:
            # Clean up file even if processing fails
            document_processor.cleanup_file(file_path)
            raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/search", response_model=SearchResponse)
async def search(query: SearchQuery):
    """
    Search the knowledge base with a natural language query
    
    Returns:
    - AI-generated answer
    - Confidence level
    - Source citations
    - Missing information
    - Enrichment suggestions
    - Auto-enrichment results (if enabled)
    """
    start_time = time.time()
    
    try:
        if not query.query or len(query.query.strip()) == 0:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        print(f"\n🔍 Search request: {query.query}")
        
        # Check if knowledge base is empty
        doc_count = vector_store.get_collection_count()
        if doc_count == 0:
            return SearchResponse(
                query=query.query,
                answer="The knowledge base is empty. Please upload documents first.",
                confidence="low",
                sources=[],
                missing_info=[],
                enrichment_suggestions=[{
                    "suggestion_type": "document",
                    "description": "Upload relevant documents to build the knowledge base",
                    "priority": "high"
                }],
                retrieved_chunks=0,
                processing_time=time.time() - start_time
            )
        
        # Run RAG pipeline
        response = rag_pipeline.search(query.query, top_k=query.top_k)
        
        # Add processing time
        response.processing_time = time.time() - start_time
        
        # Auto-enrichment if enabled
        if query.include_auto_enrichment and response.missing_info:
            print("🔄 Attempting auto-enrichment...")
            enrichment_results = simple_auto_enrichment(response.missing_info)
            if enrichment_results:
                response.auto_enrichment = enrichment_results[0]  # Use first result
                print(f"✅ Auto-enrichment: {enrichment_results[0].success}")
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/documents/count")
async def get_document_count():
    """
    Get the total number of document chunks in the knowledge base
    """
    try:
        count = vector_store.get_collection_count()
        return {"count": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get count: {str(e)}")


@app.get("/documents/list")
async def list_documents():
    """
    List all documents in the knowledge base (useful for debugging)
    """
    try:
        documents = vector_store.get_all_documents()
        
        # Group by source
        sources = {}
        for doc in documents:
            source = doc.get('metadata', {}).get('source', 'Unknown')
            if source not in sources:
                sources[source] = 0
            sources[source] += 1
        
        return {
            "total_chunks": len(documents),
            "sources": sources
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


@app.delete("/documents/reset")
async def reset_knowledge_base():
    """
    Delete all documents from the knowledge base
    ⚠️ WARNING: This cannot be undone!
    """
    try:
        vector_store.delete_collection()
        return {
            "status": "success",
            "message": "Knowledge base has been reset"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reset: {str(e)}")


# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)