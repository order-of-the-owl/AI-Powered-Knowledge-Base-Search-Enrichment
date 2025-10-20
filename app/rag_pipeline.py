import json
from typing import Dict, Any, List
from langchain_google_genai import ChatGoogleGenerativeAI
from app.vector_store import get_vector_store
from app.models import (
    SearchResponse, 
    ConfidenceLevel, 
    MissingInfo, 
    EnrichmentSuggestion,
    SourceDocument
)
from dotenv import load_dotenv
import os

load_dotenv()


class RAGPipeline:
    """
    Core RAG (Retrieval-Augmented Generation) pipeline.
    Handles document retrieval, answer generation, and completeness analysis.
    """
    
    def __init__(self):
        """Initialize the RAG pipeline with Gemini LLM and vector store"""
        # Initialize Gemini LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",  # Fast and efficient model
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.3,  # Lower = more focused, higher = more creative
            max_output_tokens=2048
        )
        
        # Get vector store instance
        self.vector_store = get_vector_store()
        
        print("✅ RAG Pipeline initialized with Gemini")
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant document chunks for a query
        
        Args:
            query: User's search question
            top_k: Number of chunks to retrieve
        
        Returns:
            List of relevant chunks with metadata and scores
        """
        print(f"🔍 Retrieving top {top_k} chunks for query: {query}")
        
        # Get chunks with similarity scores
        results = self.vector_store.similarity_search_with_score(query, k=top_k)
        
        if not results:
            print("⚠️  No relevant chunks found")
            return []
        
        # Log retrieval results
        print(f"✅ Retrieved {len(results)} chunks")
        for i, result in enumerate(results):
            score = result.get('score', 'N/A')
            source = result.get('metadata', {}).get('source', 'Unknown')
            print(f"  {i+1}. {source} (score: {score:.4f})")
        
        return results
    
    def format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Format retrieved chunks into a context string for the LLM
        
        Args:
            chunks: List of retrieved chunks with metadata
        
        Returns:
            Formatted context string
        """
        if not chunks:
            return "No relevant information found in the knowledge base."
        
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            content = chunk.get('content', '')
            metadata = chunk.get('metadata', {})
            source = metadata.get('source', 'Unknown')
            chunk_id = metadata.get('chunk_id', 'N/A')
            
            context_parts.append(
                f"[Document {i}: {source}, Chunk {chunk_id}]\n{content}\n"
            )
        
        return "\n---\n".join(context_parts)
    
    def create_prompt(self, query: str, context: str) -> str:
        """
        Create a comprehensive prompt for the LLM with instructions
        
        Args:
            query: User's question
            context: Retrieved document context
        
        Returns:
            Complete prompt string
        """
        prompt = f"""You are an intelligent assistant helping users find information from their knowledge base.

**Your Task:**
1. Answer the user's question based ONLY on the provided context
2. Assess your confidence level in the answer
3. Identify any missing information that would improve the answer
4. Suggest ways to enrich the knowledge base if information is incomplete

**Context from Knowledge Base:**
{context}

**User Question:**
{query}

**Instructions:**
- If the context contains the answer, provide a clear and concise response
- If information is partial, answer what you can add and note what's missing
- If no relevant information exists, clearly state this
- Always cite which documents you used (e.g., "According to Document 1...")
- Be honest about uncertainty

**Respond in JSON format with this exact structure:**
{{
    "answer": "Your detailed answer here, with citations like [Document 1]",
    "confidence": "high|medium|low",
    "reasoning": "Brief explanation of why you chose this confidence level",
    "missing_info": [
        {{
            "topic": "What specific information is missing",
            "reason": "Why this information would help answer the question better",
            "suggested_source": "Where this information might be found"
        }}
    ],
    "enrichment_suggestions": [
        {{
            "suggestion_type": "document|data_source|clarification",
            "description": "Specific suggestion to improve the knowledge base",
            "priority": "high|medium|low"
        }}
    ]
}}

**Confidence Level Guidelines:**
- HIGH: Information directly answers the question with strong evidence
- MEDIUM: Partial information available, or answer requires some inference
- LOW: Very limited information, or answer is highly uncertain

Now, respond in JSON format:"""
        
        return prompt
    
    def parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse LLM's JSON response, with fallback for non-JSON responses
        
        Args:
            response_text: Raw LLM output
        
        Returns:
            Parsed response dict
        """
        try:
            # Try to find JSON in the response (LLM might add extra text)
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON found in response")
            
            json_str = response_text[start_idx:end_idx]
            parsed = json.loads(json_str)
            
            # Validate required fields
            if 'answer' not in parsed:
                parsed['answer'] = response_text
            if 'confidence' not in parsed:
                parsed['confidence'] = 'medium'
            if 'missing_info' not in parsed:
                parsed['missing_info'] = []
            if 'enrichment_suggestions' not in parsed:
                parsed['enrichment_suggestions'] = []
            
            return parsed
            
        except Exception as e:
            print(f"⚠️  Failed to parse JSON response: {e}")
            # Fallback: treat entire response as answer
            return {
                "answer": response_text,
                "confidence": "medium",
                "reasoning": "Could not parse structured response",
                "missing_info": [],
                "enrichment_suggestions": []
            }
    
    def assess_completeness(
        self, 
        query: str, 
        chunks: List[Dict[str, Any]], 
        parsed_response: Dict[str, Any]
    ) -> tuple[List[MissingInfo], List[EnrichmentSuggestion]]:
        """
        Assess completeness and create enrichment suggestions
        
        Args:
            query: Original user query
            chunks: Retrieved chunks
            parsed_response: Parsed LLM response
        
        Returns:
            Tuple of (missing_info_list, enrichment_suggestions_list)
        """
        missing_info_list = []
        enrichment_list = []
        
        # Parse missing info from LLM response
        for item in parsed_response.get('missing_info', []):
            missing_info_list.append(MissingInfo(
                topic=item.get('topic', 'Unknown'),
                reason=item.get('reason', ''),
                suggested_source=item.get('suggested_source')
            ))
        
        # Parse enrichment suggestions from LLM response
        for item in parsed_response.get('enrichment_suggestions', []):
            enrichment_list.append(EnrichmentSuggestion(
                suggestion_type=item.get('suggestion_type', 'document'),
                description=item.get('description', ''),
                priority=item.get('priority', 'medium')
            ))
        
        # Add automatic suggestions based on retrieval quality
        if not chunks:
            enrichment_list.append(EnrichmentSuggestion(
                suggestion_type="document",
                description=f"No documents found related to: '{query}'. Consider uploading relevant documentation.",
                priority="high"
            ))
        elif len(chunks) < 3:
            enrichment_list.append(EnrichmentSuggestion(
                suggestion_type="document",
                description=f"Limited information found ({len(chunks)} chunks). More comprehensive documentation would improve answers.",
                priority="medium"
            ))
        
        return missing_info_list, enrichment_list
    
    def search(self, query: str, top_k: int = 5) -> SearchResponse:
        """
        Main search pipeline: retrieve, generate answer, assess completeness
        
        Args:
            query: User's search question
            top_k: Number of chunks to retrieve
        
        Returns:
            Complete SearchResponse with answer and metadata
        """
        print(f"\n{'='*60}")
        print(f"🔍 Processing Query: {query}")
        print(f"{'='*60}\n")
        
        try:
            # Step 1: Retrieve relevant chunks
            chunks = self.retrieve_relevant_chunks(query, top_k)
            
            # Step 2: Format context
            context = self.format_context(chunks)
            
            # Step 3: Create prompt
            prompt = self.create_prompt(query, context)
            
            # Step 4: Get LLM response
            print("🤖 Generating answer with Gemini...")
            llm_response = self.llm.invoke(prompt)
            response_text = llm_response.content
            
            # Step 5: Parse response
            parsed = self.parse_llm_response(response_text)
            
            # Step 6: Assess completeness
            missing_info, enrichment = self.assess_completeness(query, chunks, parsed)
            
            # Step 7: Extract and format sources
            sources = []
            for i, chunk in enumerate(chunks):
                metadata = chunk.get('metadata', {})
                source = metadata.get('source', 'Unknown')
                chunk_id = metadata.get('chunk_id', i)
                content = chunk.get('content', '')
                score = chunk.get('score', 0.0)
                
                # Create content preview (first 500 chars, properly truncated)
                if len(content) > 500:
                    # Find the last complete word within 500 characters
                    truncated = content[:500]
                    last_space = truncated.rfind(' ')
                    if last_space > 400:  # Only use last space if it's not too far back
                        content_preview = truncated[:last_space] + "..."
                    else:
                        content_preview = truncated + "..."
                else:
                    content_preview = content
                
                sources.append(SourceDocument(
                    filename=source,
                    chunk_id=chunk_id,
                    content_preview=content_preview,
                    relevance_score=1.0 - score if score else None,  # Convert distance to relevance
                    page_number=metadata.get('page_number')
                ))
            
            # Step 8: Determine confidence level
            confidence_str = parsed.get('confidence', 'medium').lower()
            if confidence_str == 'high':
                confidence = ConfidenceLevel.HIGH
            elif confidence_str == 'low':
                confidence = ConfidenceLevel.LOW
            else:
                confidence = ConfidenceLevel.MEDIUM
            
            # Step 9: Create response
            response = SearchResponse(
                query=query,
                answer=parsed.get('answer', 'No answer generated'),
                confidence=confidence,
                sources=sources,
                missing_info=missing_info,
                enrichment_suggestions=enrichment,
                retrieved_chunks=len(chunks),
                reasoning=parsed.get('reasoning')
            )
            
            print(f"\n✅ Answer generated (Confidence: {confidence.value})")
            print(f"📚 Sources used: {len(sources)}")
            print(f"⚠️  Missing info items: {len(missing_info)}")
            print(f"💡 Enrichment suggestions: {len(enrichment)}\n")
            
            return response
            
        except Exception as e:
            print(f"❌ Error in RAG pipeline: {e}")
            # Return error response
            return SearchResponse(
                query=query,
                answer=f"Error processing query: {str(e)}",
                confidence=ConfidenceLevel.LOW,
                sources=[],
                missing_info=[],
                enrichment_suggestions=[
                    EnrichmentSuggestion(
                        suggestion_type="clarification",
                        description="An error occurred. Please try rephrasing your question.",
                        priority="high"
                    )
                ],
                retrieved_chunks=0
            )


# Singleton instance
_rag_pipeline_instance = None

def get_rag_pipeline() -> RAGPipeline:
    """Get or create the RAG pipeline instance"""
    global _rag_pipeline_instance
    if _rag_pipeline_instance is None:
        _rag_pipeline_instance = RAGPipeline()
    return _rag_pipeline_instance