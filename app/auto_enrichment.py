"""
Auto-enrichment service for fetching missing information from external sources.
Provides intelligent data fetching to improve knowledge base completeness.
"""

import asyncio
import aiohttp
import json
from typing import List, Dict, Any, Optional
from urllib.parse import quote
import re

from app.models import AutoEnrichmentResult, MissingInfo
from app.logger import get_logger
from config import get_settings

logger = get_logger(__name__)
settings = get_settings()


class AutoEnrichmentService:
    """
    Service for automatically enriching the knowledge base with external data.
    Supports multiple data sources including Wikipedia, web search, and APIs.
    """
    
    def __init__(self):
        """Initialize the auto-enrichment service."""
        self.session: Optional[aiohttp.ClientSession] = None
        self.sources = {
            'wikipedia': self._fetch_from_wikipedia,
            'web_search': self._fetch_from_web_search,
            'general_api': self._fetch_from_general_api
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': 'AI-Knowledge-Base/1.0'}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def enrich_missing_info(
        self, 
        missing_info_list: List[MissingInfo],
        max_attempts: int = 3
    ) -> List[AutoEnrichmentResult]:
        """
        Attempt to enrich missing information from external sources.
        
        Args:
            missing_info_list: List of missing information items
            max_attempts: Maximum number of enrichment attempts per item
        
        Returns:
            List of enrichment results
        """
        if not self.session:
            raise RuntimeError("AutoEnrichmentService must be used as async context manager")
        
        results = []
        
        for missing_info in missing_info_list:
            logger.info(f"Attempting to enrich: {missing_info.topic}")
            
            # Try different sources in order of preference
            for source_name, fetch_func in self.sources.items():
                try:
                    result = await fetch_func(missing_info)
                    if result.success:
                        results.append(result)
                        logger.info(f"Successfully enriched from {source_name}: {missing_info.topic}")
                        break
                    else:
                        logger.warning(f"Failed to enrich from {source_name}: {result.error_message}")
                except Exception as e:
                    logger.error(f"Error enriching from {source_name}: {e}")
                    continue
            
            # If no source worked, create a failure result
            if not any(r.topic == missing_info.topic for r in results):
                results.append(AutoEnrichmentResult(
                    success=False,
                    source="none",
                    error_message="No external sources could provide this information"
                ))
        
        return results
    
    async def _fetch_from_wikipedia(self, missing_info: MissingInfo) -> AutoEnrichmentResult:
        """
        Fetch information from Wikipedia API.
        
        Args:
            missing_info: Missing information to enrich
        
        Returns:
            AutoEnrichmentResult with Wikipedia data
        """
        try:
            # Search for relevant Wikipedia articles
            search_url = f"{settings.wikipedia_api_url}/page/summary/{quote(missing_info.topic)}"
            
            async with self.session.get(search_url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Extract relevant information
                    content = ""
                    if 'extract' in data:
                        content = data['extract']
                    elif 'description' in data:
                        content = data['description']
                    
                    if content:
                        # Clean and truncate content
                        content = self._clean_content(content)
                        return AutoEnrichmentResult(
                            success=True,
                            source="wikipedia",
                            content=content,
                            confidence=0.8 
                        )
                
                return AutoEnrichmentResult(
                    success=False,
                    source="wikipedia",
                    error_message=f"Wikipedia API returned status {response.status}"
                )
                
        except Exception as e:
            return AutoEnrichmentResult(
                success=False,
                source="wikipedia",
                error_message=str(e)
            )
    
    async def _fetch_from_web_search(self, missing_info: MissingInfo) -> AutoEnrichmentResult:
        """
        Fetch information using web search (placeholder for actual implementation).
        In a real implementation, this would use a search API like Google Custom Search.
        
        Args:
            missing_info: Missing information to enrich
        
        Returns:
            AutoEnrichmentResult with web search data
        """
        # This is a placeholder implementation
        # In a real scenario, you would integrate with:
        # - Google Custom Search API
        # - Bing Search API
        # - DuckDuckGo API
        # - etc.
        
        return AutoEnrichmentResult(
            success=False,
            source="web_search",
            error_message="Web search not implemented (requires API key)"
        )
    
    async def _fetch_from_general_api(self, missing_info: MissingInfo) -> AutoEnrichmentResult:
        """
        Fetch information from general APIs based on the topic.
        
        Args:
            missing_info: Missing information to enrich
        
        Returns:
            AutoEnrichmentResult with API data
        """
        # This could be extended to support various APIs:
        # - News APIs
        # - Academic databases
        # - Government data APIs
        # - etc.
        
        return AutoEnrichmentResult(
            success=False,
            source="general_api",
            error_message="General API not implemented"
        )
    
    def _clean_content(self, content: str, max_length: int = 1000) -> str:
        """
        Clean and format content from external sources.
        
        Args:
            content: Raw content to clean
            max_length: Maximum length of cleaned content
        
        Returns:
            Cleaned content
        """
        # Remove HTML tags
        content = re.sub(r'<[^>]+>', '', content)
        
        # Remove extra whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Remove special characters that might cause issues
        content = content.replace('\n', ' ').replace('\r', ' ')
        
        # Truncate if too long
        if len(content) > max_length:
            content = content[:max_length].rsplit(' ', 1)[0] + "..."
        
        return content.strip()
    
    def _calculate_relevance_score(
        self, 
        content: str, 
        missing_info: MissingInfo
    ) -> float:
        """
        Calculate how relevant the fetched content is to the missing information.
        
        Args:
            content: Fetched content
            missing_info: Original missing information
        
        Returns:
            Relevance score between 0 and 1
        """
        # Simple keyword-based relevance scoring
        topic_words = missing_info.topic.lower().split()
        content_lower = content.lower()
        
        matches = sum(1 for word in topic_words if word in content_lower)
        return min(matches / len(topic_words), 1.0)


# Singleton instance
_auto_enrichment_service = None

def get_auto_enrichment_service() -> AutoEnrichmentService:
    """Get or create the auto-enrichment service instance."""
    global _auto_enrichment_service
    if _auto_enrichment_service is None:
        _auto_enrichment_service = AutoEnrichmentService()
    return _auto_enrichment_service
