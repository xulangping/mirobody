#!/usr/bin/env python3
"""
Web Search Service
Provides Google Search functionality via SerpApi.
"""
import os
import logging
from typing import Dict, Any, Optional
from serpapi import GoogleSearch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WebSearchService:
    """
    Web Search Service.
    
    Encapsulates SerpApi Google Search functionality.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.name = "Web Search Service"
        # Get API key from environment if not provided
        self.api_key = api_key or os.getenv('SERPAPI_API_KEY')
        
        if not self.api_key:
            logger.warning("SERPAPI_API_KEY not found in environment variables")
        else:
            logger.info("WebSearchService initialized with API key")

    def google_search(self, 
                     query: str, 
                     location: str = "Austin, Texas, United States", 
                     hl: str = "en", 
                     gl: str = "us", 
                     google_domain: str = "google.com",
                     **kwargs) -> Dict[str, Any]:
        """
        Perform a Google Search using SerpApi.

        Args:
            query (str): The search query.
            location (str): Location for the search. Defaults to "Austin, Texas, United States".
            hl (str): Language code. Defaults to "en".
            gl (str): Country code. Defaults to "us".
            google_domain (str): Google domain to use. Defaults to "google.com".
            **kwargs: Additional parameters to pass to the SerpApi.

        Returns:
            Dict[str, Any]: The search results dictionary.
        """
        if not self.api_key:
            logger.error("Cannot perform search: API Key is missing")
            return {"error": "API Key is missing"}

        try:
            params = {
                "q": query,
                "location": location,
                "hl": hl,
                "gl": gl,
                "google_domain": google_domain,
                "api_key": self.api_key
            }
            
            # Add any additional parameters
            params.update(kwargs)

            logger.info(f"Performing Google Search for query: {query}")
            search = GoogleSearch(params)
            results = search.get_dict()
            
            return results

        except Exception as e:
            logger.error(f"Google Search failed: {str(e)}")
            return {"error": str(e)}

if __name__ == "__main__":
    # Simple test if run directly
    service = WebSearchService()
    if service.api_key:
        print("Service initialized. Running test search...")
        res = service.google_search("Coffee")
        # Print only a snippet to avoid spamming
        print(str(res)[:200] + "...")
    else:
        print("Service initialized but no API key found. Skipping test.")

