#!/usr/bin/env python3
"""
Earnings Call Transcript Service - Multi-Provider Integration
Fetch complete earnings call transcript TEXT (full meeting records) from various sources

Supported Providers:
- FMP (Financial Modeling Prep) - Requires API key
- SEC EDGAR - Free but limited
- Web scraping fallback - For specific sources

This service focuses on getting the ACTUAL TEXT CONTENT of earnings calls,
not just numerical data (EPS, dates, etc.)
"""

import logging
import os
import requests
from datetime import datetime
from typing import Any, Dict, Optional

# Optional dependencies
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logging.warning("yfinance not installed. Numerical earnings data will not be available.")


class EarningsTranscriptService:
    """Earnings Call Transcript Service - Fetch COMPLETE transcript text"""

    def __init__(self):
        self.name = "Earnings Call Transcript Service"
        self.version = "2.0.0"
        
        # Get API keys from environment
        self.fmp_api_key = os.getenv("FMP_API_KEY", "")
        
        # API endpoints
        self.fmp_base_url = "https://financialmodelingprep.com/api/v3"
        self.fmp_v4_url = "https://financialmodelingprep.com/api/v4"
        
        if self.fmp_api_key:
            logging.info(f"Earnings Call Transcript Service v{self.version} initialized with FMP API")
        else:
            logging.warning(f"Earnings Call Transcript Service v{self.version} initialized WITHOUT FMP API key")
            logging.warning("Set FMP_API_KEY environment variable to use Financial Modeling Prep")
        
        logging.info(f"Earnings Call Transcript Service v{self.version} initialized")

    async def get_earnings_transcript(
        self,
        symbol: str,
        year: Optional[int] = None,
        quarter: Optional[int] = None,
        limit: Optional[int] = 10,
        provider: Optional[str] = "auto",
    ) -> Dict[str, Any]:
        """
        Fetch complete earnings call transcript TEXT (full meeting records).
        
        This tool retrieves the ACTUAL TEXT CONTENT of earnings call meetings,
        including management discussion, Q&A sessions, and full conversations.
        This is NOT numerical data like EPS - it's the actual words spoken in the meeting.
        
        Args:
            symbol (str): Stock ticker symbol (e.g., "AAPL", "TSLA", "MSFT").
                         This is a required parameter. Case-insensitive.
            
            year (Optional[int]): Specific year for the transcript (e.g., 2024, 2023).
                                 If not provided, returns the most recent transcripts.
            
            quarter (Optional[int]): Specific quarter (1, 2, 3, or 4).
                                    Only applicable when year is specified.
                                    If not provided with year, returns annual transcripts.
            
            limit (Optional[int]): Maximum number of transcripts to return. Defaults to 10.
                                  Only used when year/quarter are not specified.
            
            provider (Optional[str]): Data provider to use. Defaults to "auto".
                                     Valid values:
                                     - "auto": Automatically select best available provider
                                     - "fmp": Financial Modeling Prep (requires API key)
                                     - "yfinance": yfinance (provides numerical data only)
            
        
        Returns:
            Dict[str, Any]: A dictionary containing:
                - success (bool): Whether the operation succeeded
                - data (Dict or List): Transcript data:
                    * For single transcript: Dict with full text, date, symbol, etc.
                    * For multiple transcripts: List of transcript summaries
                    * Full text includes: management presentation, Q&A, etc.
                - metadata (Dict): Metadata about the response:
                    * symbol: Stock symbol queried
                    * provider: Data provider used
                    * transcript_count: Number of transcripts returned
                    * query_time: Timestamp of the query
                    * has_full_text: Whether full transcript text is included
                - error (Optional[str]): Error message if the operation failed
        
        Examples:
            # Get latest transcript for Apple
            result = await service.get_earnings_transcript("AAPL")
            
            # Get specific quarter transcript for Tesla
            result = await service.get_earnings_transcript(
                symbol="TSLA",
                year=2024,
                quarter=3
            )
            
            # Get last 5 transcripts for Microsoft
            result = await service.get_earnings_transcript(
                symbol="MSFT",
                limit=5
            )
        
        Note:
            - FMP (Financial Modeling Prep) provides the most complete transcript text
            - Requires FMP_API_KEY environment variable for FMP provider
            - Free tier may have limitations on historical data
            - yfinance does NOT provide full transcript text, only numerical data
        """
        try:
            # Validate required parameters
            if not symbol or not symbol.strip():
                raise ValueError("Stock symbol is required and cannot be empty")
            
            symbol = symbol.strip().upper()
            
            # Validate quarter if provided
            if quarter is not None and (quarter < 1 or quarter > 4):
                raise ValueError("Quarter must be between 1 and 4")
            
            # Validate provider
            valid_providers = ["auto", "fmp", "yfinance"]
            if provider not in valid_providers:
                raise ValueError(
                    f"Invalid provider: {provider}. "
                    f"Must be one of: {', '.join(valid_providers)}"
                )
            
            # Log request information
            logging.info(
                f"Earnings transcript request - Symbol: {symbol}, "
                f"Year: {year}, Quarter: {quarter}, Provider: {provider}"
            )
            
            # Auto-select provider
            if provider == "auto":
                if self.fmp_api_key:
                    provider = "fmp"
                    logging.info("Auto-selected FMP as provider")
                else:
                    provider = "yfinance"
                    logging.warning("No FMP API key found, falling back to yfinance (numerical data only)")
            
            # Fetch transcript based on provider
            if provider == "fmp":
                result = await self._fetch_from_fmp(symbol, year, quarter, limit)
            elif provider == "yfinance":
                result = await self._fetch_from_yfinance(symbol, year, quarter, limit)
            else:
                raise ValueError(f"Unsupported provider: {provider}")
            
            # Add provider info to metadata
            if result.get("success"):
                result["metadata"]["provider"] = provider
            
            return result
            
        except ValueError as ve:
            # Handle validation errors
            logging.error(f"Validation error: {str(ve)}")
            return {
                "success": False,
                "error": str(ve),
                "data": {},
                "metadata": {
                    "symbol": symbol if 'symbol' in locals() else None,
                    "query_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            }
            
        except Exception as e:
            # Handle unexpected errors
            logging.error(f"Unexpected error in get_earnings_transcript: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": f"An unexpected error occurred: {str(e)}",
                "data": {},
                "metadata": {
                    "symbol": symbol if 'symbol' in locals() else None,
                    "query_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            }

    async def _fetch_from_fmp(
        self,
        symbol: str,
        year: Optional[int],
        quarter: Optional[int],
        limit: int
    ) -> Dict[str, Any]:
        """
        Fetch earnings call transcript from Financial Modeling Prep API.
        FMP provides COMPLETE transcript text including full meeting content.
        """
        if not self.fmp_api_key:
            return {
                "success": False,
                "error": (
                    "FMP API key not configured. Please set FMP_API_KEY environment variable. "
                    "Get your free API key at: https://financialmodelingprep.com/developer/docs/"
                ),
                "data": {},
                "metadata": {
                    "symbol": symbol,
                    "query_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "provider": "fmp"
                }
            }
        
        try:
            # If year and quarter are specified, fetch specific transcript
            if year is not None and quarter is not None:
                url = f"{self.fmp_v4_url}/batch_earning_call_transcript/{symbol}"
                params = {
                    "year": year,
                    "quarter": quarter,
                    "apikey": self.fmp_api_key
                }
                
                logging.info(f"Fetching specific transcript: {symbol} Q{quarter} {year}")
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                
                if not data or len(data) == 0:
                    return {
                        "success": True,
                        "data": {},
                        "metadata": {
                            "symbol": symbol,
                            "year": year,
                            "quarter": quarter,
                            "transcript_count": 0,
                            "query_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "has_full_text": False,
                            "message": f"No transcript found for {symbol} Q{quarter} {year}"
                        }
                    }
                
                # Extract first transcript (should be the only one)
                transcript = data[0] if isinstance(data, list) else data
                
                return {
                    "success": True,
                    "data": {
                        "symbol": transcript.get("symbol", symbol),
                        "year": transcript.get("year", year),
                        "quarter": transcript.get("quarter", quarter),
                        "date": transcript.get("date", ""),
                        "full_text": transcript.get("content", ""),
                        "word_count": len(transcript.get("content", "").split()),
                        "preview": transcript.get("content", "")[:500] + "..." if transcript.get("content") else ""
                    },
                    "metadata": {
                        "symbol": symbol,
                        "year": year,
                        "quarter": quarter,
                        "transcript_count": 1,
                        "query_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "has_full_text": True
                    }
                }
            
            # If only year specified, fetch all transcripts for that year
            elif year is not None:
                url = f"{self.fmp_v4_url}/batch_earning_call_transcript/{symbol}"
                params = {
                    "year": year,
                    "apikey": self.fmp_api_key
                }
                
                logging.info(f"Fetching year transcripts: {symbol} {year}")
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                
                if not data or len(data) == 0:
                    return {
                        "success": True,
                        "data": [],
                        "metadata": {
                            "symbol": symbol,
                            "year": year,
                            "transcript_count": 0,
                            "query_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "has_full_text": False,
                            "message": f"No transcripts found for {symbol} in {year}"
                        }
                    }
                
                transcripts = []
                for t in data[:limit]:
                    transcripts.append({
                        "symbol": t.get("symbol", symbol),
                        "year": t.get("year"),
                        "quarter": t.get("quarter"),
                        "date": t.get("date"),
                        "preview": t.get("content", "")[:300] + "..." if t.get("content") else "",
                        "word_count": len(t.get("content", "").split()),
                        "has_full_text": bool(t.get("content"))
                    })
                
                return {
                    "success": True,
                    "data": transcripts,
                    "metadata": {
                        "symbol": symbol,
                        "year": year,
                        "transcript_count": len(transcripts),
                        "query_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "has_full_text": True,
                        "note": "Use year and quarter parameters to get full text of a specific transcript"
                    }
                }
            
            # Otherwise, fetch list of available transcripts
            else:
                url = f"{self.fmp_v4_url}/earning_call_transcript"
                params = {
                    "symbol": symbol,
                    "apikey": self.fmp_api_key
                }
                
                logging.info(f"Fetching available transcripts list: {symbol}")
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                
                if not data or len(data) == 0:
                    return {
                        "success": True,
                        "data": [],
                        "metadata": {
                            "symbol": symbol,
                            "transcript_count": 0,
                            "query_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "has_full_text": False,
                            "message": f"No transcripts found for {symbol}"
                        }
                    }
                
                transcripts = []
                for t in data[:limit]:
                    transcripts.append({
                        "symbol": t.get("symbol", symbol),
                        "year": t.get("year"),
                        "quarter": t.get("quarter"),
                        "date": t.get("date"),
                        "preview": t.get("content", "")[:300] + "..." if t.get("content") else "",
                        "word_count": len(t.get("content", "").split()) if t.get("content") else 0,
                    })
                
                return {
                    "success": True,
                    "data": transcripts,
                    "metadata": {
                        "symbol": symbol,
                        "transcript_count": len(transcripts),
                        "query_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "has_full_text": False,
                        "note": "Use year and quarter parameters to get full text of a specific transcript"
                    }
                }
                
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                error_msg = "Invalid FMP API key. Please check your FMP_API_KEY environment variable."
            elif e.response.status_code == 403:
                error_msg = "FMP API key does not have access to this endpoint. Upgrade your plan at financialmodelingprep.com"
            else:
                error_msg = f"FMP API error: {str(e)}"
            
            logging.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "data": {},
                "metadata": {
                    "symbol": symbol,
                    "query_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "provider": "fmp"
                }
            }
        
        except Exception as e:
            logging.error(f"Error fetching from FMP: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": f"Failed to fetch from FMP: {str(e)}",
                "data": {},
                "metadata": {
                    "symbol": symbol,
                    "query_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "provider": "fmp"
                }
            }

    async def _fetch_from_yfinance(
        self,
        symbol: str,
        year: Optional[int],
        quarter: Optional[int],
        limit: int
    ) -> Dict[str, Any]:
        """
        Fetch earnings data from yfinance.
        NOTE: yfinance does NOT provide full transcript text, only numerical data.
        """
        if not YFINANCE_AVAILABLE:
            return {
                "success": False,
                "error": "yfinance library is not installed. Please install with: pip install yfinance",
                "data": {},
                "metadata": {
                    "symbol": symbol,
                    "query_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "provider": "yfinance"
                }
            }
        
        logging.warning(
            "yfinance does NOT provide full earnings call transcript text. "
            "Only numerical earnings data is available. "
            "To get actual transcript text, use FMP provider with FMP_API_KEY."
        )
        
        try:
            ticker = yf.Ticker(symbol)
            
            # Get earnings history
            earnings_history = ticker.earnings_history
            
            if earnings_history is None or earnings_history.empty:
                return {
                    "success": True,
                    "data": [],
                    "metadata": {
                        "symbol": symbol,
                        "transcript_count": 0,
                        "query_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "has_full_text": False,
                        "provider": "yfinance",
                        "warning": "yfinance does not provide full transcript text, only numerical data"
                    }
                }
            
            earnings_history = earnings_history.reset_index()
            records = []
            
            for _, row in earnings_history.iterrows():
                record = {
                    "date": str(row.get("quarter", "")),
                    "eps_actual": float(row.get("epsActual", 0)) if row.get("epsActual") else None,
                    "eps_estimate": float(row.get("epsEstimate", 0)) if row.get("epsEstimate") else None,
                    "surprise_percent": float(row.get("surprisePercent", 0)) if row.get("surprisePercent") else None,
                    "note": "This is numerical data only, not transcript text"
                }
                records.append(record)
            
            return {
                "success": True,
                "data": records[:limit],
                "metadata": {
                    "symbol": symbol,
                    "transcript_count": len(records[:limit]),
                    "query_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "has_full_text": False,
                    "provider": "yfinance",
                    "warning": (
                        "yfinance does not provide full earnings call transcript text. "
                        "This is only numerical earnings data (EPS, estimates, etc.). "
                        "To get actual transcript text, set FMP_API_KEY environment variable "
                        "and use provider='fmp' or provider='auto'."
                    )
                }
            }
            
        except Exception as e:
            logging.error(f"Error fetching from yfinance: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": f"Failed to fetch from yfinance: {str(e)}",
                "data": {},
                "metadata": {
                    "symbol": symbol,
                    "query_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "provider": "yfinance"
                }
            }


# Test code
if __name__ == "__main__":
    import asyncio
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    async def test_earnings_service():
        """Test the Earnings Transcript Service with various scenarios"""
        service = EarningsTranscriptService()
        
        print("\n" + "="*80)
        print("Testing Earnings Call Transcript Service (FULL TEXT)")
        print("="*80)
        
        # Test 1: Check if FMP API key is configured
        print("\n[Test 1] Checking API configuration...")
        if service.fmp_api_key:
            print(f"✓ FMP API key configured (length: {len(service.fmp_api_key)})")
        else:
            print("✗ FMP API key NOT configured")
            print("  Set FMP_API_KEY environment variable to get full transcript text")
            print("  Get free key at: https://financialmodelingprep.com/developer/docs/")
        
        # Test 2: Get latest transcripts list for AAPL
        print("\n[Test 2] Fetching available transcripts for AAPL...")
        result1 = await service.get_earnings_transcript("AAPL", limit=5)
        print(f"Success: {result1['success']}")
        print(f"Provider: {result1['metadata'].get('provider', 'N/A')}")
        print(f"Transcript count: {result1['metadata'].get('transcript_count', 0)}")
        print(f"Has full text: {result1['metadata'].get('has_full_text', False)}")
        
        if result1['success'] and result1['data']:
            if isinstance(result1['data'], list) and len(result1['data']) > 0:
                print(f"\nFirst transcript info:")
                first = result1['data'][0]
                for key, value in first.items():
                    if key != 'preview':
                        print(f"  {key}: {value}")
                if 'preview' in first:
                    print(f"  preview: {first['preview'][:100]}...")
        
        if not result1['success']:
            print(f"Error: {result1['error']}")
        elif 'warning' in result1['metadata']:
            print(f"Warning: {result1['metadata']['warning']}")
        
        # Test 3: Try to get specific transcript (Q3 2024 for AAPL)
        print("\n[Test 3] Fetching specific transcript: AAPL Q3 2024...")
        result2 = await service.get_earnings_transcript("AAPL", year=2024, quarter=3)
        print(f"Success: {result2['success']}")
        print(f"Provider: {result2['metadata'].get('provider', 'N/A')}")
        
        if result2['success'] and result2['data']:
            if isinstance(result2['data'], dict) and 'full_text' in result2['data']:
                print(f"✓ Got full transcript text!")
                print(f"  Word count: {result2['data'].get('word_count', 0)}")
                print(f"  Date: {result2['data'].get('date', 'N/A')}")
                print(f"  Preview: {result2['data'].get('preview', '')[:200]}...")
            else:
                print(f"Data: {result2['data']}")
        
        if not result2['success']:
            print(f"Error: {result2['error']}")
        
        # Test 4: Error handling - empty symbol
        print("\n[Test 4] Testing error handling with empty symbol...")
        result3 = await service.get_earnings_transcript("")
        print(f"Success: {result3['success']}")
        print(f"Error: {result3.get('error', 'N/A')}")
        
        # Test 5: Error handling - invalid quarter
        print("\n[Test 5] Testing error handling with invalid quarter...")
        result4 = await service.get_earnings_transcript("AAPL", year=2024, quarter=5)
        print(f"Success: {result4['success']}")
        print(f"Error: {result4.get('error', 'N/A')}")
        
        print("\n" + "="*80)
        print("Testing completed")
        print("="*80 + "\n")
        
        if not service.fmp_api_key:
            print("\n" + "!"*80)
            print("IMPORTANT: To get FULL transcript text, set FMP_API_KEY:")
            print("  1. Get free key at: https://financialmodelingprep.com/developer/docs/")
            print("  2. Set environment variable: export FMP_API_KEY='your_key_here'")
            print("  3. Run this test again")
            print("!"*80 + "\n")
    
    # Run tests
    asyncio.run(test_earnings_service())
