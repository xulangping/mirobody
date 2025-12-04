#!/usr/bin/env python3
"""
Massive.com (Polygon.io) Service
Provides stock market data and technical indicators via Massive/Poly API.
Complementary to FinnhubService.
"""
import os
import logging
import datetime
import requests
from typing import Any, Dict, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MassiveService:
    """
    Massive.com (Poly) Service.
    
    Provides access to:
    - Ticker Overview (Reference Data)
    - Technical Indicators (SMA)
    
    Base URL defaults to Polygon.io as endpoints match exactly.
    Can be overridden via MASSIVE_API_URL env var.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.name = "Massive US Stock Service"
        # Strictly use POLYGON_API_KEY as requested
        self.api_key = api_key or os.getenv('POLYGON_API_KEY')
        
        # Default to Polygon.io API, but allow override if Massive.com has a different host
        self.base_url = os.getenv('MASSIVE_API_URL', 'https://api.polygon.io')
        
        if not self.api_key:
            logger.warning("POLYGON_API_KEY not found in environment variables")
        else:
            logger.info(f"MassiveService initialized with URL: {self.base_url}")

    def _format_response(self, data: Any, meta: Dict[str, Any] = None) -> Dict[str, Any]:
        """Standardized response format."""
        if data is None:
            return {
                "success": False,
                "error": "No data returned or API error",
                "metadata": {
                    "query_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    **(meta or {})
                }
            }
        return {
            "success": True,
            "found": True if data else False,
            "data": data,
            "metadata": {
                "query_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                **(meta or {})
            }
        }

    def _safe_api_call(self, endpoint: str, params: Dict[str, Any] = None) -> Any:
        """Helper to perform API calls."""
        if not self.api_key:
            logger.error("API key missing")
            return None
        
        url = f"{self.base_url}{endpoint}"
        params = params or {}
        params['apiKey'] = self.api_key
        
        try:
            logger.info(f"Requesting: {url} with params {str(params)}")
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for API level errors (Polygon style)
            if isinstance(data, dict) and data.get('status') not in ['OK', 'success', None]:
                 # Note: Some endpoints like Ticker Details might not have 'status' or use different conventions
                 # But indicators usually return status='OK'
                 if 'error' in data or 'message' in data:
                     logger.warning(f"API returned non-OK status: {data}")

            return data
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP Error: {e.response.status_code} - {e.response.text}")
            return None
        except Exception as e:
            logger.error(f"API call failed to {url}: {str(e)}")
            return None

    async def get_ticker_overview(self, ticker: str) -> Dict[str, Any]:
        """
        Retrieve comprehensive details for a single ticker.
        Endpoint: /v3/reference/tickers/{ticker}
        
        Args:
            ticker: Stock symbol (e.g., 'AAPL').
            
        Returns:
            Dict: Ticker details (market cap, description, branding, etc.)
        """
        endpoint = f"/v3/reference/tickers/{ticker.upper()}"
        
        # Note: The API might support 'date' param, defaulting to recent.
        response = self._safe_api_call(endpoint)
        
        data = None
        if response and 'results' in response:
            data = response['results']
            
        return self._format_response(data, {"symbol": ticker, "type": "overview"})

    async def get_sma(self, ticker: str, 
                      timespan: str = 'day', 
                      window: int = 20, 
                      adjusted: bool = True, 
                      limit: int = 10) -> Dict[str, Any]:
        """
        Retrieve Simple Moving Average (SMA).
        Endpoint: /v1/indicators/sma/{ticker}
        
        Args:
            ticker: Stock symbol.
            timespan: aggregate time window (minute, hour, day, week, month, quarter, year).
            window: The window size (e.g. 20 for 20-day SMA).
            adjusted: Whether to use split-adjusted prices.
            limit: Number of results to return (default 10).
            
        Returns:
            Dict: SMA values and underlying data.
        """
        endpoint = f"/v1/indicators/sma/{ticker.upper()}"
        
        params = {
            'timespan': timespan,
            'window': window,
            'series_type': 'close',
            'adjusted': str(adjusted).lower(),
            'limit': limit,
            'order': 'desc' # Get most recent first
        }
        
        response = self._safe_api_call(endpoint, params)
        
        data = None
        if response and 'results' in response:
            data = response['results']
            
        meta = {
            "symbol": ticker, 
            "indicator": "SMA", 
            "window": window,
            "timespan": timespan
        }
        return self._format_response(data, meta)

    async def get_ema(self, ticker: str, 
                      timespan: str = 'day', 
                      window: int = 20, 
                      adjusted: bool = True, 
                      limit: int = 10) -> Dict[str, Any]:
        """
        Retrieve Exponential Moving Average (EMA).
        Endpoint: /v1/indicators/ema/{ticker}
        
        Args:
            ticker: Stock symbol.
            timespan: aggregate time window (minute, hour, day, week, month, quarter, year).
            window: The window size (e.g. 20 for 20-day EMA).
            adjusted: Whether to use split-adjusted prices.
            limit: Number of results to return (default 10).
            
        Returns:
            Dict: EMA values and underlying data.
        """
        endpoint = f"/v1/indicators/ema/{ticker.upper()}"
        
        params = {
            'timespan': timespan,
            'window': window,
            'series_type': 'close',
            'adjusted': str(adjusted).lower(),
            'limit': limit,
            'order': 'desc'
        }
        
        response = self._safe_api_call(endpoint, params)
        
        data = None
        if response and 'results' in response:
            data = response['results']
            
        meta = {
            "symbol": ticker, 
            "indicator": "EMA", 
            "window": window,
            "timespan": timespan
        }
        return self._format_response(data, meta)

    async def get_macd(self, ticker: str, 
                       timespan: str = 'day', 
                       short_window: int = 12, 
                       long_window: int = 26, 
                       signal_window: int = 9, 
                       adjusted: bool = True, 
                       limit: int = 10) -> Dict[str, Any]:
        """
        Retrieve Moving Average Convergence/Divergence (MACD).
        Endpoint: /v1/indicators/macd/{ticker}
        
        Args:
            ticker: Stock symbol.
            timespan: aggregate time window.
            short_window: Short window size (default 12).
            long_window: Long window size (default 26).
            signal_window: Signal window size (default 9).
            adjusted: Whether to use split-adjusted prices.
            limit: Number of results to return (default 10).
            
        Returns:
            Dict: MACD values (histogram, signal, value) and underlying data.
        """
        endpoint = f"/v1/indicators/macd/{ticker.upper()}"
        
        params = {
            'timespan': timespan,
            'short_window': short_window,
            'long_window': long_window,
            'signal_window': signal_window,
            'series_type': 'close',
            'adjusted': str(adjusted).lower(),
            'limit': limit,
            'order': 'desc'
        }
        
        response = self._safe_api_call(endpoint, params)
        
        data = None
        if response and 'results' in response:
            data = response['results']
            
        meta = {
            "symbol": ticker, 
            "indicator": "MACD", 
            "short_window": short_window,
            "long_window": long_window,
            "signal_window": signal_window,
            "timespan": timespan
        }
        return self._format_response(data, meta)

    async def get_rsi(self, ticker: str, 
                      timespan: str = 'day', 
                      window: int = 14, 
                      adjusted: bool = True, 
                      limit: int = 10) -> Dict[str, Any]:
        """
        Retrieve Relative Strength Index (RSI).
        Endpoint: /v1/indicators/rsi/{ticker}
        
        Args:
            ticker: Stock symbol.
            timespan: aggregate time window.
            window: The window size (default 14).
            adjusted: Whether to use split-adjusted prices.
            limit: Number of results to return (default 10).
            
        Returns:
            Dict: RSI values and underlying data.
        """
        endpoint = f"/v1/indicators/rsi/{ticker.upper()}"
        
        params = {
            'timespan': timespan,
            'window': window,
            'series_type': 'close',
            'adjusted': str(adjusted).lower(),
            'limit': limit,
            'order': 'desc'
        }
        
        response = self._safe_api_call(endpoint, params)
        
        data = None
        if response and 'results' in response:
            data = response['results']
            
        meta = {
            "symbol": ticker, 
            "indicator": "RSI", 
            "window": window,
            "timespan": timespan
        }
        return self._format_response(data, meta)

if __name__ == "__main__":
    # Test Block
    import asyncio
    
    async def test():
        service = MassiveService()
        symbol = "AAPL"
        
        print(f"\n--- Testing get_ticker_overview for {symbol} ---")
        res = await service.get_ticker_overview(symbol)
        print(f"Success: {res.get('success')}")
        if res.get('success'):
            print(f"Name: {res['data'].get('name')}")
            print(f"Market Cap: {res['data'].get('market_cap')}")

        print(f"\n--- Testing get_sma for {symbol} ---")
        res = await service.get_sma(symbol, window=50)
        print(f"Success: {res.get('success')}")
        if res.get('success'):
            values = res['data'].get('values', [])
            print(f"Latest 3 SMA values: {values[:3]}")

        print(f"\n--- Testing get_ema for {symbol} ---")
        res = await service.get_ema(symbol, window=20)
        print(f"Success: {res.get('success')}")
        if res.get('success'):
            values = res['data'].get('values', [])
            print(f"Latest 3 EMA values: {values[:3]}")

        print(f"\n--- Testing get_macd for {symbol} ---")
        res = await service.get_macd(symbol)
        print(f"Success: {res.get('success')}")
        if res.get('success'):
            values = res['data'].get('values', [])
            print(f"Latest 3 MACD values: {values[:3]}")

        print(f"\n--- Testing get_rsi for {symbol} ---")
        res = await service.get_rsi(symbol, window=14)
        print(f"Success: {res.get('success')}")
        if res.get('success'):
            values = res['data'].get('values', [])
            print(f"Latest 3 RSI values: {values[:3]}")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(test())

