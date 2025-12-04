#!/usr/bin/env python3
"""
Fundamentals Service - Finnhub Integration
Fetch key fundamental metrics and financial ratios using Finnhub API.
Replaces yfinance to provide reliable, comprehensive fundamental analysis data.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional
from .finnhub_service import FinnhubService

class FundamentalsService:
    """Fundamentals Service - Fetch key fundamental metrics using Finnhub"""

    def __init__(self):
        self.name = "Fundamentals Service"
        self.version = "2.0.0"
        
        # Initialize Finnhub Service
        self.finnhub = FinnhubService()
        
        if not self.finnhub.api_key:
            logging.warning("Finnhub API key not found. Fundamentals Service will not work.")
        else:
            logging.info(f"Fundamentals Service v{self.version} initialized with Finnhub integration")

    async def _get_key_metrics(
        self,
        symbol: str,
    ) -> Dict[str, Any]:
        """
        Internal helper to fetch key metrics.
        """
        if not self.finnhub.api_key:
            return {"success": False, "error": "Finnhub API key missing"}

        today = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        
        res = await self.finnhub.get_financials(symbol, start_date, today)
        
        if not res.get("success"):
            return res
            
        raw_data = res.get("data", {}).get("basic_financials", {})
        if not raw_data:
             return {"success": False, "error": "No financial data found"}
             
        metrics = raw_data.get("metric", {})
        
        data = {
            "Valuation Metrics": {
                "market_cap": metrics.get("marketCapitalization"),
                "trailing_pe": metrics.get("peTTM"),
                "forward_pe": metrics.get("peExclExtraTTM"),
                "price_to_book": metrics.get("pbAnnual") or metrics.get("pbQuarterly"),
                "price_to_sales": metrics.get("psTTM"),
                "peg_ratio": metrics.get("pegTTM"),
                "enterprise_value": metrics.get("enterpriseValue")
            },
            "Profitability Metrics": {
                "profit_margins": metrics.get("netProfitMarginTTM"),
                "operating_margins": metrics.get("operatingMarginTTM"),
                "gross_margins": metrics.get("grossMarginTTM"),
                "return_on_equity": metrics.get("roeTTM"),
                "return_on_assets": metrics.get("roaTTM"),
            },
            "Per-Share Metrics": {
                "trailing_eps": metrics.get("epsTTM"),
                "revenue_per_share": metrics.get("revenuePerShareTTM"),
                "book_value": metrics.get("bookValuePerShareAnnual") or metrics.get("bookValuePerShareQuarterly"),
            },
            "Dividend Metrics": {
                "dividend_yield": metrics.get("dividendYieldIndicatedAnnual"),
                "payout_ratio": metrics.get("payoutRatioTTM"),
            },
            "Financial Health": {
                "debt_to_equity": metrics.get("totalDebt/totalEquityAnnual") or metrics.get("totalDebt/totalEquityQuarterly"),
                "current_ratio": metrics.get("currentRatioAnnual") or metrics.get("currentRatioQuarterly"),
                "quick_ratio": metrics.get("quickRatioAnnual") or metrics.get("quickRatioQuarterly"),
            },
            "Growth": {
                "revenue_growth_5y": metrics.get("revenueGrowth5Y"),
                "eps_growth_5y": metrics.get("epsGrowth5Y"),
                "revenue_growth_ttm": metrics.get("revenueGrowthTTMYoy"),
            },
            "Price Stats": {
                "52_week_high": metrics.get("52WeekHigh"),
                "52_week_low": metrics.get("52WeekLow"),
                "52_week_high_date": metrics.get("52WeekHighDate"),
                "52_week_low_date": metrics.get("52WeekLowDate"),
                "beta": metrics.get("beta"),
            }
        }
        
        return {
            "success": True,
            "data": data
        }

    async def get_analysis_report(self, symbol: str) -> Dict[str, Any]:
        """
        Get a comprehensive analysis report combining Fundamentals, Market Data, and Sentiment.
        Equivalent to 'get_all_stock_info' but structured for quick analysis.
        """
        today = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        
        # 1. Get Key Metrics (Internal Call)
        metrics_res = await self._get_key_metrics(symbol)
        fundamentals_data = metrics_res.get("data", {}) if metrics_res.get("success") else {}

        # 2. Get Other Data via FinnhubService aggregator
        res = await self.finnhub.get_all_stock_info(symbol, start_date, today)
        
        if not res.get("success"):
            return res
            
        raw = res.get("data", {})
        
        # Re-structure for clarity
        report = {
            "profile": raw.get("company_profile", {}).get("profile", {}),
            "quote": raw.get("market_data", {}).get("quote", {}),
            "fundamentals": fundamentals_data,
            "sentiment": self._summarize_sentiment(raw.get("sentiment", {})),
            "recommendation": self._summarize_recommendation(raw.get("analysis", {}).get("recommendation_trends", [])),
        }
        
        return {
            "success": True,
            "data": report,
            "metadata": {
                "symbol": symbol,
                "report_date": today
            }
        }

    def _summarize_sentiment(self, sentiment_data: Dict) -> Dict:
        """Helper to summarize sentiment"""
        return sentiment_data.get("sentiment", {}) 

    def _summarize_recommendation(self, trends: list) -> Dict:
        """Get latest recommendation consensus"""
        if not trends or not isinstance(trends, list):
            return {"consensus": "Unknown"}
        
        latest = trends[0]
        total = latest.get("buy", 0) + latest.get("hold", 0) + latest.get("sell", 0) + latest.get("strongBuy", 0) + latest.get("strongSell", 0)
        
        if total == 0: return {"consensus": "No Data"}
        
        score = (latest.get("strongBuy", 0)*2 + latest.get("buy", 0)*1 - latest.get("sell", 0)*1 - latest.get("strongSell", 0)*2)
        consensus = "Hold"
        if score > total * 0.5: consensus = "Strong Buy"
        elif score > 0: consensus = "Buy"
        elif score < -total * 0.5: consensus = "Strong Sell"
        elif score < 0: consensus = "Sell"
        
        return {
            "consensus": consensus,
            "breakdown": latest,
            "period": latest.get("period")
        }

if __name__ == "__main__":
    import asyncio
    import json
    
    async def test():
        service = FundamentalsService()
        symbol = "AAPL"
        
        print(f"\\nTesting Analysis Report for {symbol}...")
        res2 = await service.get_analysis_report(symbol)
        print(json.dumps(res2, indent=2))

    asyncio.run(test())
