#!/usr/bin/env python3
"""
Financial Research Agent with Real-Time Data
Date-aware, sentiment analysis, and stock price tracking
"""

import requests
import json
import time
from bs4 import BeautifulSoup
from serpapi import GoogleSearch
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import re

from context_manager import ContextManager
from deep_research_agent import (
    OllamaClient,
    WebSearcher,
    WebScraper,
    ResearchStep,
    ResearchReport
)


class FinancialDataAPI:
    """Get real-time financial data using yfinance"""

    def get_stock_price(self, symbol: str) -> Dict:
        """
        Get current stock price using yfinance library

        Falls back to web scraping if yfinance fails
        """
        # Try yfinance first (most reliable)
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Get current price (try multiple fields)
            price = info.get('currentPrice') or info.get('regularMarketPrice')

            if price:
                return {
                    'symbol': symbol,
                    'price': float(price),
                    'previous_close': info.get('previousClose'),
                    'open': info.get('open'),
                    'day_high': info.get('dayHigh'),
                    'day_low': info.get('dayLow'),
                    'volume': info.get('volume'),
                    'market_cap': info.get('marketCap'),
                    '52_week_high': info.get('fiftyTwoWeekHigh'),
                    '52_week_low': info.get('fiftyTwoWeekLow'),
                    'avg_50_day': info.get('fiftyDayAverage'),
                    'avg_200_day': info.get('twoHundredDayAverage'),
                    'timestamp': datetime.now().isoformat(),
                    'source': 'yfinance',
                    'company_name': info.get('longName', symbol)
                }

        except ImportError:
            print(f"   ⚠️ yfinance not installed, falling back to web scraping")
        except Exception as e:
            print(f"   ⚠️ yfinance failed ({e}), trying web scraping fallback")

        # Fallback: Web scraping (less reliable)
        try:
            url = f"https://finance.yahoo.com/quote/{symbol}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }

            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'lxml')

            price = None

            # Try multiple scraping methods
            price_elem = soup.find('fin-streamer', {'data-symbol': symbol, 'data-field': 'regularMarketPrice'})
            if price_elem and price_elem.get('data-value'):
                price = float(price_elem.get('data-value'))

            if not price:
                price_elem = soup.find('fin-streamer', {'data-symbol': symbol})
                if price_elem and price_elem.get('data-value'):
                    try:
                        price = float(price_elem.get('data-value'))
                    except:
                        pass

            if price:
                return {
                    'symbol': symbol,
                    'price': price,
                    'timestamp': datetime.now().isoformat(),
                    'source': 'Yahoo Finance (scraped)'
                }

        except Exception as e:
            print(f"   ❌ Web scraping also failed: {e}")

        # Both methods failed
        return {
            'symbol': symbol,
            'price': None,
            'error': 'Could not fetch price - both yfinance and web scraping failed',
            'timestamp': datetime.now().isoformat(),
            'note': 'Install yfinance: pip install yfinance'
        }


class SentimentAnalyzer:
    """Analyze sentiment of news articles"""

    def __init__(self, llm_client: OllamaClient):
        self.llm = llm_client

    def analyze_article(self, title: str, content: str) -> Dict:
        """
        Analyze sentiment of a single article

        Returns:
            {
                'sentiment': 'positive' | 'negative' | 'neutral',
                'score': -1.0 to 1.0,
                'reasoning': 'explanation'
            }
        """

        prompt = f"""Analyze the sentiment of this financial news article.

Title: {title}

Content: {content[:2000]}

Determine:
1. Overall sentiment: positive, negative, or neutral
2. Score: -1.0 (very negative) to +1.0 (very positive)
3. Brief reasoning

Respond in JSON format:
{{
    "sentiment": "positive|negative|neutral",
    "score": 0.5,
    "reasoning": "brief explanation"
}}

Respond ONLY with valid JSON, no other text."""

        response = self.llm.generate(prompt, temperature=0.1)

        try:
            # Extract JSON
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                result = json.loads(response[json_start:json_end])
                return result
            else:
                return {'sentiment': 'neutral', 'score': 0.0, 'reasoning': 'Could not parse'}
        except Exception as e:
            return {'sentiment': 'neutral', 'score': 0.0, 'reasoning': f'Error: {e}'}

    def aggregate_sentiment(self, articles: List[Dict]) -> Dict:
        """
        Aggregate sentiment across multiple articles

        Returns:
            {
                'overall_sentiment': 'positive|negative|neutral',
                'avg_score': float,
                'distribution': {'positive': count, 'negative': count, 'neutral': count},
                'trend': 'improving|declining|stable'
            }
        """

        if not articles:
            return {
                'overall_sentiment': 'neutral',
                'avg_score': 0.0,
                'distribution': {'positive': 0, 'negative': 0, 'neutral': 0},
                'trend': 'stable'
            }

        scores = [a['sentiment_score'] for a in articles if 'sentiment_score' in a]
        sentiments = [a['sentiment'] for a in articles if 'sentiment' in a]

        avg_score = sum(scores) / len(scores) if scores else 0.0

        distribution = {
            'positive': sentiments.count('positive'),
            'negative': sentiments.count('negative'),
            'neutral': sentiments.count('neutral')
        }

        # Determine overall
        if avg_score > 0.3:
            overall = 'positive'
        elif avg_score < -0.3:
            overall = 'negative'
        else:
            overall = 'neutral'

        # Determine trend (compare first half vs second half chronologically)
        if len(scores) >= 4:
            mid = len(scores) // 2
            first_half_avg = sum(scores[:mid]) / len(scores[:mid])
            second_half_avg = sum(scores[mid:]) / len(scores[mid:])

            if second_half_avg > first_half_avg + 0.2:
                trend = 'improving'
            elif second_half_avg < first_half_avg - 0.2:
                trend = 'declining'
            else:
                trend = 'stable'
        else:
            trend = 'insufficient_data'

        return {
            'overall_sentiment': overall,
            'avg_score': avg_score,
            'distribution': distribution,
            'trend': trend,
            'total_articles': len(articles)
        }


class DateAwareWebSearcher(WebSearcher):
    """Web searcher that handles date filtering"""

    def search_with_date_range(
        self,
        query: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        num_results: int = 10
    ) -> List[Dict]:
        """
        Search with date filtering

        Args:
            query: Search query
            start_date: Filter articles after this date
            end_date: Filter articles before this date
            num_results: Number of results
        """

        # Add date context to query
        date_context = ""
        if start_date and end_date:
            date_context = f" after:{start_date.strftime('%Y-%m-%d')} before:{end_date.strftime('%Y-%m-%d')}"
        elif start_date:
            date_context = f" after:{start_date.strftime('%Y-%m-%d')}"

        enhanced_query = query + date_context

        params = {
            "q": enhanced_query,
            "api_key": self.api_key,
            "num": num_results * 2,  # Request more to filter later
            "tbm": "nws"  # News search
        }

        # Try with date range parameter
        if start_date and end_date:
            params["tbs"] = f"cdr:1,cd_min:{start_date.strftime('%m/%d/%Y')},cd_max:{end_date.strftime('%m/%d/%Y')}"

        try:
            search = GoogleSearch(params)
            results = search.get_dict()

            articles = []
            if "news_results" in results:
                for result in results["news_results"]:
                    # Parse date if available
                    date_str = result.get("date", "")

                    article = {
                        "title": result.get("title", ""),
                        "link": result.get("link", ""),
                        "snippet": result.get("snippet", ""),
                        "date": date_str,
                        "source": result.get("source", "")
                    }

                    # Filter by date if we can parse it
                    if start_date and date_str:
                        # Try to filter out obviously old articles
                        if any(old_indicator in date_str.lower() for old_indicator in [
                            'year ago', 'years ago', 'months ago'
                        ]):
                            continue  # Skip very old articles

                    articles.append(article)

                    if len(articles) >= num_results:
                        break

            elif "organic_results" in results:
                # Fallback to organic results
                for result in results["organic_results"]:
                    articles.append({
                        "title": result.get("title", ""),
                        "link": result.get("link", ""),
                        "snippet": result.get("snippet", ""),
                        "date": "",
                        "source": ""
                    })

                    if len(articles) >= num_results:
                        break

            return articles[:num_results]

        except Exception as e:
            print(f"Error searching: {e}")
            return []


class FinancialResearchAgent:
    """
    Financial Research Agent with:
    - Date awareness
    - Real-time stock prices
    - Sentiment analysis
    - Volatility assessment
    """

    def __init__(
        self,
        serpapi_key: str,
        ollama_url: str = "http://localhost:11434",
        model: str = "qwen3:32b",
        max_context: int = 40960
    ):
        self.llm = OllamaClient(ollama_url, model)
        self.searcher = DateAwareWebSearcher(serpapi_key)
        self.scraper = WebScraper()
        self.financial_api = FinancialDataAPI()
        self.sentiment_analyzer = SentimentAnalyzer(self.llm)
        self.context_manager = ContextManager(max_context, ollama_url, model)

    def research_stock(
        self,
        symbol: str,
        query: str,
        reference_date: datetime = None
    ) -> tuple:
        """
        Comprehensive stock research with date awareness

        Args:
            symbol: Stock symbol (e.g., 'UNH')
            query: Research question
            reference_date: "Today's" date for the analysis (default: actual today)

        Returns:
            Tuple of (report_text: str, structured_data: Dict)
        """

        if reference_date is None:
            reference_date = datetime.now()

        print(f"\n📅 Reference Date: {reference_date.strftime('%Y-%m-%d')}")
        print(f"📊 Researching: {symbol}")
        print(f"❓ Query: {query}\n")

        # Calculate date ranges
        six_months_ago = reference_date - timedelta(days=180)
        one_week_ago = reference_date - timedelta(days=7)

        # Step 1: Get current stock price
        print("💰 Step 1: Getting current stock price...")
        price_data = self.financial_api.get_stock_price(symbol)
        print(f"   Price: ${price_data.get('price', 'N/A')}")

        # Step 2: Search for 6-month news
        print(f"\n📰 Step 2: Searching news (past 6 months: {six_months_ago.strftime('%Y-%m-%d')} to {reference_date.strftime('%Y-%m-%d')})...")
        six_month_news = self.searcher.search_with_date_range(
            f"{symbol} stock news",
            start_date=six_months_ago,
            end_date=reference_date,
            num_results=15
        )
        print(f"   Found {len(six_month_news)} articles")

        # Step 3: Search for past week news
        print(f"\n📅 Step 3: Searching recent news (past week: {one_week_ago.strftime('%Y-%m-%d')} to {reference_date.strftime('%Y-%m-%d')})...")
        recent_news = self.searcher.search_with_date_range(
            f"{symbol} stock news",
            start_date=one_week_ago,
            end_date=reference_date,
            num_results=10
        )
        print(f"   Found {len(recent_news)} recent articles")

        # Step 4: Scrape and analyze sentiment
        print("\n🔍 Step 4: Scraping and analyzing sentiment...")
        analyzed_articles = []

        # Scrape recent news first (more important)
        for article in recent_news[:5]:
            print(f"   📰 {article['title'][:60]}...")
            scraped = self.scraper.scrape(article['link'])

            if scraped['success']:
                sentiment = self.sentiment_analyzer.analyze_article(
                    article['title'],
                    scraped['content']
                )

                analyzed_articles.append({
                    'title': article['title'],
                    'url': article['link'],
                    'date': article.get('date', 'recent'),
                    'content': scraped['content'][:3000],
                    'sentiment': sentiment['sentiment'],
                    'sentiment_score': sentiment['score'],
                    'sentiment_reasoning': sentiment['reasoning'],
                    'timeframe': 'recent'
                })

                print(f"      Sentiment: {sentiment['sentiment']} ({sentiment['score']:.2f})")

        # Then scrape older news for context
        for article in six_month_news[:10]:
            if article not in recent_news:  # Don't duplicate
                print(f"   📰 {article['title'][:60]}...")
                scraped = self.scraper.scrape(article['link'])

                if scraped['success']:
                    sentiment = self.sentiment_analyzer.analyze_article(
                        article['title'],
                        scraped['content']
                    )

                    analyzed_articles.append({
                        'title': article['title'],
                        'url': article['link'],
                        'date': article.get('date', ''),
                        'content': scraped['content'][:3000],
                        'sentiment': sentiment['sentiment'],
                        'sentiment_score': sentiment['score'],
                        'sentiment_reasoning': sentiment['reasoning'],
                        'timeframe': '6-month'
                    })

        # Step 5: Aggregate sentiment
        print("\n📊 Step 5: Aggregating sentiment analysis...")
        overall_sentiment = self.sentiment_analyzer.aggregate_sentiment(analyzed_articles)

        print(f"   Overall: {overall_sentiment['overall_sentiment']}")
        print(f"   Avg Score: {overall_sentiment['avg_score']:.2f}")
        print(f"   Trend: {overall_sentiment['trend']}")
        print(f"   Distribution: {overall_sentiment['distribution']}")

        # Step 6: Generate report
        print("\n📝 Step 6: Generating comprehensive report...")
        report, structured_data = self._generate_financial_report(
            symbol=symbol,
            query=query,
            reference_date=reference_date,
            price_data=price_data,
            articles=analyzed_articles,
            sentiment=overall_sentiment
        )

        return report, structured_data

    def _generate_financial_report(
        self,
        symbol: str,
        query: str,
        reference_date: datetime,
        price_data: Dict,
        articles: List[Dict],
        sentiment: Dict
    ) -> tuple:
        """
        Generate comprehensive financial report with structured data

        Returns:
            Tuple of (report_text: str, structured_data: Dict)
        """

        # Prepare context
        recent_articles = [a for a in articles if a.get('timeframe') == 'recent']
        older_articles = [a for a in articles if a.get('timeframe') == '6-month']

        articles_summary = "\n\n".join([
            f"**{a['title']}** ({a.get('date', 'recent')})\n"
            f"Sentiment: {a['sentiment']} (score: {a['sentiment_score']:.2f})\n"
            f"Reasoning: {a['sentiment_reasoning']}\n"
            f"Content excerpt: {a['content'][:500]}...\n"
            for a in articles[:15]
        ])

        # Handle missing price
        price = price_data.get('price')
        if price:
            price_str = f"${price:.2f}"
            price_note = ""
        else:
            price_str = "N/A (scraping failed)"
            price_note = "\n**NOTE**: Current price could not be retrieved via web scraping. For production use, consider Alpha Vantage, Polygon.io, or similar financial APIs."

        synthesis_prompt = f"""You are a financial analyst. Generate a comprehensive stock analysis report.

**REFERENCE DATE**: {reference_date.strftime('%B %d, %Y')}  (This is "today" for this analysis)

**Stock**: {symbol}
**Current Price**: {price_str}{price_note}
**Query**: {query}

**Sentiment Analysis**:
- Overall Sentiment: {sentiment['overall_sentiment']}
- Average Score: {sentiment['avg_score']:.2f} (range: -1.0 to +1.0)
- Trend: {sentiment['trend']}
- Distribution: {sentiment['distribution']}
- Total Articles Analyzed: {sentiment['total_articles']}

**Recent News (Past Week)**:
{len(recent_articles)} articles analyzed

**Historical Context (Past 6 Months)**:
{len(older_articles)} articles analyzed

**Article Details**:
{articles_summary}

**YOUR TASK**:
Generate a comprehensive financial report that includes:

1. **Executive Summary** (current situation as of {reference_date.strftime('%B %d, %Y')})

2. **Current Stock Price & Market Position**
   - Price: {price_str} as of {reference_date.strftime('%B %d, %Y')}
   {'- Note: Actual price not available, focus on sentiment and trends' if not price else '- Recent performance based on news sentiment'}

3. **Sentiment Analysis Over Past 6 Months**
   - Overall sentiment: {sentiment['overall_sentiment']}
   - Sentiment trend: {sentiment['trend']}
   - Key events driving sentiment

4. **News Volatility Analysis**
   - How news sentiment correlates with stock movement
   - Major events in past 6 months
   - Volatility drivers

5. **Recent Week Analysis** (Past 7 days from {reference_date.strftime('%B %d, %Y')})
   - Latest developments
   - Recent sentiment: {sentiment['overall_sentiment']}
   - Impact on stock based on news

6. **30-Day Outlook** (Next 30 days from {reference_date.strftime('%B %d, %Y')})
   - Sentiment-based prediction
   - Expected sentiment drivers
   {'- Expected price movement direction (without specific numbers if price N/A)' if not price else '- Estimated price range based on sentiment'}
   - Confidence level

7. **Risk Factors**

8. **Investment Recommendation**

**IMPORTANT**:
- Reference date is {reference_date.strftime('%B %d, %Y')} - treat this as "today"
- Base predictions on sentiment analysis and trends
- Be specific about timeframes
- Include confidence levels for predictions
{'- Since current price is unavailable, focus on directional predictions (up/down/stable) rather than specific price targets' if not price else ''}

Write a professional, detailed report (800-1200 words)."""

        report = self.llm.generate(synthesis_prompt, temperature=0.5)

        # Add metadata
        metadata = f"""
---

## Report Metadata

- **Report Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Reference Date**: {reference_date.strftime('%Y-%m-%d')}
- **Stock Symbol**: {symbol}
- **Company**: {price_data.get('company_name', symbol)}
- **Data Source**: {price_data.get('source', 'N/A')}

### Price Information
- **Current Price**: {price_str}
{f"- **Previous Close**: ${price_data.get('previous_close'):.2f}" if price_data.get('previous_close') else ""}
{f"- **Day Range**: ${price_data.get('day_low'):.2f} - ${price_data.get('day_high'):.2f}" if price_data.get('day_low') and price_data.get('day_high') else ""}
{f"- **52 Week Range**: ${price_data.get('52_week_low'):.2f} - ${price_data.get('52_week_high'):.2f}" if price_data.get('52_week_low') and price_data.get('52_week_high') else ""}
{f"- **50 Day Average**: ${price_data.get('avg_50_day'):.2f}" if price_data.get('avg_50_day') else ""}
{f"- **200 Day Average**: ${price_data.get('avg_200_day'):.2f}" if price_data.get('avg_200_day') else ""}
{f"- **Market Cap**: ${price_data.get('market_cap'):,.0f}" if price_data.get('market_cap') else ""}
{f"- **Price Fetch Status**: Failed - {price_data.get('error', 'Unknown error')}" if not price else ""}

### Sentiment Analysis
- **Articles Analyzed**: {len(articles)}
- **Overall Sentiment**: {sentiment['overall_sentiment']}
- **Sentiment Score**: {sentiment['avg_score']:.2f} (range: -1.0 to +1.0)
- **Sentiment Trend**: {sentiment['trend']}
- **Distribution**: {sentiment['distribution']['positive']} positive, {sentiment['distribution']['negative']} negative, {sentiment['distribution']['neutral']} neutral

## Sources Analyzed

"""

        for i, article in enumerate(articles[:20], 1):
            metadata += f"[{i}] {article['title']} - {article.get('date', 'recent')} (Sentiment: {article['sentiment']})\n    {article['url']}\n\n"

        # Prepare structured data for visualizations
        structured_data = {
            'symbol': symbol,
            'company_name': price_data.get('company_name', symbol),
            'reference_date': reference_date.isoformat(),

            # Price data
            'price_data': {
                'current_price': price_data.get('price'),
                'previous_close': price_data.get('previous_close'),
                'day_high': price_data.get('day_high'),
                'day_low': price_data.get('day_low'),
                '52_week_high': price_data.get('52_week_high'),
                '52_week_low': price_data.get('52_week_low'),
                'avg_50_day': price_data.get('avg_50_day'),
                'avg_200_day': price_data.get('avg_200_day'),
                'market_cap': price_data.get('market_cap'),
                'volume': price_data.get('volume')
            },

            # Sentiment timeseries
            'sentiment_timeseries': [
                {
                    'date': article.get('date', 'recent'),
                    'title': article['title'],
                    'sentiment': article['sentiment'],
                    'sentiment_score': article['sentiment_score'],
                    'url': article['url'],
                    'timeframe': article.get('timeframe', 'unknown')
                }
                for article in articles
            ],

            # Aggregated sentiment
            'sentiment_summary': {
                'overall_sentiment': sentiment['overall_sentiment'],
                'avg_score': sentiment['avg_score'],
                'trend': sentiment['trend'],
                'distribution': sentiment['distribution'],
                'total_articles': sentiment['total_articles']
            },

            # Article breakdown by timeframe
            'articles_by_timeframe': {
                'recent': len([a for a in articles if a.get('timeframe') == 'recent']),
                'six_month': len([a for a in articles if a.get('timeframe') == '6-month'])
            },

            # Sentiment by timeframe
            'sentiment_by_timeframe': {
                'recent': {
                    'count': len([a for a in articles if a.get('timeframe') == 'recent']),
                    'avg_score': sum(a['sentiment_score'] for a in articles if a.get('timeframe') == 'recent') / max(len([a for a in articles if a.get('timeframe') == 'recent']), 1)
                },
                'six_month': {
                    'count': len([a for a in articles if a.get('timeframe') == '6-month']),
                    'avg_score': sum(a['sentiment_score'] for a in articles if a.get('timeframe') == '6-month') / max(len([a for a in articles if a.get('timeframe') == '6-month']), 1)
                }
            }
        }

        return report + metadata, structured_data


def main():
    """CLI interface"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python financial_research_agent.py \"<your query>\"")
        print("\nExample:")
        print('  python financial_research_agent.py "Analyze UNH stock with sentiment over past 6 months"')
        return

    # Parse query
    query = " ".join(sys.argv[1:])

    # Extract stock symbol (simple pattern matching)
    # Look for common patterns like "UNH stock" or "ticker:UNH"
    symbol_match = re.search(r'\b([A-Z]{1,5})\s+stock\b', query, re.IGNORECASE)
    if not symbol_match:
        symbol_match = re.search(r'\bticker:([A-Z]{1,5})\b', query, re.IGNORECASE)

    if symbol_match:
        symbol = symbol_match.group(1).upper()
    else:
        # Default or ask
        print("Could not detect stock symbol. Please specify (e.g., UNH):")
        symbol = input().strip().upper()

    SERPAPI_KEY = "89edc0de285fab16d6170e2f5d508b91069acfec757f48061b70d7a2b5bda15e"

    agent = FinancialResearchAgent(
        serpapi_key=SERPAPI_KEY
    )

    # Use custom reference date if specified
    # For your example: September 29, 2025
    reference_date = datetime(2025, 9, 29)  # Change this as needed

    report, structured_data = agent.research_stock(symbol, query, reference_date)

    print("\n" + "="*80)
    print("📊 FINANCIAL RESEARCH REPORT")
    print("="*80)
    print(report)

    print("\n" + "="*80)
    print("📈 STRUCTURED DATA (for API/visualizations)")
    print("="*80)
    print(json.dumps(structured_data, indent=2, default=str))


if __name__ == "__main__":
    main()