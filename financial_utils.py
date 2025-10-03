#!/usr/bin/env python3
"""
Financial Utilities for Research Agents
Stock price fetching, sentiment analysis, and financial calculations
"""

import requests
from bs4 import BeautifulSoup
from typing import Dict, List, Optional
from datetime import datetime
import re
import json


class StockPriceUtils:
    """Utilities for fetching stock prices"""

    @staticmethod
    def fetch_yahoo_finance_price(symbol: str) -> Dict:
        """
        Fetch stock price from Yahoo Finance

        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'UNH')

        Returns:
            Dictionary with price data or error

        Note: Web scraping is fragile. For production, use:
            - Alpha Vantage API
            - Polygon.io API
            - Yahoo Finance API (yfinance library)
            - IEX Cloud API
        """
        try:
            url = f"https://finance.yahoo.com/quote/{symbol}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }

            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'lxml')

            price = None

            # Method 1: fin-streamer with data-field
            price_elem = soup.find('fin-streamer', {'data-symbol': symbol, 'data-field': 'regularMarketPrice'})
            if price_elem and price_elem.get('data-value'):
                price = float(price_elem.get('data-value'))

            # Method 2: fin-streamer with any data-value
            if not price:
                price_elem = soup.find('fin-streamer', {'data-symbol': symbol})
                if price_elem and price_elem.get('data-value'):
                    try:
                        price = float(price_elem.get('data-value'))
                    except:
                        pass

            # Method 3: Search page text for JSON
            if not price:
                text = soup.get_text()
                price_patterns = [
                    r'regularMarketPrice["\s:]+(\d+\.\d+)',
                    r'"price"["\s:]+(\d+\.\d+)',
                    r'currentPrice["\s:]+(\d+\.\d+)'
                ]
                for pattern in price_patterns:
                    match = re.search(pattern, text)
                    if match:
                        try:
                            price = float(match.group(1))
                            break
                        except:
                            pass

            if price:
                return {
                    'symbol': symbol,
                    'price': price,
                    'timestamp': datetime.now().isoformat(),
                    'source': 'Yahoo Finance (scraped)',
                    'success': True
                }
            else:
                return {
                    'symbol': symbol,
                    'price': None,
                    'error': 'Could not parse price from Yahoo Finance',
                    'timestamp': datetime.now().isoformat(),
                    'success': False,
                    'note': 'Consider using proper financial API for production'
                }

        except Exception as e:
            return {
                'symbol': symbol,
                'price': None,
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'success': False
            }

    @staticmethod
    def fetch_price_with_yfinance(symbol: str) -> Dict:
        """
        Fetch price using yfinance library (recommended)

        Requires: pip install yfinance

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with price data
        """
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            info = ticker.info

            return {
                'symbol': symbol,
                'price': info.get('currentPrice') or info.get('regularMarketPrice'),
                'previous_close': info.get('previousClose'),
                'open': info.get('open'),
                'day_high': info.get('dayHigh'),
                'day_low': info.get('dayLow'),
                'volume': info.get('volume'),
                'market_cap': info.get('marketCap'),
                'timestamp': datetime.now().isoformat(),
                'source': 'Yahoo Finance API (yfinance)',
                'success': True
            }

        except ImportError:
            return {
                'symbol': symbol,
                'error': 'yfinance library not installed. Run: pip install yfinance',
                'success': False
            }
        except Exception as e:
            return {
                'symbol': symbol,
                'error': str(e),
                'success': False
            }


class SentimentUtils:
    """Utilities for sentiment analysis"""

    @staticmethod
    def calculate_sentiment_score(sentiment: str) -> float:
        """
        Convert sentiment label to numeric score

        Args:
            sentiment: "positive", "negative", or "neutral"

        Returns:
            Score: -1.0 (very negative) to +1.0 (very positive)
        """
        mapping = {
            'very positive': 0.9,
            'positive': 0.6,
            'slightly positive': 0.3,
            'neutral': 0.0,
            'slightly negative': -0.3,
            'negative': -0.6,
            'very negative': -0.9
        }

        return mapping.get(sentiment.lower(), 0.0)

    @staticmethod
    def aggregate_sentiment(articles: List[Dict]) -> Dict:
        """
        Aggregate sentiment across multiple articles

        Args:
            articles: List of articles with 'sentiment' and 'sentiment_score' keys

        Returns:
            Aggregated sentiment data
        """
        if not articles:
            return {
                'overall_sentiment': 'neutral',
                'avg_score': 0.0,
                'distribution': {'positive': 0, 'negative': 0, 'neutral': 0},
                'trend': 'stable',
                'total_articles': 0
            }

        scores = [a['sentiment_score'] for a in articles if 'sentiment_score' in a]
        sentiments = [a['sentiment'] for a in articles if 'sentiment' in a]

        avg_score = sum(scores) / len(scores) if scores else 0.0

        distribution = {
            'positive': sentiments.count('positive'),
            'negative': sentiments.count('negative'),
            'neutral': sentiments.count('neutral')
        }

        # Determine overall sentiment
        if avg_score > 0.3:
            overall = 'positive'
        elif avg_score < -0.3:
            overall = 'negative'
        else:
            overall = 'neutral'

        # Determine trend (compare first half vs second half chronologically)
        trend = 'stable'
        if len(scores) >= 4:
            mid = len(scores) // 2
            first_half_avg = sum(scores[:mid]) / len(scores[:mid])
            second_half_avg = sum(scores[mid:]) / len(scores[mid:])

            if second_half_avg > first_half_avg + 0.2:
                trend = 'improving'
            elif second_half_avg < first_half_avg - 0.2:
                trend = 'declining'

        return {
            'overall_sentiment': overall,
            'avg_score': avg_score,
            'distribution': distribution,
            'trend': trend,
            'total_articles': len(articles),
            'confidence': 'high' if len(articles) >= 10 else 'moderate' if len(articles) >= 5 else 'low'
        }

    @staticmethod
    def calculate_volatility_from_sentiment(articles: List[Dict]) -> Dict:
        """
        Calculate sentiment volatility (how much sentiment fluctuates)

        Args:
            articles: List of articles with sentiment scores

        Returns:
            Volatility metrics
        """
        scores = [a['sentiment_score'] for a in articles if 'sentiment_score' in a]

        if len(scores) < 2:
            return {
                'volatility': 0.0,
                'volatility_level': 'low',
                'score_range': 0.0
            }

        # Calculate standard deviation
        mean = sum(scores) / len(scores)
        variance = sum((x - mean) ** 2 for x in scores) / len(scores)
        std_dev = variance ** 0.5

        # Calculate range
        score_range = max(scores) - min(scores)

        # Classify volatility
        if std_dev > 0.5:
            level = 'high'
        elif std_dev > 0.3:
            level = 'moderate'
        else:
            level = 'low'

        return {
            'volatility': std_dev,
            'volatility_level': level,
            'score_range': score_range,
            'min_score': min(scores),
            'max_score': max(scores),
            'mean_score': mean
        }


class FinancialAnalysisUtils:
    """Utilities for financial analysis"""

    @staticmethod
    def calculate_price_target(
        current_price: float,
        sentiment_score: float,
        days_ahead: int = 30
    ) -> Dict:
        """
        Simple price target calculation based on sentiment

        Args:
            current_price: Current stock price
            sentiment_score: Sentiment score (-1.0 to +1.0)
            days_ahead: Number of days to predict

        Returns:
            Price target estimates

        Note: This is a VERY simple heuristic. Real price prediction requires:
            - Historical price data
            - Technical indicators
            - Fundamental analysis
            - Machine learning models
        """
        # Simple heuristic: sentiment correlates with price movement
        # Positive sentiment → upward movement
        # Negative sentiment → downward movement

        # Assume sentiment can drive up to 10% change over 30 days
        max_change_pct = 0.10
        sentiment_factor = sentiment_score  # -1.0 to +1.0

        # Calculate expected change
        expected_change_pct = sentiment_factor * max_change_pct * (days_ahead / 30)

        # Calculate price targets
        expected_price = current_price * (1 + expected_change_pct)
        optimistic_price = current_price * (1 + expected_change_pct * 1.5)
        pessimistic_price = current_price * (1 + expected_change_pct * 0.5)

        return {
            'current_price': current_price,
            'days_ahead': days_ahead,
            'expected_price': round(expected_price, 2),
            'optimistic_price': round(optimistic_price, 2),
            'pessimistic_price': round(pessimistic_price, 2),
            'expected_change_pct': round(expected_change_pct * 100, 2),
            'price_range': f"${pessimistic_price:.2f} - ${optimistic_price:.2f}",
            'sentiment_score': sentiment_score,
            'note': 'Simple heuristic - not financial advice'
        }

    @staticmethod
    def interpret_sentiment_trend(trend: str, current_sentiment: str) -> str:
        """
        Interpret sentiment trend for investment decisions

        Args:
            trend: "improving", "declining", or "stable"
            current_sentiment: "positive", "negative", or "neutral"

        Returns:
            Investment interpretation
        """
        interpretations = {
            ('improving', 'positive'): 'Strong bullish signal - momentum building',
            ('improving', 'neutral'): 'Moderately bullish - sentiment recovering',
            ('improving', 'negative'): 'Cautiously optimistic - sentiment turning',
            ('stable', 'positive'): 'Stable bullish - sustained optimism',
            ('stable', 'neutral'): 'Neutral - wait for clearer signals',
            ('stable', 'negative'): 'Stable bearish - sustained pessimism',
            ('declining', 'positive'): 'Weakening bullish - momentum fading',
            ('declining', 'neutral'): 'Moderately bearish - sentiment deteriorating',
            ('declining', 'negative'): 'Strong bearish signal - momentum accelerating downward'
        }

        return interpretations.get((trend, current_sentiment), 'Unclear trend')


class FinancialCalculations:
    """Advanced financial calculations"""

    @staticmethod
    def calculate_returns(prices: List[float]) -> List[float]:
        """
        Calculate period-over-period returns from price data

        Args:
            prices: List of prices in chronological order

        Returns:
            List of returns (percentage change between periods)

        Example:
            prices = [100, 105, 103]
            returns = [0.05, -0.019] (5% gain, then 1.9% loss)
        """
        if len(prices) < 2:
            return []

        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] != 0:
                ret = (prices[i] - prices[i-1]) / prices[i-1]
                returns.append(ret)

        return returns

    @staticmethod
    def calculate_volatility(prices: List[float], periods: int = 30) -> float:
        """
        Calculate rolling volatility (annualized)

        Args:
            prices: List of prices
            periods: Number of periods to use for calculation (default 30)

        Returns:
            Annualized volatility (standard deviation of returns)

        Note:
            - Uses last N periods if more data is available
            - Returns 0 if insufficient data
            - Annualization factor: sqrt(252) for daily prices
        """
        if len(prices) < 2:
            return 0.0

        returns = FinancialCalculations.calculate_returns(prices)

        if len(returns) < periods:
            # Use all available data if less than requested periods
            relevant_returns = returns
        else:
            # Use last N periods
            relevant_returns = returns[-periods:]

        if not relevant_returns:
            return 0.0

        # Calculate standard deviation
        mean = sum(relevant_returns) / len(relevant_returns)
        variance = sum((r - mean) ** 2 for r in relevant_returns) / len(relevant_returns)
        std_dev = variance ** 0.5

        # Annualize (assuming daily prices, 252 trading days/year)
        annualized_vol = std_dev * (252 ** 0.5)

        return annualized_vol

    @staticmethod
    def calculate_sharpe_ratio(prices: List[float], risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio (risk-adjusted return)

        Args:
            prices: List of prices
            risk_free_rate: Annual risk-free rate (default 2%)

        Returns:
            Sharpe ratio (higher is better)

        Formula:
            Sharpe = (Return - Risk_Free_Rate) / Volatility

        Note:
            - Returns 0 if insufficient data or zero volatility
            - Assumes daily prices for annualization
        """
        if len(prices) < 2:
            return 0.0

        returns = FinancialCalculations.calculate_returns(prices)

        if not returns:
            return 0.0

        # Calculate annualized return
        total_return = (prices[-1] - prices[0]) / prices[0]
        num_periods = len(prices) - 1
        annualized_return = ((1 + total_return) ** (252 / num_periods)) - 1

        # Calculate annualized volatility
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        std_dev = variance ** 0.5
        annualized_vol = std_dev * (252 ** 0.5)

        if annualized_vol == 0:
            return 0.0

        # Sharpe ratio
        sharpe = (annualized_return - risk_free_rate) / annualized_vol

        return sharpe

    @staticmethod
    def calculate_moving_average(values: List[float], window: int) -> List[float]:
        """
        Calculate simple moving average (SMA)

        Args:
            values: List of values (prices, scores, etc.)
            window: Window size for moving average

        Returns:
            List of moving averages (same length as input)

        Example:
            values = [1, 2, 3, 4, 5], window = 3
            sma = [1.0, 1.5, 2.0, 3.0, 4.0]
        """
        if not values or window < 1:
            return []

        sma = []
        for i in range(len(values)):
            start_idx = max(0, i - window + 1)
            window_values = values[start_idx:i+1]
            avg = sum(window_values) / len(window_values)
            sma.append(avg)

        return sma

    @staticmethod
    def calculate_exponential_moving_average(values: List[float], alpha: float = 0.3) -> List[float]:
        """
        Calculate exponential moving average (EMA)

        Args:
            values: List of values
            alpha: Smoothing factor (0 < alpha <= 1)
                  Higher alpha = more weight on recent values

        Returns:
            List of exponential moving averages

        Formula:
            EMA[0] = values[0]
            EMA[i] = alpha * values[i] + (1 - alpha) * EMA[i-1]

        Example:
            values = [1, 2, 3, 4, 5], alpha = 0.3
            ema = [1.0, 1.3, 1.81, 2.467, 3.227]
        """
        if not values or alpha <= 0 or alpha > 1:
            return []

        ema = [values[0]]

        for val in values[1:]:
            ema_value = alpha * val + (1 - alpha) * ema[-1]
            ema.append(ema_value)

        return ema

    @staticmethod
    def calculate_max_drawdown(prices: List[float]) -> Dict:
        """
        Calculate maximum drawdown (largest peak-to-trough decline)

        Args:
            prices: List of prices in chronological order

        Returns:
            Dictionary with max drawdown metrics

        Example:
            prices = [100, 110, 90, 95]
            max_drawdown = -0.1818 (18.18% decline from peak of 110 to trough of 90)
        """
        if len(prices) < 2:
            return {
                'max_drawdown': 0.0,
                'max_drawdown_pct': 0.0,
                'peak_price': prices[0] if prices else 0,
                'trough_price': prices[0] if prices else 0
            }

        peak = prices[0]
        max_dd = 0.0
        peak_price = prices[0]
        trough_price = prices[0]

        for price in prices:
            if price > peak:
                peak = price

            drawdown = (price - peak) / peak

            if drawdown < max_dd:
                max_dd = drawdown
                peak_price = peak
                trough_price = price

        return {
            'max_drawdown': max_dd,
            'max_drawdown_pct': max_dd * 100,
            'peak_price': peak_price,
            'trough_price': trough_price
        }

    @staticmethod
    def calculate_correlation(series1: List[float], series2: List[float]) -> Dict:
        """
        Calculate correlation between two series

        Args:
            series1: First series of values
            series2: Second series of values (must be same length)

        Returns:
            Dictionary with correlation metrics

        Note:
            Correlation ranges from -1 (perfect negative) to +1 (perfect positive)
        """
        if len(series1) != len(series2) or len(series1) < 2:
            return {
                'correlation': 0.0,
                'interpretation': 'insufficient_data',
                'n': len(series1)
            }

        # Calculate means
        mean1 = sum(series1) / len(series1)
        mean2 = sum(series2) / len(series2)

        # Calculate correlation coefficient
        numerator = sum((series1[i] - mean1) * (series2[i] - mean2) for i in range(len(series1)))
        denom1 = sum((x - mean1) ** 2 for x in series1) ** 0.5
        denom2 = sum((y - mean2) ** 2 for y in series2) ** 0.5

        if denom1 == 0 or denom2 == 0:
            correlation = 0.0
        else:
            correlation = numerator / (denom1 * denom2)

        # Interpret correlation
        abs_corr = abs(correlation)
        if abs_corr > 0.7:
            strength = 'strong'
        elif abs_corr > 0.3:
            strength = 'moderate'
        else:
            strength = 'weak'

        direction = 'positive' if correlation > 0 else 'negative'
        interpretation = f'{strength} {direction}' if abs_corr > 0.1 else 'negligible'

        return {
            'correlation': correlation,
            'interpretation': interpretation,
            'n': len(series1)
        }


class FinancialReportUtils:
    """Utilities for generating financial reports"""

    @staticmethod
    def format_price_change(current: float, previous: float) -> str:
        """
        Format price change with percentage

        Args:
            current: Current price
            previous: Previous price

        Returns:
            Formatted string like "+5.2 (+2.5%)" or "-3.1 (-1.8%)"
        """
        if previous == 0:
            return "N/A"

        change = current - previous
        pct_change = (change / previous) * 100

        sign = "+" if change >= 0 else ""
        return f"{sign}{change:.2f} ({sign}{pct_change:.2f}%)"

    @staticmethod
    def format_market_cap(market_cap: int) -> str:
        """
        Format market cap in B (billions) or M (millions)

        Args:
            market_cap: Market cap value

        Returns:
            Formatted string like "$500.2B" or "$45.3M"
        """
        if market_cap >= 1e9:
            return f"${market_cap / 1e9:.2f}B"
        elif market_cap >= 1e6:
            return f"${market_cap / 1e6:.2f}M"
        else:
            return f"${market_cap:,.0f}"

    @staticmethod
    def generate_risk_assessment(sentiment_data: Dict, volatility_data: Dict) -> List[str]:
        """
        Generate risk factors based on sentiment and volatility

        Args:
            sentiment_data: Aggregated sentiment
            volatility_data: Volatility metrics

        Returns:
            List of risk factors
        """
        risks = []

        # Sentiment-based risks
        if sentiment_data['overall_sentiment'] == 'negative':
            risks.append("Negative market sentiment may pressure stock price")

        if sentiment_data['trend'] == 'declining':
            risks.append("Deteriorating sentiment trend indicates potential downside")

        # Volatility-based risks
        if volatility_data['volatility_level'] == 'high':
            risks.append("High sentiment volatility suggests uncertain market conditions")

        if abs(sentiment_data['avg_score']) < 0.2:
            risks.append("Neutral sentiment provides little directional guidance")

        # Data quality risks
        if sentiment_data.get('confidence') == 'low':
            risks.append("Limited article data - analysis may not be comprehensive")

        return risks if risks else ["No major sentiment-based risks identified"]


# Tool-friendly functions

def get_stock_price(symbol: str, method: str = "scrape") -> Dict:
    """
    Tool-friendly stock price fetcher

    Args:
        symbol: Stock symbol (e.g., 'AAPL')
        method: "scrape" (Yahoo Finance scraping) or "api" (yfinance)

    Returns:
        Price data dictionary
    """
    if method == "api":
        return StockPriceUtils.fetch_price_with_yfinance(symbol)
    else:
        return StockPriceUtils.fetch_yahoo_finance_price(symbol)


def analyze_sentiment_trend(articles: List[Dict]) -> str:
    """
    Tool-friendly sentiment analysis

    Args:
        articles: List of articles with sentiment data

    Returns:
        Formatted sentiment analysis
    """
    agg = SentimentUtils.aggregate_sentiment(articles)
    vol = SentimentUtils.calculate_volatility_from_sentiment(articles)

    return f"""Sentiment Analysis:
- Overall: {agg['overall_sentiment']} (score: {agg['avg_score']:.2f})
- Trend: {agg['trend']}
- Distribution: {agg['distribution']['positive']} positive, {agg['distribution']['negative']} negative, {agg['distribution']['neutral']} neutral
- Volatility: {vol['volatility_level']} (σ={vol['volatility']:.2f})
- Confidence: {agg['confidence']} ({agg['total_articles']} articles)"""


if __name__ == "__main__":
    # Test the utilities
    print("Financial Utilities Test")
    print("=" * 60)

    # Test 1: Sentiment aggregation
    print("\n1. Sentiment Aggregation Test:")
    test_articles = [
        {'sentiment': 'positive', 'sentiment_score': 0.7},
        {'sentiment': 'positive', 'sentiment_score': 0.6},
        {'sentiment': 'neutral', 'sentiment_score': 0.1},
        {'sentiment': 'negative', 'sentiment_score': -0.4},
        {'sentiment': 'positive', 'sentiment_score': 0.8}
    ]
    agg = SentimentUtils.aggregate_sentiment(test_articles)
    print(f"   Overall: {agg['overall_sentiment']}")
    print(f"   Avg Score: {agg['avg_score']:.2f}")
    print(f"   Trend: {agg['trend']}")
    print(f"   Distribution: {agg['distribution']}")

    # Test 2: Volatility calculation
    print("\n2. Volatility Calculation:")
    vol = SentimentUtils.calculate_volatility_from_sentiment(test_articles)
    print(f"   Level: {vol['volatility_level']}")
    print(f"   Std Dev: {vol['volatility']:.3f}")
    print(f"   Range: {vol['score_range']:.2f}")

    # Test 3: Price target
    print("\n3. Price Target Calculation:")
    current_price = 350.00
    sentiment_score = 0.5  # Positive
    target = FinancialAnalysisUtils.calculate_price_target(current_price, sentiment_score, days_ahead=30)
    print(f"   Current: ${target['current_price']}")
    print(f"   Expected (30d): ${target['expected_price']}")
    print(f"   Range: {target['price_range']}")
    print(f"   Change: {target['expected_change_pct']}%")

    # Test 4: Formatting
    print("\n4. Formatting Tests:")
    print(f"   Price change: {FinancialReportUtils.format_price_change(355, 350)}")
    print(f"   Market cap: {FinancialReportUtils.format_market_cap(500_000_000_000)}")

    # Test 5: Risk assessment
    print("\n5. Risk Assessment:")
    risks = FinancialReportUtils.generate_risk_assessment(agg, vol)
    for risk in risks:
        print(f"   - {risk}")

    # Test 6: Stock price (optional - requires internet)
    print("\n6. Stock Price Fetch (optional):")
    print("   Attempting to fetch AAPL price from Yahoo Finance...")
    price_data = StockPriceUtils.fetch_yahoo_finance_price("AAPL")
    if price_data['success']:
        print(f"   ✓ Success! Price: ${price_data['price']}")
    else:
        print(f"   ✗ Failed: {price_data.get('error', 'Unknown error')}")