#!/usr/bin/env python3
"""
Web Utilities for Research Agents
Common web scraping and searching functions
"""

import requests
from bs4 import BeautifulSoup
from serpapi import GoogleSearch
from typing import List, Dict, Optional
from datetime import datetime
import re


class WebSearchUtils:
    """Utilities for web searching"""

    @staticmethod
    def search_google(
        query: str,
        api_key: str,
        num_results: int = 10,
        search_type: str = "organic"
    ) -> List[Dict]:
        """
        Search Google using SerpAPI

        Args:
            query: Search query
            api_key: SerpAPI key
            num_results: Number of results to return
            search_type: "organic" or "news"

        Returns:
            List of search results with title, link, snippet
        """
        params = {
            "q": query,
            "api_key": api_key,
            "num": num_results
        }

        if search_type == "news":
            params["tbm"] = "nws"

        try:
            search = GoogleSearch(params)
            results = search.get_dict()

            articles = []

            # Try news results first
            if "news_results" in results:
                for result in results["news_results"][:num_results]:
                    articles.append({
                        "title": result.get("title", ""),
                        "link": result.get("link", ""),
                        "snippet": result.get("snippet", ""),
                        "date": result.get("date", ""),
                        "source": result.get("source", "")
                    })

            # Fall back to organic results
            elif "organic_results" in results:
                for result in results["organic_results"][:num_results]:
                    articles.append({
                        "title": result.get("title", ""),
                        "link": result.get("link", ""),
                        "snippet": result.get("snippet", ""),
                        "date": "",
                        "source": ""
                    })

            return articles

        except Exception as e:
            print(f"Search error: {e}")
            return []

    @staticmethod
    def search_with_date_filter(
        query: str,
        api_key: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        num_results: int = 10
    ) -> List[Dict]:
        """
        Search Google with date filtering

        Args:
            query: Search query
            api_key: SerpAPI key
            start_date: Filter articles after this date
            end_date: Filter articles before this date
            num_results: Number of results

        Returns:
            List of search results
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
            "api_key": api_key,
            "num": num_results * 2,  # Request more to filter
            "tbm": "nws"  # News search
        }

        # Add Google date range parameter
        if start_date and end_date:
            params["tbs"] = f"cdr:1,cd_min:{start_date.strftime('%m/%d/%Y')},cd_max:{end_date.strftime('%m/%d/%Y')}"

        try:
            search = GoogleSearch(params)
            results = search.get_dict()

            articles = []
            if "news_results" in results:
                for result in results["news_results"]:
                    date_str = result.get("date", "")

                    # Filter out very old articles
                    if start_date and date_str:
                        if any(old in date_str.lower() for old in ['year ago', 'years ago']):
                            continue

                    articles.append({
                        "title": result.get("title", ""),
                        "link": result.get("link", ""),
                        "snippet": result.get("snippet", ""),
                        "date": date_str,
                        "source": result.get("source", "")
                    })

                    if len(articles) >= num_results:
                        break

            elif "organic_results" in results:
                for result in results["organic_results"][:num_results]:
                    articles.append({
                        "title": result.get("title", ""),
                        "link": result.get("link", ""),
                        "snippet": result.get("snippet", ""),
                        "date": "",
                        "source": ""
                    })

            return articles[:num_results]

        except Exception as e:
            print(f"Search error: {e}")
            return []


class WebScrapingUtils:
    """Utilities for web scraping"""

    DEFAULT_HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    @staticmethod
    def scrape_url(
        url: str,
        max_content_chars: int = 20000,
        timeout: int = 10,
        headers: Optional[Dict] = None
    ) -> Dict:
        """
        Scrape content from URL

        Args:
            url: URL to scrape
            max_content_chars: Maximum characters to return
            timeout: Request timeout in seconds
            headers: Custom headers (uses defaults if None)

        Returns:
            Dictionary with url, title, content, success status
        """
        if headers is None:
            headers = WebScrapingUtils.DEFAULT_HEADERS

        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'lxml')

            # Remove unwanted elements
            for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
                element.decompose()

            # Get text
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)

            return {
                "url": url,
                "title": soup.title.string if soup.title else "",
                "content": text[:max_content_chars],
                "content_length": len(text),
                "truncated": len(text) > max_content_chars,
                "success": True
            }

        except Exception as e:
            return {
                "url": url,
                "title": "",
                "content": "",
                "success": False,
                "error": str(e)
            }

    @staticmethod
    def scrape_multiple(
        urls: List[str],
        max_content_chars: int = 20000,
        max_urls: Optional[int] = None
    ) -> List[Dict]:
        """
        Scrape multiple URLs

        Args:
            urls: List of URLs
            max_content_chars: Max chars per URL
            max_urls: Maximum number of URLs to scrape (None = all)

        Returns:
            List of scrape results
        """
        if max_urls:
            urls = urls[:max_urls]

        results = []
        for url in urls:
            print(f"  🌐 Scraping: {url}")
            scraped = WebScrapingUtils.scrape_url(url, max_content_chars)
            if scraped['success']:
                results.append(scraped)

        return results

    @staticmethod
    def extract_links(html_content: str, base_url: str = "") -> List[str]:
        """
        Extract all links from HTML content

        Args:
            html_content: HTML string
            base_url: Base URL for relative links

        Returns:
            List of URLs
        """
        soup = BeautifulSoup(html_content, 'lxml')
        links = []

        for link in soup.find_all('a', href=True):
            href = link['href']

            # Convert relative to absolute
            if href.startswith('/') and base_url:
                href = base_url.rstrip('/') + href
            elif not href.startswith('http'):
                continue

            links.append(href)

        return list(set(links))  # Deduplicate

    @staticmethod
    def clean_text(text: str, max_length: Optional[int] = None) -> str:
        """
        Clean scraped text

        Args:
            text: Raw text
            max_length: Optional max length

        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\"\']', '', text)

        # Trim
        text = text.strip()

        if max_length:
            text = text[:max_length]

        return text


class URLUtils:
    """Utilities for URL handling"""

    @staticmethod
    def is_valid_url(url: str) -> bool:
        """Check if URL is valid"""
        pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain
            r'localhost|'  # localhost
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # IP
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        return pattern.match(url) is not None

    @staticmethod
    def deduplicate_urls(urls: List[str]) -> List[str]:
        """
        Deduplicate URLs, keeping order

        Args:
            urls: List of URLs

        Returns:
            Deduplicated list
        """
        seen = set()
        result = []
        for url in urls:
            if url not in seen:
                seen.add(url)
                result.append(url)
        return result

    @staticmethod
    def filter_urls(urls: List[str], exclude_domains: Optional[List[str]] = None) -> List[str]:
        """
        Filter URLs by domain

        Args:
            urls: List of URLs
            exclude_domains: Domains to exclude (e.g., ['facebook.com', 'twitter.com'])

        Returns:
            Filtered list
        """
        if not exclude_domains:
            return urls

        filtered = []
        for url in urls:
            if not any(domain in url.lower() for domain in exclude_domains):
                filtered.append(url)

        return filtered


# Tool-friendly functions

def search_web(query: str, api_key: str, num_results: int = 5, search_type: str = "organic") -> str:
    """
    Tool-friendly web search function

    Args:
        query: Search query
        api_key: SerpAPI key
        num_results: Number of results
        search_type: "organic" or "news"

    Returns:
        Formatted search results as string

    Example:
        >>> results = search_web("Python programming", api_key, 5)
        [1] Python.org
        URL: https://python.org
        Snippet: Python is a programming language...

        [2] ...
    """
    results = WebSearchUtils.search_google(query, api_key, num_results, search_type)

    if not results:
        return "No results found"

    formatted = []
    for i, result in enumerate(results, 1):
        formatted.append(
            f"[{i}] {result['title']}\n"
            f"URL: {result['link']}\n"
            f"Snippet: {result['snippet']}\n"
        )

    return "\n".join(formatted)


def scrape_url_simple(url: str, max_chars: int = 20000) -> str:
    """
    Tool-friendly scraping function

    Args:
        url: URL to scrape
        max_chars: Maximum characters

    Returns:
        Formatted scrape result as string
    """
    result = WebScrapingUtils.scrape_url(url, max_chars)

    if not result['success']:
        return f"Error scraping {url}: {result.get('error', 'Unknown error')}"

    return f"Title: {result['title']}\n\nContent:\n{result['content']}"


if __name__ == "__main__":
    # Test the utilities
    print("Web Utilities Test")
    print("=" * 60)

    # Test URL validation
    print("\n1. URL Validation:")
    test_urls = [
        "https://google.com",
        "http://example.com/path",
        "not-a-url",
        "ftp://invalid.com"
    ]
    for url in test_urls:
        valid = URLUtils.is_valid_url(url)
        print(f"   {url}: {'✓' if valid else '✗'}")

    # Test URL deduplication
    print("\n2. URL Deduplication:")
    urls = ["https://a.com", "https://b.com", "https://a.com", "https://c.com"]
    deduped = URLUtils.deduplicate_urls(urls)
    print(f"   Original: {len(urls)} URLs")
    print(f"   After dedup: {len(deduped)} URLs")

    # Test scraping (with a simple example)
    print("\n3. Web Scraping Test:")
    print("   Scraping example.com...")
    result = WebScrapingUtils.scrape_url("https://example.com", max_content_chars=500)
    if result['success']:
        print(f"   ✓ Success! Title: {result['title']}")
        print(f"   Content length: {result['content_length']} chars")
        print(f"   Truncated: {result['truncated']}")
    else:
        print(f"   ✗ Failed: {result['error']}")

    print("\n4. Text Cleaning Test:")
    dirty_text = "Hello    world!!!   \n\n  This   is    messy    text.  "
    clean = WebScrapingUtils.clean_text(dirty_text)
    print(f"   Before: '{dirty_text}'")
    print(f"   After: '{clean}'")