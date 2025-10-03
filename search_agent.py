#!/usr/bin/env python3
"""
Web Search Agent using Ollama and SerpAPI
Searches the web, scrapes content, and summarizes using local LLM
"""

import requests
import json
from bs4 import BeautifulSoup
from serpapi import GoogleSearch
from typing import List, Dict, Optional
import sys


class OllamaClient:
    """Client for interacting with Ollama REST API"""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "qwen3:32b"):
        self.base_url = base_url
        self.model = model

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text using Ollama API"""
        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }

        if system_prompt:
            payload["system"] = system_prompt

        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            print(f"Error calling Ollama API: {e}")
            return ""

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Chat with Ollama using conversation format"""
        url = f"{self.base_url}/api/chat"

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False
        }

        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()["message"]["content"]
        except Exception as e:
            print(f"Error calling Ollama chat API: {e}")
            return ""


class WebSearcher:
    """Web search using SerpAPI"""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def search(self, query: str, num_results: int = 5) -> List[Dict]:
        """Search web using SerpAPI"""
        params = {
            "q": query,
            "api_key": self.api_key,
            "num": num_results
        }

        try:
            search = GoogleSearch(params)
            results = search.get_dict()

            # Extract organic results
            organic_results = []
            if "organic_results" in results:
                for result in results["organic_results"][:num_results]:
                    organic_results.append({
                        "title": result.get("title", ""),
                        "link": result.get("link", ""),
                        "snippet": result.get("snippet", "")
                    })

            return organic_results
        except Exception as e:
            print(f"Error searching with SerpAPI: {e}")
            return []


class WebScraper:
    """Web scraper for extracting content from URLs"""

    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def scrape(self, url: str) -> Dict[str, str]:
        """Scrape content from a URL"""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'lxml')

            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()

            # Get text
            text = soup.get_text()

            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)

            # Extract links
            links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                if href.startswith('http'):
                    links.append(href)

            return {
                "url": url,
                "title": soup.title.string if soup.title else "",
                "content": text[:10000],  # Limit to first 10k chars
                "links": links[:20]  # Limit to 20 links
            }
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return {
                "url": url,
                "title": "",
                "content": "",
                "links": []
            }

    def scrape_multiple(self, urls: List[str], max_urls: int = 3) -> List[Dict]:
        """Scrape multiple URLs"""
        results = []
        for url in urls[:max_urls]:
            print(f"Scraping: {url}")
            result = self.scrape(url)
            if result["content"]:
                results.append(result)
        return results


class SearchAgent:
    """Main agent that orchestrates search, scraping, and summarization"""

    def __init__(self, serpapi_key: str, ollama_url: str = "http://localhost:11434", model: str = "qwen3:32b"):
        self.llm = OllamaClient(ollama_url, model)
        self.searcher = WebSearcher(serpapi_key)
        self.scraper = WebScraper()

    def process_query(self, query: str, deep_scrape: bool = False) -> str:
        """Process a user query: search, scrape, and summarize"""

        print(f"\n🔍 Searching for: {query}\n")

        # Step 1: Search the web
        search_results = self.searcher.search(query)

        if not search_results:
            return "No search results found."

        print(f"Found {len(search_results)} results:\n")
        for i, result in enumerate(search_results, 1):
            print(f"{i}. {result['title']}")
            print(f"   {result['link']}")
            print(f"   {result['snippet']}\n")

        # Step 2: Check if snippets contain enough information
        snippets_text = "\n\n".join([
            f"Title: {r['title']}\nURL: {r['link']}\nSnippet: {r['snippet']}"
            for r in search_results
        ])

        # Ask LLM if we need to scrape for more details
        if not deep_scrape:
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based on search results."
                },
                {
                    "role": "user",
                    "content": f"User query: {query}\n\nSearch results:\n{snippets_text}\n\nPlease provide a comprehensive answer to the user's query based on these search results. If the snippets don't contain enough information, say 'NEED_MORE_INFO'."
                }
            ]

            print("📝 Generating summary from snippets...\n")
            initial_answer = self.llm.chat(messages)

            if "NEED_MORE_INFO" not in initial_answer:
                return initial_answer

            print("🌐 Snippets insufficient, scraping full pages...\n")

        # Step 3: Scrape full content from top URLs
        urls_to_scrape = [r['link'] for r in search_results[:3]]
        scraped_data = self.scraper.scrape_multiple(urls_to_scrape)

        if not scraped_data:
            return "Could not scrape any content from the search results."

        # Step 4: Optionally scrape sub-links
        if deep_scrape:
            print("\n🔗 Scraping sub-links...\n")
            sub_links = []
            for data in scraped_data:
                sub_links.extend(data['links'][:3])  # Take first 3 links from each page

            if sub_links:
                additional_data = self.scraper.scrape_multiple(sub_links[:5], max_urls=5)
                scraped_data.extend(additional_data)

        # Step 5: Summarize all content
        full_content = "\n\n---\n\n".join([
            f"Source: {d['url']}\nTitle: {d['title']}\n\nContent:\n{d['content']}"
            for d in scraped_data
        ])

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that synthesizes information from multiple sources to answer questions comprehensively and accurately."
            },
            {
                "role": "user",
                "content": f"User query: {query}\n\nInformation gathered from web sources:\n\n{full_content}\n\nPlease provide a comprehensive, well-structured answer to the user's query based on this information. Include relevant details and cite sources where appropriate."
            }
        ]

        print("🤖 Generating comprehensive summary...\n")
        final_answer = self.llm.chat(messages)

        return final_answer


def main():
    """Main entry point"""

    SERPAPI_KEY = "89edc0de285fab16d6170e2f5d508b91069acfec757f48061b70d7a2b5bda15e"
    OLLAMA_URL = "http://localhost:11434"
    MODEL = "qwen3:32b"

    agent = SearchAgent(SERPAPI_KEY, OLLAMA_URL, MODEL)

    if len(sys.argv) > 1:
        # Query from command line
        query = " ".join(sys.argv[1:])
        deep_scrape = "--deep" in sys.argv
        result = agent.process_query(query, deep_scrape)
        print("\n" + "="*80)
        print("ANSWER:")
        print("="*80 + "\n")
        print(result)
    else:
        # Interactive mode
        print("🤖 Web Search Agent with Ollama")
        print("="*80)
        print("Commands:")
        print("  - Type your query to search")
        print("  - Add '--deep' for deep scraping with sub-links")
        print("  - Type 'quit' or 'exit' to quit")
        print("="*80 + "\n")

        while True:
            try:
                query = input("\n💬 Enter your query: ").strip()

                if query.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break

                if not query:
                    continue

                deep_scrape = "--deep" in query
                query = query.replace("--deep", "").strip()

                result = agent.process_query(query, deep_scrape)
                print("\n" + "="*80)
                print("ANSWER:")
                print("="*80 + "\n")
                print(result)

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")


if __name__ == "__main__":
    main()