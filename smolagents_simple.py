#!/usr/bin/env python3
"""
Simplified Context-Aware Research with ToolCallingAgent
More reliable than CodeAgent for Ollama models
"""

from typing import List, Dict
from smolagents import ToolCallingAgent, tool, LiteLLMModel

from context_manager import ContextManager


# Simple global state
class SimpleResearchState:
    def __init__(self):
        self.sources = []
        self.total_tokens = 0
        self.cm = None

_state = SimpleResearchState()


@tool
def search_and_scrape(query: str, num_urls: int = 3) -> str:
    """
    Search web and scrape top results in one step.

    Args:
        query: Search query
        num_urls: Number of URLs to scrape (default 3)

    Returns:
        Combined content from all scraped URLs
    """
    from serpapi import GoogleSearch
    import requests
    from bs4 import BeautifulSoup

    api_key = "89edc0de285fab16d6170e2f5d508b91069acfec757f48061b70d7a2b5bda15e"

    # Search
    params = {"q": query, "api_key": api_key, "num": num_urls}

    try:
        search = GoogleSearch(params)
        results = search.get_dict()

        if "organic_results" not in results:
            return "No results found"

        # Scrape top URLs
        combined_content = []
        headers = {'User-Agent': 'Mozilla/5.0'}

        for i, result in enumerate(results["organic_results"][:num_urls], 1):
            url = result.get('link', '')
            title = result.get('title', '')

            try:
                response = requests.get(url, headers=headers, timeout=10)
                soup = BeautifulSoup(response.content, 'lxml')

                for element in soup(["script", "style", "nav", "footer", "header"]):
                    element.decompose()

                text = soup.get_text()
                lines = (line.strip() for line in text.splitlines())
                text = ' '.join(line for line in lines if line)

                # Limit to 10KB per source
                content = text[:10000]

                # Track in state
                _state.sources.append({
                    'url': url,
                    'title': title,
                    'content': content
                })

                if _state.cm:
                    tokens = _state.cm.count_tokens(content)
                    _state.total_tokens += tokens

                combined_content.append(f"Source [{i}]: {title}\nURL: {url}\n\n{content}\n\n{'='*80}\n")

            except Exception as e:
                combined_content.append(f"Error scraping {url}: {str(e)}\n")

        return "\n".join(combined_content)

    except Exception as e:
        return f"Search error: {str(e)}"


@tool
def get_context_status() -> str:
    """
    Get current context usage status.

    Returns:
        Context usage summary
    """
    if not _state.cm:
        return "Context manager not initialized"

    usage_percent = (_state.total_tokens / _state.cm.usable_context) * 100

    status = f"""Context Status:
- Sources: {len(_state.sources)}
- Tokens: {_state.total_tokens:,} / {_state.cm.usable_context:,}
- Usage: {usage_percent:.1f}%
- Status: {"⚠️ HIGH" if usage_percent > 70 else "✅ OK"}
"""
    return status


@tool
def compress_all_sources() -> str:
    """
    Compress all gathered sources to free context space.

    Returns:
        Compression result
    """
    if not _state.cm or not _state.sources:
        return "Nothing to compress"

    old_total = _state.total_tokens

    # Compress each source
    compressed = _state.cm.compress_sources(_state.sources, target_tokens_per_source=800)

    _state.sources = compressed
    _state.total_tokens = sum(s.get('compressed_tokens', s.get('original_tokens', 0)) for s in compressed)

    saved = old_total - _state.total_tokens

    return f"Compressed {len(compressed)} sources. Saved {saved:,} tokens ({(saved/old_total)*100:.1f}%)"


@tool
def finalize_report() -> str:
    """
    Get all sources formatted for final report.

    Returns:
        All sources combined and ready for synthesis
    """
    if not _state.sources:
        return "No sources to finalize"

    formatted = []
    for i, source in enumerate(_state.sources, 1):
        formatted.append(f"""
[{i}] {source['title']}
URL: {source['url']}

{source['content']}

{'='*80}
""")

    return "\n".join(formatted)


class SimpleResearchAgent:
    """
    Simplified research agent using ToolCallingAgent.
    More reliable than CodeAgent for Ollama models.
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "qwen3:32b",
        max_context: int = 40960
    ):
        # Initialize context manager
        _state.cm = ContextManager(max_context, ollama_url, model)
        _state.sources = []
        _state.total_tokens = 0

        self.model = LiteLLMModel(
            model_id=f"ollama/{model}",
            api_base=ollama_url
        )

        tools = [
            search_and_scrape,
            get_context_status,
            compress_all_sources,
            finalize_report
        ]

        self.agent = ToolCallingAgent(
            tools=tools,
            model=self.model,
            max_steps=12
        )

        print(f"🤖 Simple Research Agent initialized")
        print(f"🔧 Tools: {[t.name for t in tools]}")

    def research(self, query: str) -> str:
        """Execute research"""

        # Reset state
        _state.sources = []
        _state.total_tokens = 0

        prompt = f"""Research this topic comprehensively: "{query}"

Your task:
1. Use search_and_scrape() to gather information (call it 3-4 times with different queries)
2. Check get_context_status() periodically
3. If context usage >70%, use compress_all_sources()
4. When done gathering, use finalize_report() to get all sources
5. Write a comprehensive report

Example:
- search_and_scrape("quantum computing theory")
- search_and_scrape("quantum computing applications 2025")
- get_context_status()
- search_and_scrape("quantum computing challenges")
- compress_all_sources() if needed
- finalize_report()
- Write final report

Begin research now."""

        print("\n" + "="*80)
        print("🔍 SIMPLE RESEARCH AGENT")
        print("="*80)
        print(f"Query: {query}\n")

        try:
            result = self.agent.run(prompt)

            print("\n" + "="*80)
            print("✅ COMPLETE")
            print("="*80)
            print(f"Sources: {len(_state.sources)}, Tokens: {_state.total_tokens:,}")

            return result

        except Exception as e:
            return f"Error: {str(e)}"


def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python smolagents_simple.py <query>")
        return

    query = " ".join(sys.argv[1:])
    agent = SimpleResearchAgent()
    result = agent.research(query)

    print("\n" + "="*80)
    print("📊 FINAL REPORT")
    print("="*80)
    print(result)


if __name__ == "__main__":
    main()