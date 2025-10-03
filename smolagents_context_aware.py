#!/usr/bin/env python3
"""
Context-Aware Deep Research using smolagents
Integrates ContextManager with smolagents framework for safe 40K token limit
"""

import os
from typing import Optional, List, Dict
from smolagents import (
    CodeAgent,
    tool,
    LiteLLMModel
)

from context_manager import ContextManager


# Global context manager instance (accessible to all tools)
_context_manager = None


def init_context_manager(max_context: int = 40960, ollama_url: str = "http://localhost:11434", model: str = "qwen3:32b"):
    """Initialize global context manager"""
    global _context_manager
    _context_manager = ContextManager(
        max_context=max_context,
        ollama_url=ollama_url,
        model=model
    )
    print(f"🧠 Context Manager initialized: {max_context:,} tokens")
    return _context_manager


def get_context_manager() -> ContextManager:
    """Get global context manager instance"""
    global _context_manager
    if _context_manager is None:
        _context_manager = init_context_manager()
    return _context_manager


# Context-aware research state (persists across tool calls)
class ResearchState:
    """
    Persistent state for research session
    Available to agent across all tool calls
    """
    def __init__(self):
        self.sources: List[Dict] = []  # All scraped sources
        self.search_results: List[Dict] = []  # All search results
        self.total_tokens: int = 0  # Running token count
        self.compressed_sources: List[Dict] = []  # Compressed versions

    def add_source(self, source: Dict):
        """Add source and track tokens"""
        cm = get_context_manager()
        tokens = cm.count_tokens(source.get('content', ''))
        source['tokens'] = tokens
        self.sources.append(source)
        self.total_tokens += tokens

        print(f"  📊 Source added: {tokens:,} tokens (total: {self.total_tokens:,})")

    def check_budget(self) -> Dict:
        """Check if we're approaching context limits"""
        cm = get_context_manager()
        usage_percent = (self.total_tokens / cm.usable_context) * 100

        return {
            'total_tokens': self.total_tokens,
            'usable_context': cm.usable_context,
            'usage_percent': usage_percent,
            'needs_compression': usage_percent > 60,  # Start compressing at 60%
            'critical': usage_percent > 85  # Stop gathering at 85%
        }


# Global research state
_research_state = None


def get_research_state() -> ResearchState:
    """Get or create research state"""
    global _research_state
    if _research_state is None:
        _research_state = ResearchState()
    return _research_state


def reset_research_state():
    """Reset research state for new query"""
    global _research_state
    _research_state = ResearchState()


# Context-aware tools

@tool
def search_web(query: str, num_results: int = 5) -> str:
    """
    Search the web using SerpAPI with context awareness.

    Args:
        query: The search query string
        num_results: Number of results to return (default 5)

    Returns:
        Formatted search results with titles, links, and snippets
    """
    from serpapi import GoogleSearch

    api_key = "89edc0de285fab16d6170e2f5d508b91069acfec757f48061b70d7a2b5bda15e"

    params = {
        "q": query,
        "api_key": api_key,
        "num": num_results
    }

    try:
        search = GoogleSearch(params)
        results = search.get_dict()

        if "organic_results" not in results:
            return "No results found"

        state = get_research_state()
        formatted_results = []

        for i, result in enumerate(results["organic_results"][:num_results], 1):
            search_result = {
                'title': result.get('title', 'No title'),
                'link': result.get('link', 'No URL'),
                'snippet': result.get('snippet', 'No snippet')
            }
            state.search_results.append(search_result)

            formatted_results.append(
                f"[{i}] {search_result['title']}\n"
                f"URL: {search_result['link']}\n"
                f"Snippet: {search_result['snippet']}\n"
            )

        return "\n".join(formatted_results)

    except Exception as e:
        return f"Error searching: {str(e)}"


@tool
def scrape_url(url: str) -> str:
    """
    Scrape content from a URL with automatic context budgeting.
    Automatically adjusts how much content to scrape based on remaining context.

    Args:
        url: The URL to scrape

    Returns:
        The scraped text content (auto-sized to fit context budget)
    """
    import requests
    from bs4 import BeautifulSoup

    state = get_research_state()
    cm = get_context_manager()

    # Check context budget
    budget_check = state.check_budget()

    if budget_check['critical']:
        return f"⚠️ Context budget critical ({budget_check['usage_percent']:.1f}%). Skipping scrape of {url}. Consider compressing sources first."

    # Calculate dynamic scrape budget
    remaining_tokens = cm.usable_context - state.total_tokens
    # Use 10% of remaining context per scrape (conservative)
    scrape_token_budget = int(remaining_tokens * 0.1)
    max_chars = scrape_token_budget * 4  # ~4 chars per token

    print(f"  📄 Scraping {url} (budget: {max_chars:,} chars, ~{scrape_token_budget:,} tokens)")

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
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

        title = soup.title.string if soup.title else "No title"

        # Truncate to budget
        content = text[:max_chars]

        # Add to state
        source = {
            'url': url,
            'title': title,
            'content': content
        }
        state.add_source(source)

        return f"Title: {title}\n\nContent:\n{content}"

    except Exception as e:
        return f"Error scraping {url}: {str(e)}"


@tool
def compress_sources(target_tokens_per_source: int = 1000) -> str:
    """
    Compress all scraped sources to free up context space.
    Uses LLM to intelligently summarize while preserving key information.

    Args:
        target_tokens_per_source: Target token count per source after compression

    Returns:
        Compression summary
    """
    state = get_research_state()
    cm = get_context_manager()

    if not state.sources:
        return "No sources to compress"

    print(f"\n🗜️  Compressing {len(state.sources)} sources...")

    compressed = cm.compress_sources(state.sources, target_tokens_per_source)

    # Update state
    old_total = state.total_tokens
    state.compressed_sources = compressed
    state.sources = compressed  # Replace with compressed versions

    # Recalculate total tokens
    state.total_tokens = sum(s.get('compressed_tokens', s.get('original_tokens', 0)) for s in compressed)

    saved_tokens = old_total - state.total_tokens
    saved_percent = (saved_tokens / old_total) * 100 if old_total > 0 else 0

    result = f"""Compression complete!
- Sources: {len(compressed)}
- Before: {old_total:,} tokens
- After: {state.total_tokens:,} tokens
- Saved: {saved_tokens:,} tokens ({saved_percent:.1f}%)

You can now scrape more URLs or proceed to synthesis."""

    print(f"  ✅ Saved {saved_tokens:,} tokens ({saved_percent:.1f}%)")

    return result


@tool
def check_context_budget() -> str:
    """
    Check current context usage and get recommendations.

    Returns:
        Context budget status and recommendations
    """
    state = get_research_state()
    budget_check = state.check_budget()

    status = f"""📊 Context Budget Status:
- Total tokens used: {budget_check['total_tokens']:,}
- Usable context: {budget_check['usable_context']:,}
- Usage: {budget_check['usage_percent']:.1f}%
- Sources scraped: {len(state.sources)}
- Search results: {len(state.search_results)}

"""

    if budget_check['critical']:
        status += "⚠️  CRITICAL: Context budget >85%. Stop gathering and compress sources!"
    elif budget_check['needs_compression']:
        status += "⚠️  WARNING: Context budget >60%. Consider compressing sources soon."
    else:
        status += "✅ HEALTHY: Plenty of context remaining."

    return status


@tool
def prepare_synthesis() -> str:
    """
    Prepare all gathered sources for final synthesis.
    Automatically compresses if needed to fit within context limits.

    Returns:
        Combined and formatted content from all sources, ready for synthesis
    """
    state = get_research_state()
    cm = get_context_manager()

    if not state.sources:
        return "No sources available for synthesis"

    print(f"\n📝 Preparing synthesis from {len(state.sources)} sources...")

    # Check if we need compression
    budget_check = state.check_budget()

    if budget_check['usage_percent'] > 70:
        print("  🗜️  Context usage high, compressing sources...")
        # Auto-compress to fit
        available_for_synthesis = int(cm.usable_context * 0.7)  # Use 70% for synthesis
        tokens_per_source = available_for_synthesis // len(state.sources)
        state.compressed_sources = cm.compress_sources(state.sources, tokens_per_source)
        sources_to_use = state.compressed_sources
    else:
        sources_to_use = state.sources

    # Format all sources
    formatted_sources = []
    for i, source in enumerate(sources_to_use, 1):
        formatted_sources.append(f"""
Source [{i}]: {source['url']}
Title: {source['title']}

Content:
{source['content']}

{'='*80}
""")

    combined = "\n".join(formatted_sources)

    # Final token check
    final_tokens = cm.count_tokens(combined)
    print(f"  ✅ Synthesis content prepared: {final_tokens:,} tokens")

    return combined


class ContextAwareResearchAgent:
    """
    Deep Research Agent with integrated context management

    Key features:
    - Automatic context budgeting
    - Progressive source compression
    - Token tracking across tool calls
    - Dynamic scrape sizing
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "qwen3:32b",
        max_context: int = 40960
    ):
        # Initialize context manager
        init_context_manager(max_context, ollama_url, model)

        # Configure LiteLLM to use Ollama
        self.model = LiteLLMModel(
            model_id=f"ollama/{model}",
            api_base=ollama_url
        )

        # Context-aware tools
        tools = [
            search_web,
            scrape_url,
            compress_sources,
            check_context_budget,
            prepare_synthesis
        ]

        # Create agent with additional imports for context manager
        self.agent = CodeAgent(
            tools=tools,
            model=self.model,
            max_steps=15,  # Limit steps to prevent infinite loops
            verbosity_level=1,  # Reduce verbosity to avoid clutter
            additional_authorized_imports=['context_manager']  # Allow importing our module
        )

        print(f"🤖 Context-Aware Agent initialized")
        print(f"🔧 Tools: {[t.name for t in tools]}")
        print(f"🧠 Max context: {max_context:,} tokens")

    def research(self, query: str) -> str:
        """
        Execute context-aware deep research

        The agent will:
        1. Plan research and check context budget
        2. Search and scrape with automatic budget management
        3. Compress sources if needed
        4. Monitor context usage throughout
        5. Prepare and synthesize final report
        """

        # Reset state for new research
        reset_research_state()

        research_prompt = f"""You are a context-aware research agent with a 40,960 token limit. Your task is to conduct comprehensive research on:

"{query}"

CRITICAL RULES:
1. ONLY write valid Python code - no regex patterns, no raw text
2. Use the provided tools: search_web(), scrape_url(), compress_sources(), check_context_budget(), prepare_synthesis()
3. Monitor context usage regularly
4. If context >60%, compress before continuing
5. If context >85%, STOP gathering and synthesize

CORRECT EXAMPLE:
```python
# Step 1: Check initial budget
budget = check_context_budget()
print(budget)

# Step 2: Search for information
results1 = search_web("quantum computing theory")
print(results1)

# Step 3: Parse results and scrape (use string methods, not regex)
# Extract URLs from results string manually
lines = results1.split('\\n')
urls_to_scrape = []
for line in lines:
    if line.startswith('URL:'):
        url = line.replace('URL:', '').strip()
        urls_to_scrape.append(url)

# Step 4: Scrape URLs
for i, url in enumerate(urls_to_scrape[:5]):
    print(f"Scraping {i+1}/{len(urls_to_scrape[:5])}: {url}")
    content = scrape_url(url)
    print(f"Scraped {len(content)} characters")

    # Check budget every 3 scrapes
    if (i + 1) % 3 == 0:
        budget = check_context_budget()
        print(budget)
        if "WARNING" in budget or "CRITICAL" in budget:
            print("Compressing sources to free space...")
            result = compress_sources(target_tokens_per_source=1000)
            print(result)

# Step 5: Do more searches if needed
results2 = search_web("quantum computing applications 2025")
# ... repeat scraping process ...

# Step 6: When done gathering, prepare synthesis
print("\\nPreparing final synthesis...")
all_sources = prepare_synthesis()

# Step 7: Write final report as a string
final_report = f\"\"\"
# Comprehensive Analysis of Quantum Computing

## Executive Summary
[Your summary based on gathered information]

## Theory
[Content from theoretical sources]

## Applications
[Content from application sources]

## Challenges
[Content about challenges]

## Future Prospects
[Content about future]

## Conclusion
[Your conclusion]

## Sources
See citations above
\"\"\"

print(final_report)
```

IMPORTANT: Do NOT use regex patterns like (.*?) or re.findall() - they cause syntax errors. Use simple string methods like split(), replace(), startswith(), in operator.

Begin your research now!"""

        print("\n" + "="*80)
        print("🔍 CONTEXT-AWARE DEEP RESEARCH")
        print("="*80)
        print(f"Query: {query}")
        print(f"Context Limit: 40,960 tokens\n")

        try:
            result = self.agent.run(research_prompt)

            print("\n" + "="*80)
            print("✅ RESEARCH COMPLETE")
            print("="*80)

            # Print final stats
            state = get_research_state()
            print(f"\nFinal Statistics:")
            print(f"  Sources gathered: {len(state.sources)}")
            print(f"  Searches performed: {len(state.search_results)}")
            print(f"  Total tokens used: {state.total_tokens:,}")
            print(f"  Context usage: {(state.total_tokens / 40960) * 100:.1f}%")

            return result

        except Exception as e:
            return f"Error during research: {str(e)}"


def main():
    """CLI interface"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python smolagents_context_aware.py <your query>")
        print("\nExample:")
        print('  python smolagents_context_aware.py "What are the latest developments in quantum computing?"')
        return

    query = " ".join(sys.argv[1:])

    agent = ContextAwareResearchAgent()
    result = agent.research(query)

    print("\n" + "="*80)
    print("📊 FINAL REPORT")
    print("="*80)
    print(result)


if __name__ == "__main__":
    main()