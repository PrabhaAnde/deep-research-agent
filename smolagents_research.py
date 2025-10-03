#!/usr/bin/env python3
"""
Deep Research using smolagents framework
Cleaner, more maintainable implementation with built-in tools
"""

import os
from typing import Optional
from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    tool,
    LiteLLMModel
)


# Custom research tool using SerpAPI
@tool
def search_web(query: str) -> str:
    """
    Search the web using SerpAPI and return relevant results.

    Args:
        query: The search query string

    Returns:
        A formatted string with search results including titles, links, and snippets
    """
    from serpapi import GoogleSearch
    import json

    api_key = "89edc0de285fab16d6170e2f5d508b91069acfec757f48061b70d7a2b5bda15e"

    params = {
        "q": query,
        "api_key": api_key,
        "num": 5
    }

    try:
        search = GoogleSearch(params)
        results = search.get_dict()

        if "organic_results" not in results:
            return "No results found"

        formatted_results = []
        for i, result in enumerate(results["organic_results"][:5], 1):
            formatted_results.append(
                f"[{i}] {result.get('title', 'No title')}\n"
                f"URL: {result.get('link', 'No URL')}\n"
                f"Snippet: {result.get('snippet', 'No snippet')}\n"
            )

        return "\n".join(formatted_results)

    except Exception as e:
        return f"Error searching: {str(e)}"


@tool
def scrape_url(url: str, max_chars: int = 15000) -> str:
    """
    Scrape content from a URL.

    Args:
        url: The URL to scrape
        max_chars: Maximum characters to return

    Returns:
        The scraped text content
    """
    import requests
    from bs4 import BeautifulSoup

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

        return f"Title: {title}\n\nContent:\n{text[:max_chars]}"

    except Exception as e:
        return f"Error scraping {url}: {str(e)}"


@tool
def summarize_sources(sources: str, query: str) -> str:
    """
    Summarize multiple sources for a research query.
    This tool uses the LLM to create a comprehensive summary.

    Args:
        sources: Combined text from multiple sources
        query: The original research query

    Returns:
        A comprehensive summary
    """
    # This will be handled by the agent's LLM naturally
    return f"Please synthesize the following information to answer: {query}\n\n{sources}"


class DeepResearchAgent:
    """
    Deep Research Agent using smolagents framework

    Advantages over custom implementation:
    - Agent writes Python code to orchestrate research
    - Built-in retry and error handling
    - Dynamic planning (not fixed iterations)
    - Cleaner, more maintainable code
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "qwen3:32b",
        agent_type: str = "code"  # "code" or "tool_calling"
    ):
        # Configure LiteLLM to use Ollama
        # LiteLLM format: ollama/<model_name>
        self.model = LiteLLMModel(
            model_id=f"ollama/{model}",
            api_base=ollama_url
        )

        # Custom tools for research
        tools = [search_web, scrape_url]

        # Create agent
        if agent_type == "code":
            self.agent = CodeAgent(
                tools=tools,
                model=self.model,
                max_steps=15,  # Allow up to 15 reasoning steps
                verbosity_level=2
            )
        else:
            self.agent = ToolCallingAgent(
                tools=tools,
                model=self.model,
                max_steps=15,
                verbosity_level=2
            )

        print(f"🤖 Initialized {agent_type} agent with {model}")
        print(f"🔧 Tools available: {[t.name for t in tools]}")

    def research(self, query: str) -> str:
        """
        Execute deep research using the agent

        The agent will autonomously:
        1. Plan the research approach
        2. Execute web searches
        3. Scrape relevant URLs
        4. Synthesize findings
        5. Iterate if needed
        """

        research_prompt = f"""You are a research agent. Your task is to conduct comprehensive research on the following query:

"{query}"

Follow this research methodology:
1. Break down the query into specific sub-questions
2. Search the web for each sub-question using search_web()
3. Scrape content from the most relevant URLs using scrape_url()
4. Analyze the information gathered
5. Identify any knowledge gaps and search for additional information if needed
6. Synthesize all findings into a comprehensive, well-structured report

Important guidelines:
- Search multiple times with different queries to get comprehensive coverage
- Scrape at least 5-10 URLs to gather diverse perspectives
- Cite sources in your final report with [1], [2], etc.
- Organize the report with clear sections
- Include specific facts, figures, and examples
- Note any contradictions or uncertainties

Begin your research now."""

        print("\n" + "="*80)
        print("🔍 DEEP RESEARCH STARTING")
        print("="*80)
        print(f"Query: {query}\n")

        try:
            # Let the agent do its thing!
            result = self.agent.run(research_prompt)

            print("\n" + "="*80)
            print("✅ RESEARCH COMPLETE")
            print("="*80)

            return result

        except Exception as e:
            return f"Error during research: {str(e)}"


class MultiAgentResearch:
    """
    Multi-agent deep research system
    Uses specialized agents for different tasks
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "qwen3:32b"
    ):
        self.model = LiteLLMModel(
            model_id=f"ollama/{model}",
            api_base=ollama_url
        )

        # Planner agent - creates research plan
        self.planner = CodeAgent(
            tools=[],
            model=self.model,
            max_steps=5,
            verbosity_level=1
        )

        # Searcher agent - executes searches
        self.searcher = CodeAgent(
            tools=[search_web, scrape_url],
            model=self.model,
            max_steps=10,
            verbosity_level=1
        )

        # Synthesizer agent - writes final report
        self.synthesizer = CodeAgent(
            tools=[],
            model=self.model,
            max_steps=5,
            verbosity_level=1
        )

    def research(self, query: str) -> str:
        """
        Multi-agent research pipeline

        1. Planner creates research plan
        2. Searcher gathers information
        3. Synthesizer writes final report
        """

        print("\n" + "="*80)
        print("🤖 MULTI-AGENT RESEARCH")
        print("="*80)

        # Step 1: Planning
        print("\n📋 Phase 1: Planning")
        plan_prompt = f"""Create a detailed research plan for the query: "{query}"

Break it down into:
1. Main topic and context
2. 3-5 specific sub-questions to answer
3. Key areas to investigate
4. Suggested search queries

Respond in a structured format."""

        plan = self.planner.run(plan_prompt)
        print(f"Plan:\n{plan}\n")

        # Step 2: Information Gathering
        print("🔍 Phase 2: Information Gathering")
        search_prompt = f"""Execute research based on this plan:

{plan}

Original query: {query}

Use search_web() to find relevant information and scrape_url() to get detailed content.
Gather comprehensive information from multiple sources (aim for 8-10 sources).
Return all gathered information in a structured format."""

        findings = self.searcher.run(search_prompt)
        print(f"Gathered information from multiple sources\n")

        # Step 3: Synthesis
        print("📝 Phase 3: Synthesis")
        synthesis_prompt = f"""Synthesize the following research findings into a comprehensive report.

Original query: {query}

Research plan:
{plan}

Gathered information:
{findings}

Create a well-structured research report that:
- Directly answers the query
- Organizes information logically
- Includes specific facts and examples
- Cites sources
- Identifies any gaps or uncertainties
- Provides actionable insights"""

        report = self.synthesizer.run(synthesis_prompt)

        print("\n" + "="*80)
        print("✅ MULTI-AGENT RESEARCH COMPLETE")
        print("="*80)

        return report


def main():
    """CLI interface"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python smolagents_research.py <your query>")
        print("\nOptions:")
        print("  --multi    Use multi-agent system")
        print("  --tool     Use ToolCallingAgent instead of CodeAgent")
        print("\nExamples:")
        print('  python smolagents_research.py "What are the latest developments in quantum computing?"')
        print('  python smolagents_research.py --multi "Compare AI chatbots in 2025"')
        return

    # Parse arguments
    use_multi = "--multi" in sys.argv
    use_tool = "--tool" in sys.argv

    # Get query (filter out flags)
    query_parts = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
    query = " ".join(query_parts)

    if not query:
        print("Error: No query provided")
        return

    # Run research
    if use_multi:
        agent = MultiAgentResearch()
    else:
        agent_type = "tool_calling" if use_tool else "code"
        agent = DeepResearchAgent(agent_type=agent_type)

    result = agent.research(query)

    print("\n" + "="*80)
    print("📊 FINAL REPORT")
    print("="*80)
    print(result)


if __name__ == "__main__":
    main()