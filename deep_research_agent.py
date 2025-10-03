#!/usr/bin/env python3
"""
Deep Research Agent - Multi-step iterative research system
Mimics Perplexity/ChatGPT deep research with local Ollama
"""

import requests
import json
import time
from bs4 import BeautifulSoup
from serpapi import GoogleSearch
from typing import List, Dict, Optional, Generator
from dataclasses import dataclass, asdict
from datetime import datetime
import asyncio

from context_manager import ContextManager


@dataclass
class ResearchStep:
    """Represents a single step in the research process"""
    step_number: int
    action: str  # "plan", "search", "scrape", "analyze", "synthesize"
    description: str
    query: Optional[str] = None
    results: Optional[List[Dict]] = None
    content: Optional[str] = None
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class ResearchReport:
    """Final research report"""
    query: str
    steps: List[ResearchStep]
    final_report: str
    sources: List[str]
    duration_seconds: float
    total_searches: int
    total_urls_scraped: int


class OllamaClient:
    """Client for Ollama REST API"""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "qwen3:32b"):
        self.base_url = base_url
        self.model = model

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        """Chat with Ollama"""
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature}
        }

        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            return response.json()["message"]["content"]
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            return ""

    def generate(self, prompt: str, system: Optional[str] = None, temperature: float = 0.7) -> str:
        """Generate with Ollama"""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature}
        }
        if system:
            payload["system"] = system

        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            return ""


class WebSearcher:
    """Web search using SerpAPI"""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def search(self, query: str, num_results: int = 5) -> List[Dict]:
        """Search web"""
        params = {
            "q": query,
            "api_key": self.api_key,
            "num": num_results
        }

        try:
            search = GoogleSearch(params)
            results = search.get_dict()

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
            print(f"Error searching: {e}")
            return []


class WebScraper:
    """Web scraper"""

    def __init__(self, max_content_chars: int = 20000):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.max_content_chars = max_content_chars

    def scrape(self, url: str) -> Dict[str, str]:
        """Scrape content from URL"""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
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
                "content": text[:self.max_content_chars],
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


class DeepResearchAgent:
    """
    Deep Research Agent - Multi-step iterative research

    Workflow:
    1. Plan: Break down query into sub-questions
    2. Search: Execute multiple targeted searches
    3. Scrape: Extract content from relevant URLs
    4. Analyze: Identify knowledge gaps
    5. Iterate: Repeat search/scrape if needed
    6. Synthesize: Generate comprehensive report
    """

    def __init__(
        self,
        serpapi_key: str,
        ollama_url: str = "http://localhost:11434",
        model: str = "qwen3:32b",
        max_iterations: int = 3,
        max_searches_per_iteration: int = 3,
        max_context: int = 40960  # Qwen3:32b context window
    ):
        self.llm = OllamaClient(ollama_url, model)
        self.searcher = WebSearcher(serpapi_key)
        self.context_manager = ContextManager(
            max_context=max_context,
            ollama_url=ollama_url,
            model=model
        )
        # Calculate dynamic scrape budget based on context
        scrape_budget = self.context_manager.estimate_scrape_budget(num_urls=5)
        self.scraper = WebScraper(max_content_chars=scrape_budget)
        self.max_iterations = max_iterations
        self.max_searches_per_iteration = max_searches_per_iteration

        print(f"  🧠 Context window: {max_context:,} tokens")
        print(f"  📄 Scrape budget: ~{scrape_budget:,} chars per URL")

    def plan_research(self, query: str) -> Dict:
        """Step 1: Create research plan and break down into sub-questions"""

        planning_prompt = f"""You are a research planner. Given a user query, create a comprehensive research plan.

User Query: {query}

Create a research plan by:
1. Breaking down the query into 3-5 specific sub-questions that need to be answered
2. Identifying key areas to investigate
3. Suggesting search queries for each sub-question

Respond in JSON format:
{{
    "main_topic": "brief description",
    "sub_questions": ["question 1", "question 2", ...],
    "search_queries": ["query 1", "query 2", ...],
    "key_areas": ["area 1", "area 2", ...]
}}

Respond ONLY with valid JSON, no other text."""

        response = self.llm.generate(planning_prompt, temperature=0.3)

        # Parse JSON response
        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                plan = json.loads(json_str)
                return plan
            else:
                # Fallback plan
                return {
                    "main_topic": query,
                    "sub_questions": [query],
                    "search_queries": [query],
                    "key_areas": ["general information"]
                }
        except Exception as e:
            print(f"Error parsing plan: {e}")
            return {
                "main_topic": query,
                "sub_questions": [query],
                "search_queries": [query],
                "key_areas": ["general information"]
            }

    def execute_searches(self, queries: List[str]) -> List[Dict]:
        """Step 2: Execute multiple searches"""
        all_results = []

        for query in queries[:self.max_searches_per_iteration]:
            print(f"  🔍 Searching: {query}")
            results = self.searcher.search(query, num_results=5)
            all_results.extend(results)

        # Deduplicate by URL
        seen_urls = set()
        unique_results = []
        for result in all_results:
            if result['link'] not in seen_urls:
                seen_urls.add(result['link'])
                unique_results.append(result)

        return unique_results

    def scrape_sources(self, search_results: List[Dict], max_scrapes: int = 5) -> List[Dict]:
        """Step 3: Scrape content from top URLs"""
        scraped_data = []
        urls_to_scrape = [r['link'] for r in search_results[:max_scrapes]]

        for url in urls_to_scrape:
            print(f"  🌐 Scraping: {url}")
            scraped = self.scraper.scrape(url)
            if scraped['success'] and scraped['content']:
                scraped_data.append(scraped)

        return scraped_data

    def analyze_gaps(self, query: str, current_findings: str) -> Dict:
        """Step 4: Analyze current findings and identify knowledge gaps"""

        gap_analysis_prompt = f"""You are a research analyst. Analyze the current findings and identify knowledge gaps.

Original Query: {query}

Current Findings:
{current_findings}

Analyze the findings and respond in JSON format:
{{
    "coverage_score": 0.0-1.0,
    "answered_aspects": ["aspect 1", "aspect 2", ...],
    "knowledge_gaps": ["gap 1", "gap 2", ...],
    "additional_queries": ["query 1", "query 2", ...],
    "recommendation": "continue" or "sufficient"
}}

Respond ONLY with valid JSON."""

        response = self.llm.generate(gap_analysis_prompt, temperature=0.3)

        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                analysis = json.loads(json_str)
                return analysis
            else:
                return {
                    "coverage_score": 0.8,
                    "recommendation": "sufficient"
                }
        except Exception as e:
            print(f"Error parsing gap analysis: {e}")
            return {
                "coverage_score": 0.8,
                "recommendation": "sufficient"
            }

    def synthesize_report(self, query: str, all_findings: List[Dict], sources: List[str]) -> str:
        """Step 6: Synthesize final comprehensive report with context management"""

        print("\n📊 Preparing synthesis with context management...")

        system_prompt = "You are an expert research analyst. Synthesize information into comprehensive, well-structured reports."

        # Use context manager to prepare sources within context limits
        findings_context, total_tokens = self.context_manager.prepare_synthesis_context(
            query=query,
            sources=all_findings,
            system_prompt=system_prompt
        )

        synthesis_prompt = f"""Original Query: {query}

Information from sources:

{findings_context}

Create a comprehensive research report that:
1. Directly answers the original query
2. Provides detailed, accurate information
3. Organizes information logically with clear sections
4. Includes specific facts, figures, and examples
5. Cites sources inline with [1], [2], etc.
6. Identifies any contradictions or uncertainties
7. Provides actionable insights where relevant

Write a thorough, professional report (aim for 500-1000 words)."""

        print(f"  🤖 Generating report with {total_tokens:,} tokens of context...")

        report = self.llm.generate(synthesis_prompt, system=system_prompt, temperature=0.5)

        # Add sources section
        sources_section = "\n\n## Sources\n\n"
        for i, source in enumerate(sources, 1):
            sources_section += f"[{i}] {source}\n"

        return report + sources_section

    def research(self, query: str, progress_callback=None) -> ResearchReport:
        """
        Execute full deep research process

        Args:
            query: Research query
            progress_callback: Optional callback(step: ResearchStep) for progress updates
        """
        start_time = time.time()
        steps = []
        all_scraped_data = []
        all_sources = []
        total_searches = 0

        # Step 1: Plan
        print("\n📋 Step 1: Planning research...")
        plan = self.plan_research(query)
        planning_step = ResearchStep(
            step_number=1,
            action="plan",
            description="Created research plan with sub-questions",
            results=[plan]
        )
        steps.append(planning_step)
        if progress_callback:
            progress_callback(planning_step)

        print(f"  Main topic: {plan['main_topic']}")
        print(f"  Sub-questions: {len(plan['sub_questions'])}")

        # Iterative search and analysis
        for iteration in range(self.max_iterations):
            print(f"\n🔄 Iteration {iteration + 1}/{self.max_iterations}")

            # Step 2: Search
            print(f"  Step 2.{iteration + 1}: Executing searches...")
            queries = plan['search_queries']
            search_results = self.execute_searches(queries)
            total_searches += len(queries)

            search_step = ResearchStep(
                step_number=len(steps) + 1,
                action="search",
                description=f"Executed {len(queries)} searches, found {len(search_results)} results",
                results=search_results
            )
            steps.append(search_step)
            if progress_callback:
                progress_callback(search_step)

            if not search_results:
                print("  No results found, stopping iteration")
                break

            # Step 3: Scrape
            print(f"  Step 3.{iteration + 1}: Scraping sources...")
            scraped_data = self.scrape_sources(search_results, max_scrapes=5)
            all_scraped_data.extend(scraped_data)
            all_sources.extend([s['url'] for s in scraped_data])

            scrape_step = ResearchStep(
                step_number=len(steps) + 1,
                action="scrape",
                description=f"Scraped {len(scraped_data)} URLs successfully",
                results=[{"url": s['url'], "title": s['title']} for s in scraped_data]
            )
            steps.append(scrape_step)
            if progress_callback:
                progress_callback(scrape_step)

            # Step 4: Analyze gaps (only if not last iteration)
            if iteration < self.max_iterations - 1:
                print(f"  Step 4.{iteration + 1}: Analyzing knowledge gaps...")

                # Prepare compressed findings for gap analysis (context aware)
                current_findings_parts = []
                for s in all_scraped_data[:10]:  # Limit to 10 most recent sources
                    excerpt = s['content'][:800]  # Limit excerpt size
                    current_findings_parts.append(f"{s['title']}: {excerpt}...")

                current_findings = "\n\n".join(current_findings_parts)

                # Ensure gap analysis prompt fits in context
                if self.context_manager.count_tokens(current_findings) > 8000:
                    print("  ⚠️  Findings too large, truncating for gap analysis...")
                    current_findings = self.context_manager.truncate_to_tokens(current_findings, 8000)

                gap_analysis = self.analyze_gaps(query, current_findings)

                analysis_step = ResearchStep(
                    step_number=len(steps) + 1,
                    action="analyze",
                    description=f"Gap analysis: {gap_analysis.get('recommendation', 'unknown')}",
                    results=[gap_analysis]
                )
                steps.append(analysis_step)
                if progress_callback:
                    progress_callback(analysis_step)

                print(f"    Coverage: {gap_analysis.get('coverage_score', 'N/A')}")
                print(f"    Recommendation: {gap_analysis.get('recommendation', 'N/A')}")

                # Check if we have enough information
                if gap_analysis.get('recommendation') == 'sufficient':
                    print("  ✓ Sufficient information gathered")
                    break

                # Update plan with additional queries
                if gap_analysis.get('additional_queries'):
                    plan['search_queries'] = gap_analysis['additional_queries']

        # Step 5: Synthesize
        print("\n📝 Final Step: Synthesizing comprehensive report...")
        final_report = self.synthesize_report(query, all_scraped_data, list(set(all_sources)))

        synthesis_step = ResearchStep(
            step_number=len(steps) + 1,
            action="synthesize",
            description="Generated final comprehensive report",
            content=final_report
        )
        steps.append(synthesis_step)
        if progress_callback:
            progress_callback(synthesis_step)

        duration = time.time() - start_time

        report = ResearchReport(
            query=query,
            steps=steps,
            final_report=final_report,
            sources=list(set(all_sources)),
            duration_seconds=duration,
            total_searches=total_searches,
            total_urls_scraped=len(all_scraped_data)
        )

        print(f"\n✅ Research complete in {duration:.1f}s")
        print(f"   Searches: {total_searches}, URLs scraped: {len(all_scraped_data)}")

        return report


def main():
    """CLI interface for deep research"""
    import sys

    SERPAPI_KEY = "89edc0de285fab16d6170e2f5d508b91069acfec757f48061b70d7a2b5bda15e"
    OLLAMA_URL = "http://localhost:11434"
    MODEL = "qwen3:32b"

    agent = DeepResearchAgent(
        serpapi_key=SERPAPI_KEY,
        ollama_url=OLLAMA_URL,
        model=MODEL,
        max_iterations=3,
        max_searches_per_iteration=3
    )

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])

        print("="*80)
        print("🤖 DEEP RESEARCH AGENT")
        print("="*80)
        print(f"Query: {query}\n")

        report = agent.research(query)

        print("\n" + "="*80)
        print("📊 RESEARCH REPORT")
        print("="*80)
        print(report.final_report)

    else:
        print("🤖 Deep Research Agent")
        print("="*80)
        print("Usage: python deep_research_agent.py <your research query>")
        print("\nExample:")
        print("  python deep_research_agent.py 'What are the latest developments in quantum computing?'")


if __name__ == "__main__":
    main()