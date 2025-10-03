"""
Context Window Management for Deep Research
Handles token counting, chunking, and progressive summarization
"""

import tiktoken
from typing import List, Dict, Tuple
import requests


class ContextManager:
    """
    Manages context window limits for LLM operations

    Key responsibilities:
    1. Token counting (approximate for Qwen via tiktoken)
    2. Intelligent content truncation
    3. Progressive summarization
    4. Context budget allocation
    """

    def __init__(
        self,
        max_context: int = 40960,  # Qwen3:32b context window
        ollama_url: str = "http://localhost:11434",
        model: str = "qwen3:32b",
        safety_margin: float = 0.2  # Reserve 20% for response
    ):
        self.max_context = max_context
        self.ollama_url = ollama_url
        self.model = model

        # Reserve context for response generation
        self.usable_context = int(max_context * (1 - safety_margin))

        # Use tiktoken with GPT-3.5 encoding (reasonable approximation for Qwen)
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text (approximate for Qwen)

        Note: Qwen uses different tokenizer, but tiktoken provides
        reasonable approximation (~10-20% variance)
        """
        if not text:
            return 0
        return len(self.encoding.encode(text))

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit"""
        if not text:
            return ""

        tokens = self.encoding.encode(text)

        if len(tokens) <= max_tokens:
            return text

        # Truncate and decode
        truncated_tokens = tokens[:max_tokens]
        return self.encoding.decode(truncated_tokens)

    def summarize_content(
        self,
        content: str,
        max_summary_tokens: int = 1000,
        context: str = ""
    ) -> str:
        """
        Summarize long content using LLM

        Args:
            content: Text to summarize
            max_summary_tokens: Target summary length
            context: Additional context for summarization
        """
        current_tokens = self.count_tokens(content)

        # If already short enough, return as-is
        if current_tokens <= max_summary_tokens:
            return content

        # Create summarization prompt
        prompt = f"""Summarize the following content concisely while preserving key information, facts, and insights.

{"Context: " + context if context else ""}

Content to summarize:
{content}

Provide a comprehensive summary that captures all important points:"""

        # Call Ollama to summarize
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": max_summary_tokens
                    }
                },
                timeout=60
            )
            response.raise_for_status()
            summary = response.json()["response"]
            return summary.strip()
        except Exception as e:
            print(f"Error summarizing content: {e}")
            # Fallback: simple truncation
            return self.truncate_to_tokens(content, max_summary_tokens)

    def chunk_content(
        self,
        content: str,
        chunk_size: int = 2000,
        overlap: int = 200
    ) -> List[str]:
        """
        Split content into overlapping chunks

        Useful for processing very long documents
        """
        tokens = self.encoding.encode(content)
        chunks = []

        start = 0
        while start < len(tokens):
            end = start + chunk_size
            chunk_tokens = tokens[start:end]
            chunks.append(self.encoding.decode(chunk_tokens))
            start = end - overlap

        return chunks

    def compress_sources(
        self,
        sources: List[Dict[str, str]],
        target_tokens_per_source: int = 1000
    ) -> List[Dict[str, str]]:
        """
        Compress scraped sources to fit context budget

        Each source gets summarized to target_tokens_per_source
        """
        compressed = []

        for source in sources:
            original_content = source.get('content', '')
            current_tokens = self.count_tokens(original_content)

            if current_tokens > target_tokens_per_source:
                # Summarize to fit budget
                summary = self.summarize_content(
                    original_content,
                    max_summary_tokens=target_tokens_per_source,
                    context=f"URL: {source.get('url', '')} | Title: {source.get('title', '')}"
                )

                compressed.append({
                    'url': source.get('url', ''),
                    'title': source.get('title', ''),
                    'content': summary,
                    'compressed': True,
                    'original_tokens': current_tokens,
                    'compressed_tokens': self.count_tokens(summary)
                })
            else:
                # Keep as-is
                compressed.append({
                    'url': source.get('url', ''),
                    'title': source.get('title', ''),
                    'content': original_content,
                    'compressed': False,
                    'original_tokens': current_tokens
                })

        return compressed

    def prepare_synthesis_context(
        self,
        query: str,
        sources: List[Dict[str, str]],
        system_prompt: str = ""
    ) -> Tuple[str, int]:
        """
        Prepare context for final synthesis, ensuring it fits within limits

        Returns: (prepared_context, total_tokens)
        """
        # Calculate token budgets
        system_tokens = self.count_tokens(system_prompt)
        query_tokens = self.count_tokens(query)
        response_reserve = 4000  # Reserve for response

        available_for_sources = self.max_context - system_tokens - query_tokens - response_reserve - 500  # buffer

        if available_for_sources <= 0:
            raise ValueError("Query and system prompt too long!")

        # Calculate tokens per source
        num_sources = len(sources)
        if num_sources == 0:
            return "", 0

        tokens_per_source = available_for_sources // num_sources

        # Compress sources to fit budget
        print(f"  📊 Context budget: {available_for_sources:,} tokens for {num_sources} sources")
        print(f"  📊 Allocation: ~{tokens_per_source:,} tokens per source")

        compressed_sources = self.compress_sources(sources, tokens_per_source)

        # Build final context
        context_parts = []
        total_tokens = system_tokens + query_tokens

        for i, source in enumerate(compressed_sources, 1):
            source_text = f"""Source [{i}]: {source['url']}
Title: {source['title']}

{source['content']}

---"""

            source_tokens = self.count_tokens(source_text)

            # Double check we're not exceeding
            if total_tokens + source_tokens > self.usable_context:
                print(f"  ⚠️  Stopping at source {i}/{num_sources} - context limit reached")
                break

            context_parts.append(source_text)
            total_tokens += source_tokens

            if source.get('compressed'):
                print(f"  📉 Source {i}: {source['original_tokens']:,} → {source['compressed_tokens']:,} tokens")
            else:
                print(f"  📄 Source {i}: {source['original_tokens']:,} tokens (kept original)")

        final_context = "\n\n".join(context_parts)

        print(f"  ✅ Final context: {total_tokens:,} / {self.max_context:,} tokens ({(total_tokens/self.max_context)*100:.1f}%)")

        return final_context, total_tokens

    def estimate_scrape_budget(self, num_urls: int) -> int:
        """
        Estimate max chars to scrape per URL to stay within context

        Rule of thumb: 1 token ≈ 4 characters
        """
        # Reserve space for prompts, responses, etc.
        available_tokens = self.usable_context // 2

        tokens_per_url = available_tokens // num_urls
        chars_per_url = tokens_per_url * 4

        # Cap at reasonable limits
        return min(chars_per_url, 20000)  # Max 20k chars per page


def test_context_manager():
    """Test context manager functionality"""
    cm = ContextManager()

    print("="*80)
    print("Context Manager Test")
    print("="*80)

    # Test 1: Token counting
    sample_text = "This is a test sentence. " * 100
    tokens = cm.count_tokens(sample_text)
    print(f"\nTest 1 - Token Counting:")
    print(f"  Text length: {len(sample_text)} chars")
    print(f"  Estimated tokens: {tokens}")
    print(f"  Ratio: {len(sample_text)/tokens:.2f} chars/token")

    # Test 2: Truncation
    long_text = "Word " * 50000
    truncated = cm.truncate_to_tokens(long_text, 1000)
    print(f"\nTest 2 - Truncation:")
    print(f"  Original: {cm.count_tokens(long_text)} tokens")
    print(f"  Truncated: {cm.count_tokens(truncated)} tokens")

    # Test 3: Chunking
    chunks = cm.chunk_content(long_text, chunk_size=500, overlap=50)
    print(f"\nTest 3 - Chunking:")
    print(f"  Original: {cm.count_tokens(long_text)} tokens")
    print(f"  Chunks: {len(chunks)}")
    print(f"  Avg chunk size: {sum(cm.count_tokens(c) for c in chunks) / len(chunks):.0f} tokens")

    # Test 4: Scrape budget
    budget = cm.estimate_scrape_budget(5)
    print(f"\nTest 4 - Scrape Budget:")
    print(f"  For 5 URLs: ~{budget:,} chars per URL")
    print(f"  Total: ~{budget * 5:,} chars")
    print(f"  Est. tokens: ~{(budget * 5) / 4:,.0f}")

    print("\n" + "="*80)


if __name__ == "__main__":
    test_context_manager()