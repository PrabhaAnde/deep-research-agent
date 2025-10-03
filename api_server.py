#!/usr/bin/env python3
"""
FastAPI Server for Deep Research Agent
Provides REST API and SSE streaming for real-time progress updates
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
import json
import uuid
from datetime import datetime
import asyncio
from queue import Queue
import threading
import os

from deep_research_agent import DeepResearchAgent, ResearchReport, ResearchStep
from financial_research_agent import FinancialResearchAgent


# Configuration
SERPAPI_KEY = os.getenv("SERPAPI_KEY", "xxxxxxx")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL = os.getenv("MODEL", "qwen3:32b")


# Pydantic models for API
class ResearchRequest(BaseModel):
    query: str = Field(..., description="Research query", min_length=3)
    max_iterations: Optional[int] = Field(3, description="Maximum research iterations", ge=1, le=5)
    max_searches_per_iteration: Optional[int] = Field(3, description="Max searches per iteration", ge=1, le=5)


class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="Conversation history")
    enable_research: Optional[bool] = Field(False, description="Enable deep research for this message")


class FinancialResearchRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol (e.g., 'AAPL', 'UNH')", min_length=1, max_length=10)
    query: str = Field(..., description="Research query about the stock", min_length=3)
    reference_date: Optional[str] = Field(None, description="Reference date for analysis (ISO format: YYYY-MM-DD)")


class NewsSearchRequest(BaseModel):
    query: str = Field(..., description="News search query", min_length=2)
    gl: Optional[str] = Field("us", description="Country code")
    hl: Optional[str] = Field("en", description="Language code")


class AISearchRequest(BaseModel):
    query: str = Field(..., description="General search query", min_length=2)


class EnhancedSearchRequest(BaseModel):
    query: str = Field(..., description="Search query", min_length=2)
    show_thinking: bool = Field(True, description="Show reasoning process")


class ResearchStatusResponse(BaseModel):
    task_id: str
    status: str  # "pending", "running", "completed", "failed"
    progress: Optional[str] = None
    result: Optional[Dict] = None
    error: Optional[str] = None


# Initialize FastAPI
app = FastAPI(
    title="Deep Research API",
    description="Multi-step iterative research API using local Ollama and SerpAPI",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# In-memory storage for research tasks
research_tasks: Dict[str, Dict] = {}
research_queues: Dict[str, Queue] = {}

# Financial research task tracking
financial_tasks: Dict[str, Dict] = {}
financial_queues: Dict[str, Queue] = {}


def create_agent(max_iterations: int = 3, max_searches: int = 3, model: str = MODEL) -> DeepResearchAgent:
    """Create a new research agent instance with context management"""
    return DeepResearchAgent(
        serpapi_key=SERPAPI_KEY,
        ollama_url=OLLAMA_URL,
        model=model,
        max_iterations=max_iterations,
        max_searches_per_iteration=max_searches,
        max_context=40960  # Qwen3:32b context window
    )


def create_financial_agent(model: str = MODEL) -> FinancialResearchAgent:
    """Create a new financial research agent instance"""
    return FinancialResearchAgent(
        serpapi_key=SERPAPI_KEY,
        ollama_url=OLLAMA_URL,
        model=model,
        max_context=40960
    )


def progress_callback(task_id: str, step: ResearchStep):
    """Callback to push progress updates to queue"""
    if task_id in research_queues:
        queue = research_queues[task_id]
        step_data = {
            "type": "step",
            "step_number": step.step_number,
            "action": step.action,
            "description": step.description,
            "timestamp": step.timestamp
        }
        queue.put(step_data)


def run_research_task(task_id: str, query: str, max_iterations: int, max_searches: int):
    """Background task to run research"""
    try:
        # Update status
        research_tasks[task_id]["status"] = "running"
        research_tasks[task_id]["started_at"] = datetime.now().isoformat()

        # Create agent
        agent = create_agent(max_iterations, max_searches)

        # Run research with progress callback
        report = agent.research(
            query,
            progress_callback=lambda step: progress_callback(task_id, step)
        )

        # Update with result
        research_tasks[task_id]["status"] = "completed"
        research_tasks[task_id]["completed_at"] = datetime.now().isoformat()
        research_tasks[task_id]["result"] = {
            "query": report.query,
            "final_report": report.final_report,
            "sources": report.sources,
            "duration_seconds": report.duration_seconds,
            "total_searches": report.total_searches,
            "total_urls_scraped": report.total_urls_scraped,
            "steps": [
                {
                    "step_number": s.step_number,
                    "action": s.action,
                    "description": s.description,
                    "timestamp": s.timestamp
                }
                for s in report.steps
            ]
        }

        # Signal completion in queue
        if task_id in research_queues:
            research_queues[task_id].put({"type": "complete", "result": research_tasks[task_id]["result"]})

    except Exception as e:
        research_tasks[task_id]["status"] = "failed"
        research_tasks[task_id]["error"] = str(e)
        research_tasks[task_id]["completed_at"] = datetime.now().isoformat()

        if task_id in research_queues:
            research_queues[task_id].put({"type": "error", "error": str(e)})


@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "name": "Deep Research API",
        "version": "1.0.0",
        "endpoints": {
            "research": {
                "POST /api/research": "Start deep research (returns task_id)",
                "GET /api/research/{task_id}": "Get research status and result",
                "GET /api/research/{task_id}/stream": "Stream research progress (SSE)"
            },
            "financial": {
                "POST /api/financial_research": "Financial stock research with sentiment analysis"
            },
            "search": {
                "POST /api/news_search": "Search Google News via SerpAPI",
                "POST /api/ai_search": "AI-powered search via SerpAPI",
                "POST /api/enhanced_search": "Web search + scraping + LLM analysis"
            },
            "chat": {
                "POST /api/chat": "Chat endpoint with optional research"
            },
            "models": {
                "GET /api/models": "List available Ollama models"
            },
            "health": {
                "GET /health": "Health check"
            }
        }
    }


@app.get("/favicon.ico")
async def favicon():
    """Return 204 for favicon to prevent 404 logs"""
    from fastapi.responses import Response
    return Response(status_code=204)


@app.get("/example_client.html")
async def serve_client():
    """Serve the example client HTML file"""
    html_path = os.path.join(os.path.dirname(__file__), "example_client.html")
    if os.path.exists(html_path):
        return FileResponse(html_path, media_type="text/html")
    raise HTTPException(status_code=404, detail="Client HTML not found")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    # Test Ollama connection
    try:
        import requests
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        ollama_status = "healthy" if response.status_code == 200 else "unhealthy"
    except:
        ollama_status = "unreachable"

    return {
        "status": "healthy",
        "ollama": ollama_status,
        "model": MODEL,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/research", response_model=ResearchStatusResponse)
async def start_research(request: ResearchRequest, background_tasks: BackgroundTasks):
    """
    Start a deep research task

    Returns immediately with a task_id. Use /api/research/{task_id} to get status,
    or /api/research/{task_id}/stream for real-time progress.
    """
    task_id = str(uuid.uuid4())

    # Initialize task
    research_tasks[task_id] = {
        "task_id": task_id,
        "query": request.query,
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "started_at": None,
        "completed_at": None,
        "result": None,
        "error": None
    }

    # Create queue for progress updates
    research_queues[task_id] = Queue()

    # Start background task
    background_tasks.add_task(
        run_research_task,
        task_id,
        request.query,
        request.max_iterations,
        request.max_searches_per_iteration
    )

    return ResearchStatusResponse(
        task_id=task_id,
        status="pending",
        progress="Research task queued"
    )


@app.get("/api/research/{task_id}", response_model=ResearchStatusResponse)
async def get_research_status(task_id: str):
    """
    Get research task status and result

    Status values: "pending", "running", "completed", "failed"
    """
    if task_id not in research_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = research_tasks[task_id]

    return ResearchStatusResponse(
        task_id=task_id,
        status=task["status"],
        progress=f"Task {task['status']}",
        result=task.get("result"),
        error=task.get("error")
    )


@app.get("/api/research/{task_id}/stream")
async def stream_research_progress(task_id: str):
    """
    Stream research progress using Server-Sent Events (SSE)

    Connect to this endpoint to receive real-time progress updates as the research proceeds.
    """
    if task_id not in research_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    if task_id not in research_queues:
        raise HTTPException(status_code=400, detail="Streaming not available for this task")

    async def event_generator():
        """Generate SSE events from queue"""
        queue = research_queues[task_id]

        # Send initial status
        yield f"data: {json.dumps({'type': 'status', 'status': research_tasks[task_id]['status']})}\n\n"

        # Stream progress updates
        last_heartbeat = datetime.now()
        while True:
            # Non-blocking queue check
            if not queue.empty():
                try:
                    event = queue.get_nowait()

                    # Send event
                    yield f"data: {json.dumps(event)}\n\n"

                    # Check if complete
                    if event.get("type") in ["complete", "error"]:
                        break

                except:
                    pass

            # Check if task completed or failed
            task_status = research_tasks[task_id]["status"]
            if task_status in ["completed", "failed"]:
                # Send final status
                if queue.empty():
                    final_event = {
                        "type": "complete" if task_status == "completed" else "error",
                        "status": task_status,
                        "result": research_tasks[task_id].get("result"),
                        "error": research_tasks[task_id].get("error")
                    }
                    yield f"data: {json.dumps(final_event)}\n\n"
                    break

            # Heartbeat every 10 seconds to keep proxies alive
            now = datetime.now()
            if (now - last_heartbeat).total_seconds() >= 10:
                try:
                    yield f"data: {json.dumps({'type': 'heartbeat', 'ts': now.isoformat()})}\n\n"
                except Exception:
                    pass
                last_heartbeat = now

            await asyncio.sleep(0.5)

        # Cleanup
        if task_id in research_queues:
            del research_queues[task_id]

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Chat endpoint with optional deep research

    If enable_research=True, the assistant will perform deep research for the user's query.
    Otherwise, it acts as a simple chat interface.
    """
    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages cannot be empty")

    last_message = request.messages[-1]

    if last_message.role != "user":
        raise HTTPException(status_code=400, detail="Last message must be from user")

    user_query = last_message.content

    # If research is enabled, run deep research
    if request.enable_research:
        agent = create_agent()
        report = agent.research(user_query)

        return {
            "role": "assistant",
            "content": report.final_report,
            "metadata": {
                "type": "research",
                "sources": report.sources,
                "duration_seconds": report.duration_seconds,
                "total_searches": report.total_searches,
                "total_urls_scraped": report.total_urls_scraped
            }
        }
    else:
        # Simple chat without research
        from deep_research_agent import OllamaClient

        llm = OllamaClient(OLLAMA_URL, MODEL)

        # Convert to Ollama format
        messages = [{"role": m.role, "content": m.content} for m in request.messages]

        response = llm.chat(messages)

        return {
            "role": "assistant",
            "content": response,
            "metadata": {
                "type": "chat"
            }
        }


@app.post("/api/financial_research")
async def financial_research(request: FinancialResearchRequest):
    """
    Financial stock research with sentiment analysis

    Performs comprehensive stock research including:
    - Current price data
    - Sentiment analysis from news articles
    - Historical trends
    - Structured data for visualizations

    Returns both narrative report and structured data for charts.
    """
    try:
        from datetime import datetime

        # Parse reference date if provided
        if request.reference_date:
            try:
                reference_date = datetime.fromisoformat(request.reference_date)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
        else:
            reference_date = datetime.now()

        # Create financial agent (always uses qwen3:32b)
        agent = create_financial_agent(model=MODEL)

        # Run research
        report, structured_data = agent.research_stock(
            symbol=request.symbol,
            query=request.query,
            reference_date=reference_date
        )

        return {
            "status": "success",
            "symbol": request.symbol,
            "query": request.query,
            "reference_date": reference_date.isoformat(),
            "model": MODEL,
            "report": report,
            "structured_data": structured_data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Research failed: {str(e)}")


# -------------------------------
# Financial Research (Async + SSE)
# -------------------------------

def financial_progress(task_id: str, event: Dict):
    """Push a progress event into the financial task queue"""
    if task_id in financial_queues:
        try:
            financial_queues[task_id].put(event)
        except Exception:
            pass


def run_financial_task(task_id: str, symbol: str, query: str, reference_date: Optional[str] = None):
    """Background task to run financial research with coarse progress updates"""
    try:
        financial_tasks[task_id]["status"] = "running"
        financial_tasks[task_id]["started_at"] = datetime.now().isoformat()

        # Prepare dates
        ref_dt = None
        if reference_date:
            try:
                ref_dt = datetime.fromisoformat(reference_date)
            except Exception:
                ref_dt = datetime.now()
        else:
            ref_dt = datetime.now()

        # Create agent
        agent = create_financial_agent(model=MODEL)

        # Step 1: Price
        financial_progress(task_id, {"type": "step", "step_number": 1, "action": "price", "description": "Fetching current stock price"})
        price_data = agent.financial_api.get_stock_price(symbol)
        financial_progress(task_id, {"type": "step", "step_number": 1, "action": "price", "description": f"Price fetched: ${price_data.get('price', 'N/A')}"})

        # Step 2: Six-month news
        financial_progress(task_id, {"type": "step", "step_number": 2, "action": "search", "description": "Searching news for past 6 months"})
        from datetime import timedelta
        six_months_ago = ref_dt - timedelta(days=180)
        six_month_news = agent.searcher.search_with_date_range(
            f"{symbol} stock news",
            start_date=six_months_ago,
            end_date=ref_dt,
            num_results=15
        )
        financial_progress(task_id, {"type": "step", "step_number": 2, "action": "search", "description": f"Found {len(six_month_news)} six-month articles"})

        # Step 3: Recent news
        financial_progress(task_id, {"type": "step", "step_number": 3, "action": "search", "description": "Searching recent news (past week)"})
        one_week_ago = ref_dt - timedelta(days=7)
        recent_news = agent.searcher.search_with_date_range(
            f"{symbol} stock news",
            start_date=one_week_ago,
            end_date=ref_dt,
            num_results=10
        )
        financial_progress(task_id, {"type": "step", "step_number": 3, "action": "search", "description": f"Found {len(recent_news)} recent articles"})

        # Step 4: Scrape + sentiment (coarse updates)
        financial_progress(task_id, {"type": "step", "step_number": 4, "action": "analyze", "description": "Scraping and analyzing sentiment"})
        analyzed_articles = []
        def analyze_batch(batch, timeframe):
            for idx, article in enumerate(batch, 1):
                scraped = agent.scraper.scrape(article.get('link', ''))
                if scraped.get('success'):
                    sent = agent.sentiment_analyzer.analyze_article(article.get('title', ''), scraped.get('content', ''))
                    analyzed_articles.append({
                        'title': article.get('title', ''),
                        'url': article.get('link', ''),
                        'date': article.get('date', '' if timeframe == '6-month' else 'recent'),
                        'content': scraped.get('content', '')[:3000],
                        'sentiment': sent.get('sentiment', 'neutral'),
                        'sentiment_score': sent.get('score', 0.0),
                        'sentiment_reasoning': sent.get('reasoning', ''),
                        'timeframe': timeframe,
                    })
            # Send periodic update
            financial_progress(task_id, {"type": "step", "step_number": 4, "action": "analyze", "description": f"Analyzed {len(analyzed_articles)} articles so far"})

        analyze_batch(recent_news[:5], 'recent')
        # Avoid duplicates: exclude recent that overlap
        six_month_filtered = [a for a in six_month_news[:10] if a not in recent_news]
        analyze_batch(six_month_filtered, '6-month')

        # Step 5: Aggregate
        financial_progress(task_id, {"type": "step", "step_number": 5, "action": "aggregate", "description": "Aggregating sentiment"})
        overall = agent.sentiment_analyzer.aggregate_sentiment(analyzed_articles)

        # Step 6: Report
        financial_progress(task_id, {"type": "step", "step_number": 6, "action": "report", "description": "Generating final report"})
        report, structured = agent._generate_financial_report(
            symbol=symbol,
            query=query,
            reference_date=ref_dt,
            price_data=price_data,
            articles=analyzed_articles,
            sentiment=overall,
        )

        # Save result
        result = {
            "status": "success",
            "symbol": symbol,
            "query": query,
            "reference_date": ref_dt.isoformat(),
            "model": MODEL,
            "report": report,
            "structured_data": structured,
        }

        financial_tasks[task_id]["status"] = "completed"
        financial_tasks[task_id]["completed_at"] = datetime.now().isoformat()
        financial_tasks[task_id]["result"] = result

        # Notify completion
        financial_progress(task_id, {"type": "complete", "result": result})

    except Exception as e:
        financial_tasks[task_id]["status"] = "failed"
        financial_tasks[task_id]["error"] = str(e)
        financial_tasks[task_id]["completed_at"] = datetime.now().isoformat()
        financial_progress(task_id, {"type": "error", "error": str(e)})


@app.post("/api/financial_research/start", response_model=ResearchStatusResponse)
async def start_financial_research(request: FinancialResearchRequest, background_tasks: BackgroundTasks):
    """Start financial research asynchronously and return task_id"""
    task_id = str(uuid.uuid4())

    financial_tasks[task_id] = {
        "task_id": task_id,
        "symbol": request.symbol,
        "query": request.query,
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "started_at": None,
        "completed_at": None,
        "result": None,
        "error": None,
    }
    financial_queues[task_id] = Queue()

    background_tasks.add_task(
        run_financial_task,
        task_id,
        request.symbol,
        request.query,
        request.reference_date,
    )

    return ResearchStatusResponse(task_id=task_id, status="pending", progress="Financial research queued")


@app.get("/api/financial_research/{task_id}", response_model=ResearchStatusResponse)
async def get_financial_research_status(task_id: str):
    if task_id not in financial_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    t = financial_tasks[task_id]
    return ResearchStatusResponse(
        task_id=task_id,
        status=t["status"],
        progress=f"Task {t['status']}",
        result=t.get("result"),
        error=t.get("error"),
    )


@app.get("/api/financial_research/{task_id}/stream")
async def stream_financial_research(task_id: str):
    if task_id not in financial_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    if task_id not in financial_queues:
        raise HTTPException(status_code=400, detail="Streaming not available for this task")

    async def event_generator():
        queue = financial_queues[task_id]
        # Initial status
        yield f"data: {json.dumps({'type': 'status', 'status': financial_tasks[task_id]['status']})}\n\n"

        last_heartbeat = datetime.now()
        while True:
            if not queue.empty():
                try:
                    ev = queue.get_nowait()
                    yield f"data: {json.dumps(ev)}\n\n"
                    if ev.get('type') in ['complete', 'error']:
                        break
                except Exception:
                    pass

            # If task ended without events
            st = financial_tasks[task_id]["status"]
            if st in ["completed", "failed"]:
                if queue.empty():
                    final_event = {
                        "type": "complete" if st == "completed" else "error",
                        "status": st,
                        "result": financial_tasks[task_id].get("result"),
                        "error": financial_tasks[task_id].get("error"),
                    }
                    yield f"data: {json.dumps(final_event)}\n\n"
                    break

            # Heartbeat every 10s
            now = datetime.now()
            if (now - last_heartbeat).total_seconds() >= 10:
                try:
                    yield f"data: {json.dumps({'type': 'heartbeat', 'ts': now.isoformat()})}\n\n"
                except Exception:
                    pass
                last_heartbeat = now

            await asyncio.sleep(0.5)

        # Cleanup
        if task_id in financial_queues:
            del financial_queues[task_id]

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@app.post("/api/news_search")
async def news_search(request: NewsSearchRequest):
    """
    Search Google News using SerpAPI

    Returns recent news articles for the given query
    """
    try:
        import requests

        # Better query - keep temporal words
        search_query = request.query.replace(' news ', ' ').strip()

        params = {
            "engine": "google_news",
            "q": search_query,
            "gl": request.gl,
            "hl": request.hl,
            "tbm": "nws",  # News tab
            "tbs": "qdr:d",  # Last 24 hours
            "api_key": SERPAPI_KEY
        }

        response = requests.get("https://serpapi.com/search", params=params, timeout=10)

        if response.status_code == 200:
            data = response.json()

            # Extract news results
            news_results = data.get("news_results", [])

            formatted_results = []
            for article in news_results[:10]:  # Top 10 results
                formatted_results.append({
                    "title": article.get("title", ""),
                    "link": article.get("link", ""),
                    "source": article.get("source", {}).get("name", ""),
                    "date": article.get("date", ""),
                    "snippet": article.get("snippet", "")
                })

            return {
                "status": "success",
                "query": search_query,
                "date_filter": "Last 24 hours",
                "results": formatted_results,
                "total": len(formatted_results)
            }
        else:
            raise HTTPException(status_code=500, detail=f"SerpAPI returned status {response.status_code}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"News search failed: {str(e)}")


@app.post("/api/ai_search")
async def ai_search(request: AISearchRequest):
    """
    AI-powered search using SerpAPI's Google AI Mode

    Returns AI-generated summary with sources
    """
    try:
        import requests

        params = {
            "engine": "google_ai_mode",
            "q": request.query,
            "api_key": SERPAPI_KEY
        }

        response = requests.get("https://serpapi.com/search", params=params, timeout=15)

        if response.status_code == 200:
            data = response.json()

            # Extract AI overview and sources
            ai_overview = data.get("ai_overview", {})

            return {
                "status": "success",
                "query": request.query,
                "ai_answer": ai_overview.get("text", ""),
                "sources": ai_overview.get("sources", []),
                "search_results": data.get("organic_results", [])[:5]  # Top 5 organic results
            }
        else:
            raise HTTPException(status_code=500, detail=f"SerpAPI returned status {response.status_code}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI search failed: {str(e)}")


@app.post("/api/enhanced_search")
async def enhanced_search(request: EnhancedSearchRequest):
    """
    Enhanced search: Web search + content scraping + LLM analysis

    Combines:
    1. SerpAPI search (news or general)
    2. Content scraping from top results
    3. LLM analysis with full context
    4. Optional thinking/reasoning display
    """
    try:
        import requests
        from bs4 import BeautifulSoup
        from datetime import datetime

        # Detect query type
        lower_query = request.query.lower()
        news_keywords = ['news', 'latest', 'breaking', 'headlines', 'recent news', 'today news']
        is_news = any(keyword in lower_query for keyword in news_keywords)

        thinking_steps = []
        scraped_content = []

        # Step 1: Perform search
        if is_news:
            thinking_steps.append("🔍 Step 1: Detected news query - using Google News search")

            # Better query construction - keep temporal context
            search_query = request.query
            # Only remove redundant "news" word, keep "latest", "today", "breaking"
            search_query = search_query.replace(' news ', ' ').strip()

            # Add today's date to query for relevance
            current_date_str = datetime.now().strftime("%B %d, %Y")
            if not any(word in lower_query for word in ['today', 'latest', 'recent']):
                search_query = f"{search_query} today"

            params = {
                "engine": "google_news",
                "q": search_query,
                "gl": "us",
                "hl": "en",
                "tbm": "nws",  # News tab
                "tbs": "qdr:d",  # Last 24 hours
                "api_key": SERPAPI_KEY
            }

            thinking_steps.append(f"   📝 Search query: '{search_query}'")
            thinking_steps.append(f"   📅 Date filter: Last 24 hours")

            response = requests.get("https://serpapi.com/search", params=params, timeout=10)
            data = response.json()
            news_results = data.get("news_results", [])[:10]  # Top 10 for news

            thinking_steps.append(f"   ✓ Found {len(news_results)} news articles")

            # Store article titles and URLs for summary
            articles_info = []
            urls = []
            for article in news_results:
                if article.get("link"):
                    urls.append(article.get("link"))
                    articles_info.append({
                        'title': article.get('title', ''),
                        'source': article.get('source', {}).get('name', ''),
                        'date': article.get('date', ''),
                        'snippet': article.get('snippet', ''),
                        'url': article.get('link')
                    })

        else:
            thinking_steps.append("🔍 Step 1: Detected general query - using AI search")

            params = {
                "engine": "google_ai_mode",
                "q": request.query,
                "api_key": SERPAPI_KEY
            }

            response = requests.get("https://serpapi.com/search", params=params, timeout=15)
            data = response.json()

            organic_results = data.get("organic_results", [])[:5]
            urls = [result.get("link") for result in organic_results if result.get("link")]

            thinking_steps.append(f"   ✓ Found {len(urls)} relevant sources")

        # Step 2: Scrape content
        thinking_steps.append(f"\n📄 Step 2: Scraping content from {len(urls)} URLs")

        for i, url in enumerate(urls[:5], 1):  # Limit to 5 URLs
            try:
                thinking_steps.append(f"   [{i}] Scraping {url[:60]}...")

                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                page_response = requests.get(url, headers=headers, timeout=10)

                if page_response.status_code == 200:
                    soup = BeautifulSoup(page_response.content, 'html.parser')

                    # Remove script and style elements
                    for script in soup(["script", "style", "nav", "footer", "header"]):
                        script.decompose()

                    # Get text
                    text = soup.get_text(separator=' ', strip=True)

                    # Clean and limit text
                    text = ' '.join(text.split())[:3000]  # First 3000 chars

                    scraped_content.append({
                        'url': url,
                        'content': text
                    })

                    thinking_steps.append(f"       ✓ Scraped {len(text)} characters")
                else:
                    thinking_steps.append(f"       ✗ Failed (status {page_response.status_code})")

            except Exception as e:
                thinking_steps.append(f"       ✗ Error: {str(e)[:50]}")
                continue

        thinking_steps.append(f"   ✓ Successfully scraped {len(scraped_content)} sources")

        # Step 3: Build LLM context
        thinking_steps.append("\n🤖 Step 3: Building context for LLM analysis")

        current_date = datetime.now().strftime("%Y-%m-%d (%A)")
        current_time = datetime.now().strftime("%H:%M:%S")

        # For news queries, include article metadata
        if is_news and 'articles_info' in locals():
            llm_prompt = f"""Current date: {current_date}
Current time: {current_time}

User Query: {request.query}

NEWS ARTICLES FOUND (Top {len(articles_info)}):

"""
            for i, article in enumerate(articles_info, 1):
                llm_prompt += f"""
Article {i}:
Title: {article['title']}
Source: {article['source']}
Date: {article['date']}
Snippet: {article['snippet']}
URL: {article['url']}
{'='*80}
"""

            if scraped_content:
                llm_prompt += f"\n\nDETAILED CONTENT FROM {len(scraped_content)} ARTICLES:\n\n"
                for i, content in enumerate(scraped_content, 1):
                    llm_prompt += f"\n[Article {i}] {content['url']}\n{content['content'][:1500]}\n{'='*80}\n"

            llm_prompt += f"""

TASK: Based on the above news articles, create a clear bullet-point summary.

Instructions:
1. List the top 10 news items as bullet points
2. Each bullet should have: Title, source, and brief summary
3. Focus on today's news ({current_date})
4. Format clearly and concisely
5. Include article dates if available

{"DO NOT show reasoning - just provide the formatted news list." if not request.show_thinking else "Show your reasoning first, then provide the news list."}
"""

        else:
            # General search context
            llm_prompt = f"""Current date: {current_date}
Current time: {current_time}

Available utilities:
- Math: calculations, algebra, statistics, conversions
- Date/Time: current date, time calculations, date arithmetic
- Text: string operations, formatting, analysis
- Logic: reasoning, comparisons, decision trees

User Query: {request.query}

Scraped Web Content from {len(scraped_content)} sources:

"""

            for i, content in enumerate(scraped_content, 1):
                llm_prompt += f"\n[Source {i}] {content['url']}\n{content['content'][:2000]}\n{'='*80}\n"

            llm_prompt += f"""

Based on the above scraped content and available utilities, please provide a comprehensive answer to the user's query.

{"Show your reasoning process step-by-step:" if request.show_thinking else "Provide a clear, concise answer."}
"""

        thinking_steps.append("   ✓ Context prepared with scraped content + utilities")
        thinking_steps.append(f"   ✓ Total context: ~{len(llm_prompt)} characters")

        # Step 4: LLM Analysis
        thinking_steps.append("\n🧠 Step 4: Analyzing with LLM...")

        from deep_research_agent import OllamaClient
        llm = OllamaClient(OLLAMA_URL, MODEL)

        llm_response = llm.chat([{"role": "user", "content": llm_prompt}])

        thinking_steps.append("   ✓ Analysis complete")

        # Format final response
        final_response = ""

        if request.show_thinking:
            final_response += "## 🧠 Thinking Process\n\n"
            final_response += "\n".join(thinking_steps)
            final_response += "\n\n---\n\n"

        final_response += "## 📊 Analysis\n\n"
        final_response += llm_response

        final_response += "\n\n---\n\n## 🔗 Sources\n\n"
        for i, content in enumerate(scraped_content, 1):
            final_response += f"{i}. [{content['url']}]({content['url']})\n"

        return {
            "status": "success",
            "query": request.query,
            "response": final_response,
            "sources_count": len(scraped_content),
            "model": MODEL
        }

    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"Enhanced search failed: {str(e)}\n{traceback.format_exc()}")


@app.get("/api/models")
async def list_models():
    """
    List available Ollama models

    Returns list of models that can be used for research.
    """
    try:
        import requests
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)

        if response.status_code == 200:
            data = response.json()
            models = [
                {
                    "name": model["name"],
                    "size": model.get("size", 0),
                    "modified": model.get("modified_at", "")
                }
                for model in data.get("models", [])
            ]
            return {
                "status": "success",
                "models": models,
                "default_model": MODEL
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to fetch models from Ollama")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching models: {str(e)}")


@app.delete("/api/research/{task_id}")
async def delete_research_task(task_id: str):
    """Delete a research task"""
    if task_id not in research_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    # Cleanup
    del research_tasks[task_id]
    if task_id in research_queues:
        del research_queues[task_id]

    return {"status": "deleted", "task_id": task_id}


@app.get("/api/tasks")
async def list_tasks():
    """List all research tasks"""
    return {
        "tasks": [
            {
                "task_id": task_id,
                "query": task["query"],
                "status": task["status"],
                "created_at": task["created_at"]
            }
            for task_id, task in research_tasks.items()
        ],
        "total": len(research_tasks)
    }


if __name__ == "__main__":
    import uvicorn

    print("="*80)
    print("🚀 Starting Deep Research API Server")
    print("="*80)
    print(f"Ollama URL: {OLLAMA_URL}")
    print(f"Model: {MODEL}")
    print(f"\nAPI will be available at: http://localhost:8000")
    print(f"API docs: http://localhost:8000/docs")
    print("="*80)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
