# 🤖 Deep Research Agent

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)](https://www.docker.com/)

A powerful AI-powered research agent that performs multi-step iterative research using local LLMs (via Ollama), web search, and intelligent content analysis. Built with FastAPI, it provides both a REST API and a modern web interface for conducting deep research on any topic.

## ✨ Features

### 🔍 Deep Research
- **Multi-step iterative research** with intelligent query refinement
- **Autonomous web search** using SerpAPI
- **Content scraping and analysis** from search results
- **Real-time progress streaming** via Server-Sent Events (SSE)
- **Context-aware analysis** using LLMs
- **Comprehensive markdown reports** with sources

### 📊 Financial Research
- **Real-time stock price tracking** via yfinance/Yahoo Finance
- **News sentiment analysis** with temporal trends
- **Multi-source financial data aggregation**
- **Interactive charts** (sentiment timeline, distribution)
- **Structured financial reports** with actionable insights

### 💬 Chat Interface
- **Multi-modal chat** with optional features:
  - 🧠 **Show Thinking** - See reasoning step-by-step
  - 🔍 **Web Search** - Search internet for current info
  - 🔬 **Deep Research** - Multi-step iterative research
- **Always-on utilities**: Math, dates, time, text operations, logic

### 🎨 Modern Web UI
- **Beautiful responsive interface** with dark/light themes
- **Real-time progress tracking** with live indicators
- **Run/Stop button controls** for all research tasks
- **Expandable process logs** with accordion UI
- **Download reports** as markdown files
- **Interactive charts** for financial data

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Web Browser                        │
│            (http://localhost:8180)                   │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│              Research Web Client                     │
│         (Python HTTP Server - Port 8180)             │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│              Research API Server                     │
│           (FastAPI - Port 8000)                      │
│  • Deep Research Agent                               │
│  • Financial Research Agent                          │
│  • Chat with multiple modes                          │
│  • SSE streaming for real-time updates              │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│                 Ollama LLM Server                    │
│              (Port 11434 - GPU Accelerated)          │
│           Auto-pulls qwen3:32b on startup            │
└─────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit
- SerpAPI key (free tier available at [serpapi.com](https://serpapi.com))

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/deep-research-agent.git
   cd deep-research-agent
   ```

2. **Configure your SerpAPI key**

   Edit `docker-compose.yaml` and replace the `SERPAPI_KEY` value:
   ```yaml
   environment:
     - SERPAPI_KEY=your_serpapi_key_here
   ```

3. **Start all services**
   ```bash
   docker-compose up -d --build
   ```

   **Note**: First startup will take 10-30 minutes as it pulls the `qwen3:32b` model (20GB+).

4. **Monitor progress**
   ```bash
   # Watch Ollama model pulling
   docker-compose logs -f ollama

   # Check all services
   docker-compose ps
   ```

5. **Access the application**
   - **Web Interface**: http://localhost:8180
   - **API Documentation**: http://localhost:8000/docs
   - **Open WebUI**: http://localhost:8080

### Verify Installation

```bash
# Check if model is loaded
docker exec ollama ollama list

# Test API health
curl http://localhost:8000/health
```

## 📖 Usage

### Web Interface

#### Deep Research Tab
1. Enter your research query
2. Click **▶ Run** to start research
3. Watch real-time progress in the accordion
4. View the comprehensive report with sources
5. Download as markdown

#### Financial Research Tab
1. Enter stock symbol (e.g., AAPL, TSLA, UNH)
2. Enter research query (e.g., "Analyze sentiment and trends")
3. Click **▶ Run**
4. View charts and detailed financial analysis
5. Download the report

#### Chat Tab
1. Enable optional features (Thinking, Web Search, Deep Research)
2. Type your question
3. Click **▶ Run** or press Enter
4. See AI response with optional reasoning

### API Usage

#### Deep Research
```bash
curl -X POST http://localhost:8000/api/research \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Latest developments in quantum computing",
    "max_iterations": 3
  }'
```

#### Financial Research
```bash
curl -X POST http://localhost:8000/api/financial_research/start \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "query": "Analyze recent sentiment and price trends"
  }'
```

#### Chat
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is the current date?"}
    ]
  }'
```

## 🛠️ Configuration

### Environment Variables

Edit `docker-compose.yaml` to customize:

```yaml
environment:
  - OLLAMA_URL=http://ollama:11434
  - SERPAPI_KEY=your_api_key
  - MODEL=qwen3:32b
```

### Adding More Models

Edit `ollama-entrypoint.sh`:
```bash
MODELS=(
    "qwen3:32b"
    "llama3:8b"     # Add more models
    "mistral:7b"
)
```

Then rebuild:
```bash
docker-compose up -d --build ollama
```

## 📁 Project Structure

```
deep-research-agent/
├── api_server.py                 # FastAPI server
├── deep_research_agent.py        # Deep research implementation
├── financial_research_agent.py   # Financial research implementation
├── financial_utils.py            # Financial data utilities
├── search_agent.py               # Web search agent
├── web_utils.py                  # Web scraping utilities
├── context_manager.py            # Context management
├── date_utils.py                 # Date/time utilities
├── example_client.html           # Web UI
├── docker-compose.yaml           # Docker orchestration
├── Dockerfile.api                # API server Dockerfile
├── Dockerfile.web                # Web client Dockerfile
├── Dockerfile.ollama             # Ollama with auto model pull
├── ollama-entrypoint.sh          # Ollama initialization script
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🎯 Use Cases

- **Academic Research**: Deep dive into academic topics with source citations
- **Market Research**: Analyze market trends and competitor insights
- **Financial Analysis**: Track stocks, sentiment, and market movements
- **Technology Scouting**: Stay updated on latest tech developments
- **Due Diligence**: Comprehensive research on companies/products
- **Content Creation**: Research for articles, blogs, and reports

## 🔧 Development

### Local Development (Without Docker)

1. **Install dependencies**
   ```bash
   python -m venv venv
   source venv/bin/activate  # or `venv\Scripts\activate` on Windows
   pip install -r requirements.txt
   ```

2. **Start Ollama separately**
   ```bash
   ollama serve
   ollama pull qwen3:32b
   ```

3. **Run the API server**
   ```bash
   export SERPAPI_KEY=your_key_here
   python api_server.py
   ```

4. **Serve the web client**
   ```bash
   python -m http.server 8180
   ```

### Running Tests

```bash
python test_api.py
```

## 📊 Performance

- **Deep Research**: 30-120 seconds per query (3 iterations)
- **Financial Research**: 20-60 seconds per stock analysis
- **Chat (no tools)**: 2-5 seconds response time
- **Model Loading**: First time ~10-30 minutes, subsequent <10 seconds

## 🐛 Troubleshooting

### Models not pulling
```bash
# Check Ollama logs
docker-compose logs -f ollama

# Manually pull model
docker exec -it ollama ollama pull qwen3:32b
```

### API can't connect to Ollama
```bash
# Check network connectivity
docker exec research-api curl http://ollama:11434/api/tags

# Restart services
docker-compose restart
```

### Out of memory errors
- Reduce model size (use smaller model like llama3:8b)
- Increase Docker memory limits
- Use CPU-only mode (remove GPU sections from docker-compose.yaml)


## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ollama](https://ollama.ai/) - Local LLM inference
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [SerpAPI](https://serpapi.com/) - Web search API
- [smolagents](https://github.com/huggingface/smolagents) - AI agent framework
- [Open WebUI](https://github.com/open-webui/open-webui) - LLM interface

## 📚 Additional Documentation

- [Docker Quick Start Guide](DOCKER_QUICKSTART.md)
- [Context Management](CONTEXT_MANAGEMENT.md)
- [Utilities Documentation](UTILITIES.md)
- [Smolagents Comparison](SMOLAGENTS_COMPARISON.md)
- [Cloudflare Setup](CLOUDFLARE_SETUP.md)

## 📧 Contact

For questions, issues, or suggestions, please open an issue on GitHub.

---

**Built with ❤️ using local LLMs and open-source tools**
