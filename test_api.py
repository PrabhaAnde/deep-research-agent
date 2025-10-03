#!/usr/bin/env python3
"""
Test script for Deep Research API
Demonstrates various ways to use the API
"""

import requests
import json
import time
import sseclient  # pip install sseclient-py


API_BASE = "http://localhost:8000"


def test_simple_research():
    """Test simple research request (polling)"""
    print("\n" + "="*80)
    print("TEST 1: Simple Research Request (Polling)")
    print("="*80)

    # Start research
    response = requests.post(
        f"{API_BASE}/api/research",
        json={"query": "What is retrieval augmented generation in AI?"}
    )

    result = response.json()
    task_id = result["task_id"]
    print(f"Task ID: {task_id}")
    print(f"Status: {result['status']}")

    # Poll for completion
    while True:
        time.sleep(2)

        status_response = requests.get(f"{API_BASE}/api/research/{task_id}")
        status_data = status_response.json()

        print(f"Status: {status_data['status']}")

        if status_data["status"] == "completed":
            print("\n✅ Research completed!")
            print("\n" + "="*80)
            print("REPORT:")
            print("="*80)
            print(status_data["result"]["final_report"])
            print(f"\nSources: {len(status_data['result']['sources'])}")
            print(f"Duration: {status_data['result']['duration_seconds']:.1f}s")
            break
        elif status_data["status"] == "failed":
            print(f"❌ Research failed: {status_data.get('error')}")
            break


def test_streaming_research():
    """Test research with SSE streaming"""
    print("\n" + "="*80)
    print("TEST 2: Streaming Research (Server-Sent Events)")
    print("="*80)

    # Start research
    response = requests.post(
        f"{API_BASE}/api/research",
        json={
            "query": "Latest developments in large language models 2025",
            "max_iterations": 2
        }
    )

    result = response.json()
    task_id = result["task_id"]
    print(f"Task ID: {task_id}")

    # Stream progress
    print("\nStreaming progress...\n")

    stream_url = f"{API_BASE}/api/research/{task_id}/stream"
    response = requests.get(stream_url, stream=True, headers={"Accept": "text/event-stream"})

    client = sseclient.SSEClient(response)

    for event in client.events():
        data = json.loads(event.data)

        if data["type"] == "status":
            print(f"📊 Status: {data['status']}")
        elif data["type"] == "step":
            print(f"  [{data['step_number']}] {data['action'].upper()}: {data['description']}")
        elif data["type"] == "complete":
            print("\n✅ Research completed!")
            print("\n" + "="*80)
            print("REPORT:")
            print("="*80)
            print(data["result"]["final_report"])
            break
        elif data["type"] == "error":
            print(f"❌ Error: {data['error']}")
            break


def test_chat_simple():
    """Test simple chat without research"""
    print("\n" + "="*80)
    print("TEST 3: Simple Chat (No Research)")
    print("="*80)

    response = requests.post(
        f"{API_BASE}/api/chat",
        json={
            "messages": [
                {"role": "user", "content": "What is Python?"}
            ],
            "enable_research": False
        }
    )

    result = response.json()
    print(f"Assistant: {result['content']}")


def test_chat_with_research():
    """Test chat with research enabled"""
    print("\n" + "="*80)
    print("TEST 4: Chat with Deep Research")
    print("="*80)

    print("Sending query with research enabled...")

    response = requests.post(
        f"{API_BASE}/api/chat",
        json={
            "messages": [
                {"role": "user", "content": "What are the main differences between GPT-4 and Claude 3?"}
            ],
            "enable_research": True
        }
    )

    result = response.json()
    print("\n" + "="*80)
    print("RESPONSE:")
    print("="*80)
    print(result["content"])

    if "metadata" in result:
        print(f"\nMetadata:")
        print(f"  Type: {result['metadata']['type']}")
        if "sources" in result["metadata"]:
            print(f"  Sources: {len(result['metadata']['sources'])}")
            print(f"  Duration: {result['metadata']['duration_seconds']:.1f}s")


def test_health():
    """Test health endpoint"""
    print("\n" + "="*80)
    print("TEST: Health Check")
    print("="*80)

    response = requests.get(f"{API_BASE}/health")
    result = response.json()

    print(f"Status: {result['status']}")
    print(f"Ollama: {result['ollama']}")
    print(f"Model: {result['model']}")


def test_list_tasks():
    """Test list all tasks"""
    print("\n" + "="*80)
    print("TEST: List All Tasks")
    print("="*80)

    response = requests.get(f"{API_BASE}/api/tasks")
    result = response.json()

    print(f"Total tasks: {result['total']}")
    for task in result["tasks"]:
        print(f"  - {task['task_id'][:8]}... | {task['status']} | {task['query'][:50]}")


if __name__ == "__main__":
    import sys

    print("="*80)
    print("🧪 Deep Research API Test Suite")
    print("="*80)
    print("Make sure the API server is running: python api_server.py")
    print("="*80)

    # Test health first
    try:
        test_health()
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        print("Make sure the server is running!")
        sys.exit(1)

    if len(sys.argv) > 1:
        test_name = sys.argv[1]

        tests = {
            "simple": test_simple_research,
            "stream": test_streaming_research,
            "chat": test_chat_simple,
            "research-chat": test_chat_with_research,
            "list": test_list_tasks
        }

        if test_name in tests:
            tests[test_name]()
        else:
            print(f"Unknown test: {test_name}")
            print(f"Available tests: {', '.join(tests.keys())}")
    else:
        print("\nUsage:")
        print("  python test_api.py <test_name>")
        print("\nAvailable tests:")
        print("  simple         - Simple research with polling")
        print("  stream         - Research with SSE streaming")
        print("  chat           - Simple chat without research")
        print("  research-chat  - Chat with deep research")
        print("  list           - List all tasks")
        print("\nExample:")
        print("  python test_api.py stream")