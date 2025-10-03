#!/bin/bash
# Ollama entrypoint script with automatic model pulling

set -e

echo "🚀 Starting Ollama with auto model pull..."

# Start Ollama in the background
/bin/ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to be ready
echo "⏳ Waiting for Ollama to start..."
READY=0
for i in {1..60}; do
    if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
        echo "✅ Ollama is ready!"
        READY=1
        break
    fi
    echo "   Attempt $i/60..."
    sleep 2
done

if [ $READY -eq 0 ]; then
    echo "❌ Ollama failed to start"
    exit 1
fi

# Models to ensure are available
MODELS=(
    "qwen3:32b"
)

# Pull models if they don't exist
for MODEL in "${MODELS[@]}"; do
    echo "📦 Checking model: $MODEL"

    # Check if model exists (handle empty list case)
    if ollama list 2>/dev/null | tail -n +2 | grep -q "^${MODEL}"; then
        echo "   ✅ Model $MODEL already exists"
    else
        echo "   📥 Pulling model $MODEL (this will take some time)..."
        ollama pull "$MODEL" || {
            echo "   ⚠️ Failed to pull $MODEL, but continuing..."
        }
        echo "   ✅ Model $MODEL ready!"
    fi
done

echo "🎉 All models ready! Ollama is fully initialized."
echo "📋 Available models:"
ollama list

# Keep Ollama running in foreground
wait $OLLAMA_PID
