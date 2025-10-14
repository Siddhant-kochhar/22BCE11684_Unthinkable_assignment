#!/bin/bash

# E-commerce Recommendation System - Run Script

echo "🚀 Starting E-commerce Recommendation System..."

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment is active: $VIRTUAL_ENV"
else
    echo "⚠️ Virtual environment not detected. Activating..."
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
        echo "✅ Virtual environment activated"
    else
        echo "❌ Virtual environment not found. Please run ./setup.sh first"
        exit 1
    fi
fi

# Check if required files exist
if [ ! -f "fastapi_app.py" ]; then
    echo "❌ fastapi_app.py not found!"
    exit 1
fi

if [ ! -f "clean_data.csv" ]; then
    echo "❌ clean_data.csv not found!"
    exit 1
fi

echo "✅ All required files found"

# Start the server
echo "🌐 Starting FastAPI server..."
echo "📱 Frontend will be available at: http://localhost:8000"
echo "📚 API Documentation at: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

uvicorn fastapi_app:app --host 0.0.0.0 --port 8000 --reload