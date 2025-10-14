#!/bin/bash

# E-commerce Recommendation System Setup Script

echo "🚀 Setting up E-commerce Recommendation System..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip."
    exit 1
fi

echo "✅ Python and pip found"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Download spaCy model
echo "🧠 Downloading spaCy English model..."
python -m spacy download en_core_web_sm

# Copy environment template
if [ ! -f ".env" ]; then
    echo "📝 Creating environment file..."
    cp .env.template .env
    echo "⚠️ Please edit .env file and add your Gemini API key!"
else
    echo "✅ Environment file already exists"
fi

# Check if data file exists
if [ ! -f "data/marketing_sample_for_walmart_com-walmart_com_product_review__20200701_20201231__5k_data.tsv" ]; then
    if [ -f "marketing_sample_for_walmart_com-walmart_com_product_review__20200701_20201231__5k_data.tsv" ]; then
        echo "📂 Moving data file to data directory..."
        mv marketing_sample_for_walmart_com-walmart_com_product_review__20200701_20201231__5k_data.tsv data/
    else
        echo "⚠️ Data file not found. Please ensure the dataset is in the project directory."
    fi
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file and add your Gemini API key"
echo "2. Activate virtual environment: source venv/bin/activate"
echo "3. Start the server: python backend/main.py"
echo "4. Open http://localhost:8000 in your browser"
echo ""
echo "API Documentation will be available at: http://localhost:8000/docs"