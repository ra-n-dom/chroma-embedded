#!/usr/bin/env bash
# Setup script for chroma-embedded Python environment

set -e

if ! command -v python3 >/dev/null 2>&1; then
    echo "❌ python3 is required but not found in PATH"
    exit 1
fi

echo "🔧 Setting up chroma-embedded Python environment..."
echo

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment with $(python3 --version)"
    python3 -m venv venv
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔌 Activating virtual environment...(runs in subshell; you'll need to activate it again in your own shell)"
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies from requirements.txt
echo "📥 Installing dependencies from requirements.txt..."
echo "  Note: Uses server-side embeddings by default; no local ML models required"

pip install -r requirements.txt

echo
echo "✅ Setup complete!"
echo
echo "Next steps:"
echo "1. Activate the virtual environment: "
echo "     source venv/bin/activate"
echo
echo "2. Run upload.py:"
echo "     python3 upload.py -c MyCollection -i /path/to/files --store source-code -e stella"
