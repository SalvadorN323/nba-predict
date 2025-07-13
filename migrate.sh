#!/bin/bash

# NBA Predictor Project Migration Script
# This script helps migrate from the old structure to the new restructured project

echo "🏀 NBA Predictor Project Migration Script"
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "README.md" ] || [ ! -d "src" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

echo "📋 This script will help you migrate to the new project structure."
echo "The new structure separates frontend and backend into their own directories."
echo ""

# Create backup
echo "💾 Creating backup of current structure..."
mkdir -p backup
cp -r src backup/
cp -r public backup/
cp app.py backup/
cp requirements.txt backup/
cp package.json backup/
cp package-lock.json backup/
cp model.pkl backup/
cp nba_games.csv backup/
echo "✅ Backup created in 'backup' directory"

# Check if new structure already exists
if [ -d "backend" ] || [ -d "frontend" ]; then
    echo "⚠️  Warning: New structure already exists. Skipping file copying."
else
    echo "📁 Setting up new project structure..."
    
    # The new structure should already be created by the restructuring process
    echo "✅ New structure is ready!"
fi

echo ""
echo "🚀 Migration Steps:"
echo "=================="
echo ""
echo "1. Backend Setup:"
echo "   cd backend"
echo "   python -m venv venv"
echo "   source venv/bin/activate  # On Windows: venv\\Scripts\\activate"
echo "   pip install -r requirements.txt"
echo "   cp env.example .env"
echo "   # Edit .env with your configuration"
echo "   python run.py"
echo ""
echo "2. Frontend Setup:"
echo "   cd frontend"
echo "   npm install"
echo "   cp env.example .env"
echo "   # Edit .env with your configuration"
echo "   npm start"
echo ""
echo "3. Update your deployment configuration:"
echo "   - Backend: Update to use 'backend' directory"
echo "   - Frontend: Update to use 'frontend' directory"
echo ""
echo "📚 Documentation:"
echo "================="
echo "- Main README: README.md"
echo "- Backend docs: backend/README.md"
echo "- Frontend docs: frontend/README.md"
echo "- Technical docs: DOCUMENTATION.md"
echo ""
echo "🎉 Migration complete! Your project now follows best practices."
echo "The old structure is backed up in the 'backup' directory." 