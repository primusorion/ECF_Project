#!/bin/bash

# Quick Start Script for Predictive Maintenance System
# This script sets up and starts all services

set -e

echo "🚀 Predictive Maintenance System - Quick Start"
echo "================================================"
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

echo "✅ Docker is running"
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python
if command_exists python3; then
    PYTHON_CMD=python3
elif command_exists python; then
    PYTHON_CMD=python
else
    echo "❌ Python not found. Please install Python 3.9+"
    exit 1
fi

echo "✅ Python found: $($PYTHON_CMD --version)"
echo ""

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p models/saved_models models/edge_optimized
mkdir -p data/raw data/processed data/synthetic
mkdir -p logs config/grafana/dashboards config/grafana/datasources
mkdir -p config/ssl

echo "✅ Directories created"
echo ""

# Install Python dependencies (optional - for local development)
read -p "📦 Install Python dependencies? (y/n): " install_deps
if [ "$install_deps" = "y" ]; then
    echo "Installing dependencies..."
    $PYTHON_CMD -m pip install --upgrade pip
    $PYTHON_CMD -m pip install -r requirements.txt
    echo "✅ Dependencies installed"
    echo ""
fi

# Generate sample data and train models (optional)
read -p "🎲 Generate sample data and train models? (y/n): " gen_data
if [ "$gen_data" = "y" ]; then
    echo "Generating sample data..."
    $PYTHON_CMD main.py --mode generate --samples 10000
    
    echo "Training models..."
    $PYTHON_CMD main.py --mode train
    echo "✅ Models trained"
    echo ""
fi

# Choose deployment mode
echo "🐳 Choose deployment mode:"
echo "1) Full stack (API + Dashboard + Database + MQTT + Monitoring)"
echo "2) Minimal (API + Dashboard only)"
echo "3) Inference only (optimized edge deployment)"
read -p "Select mode (1/2/3): " mode

case $mode in
    1)
        echo "Starting full stack..."
        docker-compose -f docker-compose.full.yml up -d
        ;;
    2)
        echo "Starting minimal stack..."
        docker-compose up -d
        ;;
    3)
        echo "Starting inference stack..."
        docker-compose -f docker-compose.inference.yml up -d
        ;;
    *)
        echo "Invalid option. Exiting."
        exit 1
        ;;
esac

echo ""
echo "⏳ Waiting for services to start..."
sleep 10

# Check service health
echo ""
echo "🔍 Checking service health..."
echo ""

# Function to check HTTP service
check_http() {
    local url=$1
    local name=$2
    if curl -s -f -o /dev/null "$url"; then
        echo "✅ $name is running - $url"
    else
        echo "⚠️  $name not responding yet - $url"
    fi
}

check_http "http://localhost:8000/health" "API Server"
check_http "http://localhost:8050" "Dashboard"

if [ "$mode" = "1" ]; then
    check_http "http://localhost:3000" "Grafana"
    check_http "http://localhost:9090" "Prometheus"
    check_http "http://localhost:5000" "MLflow"
fi

echo ""
echo "================================================"
echo "🎉 Predictive Maintenance System Started!"
echo "================================================"
echo ""
echo "📊 Access Points:"
echo "  API:            http://localhost:8000"
echo "  API Docs:       http://localhost:8000/docs"
echo "  Dashboard:      http://localhost:8050"

if [ "$mode" = "1" ]; then
    echo "  Grafana:        http://localhost:3000 (admin/admin)"
    echo "  Prometheus:     http://localhost:9090"
    echo "  MLflow:         http://localhost:5000"
fi

echo ""
echo "📝 Commands:"
echo "  View logs:      docker-compose -f <compose-file> logs -f"
echo "  Stop services:  docker-compose -f <compose-file> down"
echo "  Restart:        docker-compose -f <compose-file> restart"
echo ""
echo "📚 Documentation:"
echo "  Project Overview:  PROJECT_OVERVIEW.md"
echo "  Docker Guide:      DOCKER_RUN_GUIDE.md"
echo "  Improvements:      IMPROVEMENTS_GUIDE.md"
echo "  Getting Started:   GETTING_STARTED.md"
echo ""
echo "🧪 Test the API:"
echo '  curl -X POST "http://localhost:8000/predict" \'
echo '    -H "Content-Type: application/json" \'
echo '    -d '"'"'{"equipment_id":"Motor_001","equipment_type":"Motor","temperature":75.5,"vibration":4.2,"pressure":7.5,"rpm":1800,"power_consumption":35.0,"acoustic_emission":78.3}'"'"
echo ""
echo "Happy predicting! 🚀"
