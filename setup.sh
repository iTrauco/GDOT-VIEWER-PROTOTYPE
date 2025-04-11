#!/bin/bash
# Simple setup script for GDOT Stream Viewer

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== GDOT Stream Viewer Setup ===${NC}"
echo

# Create required directories
echo -e "Creating directories..."
mkdir -p recordings
mkdir -p scripts

# Check for required tools
echo -e "\nChecking required tools..."

# Check for Python
if command -v python3 &>/dev/null; then
    echo -e "${GREEN}✓${NC} Python 3 found"
else
    echo -e "${RED}✗${NC} Python 3 not found. Please install Python 3 to use this application."
    exit 1
fi

# Check for FFmpeg
if command -v ffmpeg &>/dev/null; then
    echo -e "${GREEN}✓${NC} FFmpeg found"
    
    # Check if FFmpeg has SSL support
    ffmpeg_ssl=$(ffmpeg -hide_banner -protocols 2>&1 | grep https)
    if [[ -z "$ffmpeg_ssl" ]]; then
        echo -e "${YELLOW}⚠${NC} Warning: Your FFmpeg installation lacks SSL support."
        echo -e "   The script will use the curl+ffmpeg method instead of direct FFmpeg."
    else
        echo -e "${GREEN}✓${NC} FFmpeg has SSL support"
    fi
else
    echo -e "${RED}✗${NC} FFmpeg not found. Please install FFmpeg to record streams."
    echo -e "   Installation instructions: https://ffmpeg.org/download.html"
    exit 1
fi

# Check for curl
if command -v curl &>/dev/null; then
    echo -e "${GREEN}✓${NC} curl found"
else
    echo -e "${RED}✗${NC} curl not found. Please install curl to download streams."
    exit 1
fi

# Create files from existing templates
echo -e "\nSetting up files..."

# Make record.sh executable
chmod +x record.sh

echo -e "\n${GREEN}Setup complete!${NC}"
echo -e "To start the server, run: ${YELLOW}python3 server.py${NC}"
echo -e "Then open your browser and navigate to: ${YELLOW}http://localhost:8000${NC}"