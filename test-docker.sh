#!/bin/bash

echo "🐳 Testing Docker Setup for RAG for Law"
echo "======================================"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}Step 1: Checking Docker installation...${NC}"
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed${NC}"
    exit 1
fi

if ! command -v docker compose &> /dev/null; then
    echo -e "${RED}❌ Docker Compose is not available${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Docker and Docker Compose are available${NC}"

echo -e "${YELLOW}Step 2: Checking environment file...${NC}"
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}⚠️  .env file not found. Creating from example...${NC}"
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo -e "${GREEN}✅ Created .env from .env.example${NC}"
        echo -e "${YELLOW}⚠️  Please edit .env with your API keys and tokens${NC}"
    else
        echo -e "${RED}❌ .env.example not found${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✅ .env file exists${NC}"
fi

echo -e "${YELLOW}Step 3: Building Docker image...${NC}"
if docker compose build; then
    echo -e "${GREEN}✅ Docker image built successfully${NC}"
else
    echo -e "${RED}❌ Failed to build Docker image${NC}"
    exit 1
fi

echo -e "${YELLOW}Step 4: Testing container startup...${NC}"
if timeout 30 docker compose run --rm rag-app python -c "print('Container is working!')"; then
    echo -e "${GREEN}✅ Container starts successfully${NC}"
else
    echo -e "${RED}❌ Container failed to start${NC}"
    exit 1
fi

echo -e "${GREEN}🎉 Docker setup is working!${NC}"
echo ""
echo "Next steps:"
echo "1. Edit .env with your API keys"
echo "2. Run: docker compose up --build"
echo "3. Or run: docker compose --profile bot up -d --build"