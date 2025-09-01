# Makefile for A2A Math Tool Agent

# Docker image name
IMAGE_NAME := ape-berkeley-v2-agent
CONTAINER_NAME := ape-berkeley-v2-container

# Default target
.DEFAULT_GOAL := help

# Build the Docker image
.PHONY: build
build:
	@echo "🏗️  Building Docker image..."
	docker build -t $(IMAGE_NAME) .
	@echo "✅ Build complete!"

# Rebuild the Docker image with no cache
.PHONY: rebuild
rebuild:
	@echo "🔄 Rebuilding Docker image (no cache)..."
	docker build --no-cache -t $(IMAGE_NAME) .
	@echo "✅ Rebuild complete!"

# Run the container (foreground, with volume mounting for development)
.PHONY: run
run:
	@echo "🚀 Starting A2A Math Tool Agent..."
	@echo "📍 Agent will be available at http://localhost:3000"
	@echo "🛑 Press Ctrl+C to stop the container"
	@echo ""
	docker run --rm -it \
		--name $(CONTAINER_NAME) \
		-p 3000:3000 \
		-v $(PWD):/app \
		--env-file .env \
		$(IMAGE_NAME)

# Stop the container (if running)
.PHONY: stop
stop:
	@echo "🛑 Stopping container..."
	@docker stop $(CONTAINER_NAME) 2>/dev/null || echo "Container not running"

# Clean up Docker image
.PHONY: clean
clean:
	@echo "🧹 Cleaning up Docker image..."
	@docker rmi $(IMAGE_NAME) 2>/dev/null || echo "Image not found"
	@docker system prune -f

# Show Docker logs
.PHONY: logs
logs:
	docker logs $(CONTAINER_NAME)

# Show running containers
.PHONY: ps
ps:
	docker ps --filter name=$(CONTAINER_NAME)

# Help target
.PHONY: help
help:
	@echo "🧮 A2A Math Tool Agent - Docker Commands"
	@echo "========================================"
	@echo ""
	@echo "Available commands:"
	@echo "  make build    - Build the Docker image"
	@echo "  make rebuild  - Rebuild the Docker image (no cache)"
	@echo "  make run      - Run the agent (foreground, Ctrl+C to stop)"
	@echo "  make stop     - Stop the running container"
	@echo "  make clean    - Remove Docker image and cleanup"
	@echo "  make logs     - Show container logs"
	@echo "  make ps       - Show running containers"
	@echo "  make help     - Show this help message"
	@echo ""
	@echo "📋 Quick Start:"
	@echo "  1. Create .env file with OPENAI_API_KEY=your_key"
	@echo "  2. make build"
	@echo "  3. make run"
	@echo ""
	@echo "🔧 Development Notes:"
	@echo "  - Code changes are live-mounted (no rebuild needed)"
	@echo "  - Only rebuild when requirements.txt changes"
	@echo "  - Agent runs on http://localhost:3000"
