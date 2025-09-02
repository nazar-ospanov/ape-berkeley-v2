# Use Python 3.11 slim image for smaller size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TZ=America/Los_Angeles

# Install system dependencies and Chromium (better ARM64 support)
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    unzip \
    curl \
    xvfb \
    ca-certificates \
    chromium \
    chromium-driver \
    tzdata \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for compatibility
RUN ln -sf /usr/bin/chromium /usr/bin/google-chrome

# Set display port to avoid crash
ENV DISPLAY=:99

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create fresh memory file on each build
RUN touch /app/memory.txt && echo "# Agent Memory - Fresh Start" > /app/memory.txt

# Expose the port the app runs on
EXPOSE 3000

# Command to run the application
CMD ["python", "__main__.py"]
