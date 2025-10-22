# Base image: official Python slim to reduce size
FROM python:3.10-slim

# Set environment
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create app directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for better caching
COPY requirements.txt /app/requirements.txt

# Install pip-tools to compile smaller sets (optional)
RUN pip install --upgrade pip setuptools wheel

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy project
COPY . /app

# Expose dashboard port
EXPOSE 8050

# Default command: run the monitoring dashboard
CMD ["python", "main.py", "--mode", "monitor"]
