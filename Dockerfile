FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY *.py .
COPY data/ ./data/

# Create directory for ChromaDB
RUN mkdir -p /app/chroma_db

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Expose the port Gradio will run on
EXPOSE 7860

# Create sample data if not exists
RUN python data_processing.py

# Command to run the application
CMD ["python", "recipe_agent.py"]