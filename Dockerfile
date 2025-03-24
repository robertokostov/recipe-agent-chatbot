# Use the correct Python version required by your final app
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies (if still needed for any Python packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
# Ensure requirements.txt specifies compatible versions, especially Gradio 5.x
RUN pip install --no-cache-dir -r requirements.txt

# Copy all Python application code files
COPY *.py .
# Copy data directory if it contains essential files needed at runtime
# COPY data/ ./data/ # Keep if needed, otherwise remove

# Remove unnecessary directory creation for in-memory DB
# RUN mkdir -p /app/chroma_db

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Ensure GOOGLE_API_KEY is set, either here or via runtime environment variables:
# ENV GOOGLE_API_KEY="your_key_here" # Alternatively, pass at runtime e.g. with docker run -e ...

# Expose the port Gradio will run on
EXPOSE 7860

# Remove command to run non-existent data processing script
# RUN python data_processing.py

# Command to run the main application script
CMD ["python", "main.py"]