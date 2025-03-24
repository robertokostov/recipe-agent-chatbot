# config.py
"""Stores configuration constants for the application."""

import os
from dotenv import load_dotenv

# Load .env file if it exists (optional)
try:
    load_dotenv()
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

# --- Core Configuration ---
VECTOR_DB_PATH = "./recipe_vectordb" # Example path for persistence
DATASET_NAME = "corbt/all-recipes"
RECIPES_CSV_PATH = "recipes_data.csv" # Cache file for parsed recipes
DEFAULT_SAMPLE_SIZE = 1000

# --- LLM Configuration ---
GEMINI_MODEL_NAME = "models/gemini-1.5-flash-latest"
# Attempt to get API key from environment
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')

# --- Embedding Model ---
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- Other ---
# You can add other configurable parameters here