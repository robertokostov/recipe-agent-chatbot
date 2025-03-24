# dependencies.py
"""Checks necessary dependencies and sets availability flags."""

import os
from logger_setup import get_logger
from config import GOOGLE_API_KEY, DOTENV_AVAILABLE # Import key from config

logger = get_logger()

# --- LangChain Core & LLM Check ---
LANGCHAIN_LLM_AVAILABLE = False
try:
    from langchain_core.documents import Document
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    from langchain_google_genai import ChatGoogleGenerativeAI
    # Check if key is actually present
    if GOOGLE_API_KEY:
        LANGCHAIN_LLM_AVAILABLE = True
        logger.info("GOOGLE_API_KEY found. LangChain LLM dependencies appear available.")
    else:
        logger.warning("GOOGLE_API_KEY environment variable not found.")
        if DOTENV_AVAILABLE: logger.info("Checked environment and .env file (if present).")
        else: logger.info("Checked environment variables.")
        LANGCHAIN_LLM_AVAILABLE = False
except ImportError as e:
    logger.error(f"Import check failed for LangChain LLM components: {e}")
    logger.error("<<<<< Please ensure 'langchain-google-genai' and core 'langchain' packages are installed >>>>>")
    LANGCHAIN_LLM_AVAILABLE = False

# --- Vector Search Imports Check ---
VECTOR_IMPORTS_AVAILABLE = False
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    from datasets import load_dataset
    import pyarrow # Ensure pyarrow is checked
    # Re-check Document was imported earlier
    from langchain_core.documents import Document
    VECTOR_IMPORTS_AVAILABLE = True
    logger.info("Vector search dependencies check: OK.")
except ImportError as e:
    logger.error(f"Import check failed for vector search dependencies: {e}")
    logger.error("<<<<< Ensure 'langchain-huggingface', 'langchain-community', 'datasets', 'pyarrow', 'chromadb', 'sentence-transformers' are installed >>>>>")
    VECTOR_IMPORTS_AVAILABLE = False

# Log final decisions
if not VECTOR_IMPORTS_AVAILABLE: logger.warning("Vector database imports failed - vector search functionality potentially limited.")
if not LANGCHAIN_LLM_AVAILABLE: logger.warning("LangChain LLM setup incomplete - LLM-dependent features (RAG, Routing, Expansion) disabled.")

# --- Export flags ---
# Export necessary components if needed elsewhere, or just flags
__all__ = ['LANGCHAIN_LLM_AVAILABLE', 'VECTOR_IMPORTS_AVAILABLE', 'GOOGLE_API_KEY']