# main.py
"""Main script to initialize the system and launch the Gradio UI."""

import gradio as gr

# Local Imports
from logger_setup import get_logger
from dependencies import LANGCHAIN_LLM_AVAILABLE, VECTOR_IMPORTS_AVAILABLE
from recipe_system import RecipeRecommendationSystem
from ui import create_interface
# Import config if needed directly here, e.g., for default sample size override
# from config import DEFAULT_SAMPLE_SIZE

logger = get_logger()

# ==============================================================================
# Main Execution Block
# ==============================================================================
if __name__ == "__main__":
    logger.info("Application starting...")

    # Log dependency status warnings
    if not LANGCHAIN_LLM_AVAILABLE:
        logger.warning("!"*20 + "\nLangChain LLM (Gemini) setup INCOMPLETE or API key missing.\n" + "LLM-dependent features will be disabled.\n" + "!"*20)
    else:
        logger.info("LangChain LLM dependencies and API key found.")

    if not VECTOR_IMPORTS_AVAILABLE:
        logger.warning("!"*20 + "\nVector search dependencies NOT FOUND.\n" + "Vector search functionality will be disabled (using fallback).\n" + "!"*20)
    else:
        logger.info("Vector search dependencies found.")

    # --- Create the single instance of the recipe system ---
    # You could potentially pass a different sample size from config here if needed
    # recipe_agent_system = RecipeRecommendationSystem(sample_size=SOME_OTHER_SIZE)
    recipe_agent_system = RecipeRecommendationSystem()

    logger.info("Creating Gradio interface...")
    # --- Create the Gradio UI, passing the system instance ---
    interface = create_interface(recipe_agent_system)

    logger.info("Launching Gradio interface...")
    # --- Launch the interface ---
    # Add server_name="0.0.0.0" to listen on all interfaces if running in Docker/cloud
    interface.launch(share=False) # Share=False for local testing

    logger.info("Gradio interface closed.")