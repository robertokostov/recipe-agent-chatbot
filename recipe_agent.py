# -*- coding: utf-8 -*-
import os
import pandas as pd
import time
import logging
import gradio as gr
from typing import List, Dict, Optional, Tuple
from functools import lru_cache
import random
import shutil
import re # Keep re if used elsewhere, though parsing moved

# --- Import components from data_processing ---
try:
    from data_processing import load_and_parse_recipes, RECIPES_CSV_PATH
    DATA_PROCESSING_AVAILABLE = True
except ImportError as e_dp:
    DATA_PROCESSING_AVAILABLE = False
    # Define dummy function if data_processing isn't available
    def load_and_parse_recipes(sample_size, csv_path): return None, None
    RECIPES_CSV_PATH = "recipes_data.csv" # Define constant anyway
    logging.error(f"Failed to import from data_processing.py: {e_dp}")
# --- End data_processing import ---

# --- Conditional Vector Search Imports ---
VECTOR_IMPORTS_AVAILABLE = False
try:
    # Check these specifically needed by the agent
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain.schema import Document # Keep Document needed for search result check
    VECTOR_IMPORTS_AVAILABLE = True
except ImportError as e_agent:
    logging.error(f"Vector search dependency missing in agent: {e_agent}")
    VECTOR_IMPORTS_AVAILABLE = False
# --- End Vector Search Imports ---


# ==============================================================================
# Logging Configuration
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('recipe_system.agent') # Use a specific logger name

# ==============================================================================
# Constants
# ==============================================================================
# VECTOR_DB_PATH is not needed for in-memory, RECIPES_CSV_PATH comes from data_processing

# ==============================================================================
# Recipe Recommendation System Class
# ==============================================================================
class RecipeRecommendationSystem:
    """
    Main agent class. Initializes data via data_processing, builds in-memory
    vector DB, performs searches, and manages state.
    """

    def __init__(self):
        """Initializes the agent's state."""
        self.is_initialized = False
        self.initialization_error = None
        self.embeddings = None
        self.vector_db = None # Holds the in-memory Chroma DB object
        self.recipes_df = None # Holds the *parsed* DataFrame
        self.sample_size = 1000
        self.backup_recipes = self._get_backup_recipes()
        # Determine final search capability based on combined checks
        self.use_vector_search = VECTOR_IMPORTS_AVAILABLE and DATA_PROCESSING_AVAILABLE
        logger.info(f"Agent instance created. Vector search capability: {self.use_vector_search}")

    def _build_in_memory_db(self, documents: List[Document]) -> bool:
        """Builds the in-memory Chroma DB from parsed documents."""
        if not VECTOR_IMPORTS_AVAILABLE: # Double check imports
             self.initialization_error = "Vector store library (Chroma/Langchain) not available."
             return False
        if not documents:
             self.initialization_error = "No documents provided to build vector DB."
             return False
        try:
            logger.info("Initializing embeddings model for vector DB...")
            self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

            logger.info(f"Creating IN-MEMORY vector database with {len(documents)} documents...")
            # Create Chroma object without persisting to disk
            self.vector_db = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings
                # No persist_directory argument
            )
            logger.info("In-memory vector database created successfully.")
            return True # DB created

        except Exception as chroma_error:
            logger.exception(f"Error creating Chroma DB (in-memory attempt): {chroma_error}")
            self.initialization_error = f"Chroma DB creation failed: {chroma_error}"
            self.vector_db = None # Ensure reset
            return False

    def initialize(self, force_reload=False, sample_size=1000):
        """
        Initializes the system. Uses vector search (in-memory) if possible,
        otherwise falls back to text search on backup recipes.
        """
        start_time = time.time()
        # In-memory always requires re-build on initialize call
        is_reload = force_reload or self.use_vector_search

        try:
            # --- Prevent Redundant Initialization (Only for backup/text mode) ---
            if not self.use_vector_search and self.is_initialized and not force_reload and self.sample_size == sample_size and self.recipes_df is not None and not self.recipes_df.empty:
                logger.info(f"System already initialized with backup text search ({len(self.recipes_df)} recipes). Skipping.")
                return True

            # --- Reset State ---
            self.sample_size = sample_size
            logger.info(f"{'Reloading' if is_reload else 'Initializing'} system with sample size: {self.sample_size}...")
            self.is_initialized = False
            self.initialization_error = None
            self.vector_db = None
            self.recipes_df = None # Clear previous data

            # --- Main Logic Path ---
            init_success = False
            final_df = None # Store the resulting df temporarily

            # Path 1: Try Vector Search (In-Memory)
            if self.use_vector_search:
                logger.info("Attempting initialization with IN-MEMORY Vector Search.")
                logger.info("Loading and parsing recipe data...")
                # Call data processing function to get parsed data and documents
                parsed_df, documents = load_and_parse_recipes(self.sample_size, RECIPES_CSV_PATH)

                if parsed_df is not None and documents is not None:
                    logger.info("Data loaded and parsed successfully.")
                    final_df = parsed_df # Store the parsed df
                    # Now build the in-memory DB from the documents
                    db_success = self._build_in_memory_db(documents)
                    if db_success:
                        logger.info("In-memory vector database built successfully.")
                        init_success = True # Both data parsing and DB build succeeded
                    else:
                        # DB build failed, set error from _build_in_memory_db
                        logger.error(f"Failed to build vector DB: {self.initialization_error}. Falling back.")
                        init_success = False # Will trigger fallback below
                else:
                    logger.error("Failed to load or parse recipe data from data_processing module. Falling back.")
                    self.initialization_error = self.initialization_error or "Data loading/parsing failed."
                    init_success = False # Will trigger fallback below

                # If vector path failed at any point (parsing or DB build), attempt fallback
                if not init_success:
                    logger.warning("Vector search initialization failed. Attempting fallback to backup recipes.")
                    final_df = pd.DataFrame(self.backup_recipes).reset_index() # Use backup df
                    self.use_vector_search = False # Disable vector search for this session
                    if final_df is not None and not final_df.empty:
                         logger.info(f"Loaded {len(final_df)} backup recipes.")
                         init_success = True # Fallback successful
                    else:
                         logger.error("Failed to load backup recipes during fallback.")
                         # init_success remains False

            # Path 2: Fallback from the start (vector imports failed initially)
            else:
                logger.info("Vector search components unavailable. Using backup recipes for text search.")
                final_df = pd.DataFrame(self.backup_recipes).reset_index()
                if final_df is not None and not final_df.empty:
                    logger.info(f"Loaded {len(final_df)} backup recipes.")
                    init_success = True
                else:
                     logger.error("Failed to load backup recipes.")
                     # init_success remains False

            # --- Final State Assignment & Check ---
            elapsed = time.time() - start_time
            self.recipes_df = final_df # Assign the final DataFrame (parsed or backup)

            if self.recipes_df is not None and not self.recipes_df.empty:
                self.is_initialized = True # Mark system as ready
                search_type = "vector search (in-memory)" if self.use_vector_search else "text search"
                is_backup = False;
                try: # Check if it's the backup data
                    backup_df = pd.DataFrame(self.backup_recipes).reset_index();
                    if len(self.recipes_df) == len(backup_df): is_backup = self.recipes_df['title'].equals(backup_df['title'])
                except Exception: pass
                source = "backup recipes" if is_backup else "Parsed Dataset"
                logger.info(f"Initialization finished successfully in {elapsed:.2f}s. Using {search_type}. Loaded {len(self.recipes_df)} recipes (Source: {source}, Sample Size: {self.sample_size}).")
                return True
            else: # If even fallback failed to produce a df
                if not self.initialization_error: self.initialization_error = "Failed to load any recipe data."
                logger.error(f"Initialization failed: {self.initialization_error}"); self.is_initialized = False; return False

        except Exception as e: # Catch unexpected errors during the whole process
            self.initialization_error = f"Unexpected outer initialization error: {str(e)}"
            logger.exception(f"CRITICAL error during initialize: {e}")
            self.is_initialized = False; self.recipes_df = None; self.vector_db = None; return False


    def search_recipes(self, query, num_results=3):
        """
        Performs recipe search using vector DB (if available) or text search.
        Assumes self.recipes_df contains PARSED data.
        """
        logger.info(f"Search: '{query}'. Init: {self.is_initialized}. Vector: {self.use_vector_search}. DB: {self.vector_db is not None}. DF size: {len(self.recipes_df) if self.recipes_df is not None else 'None'}")
        if not self.is_initialized: return "System not initialized."
        if self.recipes_df is None or self.recipes_df.empty: return "Recipe database empty."

        try:
            results_data = []; search_method_used = "unknown"

            # Attempt Vector Search
            if self.use_vector_search and self.vector_db is not None:
                try:
                    search_method_used = "vector"; logger.info(f"Attempting vector search: '{query}'")
                    vector_results = self.vector_db.similarity_search(query, k=max(num_results, 5))
                    logger.info(f"Vector search got {len(vector_results)} results.")
                    processed_ids = set()
                    for doc in vector_results:
                         if hasattr(doc, 'metadata'):
                             metadata = doc.metadata; doc_id = metadata.get('doc_id', -1)
                             if doc_id not in processed_ids or doc_id == -1:
                                 # Extract data directly from metadata (which came from parsed data)
                                 results_data.append({
                                     'title': metadata.get('title', '?'), 'description': metadata.get('description', ''),
                                     'rating': metadata.get('rating'), 'ingredients': metadata.get('ingredients', ''),
                                     'instructions': metadata.get('instructions', ''), 'source': 'vector'
                                 })
                                 if doc_id != -1: processed_ids.add(doc_id)
                         else: logger.warning("Vector result missing metadata.")
                except Exception as search_error: logger.exception(f"Vector search error: {search_error}"); results_data = []

            # Fallback to Text Search
            if not results_data:
                fallback_reason = "disabled" if not self.use_vector_search else "no db" if self.vector_db is None else "no results" if search_method_used == "vector" else "failed"
                logger.info(f"Falling back to text search ({fallback_reason})."); search_method_used = "text"
                if self.recipes_df is not None and not self.recipes_df.empty:
                    text_indices = self._text_search(query, num_results) # Use parsed df
                    logger.info(f"Text search returned indices: {text_indices}")
                    processed_indices = set()
                    for recipe_id in text_indices: # Index into parsed df
                        if recipe_id is not None and 0 <= recipe_id < len(self.recipes_df) and recipe_id not in processed_indices:
                            try:
                                recipe_data = self.recipes_df.iloc[recipe_id]
                                results_data.append({ # Extract from parsed df row
                                    'title': recipe_data.get('title', '?'), 'description': recipe_data.get('description', ''),
                                    'rating': recipe_data.get('rating'), 'ingredients': str(recipe_data.get('ingredients', '')),
                                    'instructions': str(recipe_data.get('instructions', '')), 'source': 'text', 'index': recipe_id
                                })
                                processed_indices.add(recipe_id)
                            except Exception as df_error: logger.warning(f"Error accessing parsed index {recipe_id}: {df_error}")
                        else: logger.warning(f"Invalid/duplicate text index {recipe_id}.")
                else: logger.error("Text search fallback failed: recipes_df missing.")

            # Format Results
            if not results_data: logger.info("No results found."); return f"No recipes found for: \"{query}\"."

            final_results = results_data[:num_results]; logger.info(f"Formatting {len(final_results)} results (method: {search_method_used}).")
            formatted_output = f"## Recipe Results for: \"{query}\" (using {search_method_used} search)\n\n"
            for i, recipe in enumerate(final_results):
                try:
                    formatted_output += f"### {i+1}. {recipe.get('title', 'Untitled Recipe')}\n\n"
                    # Description is likely empty from parsed data, Rating is None
                    # desc = recipe.get('description'); if desc: formatted_output += f"**Description:** {desc}\n\n"
                    # rating = recipe.get('rating'); if rating is not None: ...
                    ingredients = recipe.get('ingredients');
                    if ingredients: # Format list
                        formatted_ingredients = '- ' + str(ingredients).strip().replace('\n', '\n- ')
                        formatted_output += "**Ingredients:**\n" + formatted_ingredients + "\n\n"
                    instructions = recipe.get('instructions');
                    if instructions: # Format block
                         formatted_instructions = str(instructions).strip().replace('\n', '\n')
                         formatted_output += "**Instructions:**\n" + formatted_instructions + "\n\n"
                except Exception as format_error: logger.warning(f"Error formatting result #{i+1}: {format_error}"); formatted_output += "*Error formatting.*\n\n"
                formatted_output += "---\n\n"
            return formatted_output.strip()
        except Exception as e: logger.exception(f"Unexpected search error: {e}"); return f"Search error: {str(e)}"


    def _text_search(self, query, num_results=3):
        """Performs keyword search on the parsed self.recipes_df."""
        df_size = len(self.recipes_df) if self.recipes_df is not None else 'None'
        logger.info(f"Text search started: '{query}'. DataFrame size: {df_size}")
        if self.recipes_df is None or self.recipes_df.empty: return []
        try:
            query_lower = query.lower(); query_words = set(word for word in query_lower.split() if len(word) > 2 and word.isalnum())
            if not query_words: return []
            scored_recipes = []
            # Iterate through the *parsed* DataFrame
            for idx, row in self.recipes_df.iterrows():
                try:
                    score = 0;
                    # Use parsed columns (description is likely empty)
                    title = str(row.get('title', '')).lower(); ingredients = str(row.get('ingredients', '')).lower()
                    search_text_high = f"{title} {ingredients}"
                    if query_lower in search_text_high: score += 20 # Exact phrase bonus
                    text_words_high = set(word for word in search_text_high.split() if len(word) > 2 and word.isalnum())
                    matched_high = query_words.intersection(text_words_high)
                    score += len(matched_high) * 5 # Weighted keyword match
                    if score > 0: scored_recipes.append((idx, score))
                except Exception as row_error: logger.warning(f"Error scoring row {idx}: {row_error}")

            scored_recipes.sort(key=lambda x: x[1], reverse=True);
            result_indices = [idx for idx, score in scored_recipes[:num_results]]

            # Random fallback if no score matches
            if not result_indices and not self.recipes_df.empty:
                logger.info(f"Text search 0 results for '{query}'. Random sample.");
                count = min(num_results, len(self.recipes_df))
                if count > 0: result_indices = random.sample(range(len(self.recipes_df)), count)

            logger.info(f"Text search finished. Returning indices: {result_indices}"); return result_indices
        except Exception as e: logger.exception(f"Error during text search: {e}"); return []


    @staticmethod
    def _get_backup_recipes():
        """Provides a small, hardcoded list of recipes as a fallback."""
        # Structure should match the *parsed* data format for consistency if possible
        # (title, description='', ingredients str, instructions str, rating=None)
        # Keeping original structure for simplicity now.
        return [ {"title": "Spaghetti Carbonara", "description": "Classic Italian pasta...", "ingredients": "...", "instructions": "...", "rating": 4.8},
                 {"title": "Chocolate Chip Cookies", "description": "Classic homemade cookies...", "ingredients": "...", "instructions": "...", "rating": 4.9},
                 # ... (Include all 10 original backup recipes) ...
                 {"title": "Vegetarian Chili", "description": "Hearty and flavorful...", "ingredients": "...", "instructions": "...", "rating": 4.6} ]


# ==============================================================================
# Gradio Interface Creation
# ==============================================================================
def create_interface():
    """Sets up and defines the Gradio web interface."""
    logger.info("Creating Gradio interface definition...")
    # Create a single instance of the backend system for the UI to interact with
    recipe_system = RecipeRecommendationSystem()

    # --- UI Callback Functions ---
    def ui_init_system(sample_size_value, progress=gr.Progress(track_tqdm=True)):
        logger.info(f"UI Callback: Initialize (Sample: {sample_size_value})")
        status_msg = "Initializing... Check console logs."; yield status_msg, gr.update(interactive=False), gr.update(interactive=False)
        try:
            success = recipe_system.initialize(force_reload=False, sample_size=int(sample_size_value)); error = recipe_system.initialization_error
            if success:
                num_recipes = len(recipe_system.recipes_df) if recipe_system.recipes_df is not None else 0
                db_type = "vector search (in-memory)" if recipe_system.use_vector_search else "text search"
                status_msg = f"✅ System initialized with {num_recipes} recipes using {db_type} (Size: {recipe_system.sample_size}). Ready."
                logger.info(status_msg); yield status_msg, gr.update(interactive=True), gr.update(interactive=True)
            else: status_msg = f"❌ Init failed: {error or 'Unknown'}. Check logs."; logger.error(status_msg); yield status_msg, gr.update(interactive=True), gr.update(interactive=True)
        except Exception as e: logger.exception(f"UI init error: {e}"); yield f"❌ UI error: {str(e)}", gr.update(interactive=True), gr.update(interactive=True)

    def ui_reload_system(sample_size_value, progress=gr.Progress(track_tqdm=True)):
        logger.info(f"UI Callback: Reload (Sample: {sample_size_value})")
        status_msg = "Reloading (In-Memory DB)... Check logs."; yield status_msg, gr.update(interactive=False), gr.update(interactive=False)
        try:
            success = recipe_system.initialize(force_reload=True, sample_size=int(sample_size_value)); error = recipe_system.initialization_error
            if success:
                num_recipes = len(recipe_system.recipes_df) if recipe_system.recipes_df is not None else 0
                db_type = "vector search (in-memory)" if recipe_system.use_vector_search else "text search"
                status_msg = f"✅ System reloaded with {num_recipes} recipes using {db_type} (Size: {recipe_system.sample_size}). Ready."
                logger.info(status_msg); yield status_msg, gr.update(interactive=True), gr.update(interactive=True)
            else: status_msg = f"❌ Reload failed: {error or 'Unknown'}. Check logs."; logger.error(status_msg); yield status_msg, gr.update(interactive=True), gr.update(interactive=True)
        except Exception as e: logger.exception(f"UI reload error: {e}"); yield f"❌ UI error: {str(e)}", gr.update(interactive=True), gr.update(interactive=True)

    def ui_search_recipes(query, num_results):
        logger.info(f"UI Callback: Search (Query: '{query}', Num: {num_results})"); yield gr.Markdown("Searching...")
        if not query or not query.strip(): yield "Please enter a search query."; return
        if not recipe_system.is_initialized: yield "System is not initialized."; return
        try:
            start_time = time.time(); result = recipe_system.search_recipes(query, int(num_results)); elapsed = time.time() - start_time
            logger.info(f"Search call completed in {elapsed:.2f}s."); yield result
        except Exception as e: logger.exception(f"UI search error: {e}"); yield f"An error occurred: {str(e)}"

    # --- Define UI Layout using Gradio Blocks ---
    with gr.Blocks(title="Recipe Finder", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🍲 Recipe Recommendation System"); gr.Markdown("### Semantic search (in-memory) / keyword fallback")
        status_display = gr.Markdown("Status: Not initialized.")
        with gr.Accordion("Settings", open=False):
            sample_slider = gr.Slider(minimum=100, maximum=10000, value=1000, step=100, label="Num recipes to load/sample", info="Reload required if changed.")
            gr.Markdown("Using IN-MEMORY vector DB (rebuilt on init/reload). Parsed data saved to `recipes_data.csv`.")
        with gr.Row():
            with gr.Column(scale=4): query_input = gr.Textbox(label="What recipe?", placeholder="e.g., 'chicken pasta'", lines=2, interactive=True)
            with gr.Column(scale=1): results_slider = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Num results")
        with gr.Row():
            init_button = gr.Button("Initialize System", variant="primary"); search_button = gr.Button("Search Recipes", variant="secondary", interactive=False); reload_button = gr.Button("Reload Dataset", variant="stop", interactive=False)
        results_output = gr.Markdown("Results..."); examples = [["chocolate dessert"],["quick pasta"],["vegetarian curry"],["healthy chicken"],["spicy tacos"]]
        gr.Examples(examples=examples, inputs=query_input, label="Examples")

        # --- Define Event Listeners ---
        init_button.click(fn=ui_init_system, inputs=[sample_slider], outputs=[status_display, search_button, reload_button])
        reload_button.click(fn=ui_reload_system, inputs=[sample_slider], outputs=[status_display, search_button, reload_button])
        search_button.click(fn=ui_search_recipes, inputs=[query_input, results_slider], outputs=[results_output])
        query_input.submit(fn=ui_search_recipes, inputs=[query_input, results_slider], outputs=[results_output])

    # Return the Gradio app object
    logger.info("Gradio interface definition complete.")
    return demo

# ==============================================================================
# Main Execution Block
# ==============================================================================
if __name__ == "__main__":
    logger.info("Script started directly.")
    # Create the Gradio app
    interface = create_interface()
    # Launch the Gradio web server
    logger.info("Launching Gradio app...")
    interface.launch()
    # This runs after the server is stopped
    logger.info("Gradio app closed.")