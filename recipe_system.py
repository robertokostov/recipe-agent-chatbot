# recipe_system.py
"""Contains the main RecipeRecommendationSystem class."""

import os
import pandas as pd
import time
import logging
import re
import random
from typing import Optional, List, Dict

# LangChain Imports (conditionally imported/checked in dependencies.py)
# We still need direct imports here for type hinting and usage
try:
    from langchain_core.documents import Document
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    LANGCHAIN_IMPORTS_OK = True
except ImportError:
    LANGCHAIN_IMPORTS_OK = False # Flag potential issues

# Other Imports
try:
    from datasets import load_dataset
    import pyarrow # Keep explicit import needed by datasets
    DATASETS_OK = True
except ImportError:
    DATASETS_OK = False

# Local Imports
from config import (
    VECTOR_DB_PATH, DATASET_NAME, RECIPES_CSV_PATH,
    GEMINI_MODEL_NAME, GOOGLE_API_KEY, EMBEDDING_MODEL_NAME,
    DEFAULT_SAMPLE_SIZE
)
from logger_setup import get_logger
from dependencies import LANGCHAIN_LLM_AVAILABLE, VECTOR_IMPORTS_AVAILABLE

logger = get_logger()

# ==============================================================================
# Recipe Recommendation System Class
# ==============================================================================
class RecipeRecommendationSystem:
    """
    Manages recipe data loading, parsing, indexing (vector or text), searching,
    and optional LLM query expansion & RAG using LangChain. Includes enhanced logging
    and minimal agentic routing.
    """
    def __init__(self, sample_size=DEFAULT_SAMPLE_SIZE):
        self.is_initialized = False
        self.initialization_error = None
        self.embeddings = None
        self.vector_db = None
        self.recipes_df = None
        self.sample_size = sample_size
        self.backup_recipes = self._get_backup_recipes()
        self.lc_llm: Optional[ChatGoogleGenerativeAI] = None
        # Flags set based on dependency checks
        self.use_vector_search = VECTOR_IMPORTS_AVAILABLE
        self.use_llm = LANGCHAIN_LLM_AVAILABLE # GOOGLE_API_KEY check is within _load_llm now
        logger.info(f"System instance created. Sample Size: {self.sample_size}, Vector search enabled check: {self.use_vector_search}, LLM enabled check: {self.use_llm}")

        # Check if critical imports failed after init attempt
        if not LANGCHAIN_IMPORTS_OK:
            logger.error("Critical LangChain components failed to import during class definition.")
            self.use_llm = False # Force disable if imports failed
            self.use_vector_search = False # Chroma/HF Embeddings also depend on langchain_core sometimes indirectly
        if not DATASETS_OK:
             logger.error("Hugging Face 'datasets' library failed to import.")
             # This might prevent loading, depending on implementation - vector search might fail later
             # self.use_vector_search = False # Optional: disable vector search if dataset loading fails

    def _load_llm(self):
        if not self.use_llm:
             logger.info("LLM features disabled based on initial checks.")
             return False
        # Check for API key specifically here
        if not GOOGLE_API_KEY:
            logger.warning("Attempted to load LLM, but GOOGLE_API_KEY is missing.")
            self.use_llm = False # Update flag if key is missing now
            return False
        if self.lc_llm:
            logger.info("LangChain LLM wrapper already configured.")
            return True
        try:
            logger.info(f"Configuring LangChain Gemini LLM wrapper for model: {GEMINI_MODEL_NAME}...")
            self.lc_llm = ChatGoogleGenerativeAI(
                model=GEMINI_MODEL_NAME, google_api_key=GOOGLE_API_KEY, temperature=0.7
            )
            logger.info("LangChain Gemini LLM wrapper configured successfully.")
            return True
        except Exception as e:
            logger.exception(f"Error configuring LangChain Gemini LLM wrapper: {e}")
            self.lc_llm = None
            self.use_llm = False # Ensure flag is reset on failure
            self.initialization_error = (self.initialization_error or "") + f" | LLM Config Failed: {e}"
            return False

    def initialize(self, force_reload=False, sample_size=None):
        if sample_size is not None:
             self.sample_size = sample_size
        # If not provided, self.sample_size retains its initial value (DEFAULT_SAMPLE_SIZE)

        start_time = time.time()
        logger.info(f"Initialize called. Force reload: {force_reload}, Sample size: {self.sample_size}")

        # Re-check dependency flags and API key status on initialize call
        self.use_vector_search = VECTOR_IMPORTS_AVAILABLE
        self.use_llm = LANGCHAIN_LLM_AVAILABLE and bool(GOOGLE_API_KEY)
        llm_should_be_ready = self.use_llm # Flag if LLM *should* be usable

        # Skip if already initialized correctly
        if (self.is_initialized and not force_reload):
            current_llm_state_ok = llm_should_be_ready == (self.lc_llm is not None)
            # Check if current vector state matches expected based on availability flag
            current_vector_state_ok = self.use_vector_search == (self.vector_db is not None)
            # Ensure dataframe state matches vector state (or fallback state)
            df_state_ok = (self.use_vector_search and self.recipes_df is not None and not self.recipes_df.empty) or \
                          (not self.use_vector_search and self.recipes_df is not None and not self.recipes_df.empty)

            if current_llm_state_ok and current_vector_state_ok and df_state_ok:
                logger.info(f"System already initialized correctly ({'Vector' if self.use_vector_search else 'Text'} Search, LLM: {llm_should_be_ready}). Skipping.")
                return True
            else:
                logger.warning("System was marked initialized, but state mismatch detected. Re-initializing...")

        logger.info(f"{'Reloading' if self.is_initialized or force_reload else 'Initializing'} system...")
        # Reset state variables
        self.is_initialized = False
        self.initialization_error = None
        self.vector_db = None
        self.recipes_df = None
        # Reset LLM instance only if forced, or if it should be ready but isn't
        if force_reload or (llm_should_be_ready and not self.lc_llm):
             self.lc_llm = None
             llm_load_success = self._load_llm()
             if not llm_load_success: logger.warning("LLM configuration failed during init/reload.")
             self.use_llm = self.lc_llm is not None # Update use_llm based on outcome
        elif self.lc_llm and not llm_should_be_ready: # LLM loaded but shouldn't be
             logger.info("Disabling previously loaded LLM due to current config/availability.")
             self.lc_llm = None
             self.use_llm = False

        # Decide main path (Vector or Text)
        should_attempt_vector = self.use_vector_search # Based on import check flag
        init_success = False

        if should_attempt_vector:
            logger.info("Attempting vector search initialization pathway...")
            # Persistence logic would go here:
            # if not force_reload and load_persistent_db_if_exists(): init_success = True
            # else: create_new_db() ...
            create_success = self._create_new_db() # Currently always creates new
            if create_success:
                logger.info("Vector DB pathway initialization successful.")
                init_success = True
                self.use_vector_search = True # Ensure flag is correct
            else: # Fallback if DB creation failed
                error_msg = self.initialization_error or "DB creation failed"
                logger.error(f"{error_msg}. Falling back to text search.")
                self.recipes_df = pd.DataFrame(self._get_backup_recipes()).reset_index()
                self.use_vector_search = False # Update flag to reflect fallback
                self.vector_db = None
                if self.recipes_df is not None and not self.recipes_df.empty:
                    logger.info(f"Loaded {len(self.recipes_df)} backup recipes for fallback.")
                    init_success = True # Initialized, but in fallback mode
                else:
                     logger.error("Failed to load backup recipes during fallback.")
                     init_success = False
        else: # Fallback if vector imports/dependencies were missing from the start
            logger.info("Vector dependencies unavailable. Initializing with text search fallback.")
            self.recipes_df = pd.DataFrame(self._get_backup_recipes()).reset_index()
            # self.use_vector_search should already be False
            self.vector_db = None
            if self.recipes_df is not None and not self.recipes_df.empty:
                logger.info(f"Loaded {len(self.recipes_df)} backup recipes.")
                init_success = True
            else:
                logger.error("Failed to load backup recipes.")
                init_success = False

        # Final Status Check
        elapsed = time.time() - start_time
        if init_success and self.recipes_df is not None and not self.recipes_df.empty:
            self.is_initialized = True
            search_type = "vector search" if self.use_vector_search else "text search (fallback)"
            llm_status = "active" if self.use_llm and self.lc_llm else "inactive"
            logger.info(
                f"Initialization finished in {elapsed:.2f}s. Search: {search_type}. "
                f"LLM: {llm_status}. Recipes available: {len(self.recipes_df)}."
            )
            return True
        else:
            if not self.initialization_error: self.initialization_error = "Init failed (unknown reason)"
            logger.error(f"Initialization failed: {self.initialization_error}")
            self.is_initialized = False
            return False

    def _create_new_db(self):
        """ Creates vector DB and populates self.recipes_df. """
        if not VECTOR_IMPORTS_AVAILABLE:
             self.initialization_error = "Vector libraries not available for DB creation."
             return False
        try:
            logger.info(f"Loading dataset '{DATASET_NAME}'...")
            try: # Load dataset
                dataset = load_dataset(DATASET_NAME, split='train')
                recipes_raw_df = dataset.to_pandas()
                logger.info(f"Loaded {len(recipes_raw_df)} recipes.")
                if 'input' not in recipes_raw_df.columns: raise ValueError("Missing 'input' column")
            except Exception as e: logger.exception(f"Dataset load failed: {e}"); self.initialization_error = f"Dataset load: {e}"; return False

            # Sample data
            if 0 < self.sample_size < len(recipes_raw_df):
                logger.info(f"Sampling {self.sample_size} recipes...")
                recipes_sampled_df = recipes_raw_df.sample(self.sample_size, random_state=42).reset_index(drop=True).copy()
            else:
                logger.info(f"Using all {len(recipes_raw_df)} recipes.")
                recipes_sampled_df = recipes_raw_df.reset_index(drop=True).copy()

            # Init embeddings if needed
            if not self.embeddings:
                logger.info(f"Initializing embeddings model ({EMBEDDING_MODEL_NAME})...")
                self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
                logger.info("Embeddings model initialized.")

            # Parse and create documents
            logger.info(f"Parsing {len(recipes_sampled_df)} recipes...")
            documents: List[Document] = []; processed_data = []; skipped = 0
            log_interval = max(1, len(recipes_sampled_df) // 10)
            for idx, row in recipes_sampled_df.iterrows():
                if (idx + 1) % log_interval == 0: logger.debug(f"Parsing: {idx + 1}/{len(recipes_sampled_df)}")
                try: # Parsing logic
                    inp = row.get('input',''); lines = [ln.strip() for ln in inp.splitlines()] if isinstance(inp, str) else []
                    if not lines: skipped += 1; continue
                    title = lines[0] if lines else f'Recipe {idx}'; ingreds = []; directs = []; in_i, in_d = False, False
                    for line in lines[1:]:
                        ls = line.strip(); ll = ls.lower()
                        if ll == 'ingredients:': in_i=True; in_d=False; continue
                        elif ll == 'directions:': in_d=True; in_i=False; continue
                        if not ls and not in_i and not in_d: continue
                        if in_i: ingreds.append(ls.lstrip('- '))
                        elif in_d: directs.append(re.sub(r"^\s*[\d\W]+\.?\s*", "", ls))
                    i_str = "\n".join(ingreds).strip(); d_str = "\n".join(directs).strip()
                    if not title or not i_str or not d_str: skipped += 1; continue
                    # Store parsed data for DataFrame
                    processed_data.append({'title': title, 'ingredients': i_str, 'instructions': d_str})
                    # Create LangChain Document
                    meta = {"doc_id": int(idx), "title": title} # Keep metadata simple
                    doc_content = f"Title: {title}\n\nIngredients:\n{i_str}\n\nInstructions:\n{d_str}"
                    documents.append(Document(page_content=doc_content, metadata=meta))
                except Exception as e: logger.warning(f"Parse error row {idx}: {e}", exc_info=False); skipped += 1

            logger.info(f"Parsing complete. Docs: {len(documents)}, Data rows: {len(processed_data)}, Skipped: {skipped}")
            if not documents: self.initialization_error = "No documents parsed."; return False

            # Store DataFrame & Save CSV
            self.recipes_df = pd.DataFrame(processed_data)
            if self.recipes_df.empty: self.initialization_error = "Parsed DataFrame empty."; return False
            try: logger.info(f"Saving {len(self.recipes_df)} recipes to CSV: {RECIPES_CSV_PATH}..."); self.recipes_df.to_csv(RECIPES_CSV_PATH, index=False); logger.info("CSV saved.")
            except Exception as e: logger.warning(f"CSV save failed: {e}")

            # Create Chroma DB (In-Memory)
            logger.info(f"Creating Chroma DB with {len(documents)} documents...")
            try:
                self.vector_db = None # Ensure starting fresh
                self.vector_db = Chroma.from_documents(documents=documents, embedding=self.embeddings)
                logger.info("Chroma DB created successfully (in-memory).")
                return True
            except Exception as e: logger.exception(f"Chroma DB creation failed: {e}"); self.initialization_error = f"Chroma DB: {e}"; self.vector_db = None; return False
        except Exception as e: logger.exception(f"Outer _create_new_db error: {e}"); self.initialization_error = f"DB Creation: {e}"; self.recipes_df = None; self.vector_db = None; return False

    # --- LLM Interaction Methods ---
    def _expand_query_with_llm(self, query: str) -> Optional[str]:
        if not self.use_llm or not self.lc_llm: return None
        start_time = time.time(); logger.info(f"LCEL Chain: Expanding query: '{query}'")
        try:
            template = "Expand this recipe search query with related terms: {query}"
            prompt = PromptTemplate.from_template(template)
            output_parser = StrOutputParser()
            expansion_chain = prompt | self.lc_llm | output_parser
            expanded_query = expansion_chain.invoke({"query": query})
            elapsed = time.time() - start_time
            logger.info(f"LCEL Chain: Original: '{query}' -> Expanded: '{expanded_query}' ({elapsed:.2f}s)")
            if not expanded_query or expanded_query.lower().strip() == query.lower().strip():
                logger.info("LCEL expansion trivial."); return None
            return expanded_query.strip()
        except Exception as e: logger.exception(f"LCEL expansion error: {e}"); return None


    def _get_routing_decision(self, query: str) -> str:
        if not self.use_llm or not self.lc_llm:
            logger.warning("Router: LLM off. Defaulting to RAG.")
            return "RAG"
        logger.info(f"Router: Getting decision for query: '{query}'")
        start_time = time.time()
        routing_template = """You are a request router for a recipe system. Determine the best approach:
1. 'RAG': For specific questions about recipes (ingredients, instructions, properties like "is it vegetarian?").
2. 'TEXT_SEARCH': For general searches by name or keywords (e.g., "chocolate chip cookies", "tomato soup").
Respond ONLY 'RAG' or 'TEXT_SEARCH'. Query: {query} Approach:"""
        routing_prompt = PromptTemplate.from_template(routing_template)
        output_parser = StrOutputParser()
        try:
            routing_chain = routing_prompt | self.lc_llm | output_parser
            decision = routing_chain.invoke({"query": query}).strip().upper()
            elapsed = time.time() - start_time
            if decision in ["RAG", "TEXT_SEARCH"]: logger.info(f"Router: Decision '{decision}' ({elapsed:.2f}s)."); return decision
            else: logger.warning(f"Router: Bad response '{decision}'. Defaulting RAG."); return "RAG"
        except Exception as e: logger.exception(f"Router error: {e}. Defaulting RAG."); return "RAG"


    # --- Search Execution Methods ---
    def search_recipes(self, query, num_results=3):
        """Searches recipes using LLM-routed approach."""
        log_prefix = f"Search(Q='{query}', N={num_results})"
        logger.info(f"{log_prefix}: Called. Init: {self.is_initialized}...")
        if not self.is_initialized: return "System not initialized."
        if self.recipes_df is None or self.recipes_df.empty: return "No recipe data available for searching." # More specific message

        original_query = query; search_query = query
        expanded_query_used = False; llm_expansion_note = ""

        # Optional Expansion
        if self.use_llm:
            expanded_query = self._expand_query_with_llm(original_query)
            if expanded_query:
                search_query = expanded_query; expanded_query_used = True
                llm_expansion_note = f" (LLM expanded to: \"{search_query}\")"
                logger.info(f"{log_prefix}: Using expanded query '{search_query}'")
            else: logger.info(f"{log_prefix}: Using original query '{original_query}'")
        else: logger.info(f"{log_prefix}: LLM expansion off. Using original query.")

        search_start = time.time(); final_result = ""; search_method_used = "unknown"

        # Routing
        routing_decision = self._get_routing_decision(original_query)
        logger.info(f"{log_prefix}: Router path: {routing_decision}")

        # Initialize debug info variables
        debug_info = ""

        try:
            # RAG Path
            if routing_decision == "RAG":
                search_method_used = "vector (RAG chosen)"
                if self.use_vector_search and self.vector_db is not None:
                    try: # Attempt RAG
                        logger.info(f"{log_prefix}: Retrieving docs (Q: '{search_query}')")
                        retriever = self.vector_db.as_retriever(search_kwargs={'k': num_results})
                        retrieved_docs: List[Document] = retriever.invoke(search_query)
                        logger.info(f"{log_prefix}: Found {len(retrieved_docs)} docs.")
                        if retrieved_docs and self.lc_llm:
                            logger.info(f"{log_prefix}: Running RAG chain.")
                            def format_docs(docs): return "\n\n---\n\n".join([f"Doc {i+1} (Title: {doc.metadata.get('title','N/A')}):\n{doc.page_content}" for i, doc in enumerate(docs)])
                            context_string = format_docs(retrieved_docs)
                            rag_template_qa = """You are a helpful Recipe Assistant... [Keep full prompt] ...Context: {context} Query: {query} Answer:""" # Keep full prompt
                            rag_prompt = PromptTemplate.from_template(rag_template_qa)
                            rag_chain = ({"context": lambda x: context_string, "query": RunnablePassthrough()} | rag_prompt | self.lc_llm | StrOutputParser())
                            final_result = rag_chain.invoke(original_query)
                            search_method_used = "vector (RAG executed)"
                        elif not retrieved_docs: logger.info(f"{log_prefix}: 0 docs found for RAG. Falling back."); final_result = ""
                        else: logger.warning(f"{log_prefix}: Docs found, LLM off. Falling back."); final_result = ""
                    except Exception as rag_error: logger.exception(f"{log_prefix}: RAG error: {rag_error}"); final_result = ""
                else: logger.warning(f"{log_prefix}: RAG chosen, but vector search off/failed. Falling back."); final_result = ""

                # Fallback within RAG path
                if not final_result:
                    logger.info(f"{log_prefix}: Falling back to text search (RAG path failed or yielded no result).")
                    search_method_used = "text (RAG fallback)"
                    final_result = self._execute_text_search_and_format(original_query, search_query, num_results, llm_expansion_note, is_fallback=True)
                    # Note: _execute_text_search_and_format now returns string + debug info

            # Text Search Path
            elif routing_decision == "TEXT_SEARCH":
                search_method_used = "text (router chosen)"
                logger.info(f"{log_prefix}: Executing text search directly.")
                final_result = self._execute_text_search_and_format(original_query, search_query, num_results, llm_expansion_note, is_fallback=False)
                # Note: _execute_text_search_and_format now returns string + debug info
            else:
                 logger.error(f"{log_prefix}: Invalid router decision '{routing_decision}'.");
                 final_result = f"❌ Internal Error: Invalid routing decision."
                 search_method_used = "Error" # Set method for debug info

            # --- Final Logging and Return ---
            # Debug info is now part of final_result string from helper or RAG fallback
            search_elapsed = time.time() - search_start
            logger.info(f"{log_prefix}: Completed via '{search_method_used}' path in {search_elapsed:.2f}s.")

            # If RAG executed successfully and didn't fallback, add debug info now
            if search_method_used == "vector (RAG executed)":
                 debug_info = f"\n\n---\n`DEBUG: Router={routing_decision}, Method={search_method_used}`"
                 # Ensure final_result is a string before appending
                 final_result_str = final_result if isinstance(final_result, str) else str(final_result)
                 return final_result_str + debug_info
            elif final_result: # Text search paths already include debug info
                 return final_result
            else: # Handle cases where final_result might still be empty/None
                 debug_info = f"\n\n---\n`DEBUG: Router={routing_decision}, Method={search_method_used}`"
                 return f"😕 No results found for \"{original_query}\"." + debug_info

        except Exception as e:
            logger.exception(f"{log_prefix}: Unexpected outer error: {e}")
            # Try to provide debug info even on outer error
            debug_info = f"\n\n---\n`DEBUG: Router={routing_decision}, Method=OuterError`"
            return f"❌ An unexpected critical error occurred: {str(e)}" + debug_info

    def _execute_text_search_and_format(self, original_query, search_query, num_results, llm_expansion_note, is_fallback=False):
        """Helper to run text search and format results, including debug info."""
        log_prefix = f"Search(Q='{original_query}', N={num_results})"
        logger.info(f"{log_prefix}: Executing text search logic (Fallback={is_fallback}). Query='{search_query}'")
        method = "text (RAG fallback)" if is_fallback else "text (router chosen)" # Define method early
        debug_info = f"\n\n---\n`DEBUG: Method={method}`" # Define debug info early

        if self.recipes_df is None or self.recipes_df.empty:
            logger.error(f"{log_prefix}: Text search error: df missing.");
            return f"❌ Error: Recipe data frame is missing." + debug_info

        text_indices = self._text_search(search_query, num_results); logger.info(f"{log_prefix}: Text search found indices: {text_indices}")
        text_results_data = []; processed_indices = set()
        for recipe_id in text_indices:
            if isinstance(recipe_id, int) and 0 <= recipe_id < len(self.recipes_df) and recipe_id not in processed_indices:
                try: # Get recipe data
                    recipe_data = self.recipes_df.iloc[recipe_id]
                    title = recipe_data.get('title', f'Recipe {recipe_id}'); ingredients = str(recipe_data.get('ingredients', 'N/A')); instructions = str(recipe_data.get('instructions', 'N/A'))
                    text_results_data.append({'title': title, 'ingredients': ingredients, 'instructions': instructions}); processed_indices.add(recipe_id)
                except Exception as df_error: logger.warning(f"Text search DF access error {recipe_id}: {df_error}")
            else: logger.warning(f"Invalid or already processed text index skipped: {recipe_id}")

        # Format results if any found
        if text_results_data:
            logger.info(f"{log_prefix}: Formatting {len(text_results_data)} text results.")
            search_note = "(using _text search fallback_)" if is_fallback else "(using _text search_)"
            formatted_output = f"Found {len(text_results_data)} recipe(s) for \"**{original_query}**\"{llm_expansion_note} {search_note}:\n\n---\n\n"
            for i, recipe in enumerate(text_results_data): # Format each recipe
                try:
                    title = recipe.get('title', 'Untitled Recipe'); formatted_output += f"### {i+1}. {title}\n\n"
                    ing = recipe.get('ingredients'); inst = recipe.get('instructions')
                    if ing and ing != 'N/A': ing_list = [f"- {line.strip()}" for line in ing.strip().split('\n') if line.strip()]; formatted_output += "**Ingredients:**\n" + "\n".join(ing_list) + "\n\n"
                    if inst and inst != 'N/A': inst_list = [f"{num}. {line.strip()}" for num, line in enumerate(inst.strip().split('\n'), 1) if line.strip()]; formatted_output += "**Instructions:**\n" + "\n".join(inst_list) + "\n\n"
                except Exception as fmt_e: logger.warning(f"Error formatting text result #{i+1}: {fmt_e}"); formatted_output += f"*Err fmt recipe {i+1}*\n\n"
                if i < len(text_results_data) - 1: formatted_output += "---\n\n"
            return formatted_output.strip() + debug_info # Append debug info
        else: # No results found by text search
            logger.info(f"{log_prefix}: Text search (Fallback={is_fallback}) found 0 results.")
            return f"😕 No recipes found matching: \"{original_query}\"." + debug_info # Append debug info

    def _text_search(self, query, num_results=3):
        """Performs keyword search on self.recipes_df."""
        # ... (Keep implementation as before) ...
        if self.recipes_df is None or self.recipes_df.empty: return []
        try:
            query_lower = query.lower(); query_words = set(re.findall(r'\b\w{3,}\b', query_lower))
            if not query_words: logger.warning(f"Text Search: No valid keywords in '{query}'."); return []
            scored_recipes = []; titles = self.recipes_df.get('title', pd.Series(dtype=str)).fillna('').str.lower()
            ingredients_col = self.recipes_df.get('ingredients', pd.Series(dtype=str)).fillna('').astype(str).str.lower()
            search_texts = titles + " " + ingredients_col
            for idx, text_content in search_texts.items():
                score = 0
                try:
                    if query_lower in text_content: score += 20
                    text_words = set(word for word in re.findall(r'\b\w{3,}\b', text_content))
                    score += len(query_words.intersection(text_words)) * 5
                    title_words = set(word for word in re.findall(r'\b\w{3,}\b', titles.get(idx, '')))
                    score += len(query_words.intersection(title_words)) * 10
                except Exception as score_err: logger.warning(f"Scoring error idx {idx}: {score_err}", exc_info=False)
                if score > 0: scored_recipes.append((idx, score))
            scored_recipes.sort(key=lambda x: x[1], reverse=True)
            return [idx for idx, score in scored_recipes[:num_results]]
        except Exception as e: logger.exception(f"Unexpected error during text search for '{query}': {e}"); return []

    @staticmethod
    def _get_backup_recipes():
        """ Provides a small, hardcoded list of recipes as a fallback. """
        # ... (Keep implementation as before) ...
        # Return the full list or keep it short
        return [
            {"title": "Spaghetti Carbonara", "ingredients": "Spaghetti\nEggs\nPancetta or Guanciale\nPecorino Romano cheese\nBlack pepper", "instructions": "Cook spaghetti.\nFry pancetta.\nWhisk eggs and cheese.\nCombine pasta, pancetta fat, egg mixture off heat.\nAdd pasta water if needed.\nServe with pepper."},
            {"title": "Chocolate Chip Cookies", "ingredients": "Butter\nSugar\nBrown Sugar\nEggs\nVanilla Extract\nFlour\nBaking Soda\nSalt\nChocolate Chips", "instructions": "Cream butter and sugars.\nBeat in eggs and vanilla.\nCombine dry ingredients.\nMix wet and dry.\nStir in chocolate chips.\nDrop onto baking sheets.\nBake until golden brown."},
            {"title": "Chicken Stir Fry", "ingredients": "Chicken breast\nBroccoli\nBell peppers\nCarrots\nSoy sauce\nGinger\nGarlic\nSesame oil\nRice", "instructions": "Cut chicken and vegetables.\nStir-fry chicken until cooked.\nAdd vegetables and stir-fry until tender-crisp.\nMix sauce ingredients.\nPour sauce over stir-fry.\nServe with rice."},
            {"title": "Greek Salad", "ingredients": "Cucumber\nTomatoes\nRed onion\nKalamata olives\nFeta cheese\nOlive oil\nRed wine vinegar\nOregano", "instructions": "Chop vegetables.\nCombine vegetables and olives in a bowl.\nCrumble feta cheese over salad.\nWhisk olive oil, vinegar, and oregano for dressing.\nDrizzle dressing over salad."},
            {"title": "Easy Banana Bread", "ingredients": "Ripe bananas\nButter\nSugar\nEgg\nVanilla extract\nFlour\nBaking soda\nSalt", "instructions": "Mash bananas.\nMelt butter.\nMix melted butter, sugar, egg, and vanilla.\nCombine dry ingredients.\nMix wet and dry ingredients until just combined.\nPour into loaf pan.\nBake until a toothpick comes out clean."}
        ]