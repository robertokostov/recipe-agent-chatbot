# -*- coding: utf-8 -*-
import os
import pandas as pd
import logging
import re
from typing import List, Tuple, Dict, Any

# Attempt to import necessary libraries for data loading
try:
    from datasets import load_dataset, DatasetGenerationError
    import pyarrow # Needed by datasets for Parquet
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    DatasetGenerationError = Exception # Dummy exception
    pyarrow = None # Dummy module

# Need Document schema for creating documents
try:
    from langchain.schema import Document
except ImportError:
    # If langchain isn't available here, this module might have limited use,
    # but define a dummy for the function signature if needed.
    # However, the agent will need the real one anyway.
    # Let's assume Document schema is fundamental to the output.
    logging.error("Failed to import langchain.schema.Document. This module requires Langchain core.")
    # Re-raise or define a dummy depending on desired strictness
    raise # Fail hard if Document isn't available

logger = logging.getLogger('recipe_system.data_processing')

# --- Constants related to data ---
DATASET_NAME = "corbt/all-recipes"
RECIPES_CSV_PATH = "recipes_data.csv" # Default path to save/load parsed data

def parse_recipe_string(input_content: str, index: int) -> Dict[str, Any] | None:
    """
    Parses the multi-line string format found in the 'input' column.

    Args:
        input_content: The string content from the 'input' column.
        index: The original index for logging purposes.

    Returns:
        A dictionary containing parsed fields ('title', 'description', 'ingredients',
        'instructions', 'rating') or None if parsing fails.
    """
    try:
        if not isinstance(input_content, str) or not input_content.strip():
            return None # Skip empty input

        lines = [line.strip() for line in input_content.splitlines()]
        if not lines: return None

        title = lines[0].strip()
        description = '' # No description field in this format
        rating = None    # No rating field in this format

        ingredients_list = []
        directions_list = []
        current_section = None
        in_ingredients = False
        in_directions = False

        for i, line in enumerate(lines):
            line_strip = line.strip()
            line_lower = line_strip.lower()

            if line_lower == 'ingredients:':
                in_ingredients = True; in_directions = False; current_section = 'ingredients'; continue
            elif line_lower == 'directions:':
                in_directions = True; in_ingredients = False; current_section = 'directions'; continue

            if not line_strip: # Skip blank lines within sections
                continue # Don't reset section on blank line

            if in_ingredients:
                ingredients_list.append(line_strip.lstrip('- '))
            elif in_directions:
                # Simple cleaning of direction prefixes
                cleaned_line = re.sub(r"^\s*[\d\W]+\.?\s*", "", line_strip)
                directions_list.append(cleaned_line)

        ingredients_str = "\n".join(ingredients_list).strip()
        instructions_str = "\n".join(directions_list).strip() # Using 'directions' as 'instructions'

        # Basic validation of parsed content
        if not title or not ingredients_str or not instructions_str:
             logger.warning(f"Row {index}: Parsing resulted in empty fields (T:{bool(title)}, I:{bool(ingredients_str)}, D:{bool(instructions_str)}).")
             return None # Skip if essential parts are missing

        return {
            'title': title, 'description': description, 'ingredients': ingredients_str,
            'instructions': instructions_str, 'rating': rating
        }

    except Exception as e:
        logger.warning(f"Error parsing row index {index}: {e}.")
        return None

def load_and_parse_recipes(
    sample_size: int,
    csv_path: str = RECIPES_CSV_PATH
) -> Tuple[pd.DataFrame | None, List[Document] | None]:
    """
    Loads data from Hugging Face, parses the 'input' column, creates Documents,
    saves the parsed data to CSV, and returns the parsed DataFrame and Document list.

    Args:
        sample_size: The number of recipes to sample and process.
        csv_path: Path to save the parsed DataFrame.

    Returns:
        A tuple containing:
            - pandas.DataFrame: DataFrame with parsed columns ('title', 'ingredients', etc.).
            - list[Document]: List of LangChain Document objects for vector DB indexing.
        Returns (None, None) on failure.
    """
    if not DATASETS_AVAILABLE:
        logger.error("Hugging Face 'datasets' library not available. Cannot load data.")
        return None, None

    try:
        # --- 1. Load Raw Data ---
        logger.info(f"Loading dataset '{DATASET_NAME}' from Hugging Face...")
        dataset = load_dataset(DATASET_NAME, split='train')
        logger.info(f"Loaded dataset '{DATASET_NAME}' successfully.")
        recipes_raw_df = dataset.to_pandas()
        logger.info(f"Converted dataset to Pandas DataFrame ({len(recipes_raw_df)} rows).")
        actual_cols = list(recipes_raw_df.columns)
        logger.info(f"Columns found in raw DataFrame: {actual_cols}")
        if 'input' not in actual_cols:
            raise ValueError(f"Dataset structure unexpected. Expected 'input' column, found: {actual_cols}")

        # --- 2. Sample Data ---
        if sample_size < len(recipes_raw_df):
            logger.info(f"Sampling {sample_size} recipes from {len(recipes_raw_df)}...");
            recipes_sampled_df = recipes_raw_df.sample(sample_size, random_state=42).reset_index(drop=True).copy()
        else:
             logger.info("Using all loaded recipes.");
             recipes_sampled_df = recipes_raw_df.reset_index(drop=True).copy()

        # --- 3. Parse 'input' Column & Create Documents ---
        logger.info("Parsing 'input' column and preparing documents...")
        documents = []
        processed_data = []
        skipped_rows = 0

        for idx, row in recipes_sampled_df.iterrows():
            parsed = parse_recipe_string(row.get('input', ''), idx)
            if parsed:
                # Store parsed data for DataFrame
                processed_data.append(parsed)
                # Prepare content for embedding and metadata for Chroma
                recipe_text = f"Title: {parsed['title']}\nDescription: {parsed['description']}\nIngredients: {parsed['ingredients']}\nInstructions: {parsed['instructions']}"
                metadata = {
                    "doc_id": int(idx), # Original index in the sampled raw df
                    "title": parsed['title'],
                    "description": parsed['description'],
                    "ingredients": parsed['ingredients'],
                    "instructions": parsed['instructions'],
                    # Rating is None, so not added here
                }
                doc = Document(page_content=recipe_text, metadata=metadata)
                documents.append(doc)
            else:
                skipped_rows += 1

        logger.info(f"Finished parsing and document creation. {len(documents)} documents created, {skipped_rows} rows skipped.")

        if not documents:
            logger.error("No valid documents were created after parsing. Cannot proceed.")
            return None, None

        # --- 4. Create Parsed DataFrame & Save CSV ---
        parsed_df = pd.DataFrame(processed_data)
        if parsed_df.empty:
            logger.error("Parsed DataFrame is empty. Cannot proceed.")
            return None, None

        try:
            logger.info(f"Saving PARSED recipe data ({len(parsed_df)} rows) to {csv_path}...")
            parsed_df.to_csv(csv_path, index=False)
        except Exception as csv_error:
            logger.exception(f"Error saving parsed recipes CSV to {csv_path}: {csv_error}")
            # Proceed even if saving fails, as we have the data in memory

        logger.info("Data loading and parsing completed successfully.")
        return parsed_df, documents

    except (DatasetGenerationError, pyarrow.lib.ArrowInvalid if pyarrow else Exception) as dataset_load_error:
         logger.exception(f"Specific error loading dataset '{DATASET_NAME}': {dataset_load_error}")
         logger.error("Consider deleting HF cache: ~/.cache/huggingface/datasets/")
         return None, None
    except Exception as e:
        logger.exception(f"Unexpected error during data loading/parsing: {e}")
        return None, None

# Example usage (optional, for testing this module directly)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO) # Ensure logging is set up
    logger.info("Testing data_processing module...")
    if not DATASETS_AVAILABLE:
        logger.error("Cannot run test: 'datasets' library not available.")
    else:
        test_sample_size = 10
        df, docs = load_and_parse_recipes(sample_size=test_sample_size, csv_path="test_parsed_recipes.csv")
        if df is not None and docs is not None:
            logger.info(f"Successfully loaded and parsed {len(df)} recipes.")
            logger.info("First few rows of DataFrame:")
            print(df.head())
            logger.info("First document metadata:")
            print(docs[0].metadata if docs else "No documents")
            logger.info("Test finished.")
        else:
            logger.error("Test failed.")