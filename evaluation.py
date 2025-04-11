# -*- coding: utf-8 -*-
import os
import re
import time
import logging
from gradio_client import Client
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support
)
import pandas as pd

SPACE_ID = "rkostov/thesis-agent"

API_NAME = "/respond"
NUM_RESULTS = 3
SLEEP_BETWEEN_CALLS = 1 

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('evaluation_script')

benchmark_data = [
    {'query': 'Does spaghetti carbonara use cream?', 'intended_target': 'RAG'},
    {'query': 'What kind of cheese is in the Greek Salad recipe?', 'intended_target': 'RAG'},
    {'query': 'Is there onion in the banana bread?', 'intended_target': 'RAG'},
    {'query': 'List ingredients for chocolate chip cookies.', 'intended_target': 'RAG'},
    {'query': 'How much butter for the chocolate chip cookies?', 'intended_target': 'RAG'},
    {'query': 'Does the stir fry recipe contain peanuts?', 'intended_target': 'RAG'},
    {'query': 'What oil is recommended for the stir fry?', 'intended_target': 'RAG'},
    {'query': 'Are eggs required for the carbonara?', 'intended_target': 'RAG'},
    {'query': 'Tell me the spices in the default chicken recipe.', 'intended_target': 'RAG'},
    {'query': 'Any garlic in the greek salad?', 'intended_target': 'RAG'},
    {'query': 'What type of flour is used in the banana bread?', 'intended_target': 'RAG'},
    {'query': 'How many eggs in the carbonara?', 'intended_target': 'RAG'},
    {'query': 'Does the banana bread use baking soda or baking powder?', 'intended_target': 'RAG'},
    {'query': 'Are fresh tomatoes needed for the greek salad?', 'intended_target': 'RAG'},
    {'query': 'What cut of chicken for the stir fry?', 'intended_target': 'RAG'},
    # Instructions & Timing
    {'query': 'How long do I bake the chocolate chip cookies?', 'intended_target': 'RAG'},
    {'query': 'What temperature to bake cookies?', 'intended_target': 'RAG'},
    {'query': 'What is the first step for the chicken stir fry?', 'intended_target': 'RAG'},
    {'query': 'How do you make the dressing for the Greek Salad?', 'intended_target': 'RAG'},
    {'query': 'Tell me how to cook spaghetti carbonara.', 'intended_target': 'RAG'},
    {'query': 'Summarize the banana bread instructions.', 'intended_target': 'RAG'},
    {'query': 'How many steps are there to make the cookies?', 'intended_target': 'RAG'},
    {'query': 'What do I do after frying the pancetta in carbonara?', 'intended_target': 'RAG'},
    {'query': 'How should I prepare the vegetables for the stir fry?', 'intended_target': 'RAG'},
    {'query': "What's the final step for the Greek salad?", 'intended_target': 'RAG'},
    {'query': 'How long does the banana bread need to cool?', 'intended_target': 'RAG'},
    {'query': 'At what point are the chocolate chips added?', 'intended_target': 'RAG'},
    {'query': 'How long to cook the chicken in the stir fry?', 'intended_target': 'RAG'},
    {'query': 'When is the pasta water used in carbonara?', 'intended_target': 'RAG'},
    {'query': 'Should the feta be crumbled or cubed for the salad?', 'intended_target': 'RAG'},
    # Properties/Suitability
    {'query': 'Is the Greek Salad vegetarian?', 'intended_target': 'RAG'},
    {'query': 'Are the chocolate chip cookies gluten-free?', 'intended_target': 'RAG'},
    {'query': 'Is the banana bread recipe vegan?', 'intended_target': 'RAG'},
    {'query': 'Can the carbonara be made ahead of time?', 'intended_target': 'RAG'},
    {'query': 'Is the chicken stir fry spicy?', 'intended_target': 'RAG'},
    {'query': 'Approximate prep time for banana bread?', 'intended_target': 'RAG'},
    {'query': 'Which of the backup recipes are vegetarian?', 'intended_target': 'RAG'},
    {'query': 'Difficulty level of the carbonara?', 'intended_target': 'RAG'},
    {'query': 'Does the cookie recipe yield many cookies?', 'intended_target': 'RAG'},
    {'query': 'Is the stir fry low-carb?', 'intended_target': 'RAG'},
    # Technique/Tools
    {'query': 'How do I cream butter and sugar?', 'intended_target': 'RAG'},
    {'query': "What does 'fold in' mean for the banana bread?", 'intended_target': 'RAG'},
    {'query': 'What pan size for the banana bread?', 'intended_target': 'RAG'},
    {'query': 'Do I need a whisk for the carbonara?', 'intended_target': 'RAG'},
    {'query': "What does 'tender-crisp' mean for stir fry vegetables?", 'intended_target': 'RAG'},
    {'query': 'How to mash bananas properly?', 'intended_target': 'RAG'},
    {'query': 'What kind of pan for stir fry?', 'intended_target': 'RAG'},
    {'query': 'How to chop an onion for the salad?', 'intended_target': 'RAG'},
    {'query': "What does 'al dente' mean for spaghetti?", 'intended_target': 'RAG'},
    {'query': 'Why mix wet and dry ingredients separately for cookies?', 'intended_target': 'RAG'},

    # === Text Search - General Queries ===
    # Recipe Name
    {'query': 'Spaghetti Carbonara', 'intended_target': 'TEXT_SEARCH'},
    {'query': 'Easy Banana Bread', 'intended_target': 'TEXT_SEARCH'},
    {'query': 'Chicken Stir Fry', 'intended_target': 'TEXT_SEARCH'},
    {'query': 'Greek Salad', 'intended_target': 'TEXT_SEARCH'},
    {'query': 'Chocolate Chip Cookies', 'intended_target': 'TEXT_SEARCH'},
    {'query': 'Recipe for carbonara', 'intended_target': 'TEXT_SEARCH'},
    {'query': 'Show me banana bread', 'intended_target': 'TEXT_SEARCH'},
    {'query': 'cookies', 'intended_target': 'TEXT_SEARCH'},
    {'query': 'salad', 'intended_target': 'TEXT_SEARCH'},
    {'query': 'pasta', 'intended_target': 'TEXT_SEARCH'},
    # Main Ingredient(s)
    {'query': 'recipes with chicken breast', 'intended_target': 'TEXT_SEARCH'},
    {'query': 'broccoli soup', 'intended_target': 'TEXT_SEARCH'},
    {'query': 'something with eggs and pancetta', 'intended_target': 'TEXT_SEARCH'},
    {'query': 'Find recipes using feta cheese.', 'intended_target': 'TEXT_SEARCH'},
    {'query': 'pasta with eggs', 'intended_target': 'TEXT_SEARCH'},
    {'query': 'banana recipes', 'intended_target': 'TEXT_SEARCH'},
    {'query': 'cookies with chocolate', 'intended_target': 'TEXT_SEARCH'},
    {'query': 'salad with olives', 'intended_target': 'TEXT_SEARCH'},
    {'query': 'dinner with chicken', 'intended_target': 'TEXT_SEARCH'},
    {'query': 'recipes using ripe bananas', 'intended_target': 'TEXT_SEARCH'},
    {'query': 'find recipes with bell peppers', 'intended_target': 'TEXT_SEARCH'},
    {'query': 'Pecorino Romano recipes', 'intended_target': 'TEXT_SEARCH'},
    {'query': 'What can I make with butter and sugar?', 'intended_target': 'TEXT_SEARCH'},
    {'query': 'Search for recipes with cucumber', 'intended_target': 'TEXT_SEARCH'},
    {'query': 'Got extra eggs, what can I make?', 'intended_target': 'TEXT_SEARCH'},
    # Meal Type/Descriptor
    {'query': 'quick weeknight dinner', 'intended_target': 'TEXT_SEARCH'},
    {'query': 'healthy dessert', 'intended_target': 'TEXT_SEARCH'},
    {'query': 'vegetarian main course', 'intended_target': 'TEXT_SEARCH'},
    {'query': 'party appetizer', 'intended_target': 'TEXT_SEARCH'},
    {'query': 'easy baking recipes', 'intended_target': 'TEXT_SEARCH'},
    {'query': 'low carb meals', 'intended_target': 'TEXT_SEARCH'},
    {'query': 'comfort food', 'intended_target': 'TEXT_SEARCH'},
    {'query': 'salad recipes', 'intended_target': 'TEXT_SEARCH'},
    {'query': 'budget friendly ideas', 'intended_target': 'TEXT_SEARCH'},
    {'query': 'simple lunch', 'intended_target': 'TEXT_SEARCH'},
    {'query': 'italian pasta', 'intended_target': 'TEXT_SEARCH'},
    {'query': 'something sweet', 'intended_target': 'TEXT_SEARCH'},
    {'query': 'savory dishes', 'intended_target': 'TEXT_SEARCH'},
    {'query': 'recipes for beginners', 'intended_target': 'TEXT_SEARCH'},
    {'query': '30 minute meals', 'intended_target': 'TEXT_SEARCH'},

    # === Ambiguous Queries (Assigning a default target for evaluation) ===
    {'query': 'ingredients for healthy vegetarian soup', 'intended_target': 'TEXT_SEARCH'},
    {'query': 'how to make vegetarian lasagna', 'intended_target': 'TEXT_SEARCH'},
    {'query': 'best chocolate chip cookie recipe', 'intended_target': 'TEXT_SEARCH'},
    {'query': 'carbonara no cream', 'intended_target': 'TEXT_SEARCH'},
    {'query': 'information about banana bread', 'intended_target': 'RAG'},
    {'query': 'Greek salad dressing instructions', 'intended_target': 'RAG'},
    {'query': 'quick vegetarian pasta', 'intended_target': 'TEXT_SEARCH'},
    {'query': 'tell me about stir fry', 'intended_target': 'RAG'},
    {'query': 'carbonara recipe details', 'intended_target': 'RAG'},
    {'query': 'cookie variations', 'intended_target': 'TEXT_SEARCH'},
    {'query': 'Can you find a low-sugar banana bread?', 'intended_target': 'TEXT_SEARCH'},
    {'query': 'What are some salads with cucumber?', 'intended_target': 'TEXT_SEARCH'},
    {'query': 'Talk me through the carbonara recipe', 'intended_target': 'RAG'},
    {'query': 'Nutritional info for cookies', 'intended_target': 'RAG'},
    {'query': 'Compare carbonara and stir fry', 'intended_target': 'RAG'},

    # === Edge Cases ===
    {'query': 'choclate chip cookis', 'intended_target': 'TEXT_SEARCH'},
    {'query': 'soup', 'intended_target': 'TEXT_SEARCH'}, 
    {'query': 'Does any recipe use saffron?', 'intended_target': 'RAG'}, 
    {'query': 'asdfghjkl', 'intended_target': 'TEXT_SEARCH'},
    {'query': 'tell me a joke about cooking', 'intended_target': 'RAG'} 
]

# --- Helper function to extract routing decision ---
def extract_routing_decision(response_content):
    if not isinstance(response_content, str):
        return "PARSE_ERROR" # Handle non-string content
    # Pattern to find Router=VALUE within the debug string `DEBUG: Router=VALUE,...`
    pattern = r"Router=([^,`]+)"
    match = re.search(pattern, response_content)
    if match:
        decision = match.group(1).strip()
        if decision in ["RAG", "TEXT_SEARCH"]:
            return decision
        else:
            logger.warning(f"Parsed unexpected decision value: {decision}")
            return "PARSE_ERROR" # Unexpected value
    else:
        # Check if Method= only is present (e.g. from text search helper)
        method_pattern = r"Method=([^`]+)"
        method_match = re.search(method_pattern, response_content)
        if method_match:
            method_used = method_match.group(1).strip()
            # Infer routing based on method if Router= missing
            if "text (router chosen)" in method_used:
                 logger.warning("Router= missing, inferred TEXT_SEARCH from method.")
                 return "TEXT_SEARCH"
            elif "text (RAG fallback)" in method_used:
                 logger.warning("Router= missing, inferred RAG (fallback) from method.")
                 return "RAG" # It was intended RAG, even if it failed
            elif "vector (RAG executed)" in method_used:
                 logger.warning("Router= missing, inferred RAG from method.")
                 return "RAG"
        logger.warning(f"Could not parse routing decision or infer from method in response.")
        return "PARSE_ERROR" # Pattern not found


# --- Main Evaluation Logic ---
if not SPACE_ID:
    logger.error("Error: SPACE_ID not configured. Set the environment variable or hardcode it.")
    exit()

logger.info(f"Connecting to Gradio Space: {SPACE_ID}")
try:
    # Increase timeout if needed client = Client(SPACE_ID, hf_token=...)
    client = Client(SPACE_ID)
    logger.info("Connection successful.")
except Exception as e:
    logger.error(f"Failed to connect to Gradio Space: {e}")
    exit()

# Lists to store labels and predictions
y_true = [] # Your manual labels ('intended_target')
y_pred = [] # Agent's actual routing decisions

logger.info(f"Starting evaluation for {len(benchmark_data)} queries...")

for i, item in enumerate(benchmark_data):
    query = item['query']
    intended_target = item['intended_target']
    logger.info(f"Processing query {i+1}/{len(benchmark_data)}: '{query}' (Expected: {intended_target})")

    actual_decision = "API_ERROR" # Default if API call fails

    try:
        # Make the API call (stateless) - Requires API to accept only message & num_results
        result = client.predict(
            message=query,
            num_results_value=NUM_RESULTS,
            api_name=API_NAME # Use "/respond"
            # No chat_history argument here due to API bug
        )

        # Process the result
        # Expected result (based on corrected respond): tuple (chat_history_list, "")
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], list) and result[0]:
             # Get the last message added (should be the assistant's response)
             last_message = result[0][-1]
             if isinstance(last_message, dict) and last_message.get("role") == "assistant":
                 bot_content = last_message.get("content")
                 actual_decision = extract_routing_decision(bot_content) # Parse debug info
             else:
                 logger.warning(f"Unexpected structure in last message: {last_message}")
                 actual_decision = "PARSE_ERROR"
        elif result is None:
             logger.error(f"API call for query '{query}' returned None.")
             actual_decision = "API_NONE_RETURN"
        else:
             logger.warning(f"Unexpected API result structure: {type(result)} | Content: {result}")
             actual_decision = "API_STRUCT_ERROR"

    except Exception as e:
        logger.error(f"API call failed for query '{query}': {e}")
        actual_decision = "API_ERROR"

    # Append results, ensuring labels are consistent
    y_true.append(intended_target)
    y_pred.append(actual_decision)

    logger.info(f" -> Actual Decision: {actual_decision}")

    # Wait briefly to avoid hitting potential rate limits on free Spaces
    time.sleep(SLEEP_BETWEEN_CALLS)

logger.info("Evaluation loop finished.")

# --- Calculate and Print Metrics ---
logger.info("\n--- Evaluation Results ---")

# Define the valid labels we expect the parser to return
valid_labels = ['RAG', 'TEXT_SEARCH']
filtered_y_true = []
filtered_y_pred = []
# Tally errors
error_codes = ["API_ERROR", "PARSE_ERROR", "API_STRUCT_ERROR", "API_NONE_RETURN"]
error_counts = {code: 0 for code in error_codes}
unknown_preds = []

for true_label, pred_label in zip(y_true, y_pred):
    if pred_label in valid_labels:
        filtered_y_true.append(true_label)
        filtered_y_pred.append(pred_label)
    elif pred_label in error_counts:
        error_counts[pred_label] += 1
    else: # Catch any unexpected prediction labels
         logger.error(f"Encountered unexpected predicted label: {pred_label} for true label: {true_label}")
         unknown_preds.append(pred_label)


total_processed = len(filtered_y_true)
total_errors = sum(error_counts.values())
logger.info(f"Total Queries Run: {len(benchmark_data)}")
logger.info(f"Successfully Parsed Predictions: {total_processed}")
logger.info(f"API/Parse Errors: {total_errors}")
for code, count in error_counts.items():
    if count > 0: logger.info(f"  - {code}: {count}")
if unknown_preds: logger.warning(f"Unknown predicted labels encountered: {set(unknown_preds)}")


if total_processed > 0:
    # Overall Accuracy
    accuracy = accuracy_score(filtered_y_true, filtered_y_pred)
    logger.info(f"\nOverall Routing Accuracy (on {total_processed} successful predictions): {accuracy:.2%}")

    # Confusion Matrix
    logger.info("\nConfusion Matrix (Rows: Actual/Intended, Columns: Predicted by Agent):")
    # Ensure consistent labeling for the matrix
    cm = confusion_matrix(filtered_y_true, filtered_y_pred, labels=valid_labels)
    logger.info(f"Labels: {valid_labels}")
    # Print matrix with labels
    cm_df = pd.DataFrame(cm, index=[f'Actual_{l}' for l in valid_labels], columns=[f'Predicted_{l}' for l in valid_labels])
    logger.info(f"\n{cm_df}\n")
    # Explanation (assuming RAG=0, TEXT_SEARCH=1) -> Use labels instead
    logger.info(f"TN (Actual RAG, Predicted RAG):                 {cm[0][0]}")
    logger.info(f"FP (Actual RAG, Predicted TEXT_SEARCH):         {cm[0][1]}")
    logger.info(f"FN (Actual TEXT_SEARCH, Predicted RAG):         {cm[1][0]}")
    logger.info(f"TP (Actual TEXT_SEARCH, Predicted TEXT_SEARCH): {cm[1][1]}")


    # Classification Report (Precision, Recall, F1 per class)
    logger.info("\nClassification Report:")
    # Use dict output for easier logging if needed, default string is fine too
    report = classification_report(
        filtered_y_true,
        filtered_y_pred,
        labels=valid_labels,
        target_names=valid_labels,
        zero_division=0 # Report 0 instead of warning for classes with no support/predictions
    )
    logger.info(f"\n{report}")
else:
    logger.warning("No successful predictions were parsed, cannot calculate metrics.")

logger.info("--- Evaluation Complete ---")
