# Recipe Agent Chatbot with LangChain, HuggingFace, and Gradio

A personalized recipe search and recommendation system built with LangChain, HuggingFace models, ChromaDB (in-memory), and Gradio.

## Features

- 🍲 Recipe search and recommendation based on user query
- 💬 Basic conversational interface via Gradio for search interaction
- 📊 Vector-based recipe search (using ChromaDB **in-memory**) for semantic matching
- 📉 Fallback Text-based keyword search if vector search fails or is unavailable
- 🖥️ Simple web interface built with Gradio

## Project Structure

```
recipe-agent/
├── recipe_agent.py       # Main agent implementation & Gradio UI
├── data_processing.py    # Data loading (from HF Hub) & parsing utilities
├── requirements.txt      # Python dependencies
├── Dockerfile             # (Optional) Docker container definition
├── docker-compose.yml    # (Optional) Docker Compose configuration
├── recipes_data.csv      # CACHE: Stores parsed data from last successful init/reload
└── .env                  # Optional: For API keys if needed later (e.g., for LLM inference)
```

## Current Workflow & Limitations

- **Initialization:** On first run or reload, the app loads the `corbt/all-recipes` dataset from Hugging Face Hub, samples it, parses the specific string format found in the dataset's 'input' column (Title\n\nIngredients:\n...\n\nDirections:\n...), builds an **in-memory** vector database (using ChromaDB and sentence-transformers), and caches the parsed data to `recipes_data.csv`.
- **In-Memory DB:** The vector database is rebuilt each time the application starts or is initialized/reloaded. It is **not persisted** to disk in the current configuration, meaning initialization time will depend on data loading/parsing/embedding on each run.
- **Search:** Performs semantic search against the in-memory vector DB. If vector search fails or components are unavailable, it falls back to basic keyword text search on the parsed recipe data (title and ingredients).
- **LLM Features:** Does not currently include advanced conversational AI or recipe adaptation features using a large language model (LLM). The focus is on search and retrieval based on the input query.
- **Data Format Dependency:** Relies heavily on the specific multi-line string format within the 'input' column of the `corbt/all-recipes` dataset. Changing the dataset requires adapting the parsing logic in `data_processing.py`.
- **No User Context/Memory:** The current version does not retain user preferences or conversation history between searches.

## Getting Started

### Prerequisites

- Python 3.9+
- `git` (for cloning the repository)

### Setup

1.  Clone this repository:
    ```bash
    git clone [https://github.com/yourusername/recipe-agent.git](https://github.com/yourusername/recipe-agent.git)
    cd recipe-agent
    ```
    _(Replace `yourusername` with the actual path if applicable)_
2.  Set up a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    _(Note: Ensure `pyarrow` is included in `requirements.txt` as it's crucial for the `datasets` library)._

### Running the Application

1.  Start the chatbot application:
    ```bash
    python recipe_agent.py
    ```
2.  Open the Gradio interface (usually `http://localhost:7860` or the URL provided in the terminal).
3.  Click **"Initialize System"** (or "Reload Dataset").
    - The _first time_ you initialize, it will:
      - Download the `corbt/all-recipes` dataset from Hugging Face Hub (this might take time).
      - Sample the data.
      - Parse the recipe text from the `input` column.
      - Calculate embeddings for the recipes.
      - Build the **in-memory** vector database.
      - Save the parsed data to `recipes_data.csv`.
    - Subsequent initializations will rebuild the in-memory database. Check the logs for details.

### Docker Deployment (Example)

_Note: `Dockerfile` and `docker-compose.yml` need to be created and configured correctly for this project._

To run using Docker (once configured):

1.  Build and start the container:
    ```bash
    docker-compose up -d
    ```
2.  Access the interface at `http://localhost:7860`

## Customization

### Changing the Data Source

The system currently uses the `corbt/all-recipes` dataset and parses its `input` column string format. To use different data:

1.  **Use a different Hugging Face Dataset:**
    - If the new dataset has the _exact same structure_ (a single 'input' column with the Title/Ingredients/Directions format), you can change the `DATASET_NAME` constant in `data_processing.py`.
    - If the new dataset has a _different structure_ (e.g., columns like 'title', 'ingredients', 'directions'), you must modify the `load_and_parse_recipes` function and potentially `parse_recipe_string` in `data_processing.py` to handle the new column names and data format.
2.  **Use a Local CSV File:**
    - Modify `load_and_parse_recipes` in `data_processing.py` to load your CSV using `pd.read_csv()` instead of `load_dataset()`.
    - Ensure your CSV has columns that can be mapped to 'title', 'ingredients', and 'instructions'.
    - Adjust the parsing/document creation loop to read from your CSV columns.

### Changing the Language Model (Future Feature)

The current version focuses on search/retrieval and does **not** yet incorporate a large language model (LLM) for advanced chat or recipe adaptation.

When an LLM (like one from `HuggingFaceHub` or locally run) is added, you would typically configure and integrate it within the `RecipeRecommendationSystem` class in `recipe_agent.py`.

## Cost-Effective Implementation

This project aims to be cost-effective:

- Uses local sentence-transformer models for embeddings (no API calls for this).
- Implements ChromaDB **in-memory** by default, avoiding persistent disk storage costs (but requires re-computation on startup).
- Built primarily with open-source components (LangChain, Gradio, `datasets`, etc.).
- Can run entirely locally after initial dataset download.

## Next Steps

- Implement LLM integration for conversational recipe adaptation and Q&A.
- Add LangChain memory to retain user preferences and context.
- Implement more sophisticated NLU for query understanding (dietary needs, cuisine, etc.).
- Add image generation for recipes (e.g., using Stable Diffusion).
- Integrate grocery list export functionality.
- Implement user accounts for persistent preferences across sessions (would require a persistent backend).
- Add health and nutrition-focused recommendations.
- Explore alternative/persistent vector store options if needed.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
