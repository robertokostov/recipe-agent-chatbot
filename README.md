# Recipe Agent: Conversational RAG Chatbot for Recipe Queries

An intelligent conversational agent designed to help users find recipes and answer specific questions about them using Retrieval-Augmented Generation (RAG), keyword search, and LLM-based routing. Built with LangChain, Google Gemini, ChromaDB, Sentence Transformers, and Gradio.

## Overview

This project explores a hybrid approach to handling recipe-related queries. Users can interact with the agent through a chat interface. The agent analyzes the user's query using an LLM (Gemini Flash) to decide whether it's a specific question best suited for RAG or a general search term best handled by keyword search.

- For specific questions, the system retrieves relevant recipe text from a vector database (ChromaDB using `all-MiniLM-L6-v2` embeddings) and uses the LLM to generate an answer grounded in that context [cite: 83-84, 191].
- For general searches, it uses a keyword-based search against a parsed recipe dataset[cite: 81, 192].
- An optional LLM query expansion step can broaden search terms before retrieval[cite: 87].

## Features

- **Conversational Interface:** Stateful chat UI built with Gradio (`gr.Chatbot`)[cite: 127, 193].
- **Agentic Routing:** LLM classifies user intent to route queries to RAG or Text Search paths[cite: 85, 190].
- **RAG for Q&A:** Answers specific questions about recipes using retrieved context to ground LLM responses and reduce hallucinations [cite: 83-84, 196].
- **Semantic Search:** Finds relevant recipes based on meaning using vector embeddings (`all-MiniLM-L6-v2`) and ChromaDB[cite: 70, 72].
- **Keyword Text Search:** Fallback mechanism using keyword matching on titles and ingredients[cite: 81].
- **Query Expansion:** Optional LLM step to add related terms to the search query[cite: 87].
- **In-Memory Vector Store:** Uses ChromaDB configured for in-memory operation (no persistence implemented).

## Project Structure

```
recipe-agent/
├── logger_setup.py       # Stores constants like paths, model names
├── config.py              # Sets up the logger
├── ui.py                 # Gradio UI creation function (create_interface) and helpers
├── main.py               # Main script to initialize system and launch UI
├── dependencies.py       # Checks imports, sets flags & API key
├── recipe_system.py      # Contains the RecipeRecommendationSystem class
├── requirements.txt      # Python dependencies
├── Dockerfile             # (Optional) Docker container definition
├── docker-compose.yml    # (Optional) Docker Compose configuration
├── recipes_data.csv      # CACHE: Stores parsed data from last successful init/reload
├── images/               # Optional: For UI screenshots or diagrams used in docs
└── .env                  # Optional: For API keys if needed later (e.g., for LLM inference)
```

## Current Limitations

- **No Persistence:** The ChromaDB vector store is built **in-memory** and is lost when the application stops. Initialization requires re-loading, re-parsing, and re-embedding data on each fresh start[cite: 75, 124, 201]. `recipes_data.csv` caches parsed text but not embeddings/index.
- **No Backend Conversational Memory:** While the Gradio UI maintains chat history visually using `gr.State`, the backend logic (`search_recipes`) currently processes each query independently without deep context from previous turns. True conversational follow-up is limited [cite: 184-187, 202].
- **Stateless API Client Interaction:** Due to identified issues with Gradio's API specification generation for `gr.State` components (tested up to v5.x on Python 3.12), interaction via `gradio_client` is effectively stateless, preventing automated testing of multi-turn conversations[cite: 103, 204].
- **Limited Dataset/Parsing:** Uses a sample (e.g., 1000 recipes) from `corbt/all-recipes`. Parsing depends on specific text format and may skip recipes; relevance is limited by the sample [cite: 79-80, 82, 118-122].
- **RAG Hallucinations:** While RAG aims to ground responses, the LLM may still hallucinate information if the retrieved context is incomplete, irrelevant, or if the prompt instructions are not perfectly followed [cite: 166-174, 197, 203].
- **Routing Ambiguity:** The LLM-based router may misclassify ambiguous user queries, sometimes sending general searches to the RAG path or specific questions to text search [cite: 144-145, 152-154].
- **No User Personalization:** The system does not currently store or utilize user profiles, dietary restrictions, or past preferences [cf. 94, 206].

## Getting Started

### Prerequisites

- Python **3.12** [cite: 109]
- `pip` and `venv`
- `git` (for cloning)
- Google API Key with Gemini access enabled (set as `GOOGLE_API_KEY` environment variable)

### Setup

1.  **Clone Repository:**
    ```bash
    git clone [YOUR_REPOSITORY_URL]
    cd recipe-agent
    ```
2.  **Create & Activate Virtual Environment:**
    ```bash
    python3.12 -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate    # Windows CMD
    # venv\Scripts\Activate.ps1 # Windows PowerShell
    ```
3.  **Install Dependencies:**
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
4.  **Set API Key:**
    - Create a `.env` file in the project root:
      ```text
      GOOGLE_API_KEY='your_google_api_key_here'
      ```
    - Or, set the environment variable directly in your terminal:
      ```bash
      export GOOGLE_API_KEY='your_google_api_key_here' # Linux/macOS
      # set GOOGLE_API_KEY=your_google_api_key_here    # Windows CMD
      # $env:GOOGLE_API_KEY='your_google_api_key_here' # Windows PowerShell
      ```

### Running the Application

1.  **Start the App:**
    ```bash
    python app.py # Or your main script name
    ```
2.  **Access UI:** Open the Gradio interface URL shown in the terminal (usually `http://127.0.0.1:7860`).
3.  **Initialize:** Click **"Initialize System"**. The _first_ time, this will download the dataset, parse recipes, generate embeddings, and build the in-memory vector store (this can take several minutes). Subsequent clicks (without `force_reload`) might be faster if data loading/parsing is skipped, but embedding/indexing still happens in memory. Check logs for progress.
4.  **Interact:** Once initialized, use the chat interface to ask questions or search for recipes.

## Customization

- **Dataset:** Change `DATASET_NAME` or modify data loading/parsing logic in `app.py` (within `_create_new_db`). Requires adapting parsing to the new data structure.
- **Sample Size:** Adjust `sample_size` in the `RecipeRecommendationSystem` initialization or `initialize` method call.
- **LLM:** Change `GEMINI_MODEL_NAME` or `ChatGoogleGenerativeAI` parameters. Ensure the API key has access.
- **Embedding Model:** Change the `model_name` in the `HuggingFaceEmbeddings` initialization in `_create_new_db`.

## Future Work

Based on the current implementation and limitations[cite: 206]:

- **Implement Persistence:** Save/load the ChromaDB index to/from disk (`persist_directory`) for faster startups.
- **Add Backend Conversational Memory:** Integrate LangChain memory modules or manually pass history to LLM prompts for true contextual follow-up conversations.
- **Improve Routing:** Refine the routing prompt with few-shot examples or better logic to handle ambiguity.
- **Enhance RAG:** Improve retrieval (hybrid search, re-ranking) and refine RAG prompts further to minimize hallucinations; potentially add verification steps.
- **Implement Personalization:** Add mechanisms to store and utilize user profiles (diet, allergies, preferences) to filter/tailor results.
- **Explore UI Enhancements:** Richer display of recipes, interaction buttons (e.g., "Save Recipe").
- **Expand Toolset (Advanced):** Integrate external tools (web search for nutrition, unit conversion).

## License

MIT License (or your chosen license - update accordingly)
