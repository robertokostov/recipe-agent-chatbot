# Recipe Agent Chatbot with LangChain, HuggingFace, and Gradio

A personalized recipe recommendation and adaptation system built with LangChain, HuggingFace models, ChromaDB, and Gradio.

## Features

- 🍲 Recipe search and recommendation based on user preferences
- 🔄 Automatic recipe adaptation to match dietary needs
- 💬 Conversational interface for natural interaction
- 🧠 Memory of user preferences across the conversation
- 📊 Vector-based recipe search for semantic matching
- 🖥️ Simple web interface built with Gradio

## Project Structure

```
recipe-agent/
├── recipe_agent.py       # Main agent implementation
├── data_processing.py    # Data processing utilities
├── requirements.txt      # Python dependencies
├── Dockerfile            # Docker container definition
├── docker-compose.yml    # Docker Compose configuration
├── data/                 # Data directory
│   └── recipes.csv       # Recipe dataset
└── chroma_db/            # ChromaDB vector storage
```

## Getting Started

### Prerequisites

- Python 3.9+
- HuggingFace account with API token

### Setup

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/recipe-agent.git
   cd recipe-agent
   ```

2. Set up a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your HuggingFace API token:
   ```
   HUGGINGFACEHUB_API_TOKEN=your_token_here
   ```

5. Generate sample recipe data:
   ```
   python data_processing.py
   ```

### Running the Application

Start the chatbot with:

```
python recipe_agent.py
```

The Gradio interface will be available at `http://localhost:7860`.

### Docker Deployment

To run using Docker:

1. Build and start the container:
   ```
   docker-compose up -d
   ```

2. Access the interface at `http://localhost:7860`

## Customization

### Using Your Own Recipe Data

1. Prepare a CSV file with the following columns:
   - name: Recipe name
   - cuisine: Cuisine type
   - diet_types: Comma-separated diet categories
   - ingredients: List of ingredients
   - instructions: Step-by-step cooking instructions
   - cook_time: Cooking time
   - difficulty: Recipe difficulty level
   - nutritional_info: Nutritional information

2. Place your CSV in the `data/` directory and update the `recipe_data_path` parameter when initializing the `RecipeAgent`.

### Changing the Language Model

To use a different model:

1. Update the `repo_id` parameter in the `HuggingFaceHub` initialization:
   ```python
   self.llm = HuggingFaceHub(
       repo_id="your-preferred-model",
       model_kwargs={"temperature": 0.7, "max_length": 512}
   )
   ```

## Cost-Effective Implementation

This project was designed to be cost-effective:

- Uses smaller, efficient models like FLAN-T5-Base
- Implements ChromaDB for efficient vector storage
- Uses open-source components throughout
- Can run entirely locally for zero API costs
- Optimized for minimal resource usage

## Next Steps

- Implement more sophisticated NLU capabilities
- Add image generation for recipes
- Integrate grocery list export
- Implement user accounts for persistent preferences
- Add health and nutrition-focused recommendations

## License

This project is licensed under the MIT License - see the LICENSE file for details.
