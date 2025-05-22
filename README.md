## Installation
- Create a Python virtual environment. For example using venv:
```
python -m venv .llm
source .llm/bin/activate
```
- Install the requirements:
```
pip install -r requirements.txt
```
- Download the .env file and replace the OPENAI API key with your own if so choose: https://drive.google.com/file/d/14_6HYo3S4nQW9PNqjRbHEZG5-EPyNTWR/view?usp=sharing
- Go to the appropriate project directory, for example HED tagging bot:
```
cd hed-bot/hed-tagging-bot
```
- Run the bot in the desired port, for example 55000:
```
chainlit run --port 55000 app.py -w
```

## Tech Stack
- **Python**: Main programming language.
- **Chainlit**: Conversational AI web UI framework.
- **LangChain & LangGraph**: LLM orchestration, prompt management, and agent workflows.
- **OpenAI API**: For LLM completions (via `langchain-openai`).
- **HED Python Library**: For Hierarchical Event Descriptor (HED) string validation and manipulation.
- **BeautifulSoup & Requests**: For web scraping and XML parsing (fetching/parsing HED vocabularies).
- **dotenv**: For environment variable management.

## Codebase Overview
- The main project is organized under `hed-bot/`, with subfolders for each bot:
  - `hed-tagging-bot`: Translates event descriptions to HED annotations and validates them.
  - `hed-doc-bot`, `hed-rag-tagging-bot`, `hed-repo-bot`, `hed-vocab-bot`: Other specialized bots, each with a similar structure.
- Each bot contains:
  - `app.py`: Main application logic, including Chainlit event handlers and LLM agent setup.
  - `.chainlit/`: Chainlit configuration (e.g., `config.toml` for UI and feature settings).
  - `chainlit.md`: Optional welcome screen/documentation for the bot UI.
  - (Some bots) Data files or additional resources (e.g., `HEDLatest-terms`).
- **Key Components in `hed-tagging-bot/app.py`:**
  - Loads environment variables and sets up the OpenAI LLM via LangChain.
  - Defines a tool for HED string validation using the HED Python library.
  - Fetches HED vocabulary from a local file or the official HED schema repository.
  - Sets up a Chainlit chat interface, where users can input event descriptions and receive HED annotations, with real-time validation feedback.
  - Handles image uploads and integrates them into the LLM prompt if provided.

### Extending or Transferring
- To add new bots, replicate the structure of an existing bot directory.
- To change the LLM or prompt logic, modify the `app.py` in the relevant bot.
- To update the HED schema or vocab, replace the `HEDLatest-terms` file or adjust the fetching logic.
- Chainlit configuration and UI can be customized via `.chainlit/config.toml` and `chainlit.md`.