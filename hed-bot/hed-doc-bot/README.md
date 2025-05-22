# HED Doc Bot

## Purpose
Provides a conversational interface to answer questions about HED documentation using Retrieval-Augmented Generation (RAG).

## Architecture & Main Components
- **RAG Pipeline**: Loads HED documentation from official web pages, splits into chunks, and indexes with Chroma vector store.
- **LLM Integration**: Uses OpenAI's GPT-4o-mini via LangChain for question answering.
- **Retriever**: Uses vector search to find relevant documentation snippets for each user query.
- **Chainlit UI**: Web interface for users to ask questions and receive answers with source links.

## How it Works
1. **Startup**: Loads and indexes HED documentation from the hed documentation pages.
2. **User Input**: Users ask questions via the Chainlit UI.
3. **Retrieval**: Relevant document chunks are retrieved using vector search.
4. **Answer Generation**: The LLM generates concise answers using the retrieved context.
5. **Session Management**: Maintains chat history and context for follow-up questions.

## Key Files
- `app.py`: Main application logic.
- `.chainlit/config.toml`: UI and feature configuration.

## Extending
- Add or change documentation sources in `app.py`.
- Update prompt templates or LLM model as needed.
- Customize UI via `.chainlit/config.toml` and `chainlit.md`. 