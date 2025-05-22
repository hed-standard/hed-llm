# HED RAG Tagging Bot

## Purpose
Combines HED annotation, validation, and documentation retrieval in a single conversational interface using Retrieval-Augmented Generation (RAG). **This bot is still experimental and not quite functional**

## Architecture & Main Components
- **RAG Pipeline**: Loads and indexes HED documentation for retrieval.
- **LLM Integration**: Uses OpenAI's GPT-4o-mini via LangChain for both annotation and question answering.
- **HED Validation**: Integrates the HED Python library for validating user or LLM-generated HED strings.
- **Retriever Tool**: Allows the agent to fetch relevant documentation snippets as needed.
- **Chainlit UI**: Web interface for users to describe events, ask questions, and receive HED annotations or documentation answers.

## How it Works
1. **Startup**: Loads environment variables, initializes LLM, retriever, and HED validator.
2. **User Input**: Users can describe events or ask documentation questions.
3. **Annotation/QA**: The agent decides whether to generate HED annotations, validate them, or retrieve documentation context.
4. **Validation**: HED strings are validated and corrected as needed.
5. **Session Management**: Maintains chat and agent state for multi-turn interactions.

## Key Files
- `app.py`: Main application logic.
- `.chainlit/config.toml`: UI and feature configuration.

## Extending
- Update document sources, prompt logic, or LLM model in `app.py`.
- Add new tools or validation logic as needed.
- Customize UI via `.chainlit/config.toml` and `chainlit.md`. 