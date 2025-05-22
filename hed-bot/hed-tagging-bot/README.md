# HED Tagging Bot

## Purpose
Translates event descriptions into HED (Hierarchical Event Descriptor) annotations and validates them using the HED Python library. Provides real-time feedback and suggestions for error-free HED annotations.

## Architecture & Main Components
- **LLM Integration**: Uses OpenAI's GPT-4o-mini via LangChain for natural language understanding and annotation generation.
- **HED Validation**: Integrates the HED Python library to validate and correct HED strings.
- **Vocabulary Fetching**: Loads HED vocabulary from a local file or fetches the latest from the official HED schema repository (parsing XML with BeautifulSoup).
- **Chainlit UI**: Provides a conversational web interface for users to input event descriptions and receive HED annotations, with support for image uploads.

## How it Works
1. **Startup**: Loads environment variables and initializes the LLM and HED validator.
2. **User Input**: Users describe events (optionally upload images) via the Chainlit UI.
3. **Annotation Generation**: The LLM generates HED annotations using only allowed vocabulary.
4. **Validation**: The annotation is validated; if issues are found, the hed tagging agent is prompted to revise until error-free.
5. **Session Management**: Uses Chainlit's session and memory features to manage chat history and agent state.

## Key Files
- `app.py`: Main application logic.
- `HEDLatest-terms`: (Optional) Local HED vocabulary file.
- `.chainlit/config.toml`: UI and feature configuration.

## Extending
- Update prompts or LLM model in `app.py`.
- Change HED schema or vocab by replacing `HEDLatest-terms` or editing fetch logic.
- Customize UI via `.chainlit/config.toml` and `chainlit.md`. 