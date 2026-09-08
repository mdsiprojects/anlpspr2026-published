# Session 8 — Building LLM Apps with APIs and RAG

**Date:** 14 September 2026

## Before class

Spend 15–20 minutes on the core sections of [pre-reading.md](pre-reading.md) and the [vector similarity guide](vector_similarity.md). The pre-reading also contains optional deeper material on retrieval strategies and vector stores. Slides are provided separately on Canvas.

## Topics

- LLM API access with the OpenAI SDK
- Chat Completions vs Responses API
- streaming responses for interactive apps
- Provider compatibility: OpenAI and Ollama
- Retrieval-Augmented Generation (RAG)
- Vector stores: FAISS hands-on; OpenAI vector stores and Azure AI Search as conceptual comparisons
- Tool use and function calling as a bridge to Session 9
- simple UI demos with Gradio and Streamlit

## Learning Objectives

By the end of this session, students should be able to:

- call language models from Python using the OpenAI SDK
- explain the difference between Chat Completions and the Responses API
- implement streaming output with the Responses API
- compare OpenAI and Ollama endpoints and identify feature-compatibility limits
- explain why PDF documents need parsing and chunking before retrieval
- build a minimal FAISS-based RAG pipeline over handbook and policy PDFs
- compare answers with and without retrieval grounding
- wrap model calls in a small Gradio demo
- build a streaming Streamlit RAG app
- explain how a model requests a function call and how the application returns its result

## Folder Structure

```text
material/Session 8/
├── README.md
├── pre-reading.md
├── data/
│   ├── README.md
│   ├── images/
│   ├── questions.json
│   └── pdfs/
├── helpers/
│   ├── pdf_utils.py
│   ├── agent_tools.py
│   └── rag_utils.py
├── notebooks/
│   ├── 01_llm_api_access.ipynb
│   ├── 02_embeddings_and_similarity.ipynb
│   ├── 03_streaming_responses.ipynb
│   ├── 04_rag_with_pdfs_and_faiss.ipynb
│   ├── 05_gradio_chat_demo.ipynb
│   └── 06_tool_use.ipynb
├── 06_streamlit_rag_app.py
├── 07_streamlit_agentic_rag_app.py
├── .streamlit/
│   └── secrets.toml.example
└── vector_similarity.md
```

## Notebook Order

### `01_llm_api_access.ipynb`

This notebook introduces:

- OpenAI SDK setup
- Chat Completions
- Responses API
- structured output
- image and multimodal input
- provider swap patterns for OpenAI and Ollama
- a short Claude comparison

### `02_embeddings_and_similarity.ipynb`

This notebook introduces:

- what embeddings are and why they are useful
- OpenAI embedding creation with the SDK
- cosine similarity for semantic comparison
- a tiny semantic search example
- the retrieval idea that later becomes RAG

### `03_streaming_responses.ipynb`

This notebook introduces:

- why streaming matters in user-facing apps
- how to iterate over Response API events
- how to accumulate streamed text in Python
- how the same pattern connects naturally to UI tools later in the session

### `04_rag_with_pdfs_and_faiss.ipynb`

This notebook builds a minimal RAG pipeline:

- parse PDF documents
- inspect extracted text
- chunk the text with overlap
- create embeddings
- build a FAISS index
- retrieve relevant chunks
- compare no-RAG vs RAG answers

### `05_gradio_chat_demo.ipynb`

This notebook introduces:

- a minimal chat UI using Gradio
- how to connect streaming model output to a lightweight interface
- how quickly an API call can become an interactive demo

### `06_tool_use.ipynb`

This notebook covers tool use and function calling:

- anatomy of a tool definition (JSON schema)
- a custom weather tool — the classic first example
- the 4-step agentic loop: request → detect → execute → feed back
- a `run_tool_loop` helper for multi-turn agentic conversations
- built-in tools: `web_search` and `code_interpreter`
- parallel tool calls — two tools in one turn
- RAG as a tool call — the FAISS index wrapped as a function the model can invoke

### `06_streamlit_rag_app.py`

This is the main app build for Session 8:

- a streaming chat interface in Streamlit
- retrieval over the Session 8 PDF corpus
- grounded answers with visible sources
- editable system prompt in the sidebar with a reset-to-default button
- a stronger end-to-end example than the Gradio demo

### `07_streamlit_agentic_rag_app.py`

This second app applies notebook 06's tool loop to the same PDF corpus. The first app retrieves for every question; this version lets the model choose whether to search, check the weather, read the clock, or answer directly.

```bash
python -m streamlit run "material/Session 8/07_streamlit_agentic_rag_app.py"
```

- `search_documents(query)` returns PDF passages and source/chunk IDs. The FAISS index is built and cached only when the model first requests search.
- `get_weather(location, units)` uses the same wttr.in service as notebook 06. It needs internet access but no separate weather key; service failures are returned to the model.
- `get_current_datetime()` extends the notebook's date tool with the time and UTC offset from the computer running Streamlit. On deployment, this is the server's timezone.
- `tool_choice="auto"` allows an answer without tools. The app executes registered functions, returns each result with its call ID, and asks the model again. After four tool rounds, it requests a final answer with tools disabled.
- Answers stream, and each turn has an expandable record of tool arguments and results. Conversation history includes earlier tool results, so follow-up questions can reuse evidence. **Clear conversation** starts fresh.
- The default system prompt asks for evidence on company-specific claims and citations to returned passages. Tool choice is a model decision: inspect the trace and check whether the selected evidence supports the answer.

Try a greeting, a policy question, a weather question, a date/time question, then a combined request. Compare which tools are called. A greeting should not trigger PDF indexing.

This app uses the same OpenAI key, optional organisation/project settings, and model environment variables as the original. It does not implement the notebook's Ollama provider switch. No additional Python packages are required.

References: [Responses function calling](https://developers.openai.com/api/docs/guides/function-calling), [wttr.in](https://github.com/chubin/wttr.in).

## Setup

Use **Python 3.12** and the course `.venv`. Update your fork from the [published course repository](https://github.com/mdsiprojects/anlpspr2026-published) before installing dependencies. Run the following from the repository root, not from the session folder.

If `.venv` does not exist, create it with `py -3.12 -m venv .venv` on Windows or `python3.12 -m venv .venv` on macOS/Linux.

Activate the course virtual environment from the repo root:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Open the notebooks in VS Code or Jupyter and select this `.venv` as the kernel. Work through notebook 01 first, then follow the numbered notebooks. First-time downloads and dependency installation can take longer than the pre-reading.

## Environment Variables

Most notebook activities and the Streamlit chat app require **OpenAI API access with billing and model access**. A ChatGPT subscription alone does not provide API credit.

If you have OpenAI API access, create a `.env` file at the repository root and add your key:

```dotenv
OPENAI_API_KEY=your_api_key_here
```

Add `OPENAI_ORG_ID` and `OPENAI_PROJECT_ID` if those values were supplied for your account. The examples pass them to the SDK when available. Keep `.env` and real `secrets.toml` files out of Git and screenshots.

Optional comparisons use a separate `ANTHROPIC_API_KEY` for Claude or Ollama at `http://localhost:11434/v1`. Skip the relevant optional cells if those services are unavailable. Local embedding examples run without an OpenAI key after their models have downloaded; they do not make the entire session or the Streamlit generator work offline.

### Using Ollama without an OpenAI key

Reuse the local Ollama setup discussed in Session 7, with a model such as **Llama** or **gpt-oss** that runs on your machine. Download your chosen model and check that it responds before class. Use the exact installed model tag shown by `ollama list`.

For notebook 01's provider comparison, add these settings to your root `.env`, replacing the model tag if needed:

```dotenv
SESSION8_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_CHAT_MODEL=llama3.2:latest
```

Start Ollama, run notebook 01's setup and client-helper cells, then run its provider-comparison section. Skip cells marked `Requires OPENAI_API_KEY` if you do not have a key; do not use **Run All** for that route. No OpenAI key is needed for the local comparison. Local sentence-transformers embedding examples can also be used after their model weights have downloaded.

This selector applies only to the provider comparison. It does not switch the other notebooks or the Streamlit RAG app to Ollama, and local models may support different features. Students without an OpenAI key can work through the local activities, read the OpenAI-specific code, and follow the demonstrations.

## Streamlit Secrets (for deployment)

A secrets template is provided at `.streamlit/secrets.toml.example`. To use it locally or deploy to Streamlit Cloud:

```bash
mkdir -p .streamlit
cp "material/Session 8/.streamlit/secrets.toml.example" .streamlit/secrets.toml
# fill in your values — secrets.toml is gitignored
```

For Streamlit Cloud, paste the contents of the template into **App Settings → Secrets**.

## Running the Streamlit App

From the repo root:

```bash
streamlit run "material/Session 8/06_streamlit_rag_app.py"
```

## Dataset Notes

The PDF corpus uses generic handbook and benefits documents because they produce clearer retrieval-style questions than course notes.

See [README.md](data/README.md) inside the `data/` folder for details.

## Related References

- [pre-reading.md](pre-reading.md)
- [vector_similarity.md](vector_similarity.md)

## API compatibility (checked 5 September 2026)

GitHub Models retired on 30 July 2026. Its playground and inference endpoints are no longer available; a GitHub token cannot provide model access. The examples now use the OpenAI API directly, with Ollama for local comparisons and a separate optional Claude example.

- `SESSION8_MODEL`: generation model; defaults to `gpt-4.1-mini`, a small model supporting the session's text, image, structured-output, and tool examples. This is a teaching default, not a claim that it is the newest model.
- `SESSION8_EMBEDDING_MODEL`: defaults to `text-embedding-3-small`. Rebuild the index whenever the embedding model changes.
- `SESSION8_PROVIDER`: `openai` or `ollama`, for notebook 01's provider comparison only.
- `OLLAMA_CHAT_MODEL`: defaults to `llama3.2:latest`; run `ollama pull llama3.2:latest` first.
- `OLLAMA_EMBEDDING_MODEL`: defaults to `nomic-embed-text`; run `ollama pull nomic-embed-text` first.
- `ANTHROPIC_MODEL`: defaults to `claude-sonnet-4-6`; requires separate Anthropic API access.

Use the course `.venv` as the Jupyter kernel. OpenAI examples require an API key with billing and model access; a ChatGPT or Copilot subscription alone does not supply this. Skip optional Ollama/Claude cells if those services are unavailable. Ollama's stateless Responses endpoint requires version 0.13.3 or later. Local sentence-transformers examples download model weights on first use and can then run offline.

The tool-use notebook makes live weather requests and uses billed OpenAI web-search and code-interpreter tools. Remote services and model outputs can vary. Notebook outputs are cleared so old executions cannot be mistaken for current results.

Sources: [GitHub retirement](https://github.blog/changelog/2026-07-30-github-models-is-now-retired/), [OpenAI model capabilities](https://developers.openai.com/api/docs/models/gpt-4.1-mini), [Ollama compatibility](https://docs.ollama.com/api/openai-compatibility), [Gradio 6 migration](https://www.gradio.app/guides/gradio-6-migration-guide).

## What the teaching app does

The app reads the four supplied PDFs from `data/pdfs/`, splits their extracted text into 500-character chunks with 100-character overlap, and caches an exact FAISS index. Its top-k slider defaults to four chunks. You can edit or reset the system prompt and inspect retrieved sources.

Each question is retrieved and answered independently; earlier messages remain visible but are not sent as conversation history. Ask self-contained questions. The app has no PDF upload widget or provider-switching widget. Restart it after changing the model or PDF corpus so the cached index is rebuilt.

The examples demonstrate the mechanics of RAG. They do not establish the correctness of every answer or provide production authentication, permissions, or service monitoring.
