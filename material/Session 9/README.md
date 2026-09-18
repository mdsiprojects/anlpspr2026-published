# Session 9 — AI Agents

**Date:** 28 September 2026
**Week:** 10

Session 9 extends the retrieval and tool-use patterns from Session 8 into agent loops, multi-agent orchestration, Model Context Protocol (MCP), and human approval boundaries.

## Learning outcomes

By the end of the session, you should be able to:

- distinguish a deterministic workflow from a model-directed agent;
- design useful, narrow tools with clear descriptions;
- implement routing, agents-as-tools, and handoff patterns;
- explain how MCP replaces one-off tool integrations with a standard client/server interface;
- identify where guardrails, tracing, and human approval belong in an agent system.

## Session agenda

| Block | Focus | Resource |
|---|---|---|
| 1 | From RAG to agentic RAG; workflows versus agents | Slides + notebook 01 |
| 2 | Prompt chaining, routing, parallel work, evaluator–optimizer | Slides + notebook 01 |
| 3 | MCP architecture and a Python 2.x `MCPServer` | Notebook 01 |
| 4 | Build an agent with tools and inspect its run | Notebook 02 |
| 5 | Agents as tools, handoffs, reliability, and human approval | Notebook 02 + design activity |
| Extension | Streaming, context, hooks, Gradio, approval UI, and local-model tracing | Notebooks 03–05 |

## Slides

[Session 9 lecture slides](ANLP%20Session9_Week10_REVISED_OLLAMA_MLFLOW.pptx) include the optional Ollama and MLflow lab.

## Notebooks

### Core

1. [`01_agents_concepts.ipynb`](notebooks/01_agents_concepts.ipynb) — agentic RAG, workflow patterns, and MCP.
2. [`02_agent_with_tools.ipynb`](notebooks/02_agent_with_tools.ipynb) — OpenAI Agents SDK tools, tracing, agents as tools, and handoffs.

### Optional extensions

3. [`03_beyond_basics.ipynb`](notebooks/03_beyond_basics.ipynb) — streaming, state, hooks, typed output, and orchestration controls.
4. [`04_agent_gradio_app.ipynb`](notebooks/04_agent_gradio_app.ipynb) — streaming Gradio interface with an explicit human-approval checkpoint.
5. [`05_ollama_model_switch.ipynb`](notebooks/05_ollama_model_switch.ipynb) — run the same Agents SDK agent with Ollama models and inspect local MLflow traces; no OpenAI key is needed for the default path.

The complete approved Session 9 folder will be published at [anlpspr2026-published / Session 9](https://github.com/mdsiprojects/anlpspr2026-published/tree/main/material/Session%209).

## Setup

Run these commands from the repository root using Python 3.12.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS or Linux, activate with `source .venv/bin/activate`.

For notebooks 01–04, create a repository-root `.env` file or set the same values in your shell:

```text
OPENAI_API_KEY=...
OPENAI_ORG_ID=...       # optional
OPENAI_PROJECT_ID=...   # optional
OPENAI_MODEL=gpt-5.6-luna
```

Never commit `.env` or API credentials. The OpenAI-backed live examples use API quota; notebook 05 uses local inference by default. The notebooks use a restricted arithmetic evaluator; do not replace it with Python `eval()`.

## Current implementation notes

- Notebooks 01–04 target `openai-agents` 0.22.x and use the OpenAI Responses API. Notebook 05 uses the same Agents SDK with Ollama's OpenAI-compatible Chat Completions endpoint.
- The MCP example targets the stable MCP Python SDK 2.x API: `from mcp.server import MCPServer`.
- MCP was donated to the Linux Foundation's Agentic AI Foundation in December 2025.
- Microsoft Agent Framework 1.0 is the current Microsoft framework for new Python/.NET agent development; AutoGen and Semantic Kernel are discussed as predecessor projects.
- Notebook 03's interactive REPL is opt-in so the notebook can execute unattended.
- Notebook 05 is an optional local extension. Install and start Ollama, pull `qwen3:4b`, and follow its MLflow setup cells. Set `ANLP_OLLAMA_SECOND_MODEL` to compare another installed model. It is validated separately from the four OpenAI-backed notebooks.

## Primary references

- [Anthropic: Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/)
- [Ollama OpenAI-compatible API](https://docs.ollama.com/api/openai-compatibility)
- [MLflow tracing for the Agents SDK](https://mlflow.org/docs/latest/genai/tracing/integrations/listing/openai-agent/)
- [MCP Python SDK v2](https://github.com/modelcontextprotocol/python-sdk)
- [Anthropic: MCP donation to the Agentic AI Foundation](https://www.anthropic.com/news/donating-the-model-context-protocol-and-establishing-of-the-agentic-ai-foundation)
- [Microsoft Agent Framework](https://learn.microsoft.com/en-us/agent-framework/)
