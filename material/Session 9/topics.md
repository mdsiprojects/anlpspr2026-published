# Session 9 topic guide

**Delivery date:** 28 September 2026

## 1. From RAG to agents

A conventional RAG pipeline follows a path chosen by the developer: retrieve, augment, generate. An agent can decide whether to retrieve, which tool to call, whether the result is sufficient, and what to do next. Use that flexibility only when the task genuinely needs model-directed decisions; deterministic code remains easier to test and cheaper to run.

## 2. The augmented LLM

An agent starts with a model and adds:

- instructions and an explicit objective;
- tools for retrieval or action;
- working context or persistent session state;
- an execution loop that observes results and selects the next step;
- guardrails, tracing, and approval boundaries.

## 3. Workflow patterns

| Pattern | Use when | Main risk |
|---|---|---|
| Prompt chaining | Steps and validation gates are known | A weak early result propagates |
| Routing | Inputs map to distinct specialists | Misclassification sends work to the wrong path |
| Parallelization | Subtasks are independent | Cost and result aggregation |
| Orchestrator–workers | Task decomposition is not known in advance | Extra latency and coordination errors |
| Evaluator–optimizer | Quality criteria can be stated clearly | Endless or expensive revision loops |

The core design choice is who controls the flow: code, an LLM, or a deliberate combination of both.

## 4. Tools and reliability

Tool descriptions form part of the model's decision surface. State when to use the tool, define inputs precisely, return compact structured results, and surface errors clearly. Validate arguments, constrain side effects, and require human approval for consequential actions.

For a ten-step workflow with independent 85% step accuracy, the probability that every step succeeds is approximately `0.85 ** 10 = 0.1969`. This simplified calculation illustrates why long agent chains need checkpoints, tests, and recovery paths.

## 5. OpenAI Agents SDK

The practical work uses the lightweight Python SDK and its current primitives:

| Primitive | Role |
|---|---|
| `Agent` | Instructions, model, tools, handoffs, and output contract |
| `Runner` | Executes the agent loop and returns run state/results |
| Function tools | Typed Python functions exposed to the model |
| Agents as tools | A manager retains control while calling specialists |
| Handoffs | A specialist takes control of the conversation |
| Guardrails | Input/output validation boundaries |
| Tracing | Visibility into model calls, tools, and transfers |

The SDK uses the Responses API by default for OpenAI models. The course model is selected through `OPENAI_MODEL`, with `gpt-5.6-luna` as the teaching default.

Optional notebook 05 keeps `Agent` and `Runner` but supplies an `OpenAIChatCompletionsModel` pointed at a local Ollama endpoint. This demonstrates that the agent framework, model provider, and trace destination are separate choices. The default local trace destination is MLflow; OpenAI trace export is opt-in.

## 6. Model Context Protocol

MCP is an open protocol for connecting AI applications to tools and context sources. Instead of every application building a custom integration for every tool, applications implement an MCP client and tools expose an MCP server.

The stable Python SDK 2.x renamed the high-level `FastMCP` class to `MCPServer`:

```python
from mcp.server import MCPServer

mcp_server = MCPServer("ANLP Course Server")
```

MCP was introduced by Anthropic in November 2024 and donated to the Linux Foundation's Agentic AI Foundation on 9 December 2025. The official Python SDK 2.x is the current stable line and supports the 2026-07-28 protocol revision while negotiating with earlier clients.

## 7. Multi-agent design

Use one agent until specialization creates a clear benefit. Choose agents as tools when one manager should own the final answer. Choose handoffs when a specialist should take over the conversation. Keep each specialist's scope narrow and make transfer criteria explicit.

Framework positioning for this session:

- **OpenAI Agents SDK:** hands-on course framework; Python-first and intentionally small.
- **LangGraph:** graph/state-machine control for explicit, durable workflows.
- **CrewAI:** role-oriented teams and task delegation.
- **Microsoft Agent Framework 1.0:** Microsoft's supported Python/.NET framework, converging work from AutoGen and Semantic Kernel.

## 8. Safety and human control

Treat retrieved text, tool results, and user content as untrusted input. Apply least privilege, validate tool arguments, isolate code execution, log decisions, and require approval before irreversible or high-impact actions. Human-in-the-loop is a system boundary, not a substitute for validation.

## References

- [Anthropic — Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)
- [OpenAI Agents SDK documentation](https://openai.github.io/openai-agents-python/)
- [OpenAI Agents SDK orchestration](https://openai.github.io/openai-agents-python/multi_agent/)
- [OpenAI Agents SDK non-OpenAI models and tracing](https://openai.github.io/openai-agents-python/models/)
- [Ollama OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility)
- [MLflow tracing for the OpenAI Agents SDK](https://mlflow.org/docs/latest/genai/tracing/integrations/listing/openai-agent/)
- [MCP Python SDK v2](https://github.com/modelcontextprotocol/python-sdk)
- [MCP Python SDK: what's new in v2](https://github.com/modelcontextprotocol/python-sdk/blob/main/docs/whats-new.md)
- [Anthropic — MCP donation and AAIF](https://www.anthropic.com/news/donating-the-model-context-protocol-and-establishing-of-the-agentic-ai-foundation)
- [Microsoft Agent Framework](https://learn.microsoft.com/en-us/agent-framework/)
- [Microsoft Agent Framework 1.0 announcement](https://devblogs.microsoft.com/agent-framework/microsoft-agent-framework-at-build-2026-announce/)
